from ema_pytorch import EMA
from pathlib import Path
import tqdm
import wandb
import gc
import re

import torch
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

from .base import BaseTrainer

from ..utils import freeze, Timer
from ..schedulers import get_scheduler_cls
from ..models import get_model_cls
from ..sampling import get_sampler_cls
from ..data import get_loader
from ..utils.logging import LogHelper, to_wandb_samples
from ..utils.owl_vae_bridge import get_audio_encoder_decoder, make_batched_audio_encode_fn, make_batched_audio_decode_fn


class AudioRFTTrainer(BaseTrainer):
    """
    Trainer for audio rectified flow transformer
    
    Loads raw waveforms [b,88200,2] and encodes them to latents [b,120,32] 
    for training the diffusion model.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        model_id = self.model_cfg.model_id
        self.model = get_model_cls(model_id)(self.model_cfg).train()

        # Print model size
        if self.rank == 0:
            n_params = sum(p.numel() for p in self.model.parameters())
            print(f"Model has {n_params:,} parameters")

        self.ema = None
        self.opt = None
        self.scheduler = None

        self.total_step_counter = 0
        
        # Load both encoder and decoder for audio VAE
        self.encoder, self.decoder = get_audio_encoder_decoder(
            self.train_cfg.vae_cfg_path,
            self.train_cfg.vae_ckpt_path
        )
        
        # Freeze VAE components
        freeze(self.encoder)
        freeze(self.decoder)

    @staticmethod
    def get_raw_model(model):
        return getattr(model, "module", model)

    def save(self):
        if self.rank != 0:
            return

        save_dict = {
            'model': self.get_raw_model(self.model).state_dict(),
            'ema': self.get_raw_model(self.ema).state_dict(),
            'opt': self.opt.state_dict(),
            'steps': self.total_step_counter
        }
        if self.scheduler is not None:
            save_dict['scheduler'] = self.scheduler.state_dict()
        super().save(save_dict)

    def load(self) -> None:
        """Build runtime objects and optionally restore a checkpoint."""
        # ----- model & helpers -----
        ckpt = getattr(self.train_cfg, "resume_ckpt", None)
        state = None
        if ckpt:
            state = super().load(ckpt)

            # Allow legacy checkpoints: strip module and _orig_mod
            pat = r'^(?:(?:_orig_mod\.|module\.)+)?([^.]+\.)?(?:(?:_orig_mod\.|module\.)+)?'
            state["model"] = {re.sub(pat, r'\1', k): v for k, v in state["model"].items()}
            state["ema"] = {re.sub(pat, r'\1', k): v for k, v in state["ema"].items()}

            self.model.load_state_dict(state["model"], strict=True)
            self.total_step_counter = state.get("steps", 0)

        self.model = self.model.cuda()
        if self.world_size > 1:
            self.model = DDP(self.model, device_ids=[self.local_rank])
        else:
            self.model = self.model
        self.model = torch.compile(self.model)

        # Setup VAE encoder and decoder
        self.encoder = self.encoder.cuda().eval().bfloat16()
        self.decoder = self.decoder.cuda().eval().bfloat16()
        
        # Compile VAE functions for speed
        self.encode_fn = torch.compile(make_batched_audio_encode_fn(self.encoder, self.train_cfg.vae_batch_size))
        self.decode_fn = torch.compile(make_batched_audio_decode_fn(self.decoder, self.train_cfg.vae_batch_size, max_seq_len=120))

        # ----- EMA, optimiser, scheduler -----
        self.ema = EMA(self.model, beta=0.999, update_after_step=0, update_every=1)

        if self.train_cfg.opt.lower() == "muon":
            from ..muon import init_muon
            self.opt = init_muon(self.model, rank=self.rank, world_size=self.world_size, **self.train_cfg.opt_kwargs)
        else:
            self.opt = getattr(torch.optim, self.train_cfg.opt)(self.model.parameters(), **self.train_cfg.opt_kwargs)

        if self.train_cfg.scheduler:
            sched_cls = get_scheduler_cls(self.train_cfg.scheduler)
            self.scheduler = sched_cls(self.opt, **self.train_cfg.scheduler_kwargs)

        # ----- optional checkpoint restore -----
        if ckpt:
            self.ema.load_state_dict(state["ema"])
            self.opt.load_state_dict(state["opt"])
            if self.scheduler and "scheduler" in state:
                self.scheduler.load_state_dict(state["scheduler"])

        del state

    @torch.no_grad()
    def update_buffer(self, name: str, value: torch.Tensor, value_ema: torch.Tensor | None = None):
        """Set the buffer `name` (e.g. 'core.transformer.foo') across ranks and EMA."""
        online = self.model.module if isinstance(self.model, DDP) else self.model
        buf_online = online.get_buffer(name)
        buf_ema = self.ema.ema_model.get_buffer(name)

        if self.rank == 0:
            buf_online.copy_(value.to(buf_online))
        if self.world_size > 1:
            dist.broadcast(buf_online, 0)

        buf_ema.copy_(buf_online)

    def train(self):
        torch.cuda.set_device(self.local_rank)
        print(f"Device used: rank={self.rank}")

        # Grad accum setup and scaler
        accum_steps = self.train_cfg.target_batch_size // self.train_cfg.batch_size // self.world_size
        accum_steps = max(1, accum_steps)
        ctx = torch.amp.autocast('cuda', torch.bfloat16)

        self.load()

        # Timer reset
        timer = Timer()
        timer.reset()
        metrics = LogHelper()

        if self.rank == 0:
            wandb.watch(self.get_module(), log='all')

        # Dataset setup - loads raw waveforms
        loader = get_loader(
            self.train_cfg.data_id,
            self.train_cfg.batch_size,
            **self.train_cfg.data_kwargs
        )

        # For sampling
        n_samples = (self.train_cfg.n_samples + self.world_size - 1) // self.world_size
        sample_loader = get_loader(
            self.train_cfg.data_id,
            n_samples,
            **self.train_cfg.data_kwargs
        )
        sample_loader = iter(sample_loader)
        
        # Initialize audio sampler
        sampler = get_sampler_cls(self.train_cfg.sampler_id)(**self.train_cfg.sampler_kwargs)

        local_step = 0
        for epoch in range(self.train_cfg.epochs):
            for batch in tqdm.tqdm(loader, total=len(loader), disable=self.rank != 0, desc=f"Epoch: {epoch}"):
                # batch is raw waveforms [b, 88200, 2]
                waveforms = batch.cuda()
                
                # Encode waveforms to latents [b, 120, 32]
                with torch.no_grad():
                    audio_latents = self.encode_fn(waveforms.bfloat16()).to(waveforms.dtype)
                    audio_latents = audio_latents / self.train_cfg.vae_scale

                with ctx:
                    # Train diffusion model on latents
                    loss = self.model(audio_latents)
                    loss = loss / accum_steps
                    loss.backward()

                metrics.log('diffusion_loss', loss)

                local_step += 1
                if local_step % accum_steps == 0:

                    # Optimizer updates
                    if self.train_cfg.opt.lower() != "muon":
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)
                    self.opt.step()
                    self.opt.zero_grad(set_to_none=True)

                    if self.scheduler is not None:
                        self.scheduler.step()
                    self.ema.update()

                    # Do logging
                    with torch.no_grad():
                        wandb_dict = metrics.pop()
                        wandb_dict['time'] = timer.hit()
                        timer.reset()

                        # Audio sampling
                        if self.total_step_counter % self.train_cfg.sample_interval == 0:
                            with ctx:
                                eval_wandb_dict = self.eval_step(sample_loader, sampler)
                                gc.collect()
                                torch.cuda.empty_cache()
                                if self.rank == 0:
                                    wandb_dict.update(eval_wandb_dict)

                        if self.rank == 0:
                            wandb.log(wandb_dict)

                    self.total_step_counter += 1
                    if self.total_step_counter % self.train_cfg.save_interval == 0:
                        self.save()

                    self.barrier()

    def _gather_concat_cpu(self, t: torch.Tensor, dim: int = 0):
        """Gather *t* from every rank onto rank 0 and return concatenated copy."""
        if self.world_size == 1:
            return t.cpu()
        if self.rank == 0:
            parts = [t.cpu()]
            scratch = torch.empty_like(t)
            for src in range(1, self.world_size):
                dist.recv(scratch, src=src)
                parts.append(scratch.cpu())
            return torch.cat(parts, dim=dim)
        dist.send(t, dst=0)

    def eval_step(self, sample_loader, sampler):
        """Audio sampling evaluation"""
        ema_model = self.get_module(ema=True).core

        # Get sample audio waveforms for context
        sample_waveforms = next(sample_loader).cuda()
        
        # Encode to latents for initial context
        with torch.no_grad():
            sample_latents = self.encode_fn(sample_waveforms) / self.train_cfg.vae_scale
        
        # Generate audio using the sampler (returns full sequence: context + generated)
        generated_latents, full_generated_waveforms = sampler(
            ema_model, 
            sample_latents,
            decode_fn=self.decode_fn,
            vae_scale=self.train_cfg.vae_scale
        )
        
        # Decode original context for comparison (ground truth)
        with torch.no_grad():
            context_waveforms = self.decode_fn(sample_latents * self.train_cfg.vae_scale)
        
        # Gather results across ranks
        full_generated_waveforms = self._gather_concat_cpu(full_generated_waveforms)
        context_waveforms = self._gather_concat_cpu(context_waveforms)
        
        eval_wandb_dict = {}
        if self.rank == 0:
            # Log shapes and basic info
            eval_wandb_dict['generated_audio_samples'] = full_generated_waveforms.shape[0]
            eval_wandb_dict['context_length'] = context_waveforms.shape[1]
            eval_wandb_dict['full_length'] = full_generated_waveforms.shape[1]
            eval_wandb_dict['generated_length'] = full_generated_waveforms.shape[1] - context_waveforms.shape[1]
            
            # Log audio samples to wandb
            # Convert to numpy and log first sample from batch
            context_np = context_waveforms[0].float().cpu().numpy()  # [n_samples, 2]
            generated_np = full_generated_waveforms[0].float().cpu().numpy()  # [full_length, 2]
            
            # Log stereo audio (wandb.Audio handles stereo arrays)
            eval_wandb_dict['context_audio'] = wandb.Audio(context_np, sample_rate=44100)
            eval_wandb_dict['generated_audio'] = wandb.Audio(generated_np, sample_rate=44100)
        
        return eval_wandb_dict