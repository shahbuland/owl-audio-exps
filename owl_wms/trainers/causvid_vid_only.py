import torch
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

from .base import BaseTrainer
from ..configs import Config
from ..utils import versatile_load, freeze, unfreeze, Timer
from ..models import get_model_cls
from ..utils.owl_vae_bridge import get_decoder_only, make_batched_decode_fn, make_batched_audio_decode_fn
from ..utils import batch_permute_to_length
from ..nn.kv_cache import KVCache
from ..utils.logging import LogHelper, to_wandb_av
from ..data import get_loader
from ..sampling import get_sampler_cls

from copy import deepcopy
from ema_pytorch import EMA
import einops as eo
from contextlib import nullcontext
import random
import wandb
import gc
from pathlib import Path

# === ROLLOUTS ===

def zlerp(x, t):
    z = torch.randn_like(x)
    return x * (1. - t) + z * t

def zlerp_batched(x, t):
    z = torch.randn_like(x)
    t = t[:,:,None,None,None]
    return x * (1. - t) + z * t

def lerp_batched(x, z, t):
    t = t[:,:,None,None,None]
    return x * (1. - t) + z * t

class RolloutManager:
    def __init__(self, model_cfg, min_rollout_frames=8, rollout_steps = 1):
        self.model_cfg = model_cfg
        self.min_rollout_frames = min_rollout_frames
        self.rollout_steps = rollout_steps
        self.noise_prev = 0.2
    
    def get_rollouts(
        self, 
        model,
        video,
        mouse,
        btn
    ):

        with torch.no_grad():
            # Step 1: Pick half of each video to be generated
            gen_mask = torch.rand(video.shape[0], video.shape[1]) < 0.5
            gen_mask = gen_mask.to(device = video.device, dtype = torch.bool)

            ts = torch.rand(video.shape[0], video.shape[1]) * 0.96 + 0.02 # U(0.02, 0.98)
            ts = ts.to(device = video.device, dtype = video.dtype)

            orig_video = video.clone() # Keep the clean video
            noisy_video = zlerp_batched(video, ts)

        v_pred = model(
            noisy_video,
            ts,
            mouse,
            btn
        )

        return noisy_video - v_pred*ts[:,:,None,None,None], mouse, btn

# === LOSSES ===

def get_critic_loss(
    student, critic,
    video,
    mouse,
    btn,
    rollout_manager
):
    # Get rollout
    with torch.no_grad():
        video, mouse, btn = rollout_manager.get_rollouts(
            model = student,
            video = video,
            mouse = mouse,
            btn = btn
        )

        # Get ts ~ U(0.02, 0.98)
        ts = torch.rand(video.shape[0], video.shape[1]) * 0.96 + 0.02
        ts = ts.to(video.device, video.dtype)

        noise = torch.randn_like(video)
        noisy_vid = lerp_batched(video, noise, ts)
        target_vid = (noise - video)

    pred_vid = critic(
        noisy_vid,
        ts,
        mouse,
        btn
    )

    vid_loss = F.mse_loss(pred_vid, target_vid)
    return vid_loss

def get_dmd_loss(
    student, critic, teacher,
    video,
    mouse,
    btn,
    rollout_manager,
    cfg_scale = 1.5
):
    # Get rollout
    video, mouse, btn = rollout_manager.get_rollouts(
        model = student,
        video = video,
        mouse = mouse,
        btn = btn
    )
    with torch.no_grad():
        # Get ts ~ U(0.02, 0.98)
        ts = torch.rand(video.shape[0], video.shape[1]) * 0.96 + 0.02
        ts = ts.to(video.device, video.dtype)

        # Get noise
        noise = torch.randn_like(video)
        noisy_vid = lerp_batched(video, noise, ts)
        
        # Get velocities from teacher
        if cfg_scale != 1.0:
            pred_vid_uncond = teacher(
                noisy_vid,
                ts,
                torch.zeros_like(mouse),
                torch.zeros_like(btn)
            )

        pred_vid_cond = teacher(
            noisy_vid,
            ts,
            mouse,
            btn
        )
        
        if cfg_scale != 1.0:
            v_teacher_vid = pred_vid_uncond + cfg_scale * (pred_vid_cond - pred_vid_uncond)
        else:
            v_teacher_vid = pred_vid_cond

        # Velocities from critic
        v_critic_vid = critic(
            noisy_vid,
            ts,
            mouse,
            btn
        )

        # Get predictions mu_real, mu_fake
        mu_teacher_vid = noisy_vid - ts[:,:,None,None,None] * v_teacher_vid
        mu_critic_vid = noisy_vid - ts[:,:,None,None,None] * v_critic_vid
        
        # Get normalizers
        vid_normalizer = torch.abs(video - mu_teacher_vid).mean(dim=[1,2,3,4],keepdim=True)

        # Get gradients
        grad_vid = (mu_critic_vid - mu_teacher_vid) / vid_normalizer
        grad_vid = torch.nan_to_num(grad_vid, nan=0.0)

        # Get targets
        target_vid = (video.double() - grad_vid.double()).detach()

    # Get losses
    vid_loss = 0.5 * F.mse_loss(
        video.double(),
        target_vid,
        reduction = 'mean'
    )
    return vid_loss

class CausVidTrainer(BaseTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Ensure causal model and no cfg
        self.model_cfg.cfg_prob = 0.0
        self.model_cfg.causal = True

        # === init teacher ===
        teacher_cfg_path = self.train_cfg.teacher_cfg
        teacher_ckpt_path = self.train_cfg.teacher_ckpt
        teacher_cfg = Config.from_yaml(teacher_cfg_path).model
        teacher_ckpt = versatile_load(teacher_ckpt_path)

        self.teacher = get_model_cls(teacher_cfg.model_id)(teacher_cfg)
        try:
            self.teacher.load_state_dict(teacher_ckpt)
        except:
            self.teacher.core.load_state_dict(teacher_ckpt)
    
        # === init student (and fake score fn) ===
        student_ckpt_path = self.train_cfg.student_ckpt
        student_ckpt = versatile_load(student_ckpt_path)

        self.student = get_model_cls(self.model_cfg.model_id)(self.model_cfg)
        try:
            self.student.load_state_dict(student_ckpt)
        except:
            self.student.core.load_state_dict(student_ckpt)

        self.critic = deepcopy(self.student)

        # All models should be cores only
        # (idiosyncracy with model impls)
        self.teacher = self.teacher.core
        self.student = self.student.core
        self.critic = self.critic.core

        # Print model size for logging
        if self.rank == 0:
            n_params = sum(p.numel() for p in self.student.parameters())
            print(f"Model has {n_params:,} parameters")

        # Initialize parameters for organizations sake
        self.ema = None
        self.opt = None
        self.critic_opt = None
        self.scaler = None
        self.critic_scaler = None
        self.total_step_counter = 0

        # === decoders for sampling ===
        self.decoder = get_decoder_only(
            None,
            self.train_cfg.vae_cfg_path,
            self.train_cfg.vae_ckpt_path
        )
    
    def save(self):
        save_dict = {
            'model' : self.student.state_dict(),
            'ema' : self.ema.state_dict(),
            'opt' : self.opt.state_dict(),
            'scaler' : self.scaler.state_dict(),
            'critic' : self.critic.state_dict(),
            'critic_opt' : self.critic_opt.state_dict(),
            'critic_scaler' : self.critic_scaler.state_dict(),
        }
        super().save(save_dict)

    def load(self):
        has_ckpt = False
        if hasattr(self.train_cfg, 'resume_ckpt') and self.train_cfg.resume_ckpt is not None:
            save_dict = super().load(self.train_cfg.resume_ckpt)
            has_ckpt = True
        
        if not has_ckpt:
            return
        
        self.student.load_state_dict(save_dict['model'])
        self.ema.load_state_dict(save_dict['ema'])
        self.opt.load_state_dict(save_dict['opt'])
        self.scaler.load_state_dict(save_dict['scaler'])
        self.critic.load_state_dict(save_dict['critic'])
        self.critic_opt.load_state_dict(save_dict['critic_opt'])
        self.critic_scaler.load_state_dict(save_dict['critic_scaler'])
        self.total_step_counter = save_dict['steps']

    def train(self):
        torch.cuda.set_device(self.local_rank)

        # Inference only modules are frozen eval+cuda+bf16
        self.teacher = self.teacher.cuda().eval().bfloat16()
        self.decoder = self.decoder.cuda().eval().bfloat16()

        # Training modules are train+cuda
        self.student = self.student.cuda().train()
        self.critic = self.critic.cuda().train()

        # DDP
        if self.world_size > 1:
            self.student = DDP(self.student, find_unused_parameters=True)
            self.critic = DDP(self.critic, find_unused_parameters=True)

        # Ema model
        self.ema = EMA(
            self.student,
            beta = 0.99,
            update_after_step = 0,
            update_every = 1
        )
        ema_module = lambda: self.ema.ema_model.module if self.world_size > 1 else self.ema.ema_model
        
        # Prepare decode functions for sampling
        frame_decode_fn = make_batched_decode_fn(self.decoder, batch_size=self.train_cfg.vae_batch_size)

        # Initialize optimizers and scalers and amp context
        self.opt = getattr(torch.optim, self.train_cfg.opt)(self.student.parameters(), **self.train_cfg.opt_kwargs)
        self.critic_opt = getattr(torch.optim, self.train_cfg.opt)(self.critic.parameters(), **self.train_cfg.d_opt_kwargs)
        self.scaler = torch.amp.GradScaler()
        self.critic_scaler = torch.amp.GradScaler()
        ctx = torch.amp.autocast('cuda', torch.bfloat16)

        self.load()

        # Logging helpers
        timer = Timer()
        timer.reset()
        metrics = LogHelper()
        if self.rank == 0:
            wandb.watch(self.student, log = 'all')
        
        # Dataset and sampling prep
        loader = get_loader(self.train_cfg.data_id, self.train_cfg.batch_size, **self.train_cfg.data_kwargs)
        loader = iter(loader)
        sample_loader = get_loader(self.train_cfg.sample_data_id, self.train_cfg.batch_size, **self.train_cfg.sample_data_kwargs)
        sample_loader = iter(sample_loader)

        # Sampler
        sampler = get_sampler_cls(self.train_cfg.sampler_id)(**self.train_cfg.sampler_kwargs)
        rollout_manager = RolloutManager(self.model_cfg, self.train_cfg.min_rollout_frames, self.train_cfg.rollout_steps)

        # Gradient accumulation setup
        accum_steps = self.train_cfg.target_batch_size // self.train_cfg.batch_size // self.world_size
        accum_steps = max(1, accum_steps)

        # optimizer step
        def optimizer_step(model, scaler, optimizer):
            scaler.unscale_(optimizer)
            g_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            optimizer.zero_grad(set_to_none=True)
            scaler.update()
            return g_norm

        # simplify getting batches
        def get_batch():
            vid, mouse, btn = next(loader)
            vid = vid / self.train_cfg.vae_scale
            return vid.to(self.device), mouse.to(self.device), btn.to(self.device)

        # === training loop ===
        while True:
            freeze(self.student)
            unfreeze(self.critic)

            for _ in range(self.train_cfg.update_ratio):
                for _ in range(accum_steps):
                    vid, mouse, btn = get_batch()
                    
                    with ctx:
                        critic_loss = get_critic_loss(
                            student = self.student,
                            critic = self.critic,
                            video = vid,
                            mouse = mouse,
                            btn = btn,
                            rollout_manager = rollout_manager
                        ) / accum_steps
                        metrics.log('critic_loss', critic_loss)
                        self.critic_scaler.scale(critic_loss).backward()

                optimizer_step(self.critic, self.critic_scaler, self.critic_opt)

            freeze(self.critic)
            unfreeze(self.student)

            for _ in range(accum_steps):
                vid, mouse, btn = get_batch()
                
                with ctx:
                    dmd_loss = get_dmd_loss(
                        student = self.student,
                        critic = self.critic,
                        teacher = self.teacher,
                        video = vid,
                        mouse = mouse,
                        btn = btn,
                        rollout_manager = rollout_manager
                    ) / accum_steps
                    metrics.log('dmd_loss', dmd_loss)
                    self.scaler.scale(dmd_loss).backward()

                g_norm = optimizer_step(self.student, self.scaler, self.opt)
                metrics.log('g_norm', g_norm)

            with torch.no_grad():
                # Logging
                self.ema.update()
                wandb_dict = metrics.pop()
                wandb_dict['time'] = timer.hit()
                timer.reset()

                # Sampling
                if self.total_step_counter % self.train_cfg.sample_interval == 0:
                    with ctx:
                        eval_wandb_dict = self.eval_step(sample_loader, sampler, frame_decode_fn)
                        gc.collect()
                        torch.cuda.empty_cache()
                        if self.rank == 0:
                            wandb_dict.update(eval_wandb_dict)

                if self.rank == 0:
                    wandb.log(wandb_dict)

                self.total_step_counter += 1
                if self.total_step_counter % self.train_cfg.save_interval == 0:
                    if self.rank == 0:
                        self.save()

    @torch.no_grad()
    def eval_step(self, sample_loader, sampler, decode_fn = None):
        model = self.ema.ema_model.module if self.world_size > 1 else self.ema.ema_model

        """
        In order to get nice samples to draw,
        We take many many samples then take controls from them.
        But we only care about the first video
        """
        vid, mouse, btn = next(sample_loader)

        mouses = [mouse]
        btns = [btn]
        for _ in range(15):
            _, new_mouse, new_btn = next(sample_loader)
            mouses.append(new_mouse)
            btns.append(new_btn)

        mouses = torch.cat(mouses, dim=0)
        btns = torch.cat(btns, dim=0)

        mouse, btn = batch_permute_to_length(mouses, btns, sampler.num_frames + vid.size(1))
        mouse = mouse[:1] # First batch element
        btn = btn[:1] # First batch element

        latent_vid = sampler(model, vid, mouse, btn)

        if self.sampler_only_return_generated:
            latent_vid = latent_vid[:, vid.size(1):]
            mouse = mouse[:, vid.size(1):]
            btn = btn[:, vid.size(1):]

        video_out = decode_fn(latent_vid * self.train_cfg.vae_scale) if decode_fn is not None else None

        gc.collect()
        torch.cuda.empty_cache()

        def gather_concat_cpu(t, dim=0):
            if self.rank == 0:
                parts = [t.cpu()]
                scratch = torch.empty_like(t)
                for src in range(self.world_size):
                    if src == 0:
                        continue
                    dist.recv(scratch, src=src)
                    parts.append(scratch.cpu())
                return torch.cat(parts, dim=dim)
            else:
                dist.send(t, dst=0)
                return None

        # ---- Save Latent Artifacts ----
        if getattr(self.train_cfg, "eval_sample_dir", None):
            latent_vid = gather_concat_cpu(latent_vid)
            if self.rank == 0:
                eval_dir = Path(self.train_cfg.eval_sample_dir)
                eval_dir.mkdir(parents=True, exist_ok=True)
                torch.save(latent_vid, eval_dir / f"vid.{self.total_step_counter}.pt")
        
        # ---- Generate Media Artifacts ----
        video_out, mouse, btn = [
            gather_concat_cpu(x, dim=0) for x in [video_out, mouse, btn]
        ]

        if self.rank == 0:
            wandb_av_out = to_wandb_av(video_out, None, mouse, btn)
            if len(wandb_av_out) == 3:
                video, depth_gif, flow_gif = wandb_av_out
                eval_wandb_dict = dict(samples=video, depth_gif=depth_gif, flow_gif=flow_gif)
            elif len(wandb_av_out) == 2:
                video, depth_gif = wandb_av_out
                eval_wandb_dict = dict(samples=video, depth_gif=depth_gif)
            else:
                eval_wandb_dict = dict(samples=wandb_av_out)
        else:
            eval_wandb_dict = None
        self.barrier()

        return eval_wandb_dict
