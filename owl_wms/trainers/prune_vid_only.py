"""
TODO this doesnt work right now
"""

import torch
from torch import nn
import torch.nn.functional as F
import einops as eo
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import gc
import random
import wandb
from ema_pytorch import EMA

from ..nn.rope import cast_rope_buffers_to_fp32
from ..data import get_loader
from ..models import get_model_cls
from ..muon import init_muon
from ..utils import Timer, freeze, unfreeze, versatile_load
from ..utils import batch_permute_to_length
from ..utils.logging import LogHelper, to_wandb, to_wandb_av
from .base import BaseTrainer
from ..configs import Config
from ..sampling import get_sampler_cls
from ..sampling.schedulers import get_sd3_euler
from ..utils.owl_vae_bridge import get_decoder_only, make_batched_decode_fn

class SoftResetIterator:
    """
    Wraps an iterable (e.g., DataLoader) so that when exhausted,
    it automatically resets and continues yielding items.
    """
    def __init__(self, iterable):
        self._iterable = iterable
        self._reset_iter()

    def _reset_iter(self):
        self._iterator = iter(self._iterable)

    def __iter__(self):
        return self

    def __next__(self):
        try:
            return next(self._iterator)
        except StopIteration:
            self._reset_iter()
            return next(self._iterator)

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

class DistillODETrainer(BaseTrainer):
    """
    Trainer for distilling a teacher WM into some student WM.
    New config values needed:
    - teacher_ckpt
    - teacher_cfg
    - cfg_scale (when getting teacher trajectories for distillation)
    - subsample (how many solution pairs to actually regress on)
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Load the teacher model
        teacher_ckpt_path = self.train_cfg.teacher_ckpt
        teacher_cfg_path = self.train_cfg.teacher_cfg
        teacher_ckpt = versatile_load(teacher_ckpt_path)
        teacher_cfg = Config.from_yaml(teacher_cfg_path).model
        teacher = get_model_cls(teacher_cfg.model_id)(teacher_cfg)
        teacher.load_state_dict(teacher_ckpt)
        self.teacher = teacher.core.to(self.device).bfloat16()

        cast_rope_buffers_to_fp32(self.teacher)
    
        # Instantiate student
        student_cfg = self.model_cfg
        self.student = get_model_cls(student_cfg.model_id)(student_cfg)
        self.load_teacher_into_student(self.student, teacher_ckpt, teacher_cfg, student_cfg)
        self.student = self.student.core.to(self.device).train()

        # Model size and init everything else
        if self.rank == 0:
            model_params = sum(p.numel() for p in self.student.parameters())
            print(f"Student model parameters: {model_params:,}")

        self.opt = None
        self.scaler = None
        self.total_step_counter = 0
        self.ema = None

        self.decoder = get_decoder_only(
            self.train_cfg.vae_id,
            self.train_cfg.vae_cfg_path,
            self.train_cfg.vae_ckpt_path
        )

        freeze(self.decoder)
        self.decoder = self.decoder.bfloat16().to(self.device).eval()

    def load_teacher_into_student(self, student_model, teacher_state_dict, teacher_cfg, student_cfg):
        """
        Improved weight-transfer from teacher → student.

        1. Handles 'module.' / 'core.' prefixes in the teacher checkpoint.
        2. Copies projection / embedding / final layers verbatim.
        3. Copies a subset of transformer blocks such that the first and last
           blocks are retained and the intermediate ones are chosen with
           uniform spacing.
        4. Works for both DiffusionDecoder (wrapper) and DiffusionDecoderCore.
        """

        # ── 0. get the real core we want to load into ────────────────────────────
        student_core = student_model.core if hasattr(student_model, "core") else student_model

        # ── 1. clean the teacher state-dict prefixes ────────────────────────────
        clean_teacher = {}
        for k, v in teacher_state_dict.items():
            if k.startswith("module."):
                k = k[len("module."):]
            if k.startswith("core."):
                k = k[len("core."):]
            clean_teacher[k] = v

        # ── 2. determine transformer depth of teacher & student ────────────────
        n_teacher = getattr(teacher_cfg, "n_layers", None)
        n_student = getattr(student_cfg, "n_layers", None)
        if n_teacher is None or n_student is None:
            raise ValueError("Both cfgs must expose `n_layers`")

        # Map indices: always keep first & last, interpolate the rest
        if n_student == 1:
            index_map = {0: 0}
        else:
            t_ids = [
                round(i * (n_teacher - 1) / (n_student - 1))
                for i in range(n_student)
            ]
            index_map = {s_idx: t_idx for s_idx, t_idx in enumerate(t_ids)}

        new_state_dict = {}

        # ── 3. copy projection / embedding / final layers verbatim ─────────────
        for k, v in clean_teacher.items():
            if not k.startswith("transformer.blocks."):
                new_state_dict[k] = v

        # ── 4. copy the selected transformer blocks ────────────────────────────
        for s_idx, t_idx in index_map.items():
            t_pref = f"transformer.blocks.{t_idx}."
            s_pref = f"transformer.blocks.{s_idx}."
            for k, v in clean_teacher.items():
                if k.startswith(t_pref):
                    new_state_dict[s_pref + k[len(t_pref):]] = v

        # ── 5. load into the student core ──────────────────────────────────────
        student_core.load_state_dict(new_state_dict, strict=True)

    def get_ema_core(self):
        if self.world_size > 1:
            return self.ema.ema_model.module
        return self.ema.ema_model

    def save(self):
        save_dict = {
            'model': self.student.state_dict(),
            'ema': self.ema.state_dict(),
            'opt': self.opt.state_dict(),
            'scaler': self.scaler.state_dict(),
            'steps': self.total_step_counter
        }
        super().save(save_dict)

    def load(self):
        resume_ckpt = getattr(self.train_cfg, 'resume_ckpt', None)
        if resume_ckpt is None:
            return
        save_dict = super().load(resume_ckpt)
        self.student.load_state_dict(save_dict['model'], strict=False)
        self.ema.load_state_dict(save_dict['ema'])
        self.opt.load_state_dict(save_dict['opt'])
        self.scaler.load_state_dict(save_dict['scaler'])
        self.total_step_counter = save_dict['steps']

    def get_ema_core(self):
        if self.world_size > 1:
            return self.ema.ema_model.module
        return self.ema.ema_model

    def train(self):
        torch.cuda.set_device(self.local_rank)

        self.cfg_scale = self.train_cfg.cfg_scale
        self.subsample = self.train_cfg.subsample

        # Prepare model, ema
        self.student = self.student.to(self.device).train()
        if self.world_size > 1:
            self.student = DDP(self.student)

        #self.teacher = torch.compile(self.teacher, mode='max-autotune', dynamic=False, fullgraph=True)

        self.ema = EMA(
            self.student,
            beta=0.999,
            update_after_step=0,
            update_every=1
        )

        # Optimizer
        if self.train_cfg.opt.lower() == "muon":
            self.opt = init_muon(self.student, rank=self.rank, world_size=self.world_size, **self.train_cfg.opt_kwargs)
        else:
            opt_cls = getattr(torch.optim, self.train_cfg.opt)
            self.opt = opt_cls(self.student.parameters(), **self.train_cfg.opt_kwargs)

        self.scaler = torch.amp.GradScaler()
        self.ctx = torch.amp.autocast(self.device, torch.bfloat16)

        self.accum_steps = self.train_cfg.target_batch_size // self.train_cfg.batch_size // self.world_size
        self.accum_steps = max(1, self.accum_steps)

        decode_fn = make_batched_decode_fn(self.decoder, self.train_cfg.vae_batch_size)

        # Dataset setup
        loader = get_loader(self.train_cfg.data_id, self.train_cfg.batch_size, **self.train_cfg.data_kwargs)
        loader = SoftResetIterator(loader)

        sample_loader = get_loader(self.train_cfg.sample_data_id, self.train_cfg.n_samples, **self.train_cfg.sample_data_kwargs)
        sample_loader = SoftResetIterator(sample_loader)
        
        sampler = get_sampler_cls(self.train_cfg.sampler_id)(**self.train_cfg.sampler_kwargs)

        # Timer and logging
        timer = Timer()
        timer.reset()
        metrics = LogHelper()
        if self.rank == 0:
            wandb.watch(self.student, log='all')

        self.load()

        local_step = 0
        input_shape = (
            self.model_cfg.channels,
            self.model_cfg.sample_size,
            self.model_cfg.sample_size,
        )

        def get_dummy(z):
            return torch.randn(z.shape[0],z.shape[1],*input_shape, device=z.device, dtype=z.dtype)

        @torch.no_grad()
        def sample_with_teacher(vid, mouse, btn, n_steps=self.train_cfg.rollout_steps, subsample=self.train_cfg.subsample, gen_p=self.train_cfg.gen_p):
            """
            Sample using teacher (video only, no cond/uncond mask, just zeros_like controls)
            """
            alpha = 0.2
            gen_mask = torch.rand(vid.shape[0], vid.shape[1], device=vid.device, dtype=vid.dtype) < gen_p
            noisy_video = torch.where(
                gen_mask[:,:,None,None,None],
                torch.randn_like(vid),
                zlerp(vid, alpha)
            )
            ts = torch.where(
                gen_mask[:,:,None,None,None],
                torch.ones_like(vid[:,:,0,0,0]),
                torch.ones_like(vid[:,:,0,0,0]) * alpha
            )

            dt_list = get_sd3_euler(n_steps)
            t = torch.ones(mouse.shape[0], mouse.shape[1], device=mouse.device, dtype=mouse.dtype)

            video_inputs = []
            mouse_inputs = []
            btn_inputs = []
            ts = []
            video_outputs = []
            gen_masks = []

            zero_mouse = torch.zeros_like(mouse)
            zero_btn = torch.zeros_like(btn)

            for dt in dt_list:
                pred_video_uncond = self.teacher(noisy_video, t, zero_mouse, zero_btn)
                pred_video_cond = self.teacher(noisy_video, t, mouse, btn)

                pred_video = pred_video_uncond + self.cfg_scale * (pred_video_cond - pred_video_uncond)

                video_inputs.append(noisy_video.clone()) # b n c h w
                video_outputs.append(pred_video.clone()) # b n c h w
                mouse_inputs.append(mouse.clone())
                btn_inputs.append(btn.clone())
                ts.append(t.clone()) # b n
                gen_masks.append(gen_mask.clone()) # b n

                noisy_video = torch.where(
                    gen_mask[:,:,None,None,None],
                    noisy_video - dt * pred_video,
                    noisy_video
                )
                t = torch.where(
                    gen_mask,
                    t - dt,
                    t
                )

            # Concatenate on batch dim
            video_inputs = torch.cat(video_inputs, dim=0) # (steps b) n c h w
            video_outputs = torch.cat(video_outputs, dim=0)
            mouse_inputs = torch.cat(mouse_inputs, dim=0)
            btn_inputs = torch.cat(btn_inputs, dim=0)
            ts = torch.cat(ts, dim=0) # (steps b) n 
            gen_masks = torch.cat(gen_masks, dim=0) # (steps b) n

            if subsample < 1.0:
                inds = torch.randperm(video_inputs.shape[0])[:int(video_inputs.shape[0] * subsample)]
                video_inputs = video_inputs[inds]
                video_outputs = video_outputs[inds]
                mouse_inputs = mouse_inputs[inds]
                btn_inputs = btn_inputs[inds]
                ts = ts[inds]
                gen_masks = gen_masks[inds]

            return video_inputs, video_outputs, mouse_inputs, btn_inputs, ts, gen_masks

        for (batch_vid, batch_mouse, batch_btn) in loader:
            batch_vid = batch_vid.to(self.device).bfloat16() / self.train_cfg.vae_scale
            batch_mouse = batch_mouse.to(self.device).bfloat16()
            batch_btn = batch_btn.to(self.device).bfloat16()
            
            with self.ctx:
                (
                    video_inputs,
                    video_outputs,
                    mouse_inputs,
                    btn_inputs,
                    ts,
                    gen_masks
                ) = sample_with_teacher(batch_vid, batch_mouse, batch_btn)

                preds_video = self.student(video_inputs, ts, mouse_inputs, btn_inputs)
                gen_masks = gen_masks[:,:,None,None,None]
                loss_video = F.mse_loss(preds_video * gen_masks, video_outputs * gen_masks) / self.accum_steps
                loss = loss_video

            metrics.log('loss', loss)
            self.scaler.scale(loss).backward()

            local_step += 1
            if local_step % self.accum_steps == 0:
                # Updates
                self.scaler.unscale_(self.opt)
                torch.nn.utils.clip_grad_norm_(self.student.parameters(), max_norm=10.0)

                self.scaler.step(self.opt)
                self.opt.zero_grad(set_to_none=True)
                self.scaler.update()
                self.ema.update()

                # Logging and sampling
                with torch.no_grad():
                    wandb_dict = metrics.pop()
                    wandb_dict['time'] = timer.hit()
                    timer.reset()

                # Sampling
                if self.total_step_counter % self.train_cfg.sample_interval == 0:
                    with self.ctx:
                        eval_wandb_dict = self.eval_step(sample_loader, sampler, decode_fn)
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

                self.barrier()

    @torch.no_grad()
    def eval_step(self, sample_loader, sampler, decode_fn=None):
        """
        Evaluation step for ODE regression trainer.
        Follows the pattern of sf_vid_only.py: sample a batch, sample controls, run model, decode, and log.
        """
        model = self.ema.ema_model.module if self.world_size > 1 else self.ema.ema_model
        model = model.eval()

        # Get a batch of data
        vid, mouse, btn = next(sample_loader)
        vid = vid / self.train_cfg.vae_scale

        # Collect multiple mouse/btn for control sampling
        mouses = [mouse]
        btns = [btn]
        for _ in range(15):
            _, new_mouse, new_btn = next(sample_loader)
            mouses.append(new_mouse)
            btns.append(new_btn)

        mouses = torch.cat(mouses, dim=0)
        btns = torch.cat(btns, dim=0)

        # Permute controls to match sample length
        mouse, btn = batch_permute_to_length(
            mouses, btns, sampler.num_frames + vid.size(1)
        )
        mouse = mouse[:vid.size(0)]
        btn = btn[:vid.size(0)]

        # Run the sampler (should output latent video)
        latent_vid = sampler(model, vid.cuda(), mouse.cuda(), btn.cuda())

        # Remove context frames if needed
        if True:
            latent_vid = latent_vid[:, vid.size(1):]
            mouse = mouse[:, vid.size(1):]
            btn = btn[:, vid.size(1):]

        # Decode video if decode_fn is provided
        video_out = decode_fn(latent_vid * self.train_cfg.vae_scale) if decode_fn is not None else None

        gc.collect()
        torch.cuda.empty_cache()

        def gather_concat_cpu(t, dim=0):
            if self.rank == 0:
                t_gpu = t.cuda() if t.device.type == 'cpu' else t
                parts = [t_gpu.cpu()]
                scratch = torch.empty_like(t_gpu)
                for src in range(self.world_size):
                    if src == 0:
                        continue
                    dist.recv(scratch, src=src)
                    parts.append(scratch.cpu())
                return torch.cat(parts, dim=dim)
            else:
                t_gpu = t.cuda() if t.device.type == 'cpu' else t
                dist.send(t_gpu, dst=0)
                return None

        # Save latent samples if requested
        if getattr(self.train_cfg, "eval_sample_dir", None):
            latent_vid = gather_concat_cpu(latent_vid)
            if self.rank == 0:
                eval_dir = Path(self.train_cfg.eval_sample_dir)
                eval_dir.mkdir(parents=True, exist_ok=True)
                torch.save(latent_vid, eval_dir / f"vid.{self.total_step_counter}.pt")

        # Gather outputs for logging
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
        model = model.train()
        return eval_wandb_dict