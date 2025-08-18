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
from ..nn.rope import cast_rope_buffers_to_fp32

from copy import deepcopy
from ema_pytorch import EMA
import einops as eo
from contextlib import nullcontext
import random
import wandb
import gc
from pathlib import Path
import wandb
import contextvars

torch.autograd.set_detect_anomaly(True)

def backward_and_check(model, tensor):
    tensor.mean().backward()
    total = 0
    for n, p in model.named_parameters():
        if p.grad is not None:
            total += (p.grad.abs().sum().item() > 0)
    print(f"{total} parameters received gradient")  
    exit()

@torch.no_grad()
def log_decoded_images(tensor, decode_fn, key, n_images = 16):
    """
    Given a tensor of shape [b, c, h, w], decode and log images to wandb.

    Args:
        tensor: torch.Tensor of shape [b, c, h, w]
        decode_fn: function that takes [b, c, h, w] and returns images in [-1, 1] (float32)
        key: str, wandb log key
    """
    # Flatten tensor from [b, n, c, h, w] to [b * n, c, h, w] if needed
    if tensor.dim() == 5:
        b, n, c, h, w = tensor.shape
        tensor = tensor.reshape(b * n, c, h, w)
    idx = torch.randperm(tensor.shape[0])[:n_images]
    tensor = tensor[idx]
    tensor = tensor.unsqueeze(0)
    with torch.no_grad():
        # Ensure tensor is on cpu and float32
        tensor = tensor.detach().cuda().bfloat16()
        # Decode to images in [-1, 1]
        imgs = decode_fn(tensor).float().cpu().squeeze(0)
        # Clamp to [-1, 1]
        imgs = torch.clamp(imgs, -1, 1)
        # Map to [0, 1]
        imgs = (imgs + 1) / 2
        # Convert to numpy for wandb
        imgs = imgs.permute(0, 2, 3, 1).cpu().numpy()  # [b, h, w, c]
        # If single channel, repeat to 3 channels for visualization
        if imgs.shape[-1] == 1:
            imgs = imgs.repeat(1, 1, 1, 3)
        # Log as a list of wandb.Image
        wandb_imgs = [wandb.Image(img) for img in imgs]
        wandb.log({key: wandb_imgs})

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

class RolloutManager:
    def __init__(self, model_cfg, min_rollout_frames=8, rollout_steps=1):
        self.model_cfg = model_cfg
        self.min_rollout_frames = min_rollout_frames
        self.rollout_steps = rollout_steps

    def get_rollouts(
        self,
        model,
        video,
        mouse,
        btn
    ):

        def get_module():
            if hasattr(model, 'module'):
                return model.module
            else:
                return model

        # === init kv cache ===
        kv_cache = KVCache(self.model_cfg)
        kv_cache.reset(video.shape[0])

        # === init generation params ===
        window_length = video.shape[1]
        rollout_frames = self.min_rollout_frames  # fixed for now, could be randomized if needed
        if dist.is_initialized():
            rollout_frames_tensor = torch.tensor(rollout_frames, device='cuda')
            dist.broadcast(rollout_frames_tensor, src=0)
            rollout_frames = rollout_frames_tensor.item()

        # === extend control inputs for open-ended generation ===
        ext_mouse, ext_btn = batch_permute_to_length(mouse, btn, window_length + rollout_frames)
        rollout_mouse = ext_mouse[:, window_length:]
        rollout_btn = ext_btn[:, window_length:] 

        ts = torch.zeros_like(video[:, :, 0, 0, 0])  # [b, n]

        # === cache context frames ===

        # Warmup call prevents autograd errors (???)
        _ = model(torch.randn_like(video)[:,:1], torch.randn_like(ts)[:,:1], torch.randn_like(mouse)[:,:1], torch.randn_like(btn)[:,:1])

        with torch.no_grad():
            kv_cache.enable_cache_updates()
            model(video, ts, mouse, btn, kv_cache=kv_cache)
            kv_cache.disable_cache_updates()
  
        # Mark context frames as False, generated frames as True
        gen_mask = torch.zeros((video.shape[0], window_length + rollout_frames), dtype=torch.bool, device=video.device)
        gen_mask[:, window_length:] = True

        # === rollout time! ===
        get_module().transformer.enable_decoding()

        for frame_idx in range(rollout_frames):
            ts_1 = torch.ones_like(video[:, 0:1, 0, 0, 0])  # [b,1]
            frame_1 = torch.randn_like(video[:, 0:1])  # [b,1,c,h,w]
            mouse_1 = rollout_mouse[:, frame_idx:frame_idx+1]
            btn_1 = rollout_btn[:, frame_idx:frame_idx+1]

            end_frame = random.randint(1, self.rollout_steps)
            dt = 1. / self.rollout_steps

            for step_idx in range(end_frame):
                early_stop = (step_idx == end_frame - 1)
                
                if early_stop:
                    vid_pred = model(
                        frame_1,
                        ts_1,
                        mouse_1,
                        btn_1,
                        kv_cache=kv_cache
                    )
                    frame_1 = frame_1 - ts_1 * vid_pred
                    ts_1 = torch.zeros_like(ts_1) # -> 0
                    break
                else:
                    with torch.no_grad():
                        vid_pred = model(
                            frame_1,
                            ts_1,
                            mouse_1,
                            btn_1,
                            kv_cache=kv_cache
                        )
                        frame_1 = frame_1 - dt * vid_pred
                        ts_1 = ts_1.clone() - dt

            with torch.no_grad():
                kv_cache.enable_cache_updates()
                model(
                    frame_1,
                    ts_1 * 0,
                    mouse_1,
                    btn_1,
                    kv_cache=kv_cache
                )
                kv_cache.disable_cache_updates()
                kv_cache.truncate(1, front=False) # eject oldest

            video = torch.cat([video, frame_1], dim=1)            

        get_module().transformer.disable_decoding()

        # Only return fixed window of generated frames and grad mask
        video = video[:, -window_length:]
        mouse = ext_mouse[:, -window_length:]
        btn = ext_btn[:, -window_length:]
        grad_mask = gen_mask[:, -window_length:]

        return video, mouse, btn, grad_mask

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
        video, mouse, btn, grad_mask = rollout_manager.get_rollouts(
            model = student,
            video = video,
            mouse = mouse,
            btn = btn
        )

        ts = torch.randn(video.shape[0], video.shape[1], device=video.device, dtype = video.dtype).sigmoid()

        noise = torch.randn_like(video)
        noisy_vid = lerp_batched(video, noise, ts)
        target_vid = (noise - video)

    pred_vid = critic(
        noisy_vid,
        ts,
        mouse,
        btn
    )

    grad_mask_exp = grad_mask[:, :, None, None, None]
    vid_loss = F.mse_loss(pred_vid * grad_mask_exp, target_vid * grad_mask_exp)
    return vid_loss

def get_dmd_loss(
    student, critic, teacher,
    video,
    mouse,
    btn,
    rollout_manager,
    cfg_scale = 1.5,
    decode_fn = None,
    log_predictions = False
):
    # Get rollout
    video, mouse, btn, grad_mask = rollout_manager.get_rollouts(
        model = student,
        video = video,
        mouse = mouse,
        btn = btn
    )
    with torch.no_grad():
        ts = torch.randn(video.shape[0], video.shape[1], device=video.device, dtype = video.dtype).sigmoid()

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
        vid_normalizer = torch.abs(video - mu_teacher_vid).mean(dim=[1,2,3,4], keepdim=True)

        # Get gradients
        grad_vid = (mu_critic_vid - mu_teacher_vid) / vid_normalizer

        if torch.isnan(grad_vid).any():
            print("Warning: grad_vid contains NaNs")

        grad_vid = torch.nan_to_num(grad_vid, nan=0.0)

        # Get targets
        target_vid = (video.double() - grad_vid.double()).detach()

        if log_predictions:
            log_decoded_images(video[grad_mask], decode_fn, "video")
            log_decoded_images(target_vid[grad_mask], decode_fn, "target_vid")
            log_decoded_images(mu_teacher_vid[grad_mask], decode_fn, "mu_teacher_vid")
            log_decoded_images(mu_critic_vid[grad_mask], decode_fn, "mu_critic_vid")


        target_vid = target_vid[grad_mask]

    # Get losses, only where grad_mask is true
    dmd_loss = 0.5 * F.mse_loss(
        video[grad_mask].double(),
        target_vid,
        reduction='mean'
    )

    return dmd_loss

class SelfForceTrainer(BaseTrainer):
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

        freeze(self.teacher)
    
    def save(self):
        save_dict = {
            'model' : self.student.state_dict(),
            'ema' : self.ema.state_dict(),
            'opt' : self.opt.state_dict(),
            'scaler' : self.scaler.state_dict(),
            'critic' : self.critic.state_dict(),
            'critic_opt' : self.critic_opt.state_dict(),
            'critic_scaler' : self.critic_scaler.state_dict(),
            'steps' : self.total_step_counter
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
        self.teacher = self.teacher.cuda().bfloat16()
        self.decoder = self.decoder.cuda().eval().bfloat16()

        cast_rope_buffers_to_fp32(self.teacher)

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
        loader = SoftResetIterator(loader)
        sample_loader = get_loader(self.train_cfg.sample_data_id, self.train_cfg.batch_size, **self.train_cfg.sample_data_kwargs)
        sample_loader = SoftResetIterator(sample_loader)

        # Sampler
        sampler = get_sampler_cls(self.train_cfg.sampler_id)(**self.train_cfg.sampler_kwargs)
        rollout_manager = RolloutManager(self.model_cfg, self.train_cfg.rollout_steps, self.train_cfg.min_rollout_frames)

        # Gradient accumulation setup
        accum_steps = self.train_cfg.target_batch_size // self.train_cfg.batch_size // self.world_size
        accum_steps = max(1, accum_steps)

        # optimizer step
        def optimizer_step(model, scaler, optimizer):
            scaler.unscale_(optimizer)
            g_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
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
                        rollout_manager = rollout_manager,
                        decode_fn = frame_decode_fn,
                        log_predictions = (self.rank == 0 and self.total_step_counter % self.train_cfg.sample_interval == 0 and self.train_cfg.log_predictions)
                    )
                    dmd_loss = dmd_loss / accum_steps
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
        vid = vid / self.train_cfg.vae_scale

        mouses = [mouse]
        btns = [btn]
        for _ in range(15):
            _, new_mouse, new_btn = next(sample_loader)
            mouses.append(new_mouse)
            btns.append(new_btn)

        mouses = torch.cat(mouses, dim=0)
        btns = torch.cat(btns, dim=0)

        mouse, btn = batch_permute_to_length(mouses, btns, sampler.num_frames + vid.size(1))
        mouse = mouse[:vid.size(0)] # First batch element
        btn = btn[:vid.size(0)] # First batch element

        latent_vid = sampler(model, vid.cuda(), mouse.cuda(), btn.cuda())

        if True:
            latent_vid = latent_vid[:, vid.size(1):]
            mouse = mouse[:, vid.size(1):]
            btn = btn[:, vid.size(1):]

        video_out = decode_fn(latent_vid * self.train_cfg.vae_scale) if decode_fn is not None else None

        gc.collect()
        torch.cuda.empty_cache()

        def gather_concat_cpu(t, dim=0):
            if self.rank == 0:
                # Ensure tensor is on GPU for distributed communication
                t_gpu = t.cuda() if t.device.type == 'cpu' else t
                parts = [t_gpu.cpu()]
                scratch = torch.empty_like(t_gpu)  # scratch on GPU
                for src in range(self.world_size):
                    if src == 0:
                        continue
                    dist.recv(scratch, src=src)
                    parts.append(scratch.cpu())
                return torch.cat(parts, dim=dim)
            else:
                # Ensure tensor is on GPU for distributed communication
                t_gpu = t.cuda() if t.device.type == 'cpu' else t
                dist.send(t_gpu, dst=0)
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
