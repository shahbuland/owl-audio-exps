import torch
from tqdm import tqdm

from ..nn.kv_cache import KVCache

from .schedulers import get_sd3_euler


def zlerp(x, alpha):
    z = torch.randn_like(x)
    return x * (1. - alpha) + z * alpha


class AVCachingSampler:
    """
    Parameters
    ----------
    :param n_steps: Number of diffusion steps for each frame
    :param cfg_scale: Must be 1.0
    :param num_frames: Number of new frames to sample
    :param noise_prev: Noise previous frame
    :param only_return_generated: Whether to only return the generated frames
    """
    def __init__(
        self,
        n_steps: int = 16,
        cfg_scale: float = 1.0,
        num_frames: int = 60,
        noise_prev: float = 0.2,
        only_return_generated: bool = False,
    ) -> None:
        if cfg_scale != 1.0:
            raise NotImplementedError("cfg_scale must be 1.0")
        if only_return_generated:
            raise NotImplementedError("only_return_generated must be False")

        self.n_steps = n_steps
        self.num_frames = num_frames
        self.noise_prev = noise_prev
        self.only_return_generated = only_return_generated

    @torch.no_grad()
    def __call__(
            self,
            model,
            video: torch.Tensor, audio: torch.Tensor, mouse: torch.Tensor, btn: torch.Tensor,
            decode_fn=None, audio_decode_fn=None, image_scale=1, audio_scale=1
    ):
        """Generate `num_frames` new frames and return updated tensors."""
        batch_size, init_len = video.shape[:2]

        dt = get_sd3_euler(self.n_steps).to(device=video.device, dtype=video.dtype)

        kv_cache = KVCache(model.config)
        kv_cache.reset(batch_size)

        video_out = [] if self.only_return_generated else [video]
        audio_out = [] if self.only_return_generated else [audio]

        # History for the first frame generation step = full clean clip
        prev_video, prev_audio = video, audio
        prev_mouse, prev_btn = mouse[:, :init_len], btn[:, :init_len]

        for idx in tqdm(range(self.num_frames), desc="Sampling frames"):
            start = min(init_len + idx, mouse.size(1) - 1)
            curr_mouse, curr_btn = mouse[:, start: start + 1], btn[:, start: start + 1]

            new_video, new_audio = self.denoise_frame(
                model, kv_cache,
                prev_video, prev_audio, prev_mouse, prev_btn,
                curr_mouse, curr_btn,
                dt=dt,
            )

            video_out.append(new_video)
            audio_out.append(new_audio)

            # all history kv cached except for newly generated from - set the previous as the new state
            prev_video, prev_audio = new_video, new_audio
            prev_mouse, prev_btn = curr_mouse, curr_btn

        video_out, audio_out = torch.cat(video_out, dim=1), torch.cat(audio_out, dim=1)
        if decode_fn is not None:
            video_out = decode_fn(video_out * image_scale)
        if audio_decode_fn is not None:
            audio = audio_decode_fn(audio * audio_scale)

        return video_out, audio_out, mouse, btn

    def denoise_frame(
        self,
        model,
        kv_cache: KVCache,
        prev_video: torch.Tensor,
        prev_audio: torch.Tensor,
        prev_mouse: torch.Tensor,
        prev_btn: torch.Tensor,
        curr_mouse: torch.Tensor,
        curr_btn: torch.Tensor,
        dt: torch.Tensor,
    ):
        """Run all denoising steps for new frame"""
        batch_size = prev_video.size(0)

        # Partially re-noise history
        prev_vid = zlerp(prev_video, self.noise_prev)
        prev_aud = zlerp(prev_audio, self.noise_prev)
        t_prev = prev_video.new_full((batch_size, prev_vid.size(1)), self.noise_prev)

        # Create new pure-noise frame
        new_vid = torch.randn_like(prev_video[:, :1])
        new_aud = torch.randn_like(prev_audio[:, :1])
        t_new = t_prev.new_ones(batch_size, 1)

        # update kv cache with previous uncached frames
        kv_cache.enable_cache_updates()
        eps_v, eps_a = model(
            torch.cat([prev_vid, new_vid], dim=1),
            torch.cat([prev_aud, new_aud], dim=1),
            torch.cat([t_prev, t_new], dim=1),
            torch.cat([prev_mouse, curr_mouse], dim=1),
            torch.cat([prev_btn, curr_btn], dim=1),
            kv_cache=kv_cache,
        )
        kv_cache.disable_cache_updates()
        kv_cache.truncate(1, front=False)  # new "still-being-denoised" frame from kv cache

        # Euler update for step‑0 (affects only the *last* frame)
        new_vid -= eps_v[:, -1:] * dt[0]
        new_aud -= eps_a[:, -1:] * dt[0]
        t_new -= dt[0]

        # Remaining diffusion steps with cached history, denoising denoising only new frame
        for step in range(1, self.n_steps):
            eps_vid, eps_aud = model(
                new_vid, new_aud, t_new, curr_mouse, curr_btn, kv_cache=kv_cache
            )
            new_vid -= eps_vid * dt[step]
            new_aud -= eps_aud * dt[step]
            t_new -= dt[step]

        # Clean frame will be cached automatically in the *next* step‑0
        return new_vid, new_aud
