import torch
from torch import nn
import torch.nn.functional as F
from tqdm import tqdm

from ..utils import batch_permute_to_length
from ..nn.kv_cache import KVCache

class AVCachingSampler:
    """
    Caching sampler that uses KV cache to avoid recomputing attention for previous frames.
    Samples new frames one by one, using cached keys/values from previous frames.

    :param n_steps: Number of diffusion steps for each frame (default 4)
    :param cfg_scale: CFG scale for each frame
    :param num_frames: Number of new frames to sample
    :param only_return_generated: Whether to only return the generated frames
    :param cache_after_denoise: Whether to cache clean frame after denoising (vs caching final noisy frame)
    """
    def __init__(self, n_steps=4, num_frames=60, only_return_generated=False, cache_after_denoise=True):
        self.n_steps = n_steps
        self.num_frames = num_frames
        self.only_return_generated = only_return_generated
        self.cache_after_denoise = cache_after_denoise

    @torch.no_grad()
    def __call__(self, model, dummy_batch, audio, mouse, btn, decode_fn=None, audio_decode_fn=None, image_scale=1, audio_scale=1):
        # Initialize outputs
        clean_history = dummy_batch.clone()
        clean_audio_history = audio.clone()
        
        # Get extended controls
        extended_mouse, extended_btn = batch_permute_to_length(mouse, btn, self.num_frames + dummy_batch.shape[1])

        # Initialize KV cache
        kv_cache = KVCache(model.config)
        kv_cache.reset(dummy_batch.shape[0])

        # Cache context frames
        kv_cache.enable_cache_updates()
        ts = torch.ones_like(clean_history[:,:,0,0,0])
        _ = model(clean_history, clean_audio_history, ts, mouse, btn, kv_cache=kv_cache)
        kv_cache.disable_cache_updates()

        dt = 1. / self.n_steps

        # Generate new frames
        for frame_idx in tqdm(range(self.num_frames)):
            # Initialize new frame with noise
            new_frame = torch.randn_like(clean_history[:,0:1])
            new_audio = torch.randn_like(clean_audio_history[:,0:1])

            curr_mouse = extended_mouse[:,frame_idx:frame_idx+1]
            curr_btn = extended_btn[:,frame_idx:frame_idx+1]

            b = new_frame.shape[0]
            ts = torch.ones_like(new_frame[:,0,0,0,0]).unsqueeze(1)

            # Denoise
            for step in range(self.n_steps):
                is_final_step = (step == self.n_steps - 1)

                if is_final_step and not self.cache_after_denoise:
                    kv_cache.enable_cache_updates()
                pred_video, pred_audio = model(new_frame, new_audio, ts, curr_mouse, curr_btn, kv_cache=kv_cache)
                if is_final_step and not self.cache_after_denoise:
                    kv_cache.disable_cache_updates()

                # Update
                new_frame = new_frame - pred_video * dt
                new_audio = new_audio - pred_audio * dt
                ts = ts - dt

            # Cache the frame
            if self.cache_after_denoise:
                # Cache clean frame with noise level 1.0
                kv_cache.enable_cache_updates()
                ts = torch.ones_like(new_frame[:,0,0,0,0]).unsqueeze(1)
                _ = model(new_frame, new_audio, ts, curr_mouse, curr_btn, kv_cache=kv_cache)
                kv_cache.disable_cache_updates()

            # Add to history
            clean_history = torch.cat([clean_history, new_frame], dim=1)
            clean_audio_history = torch.cat([clean_audio_history, new_audio], dim=1)

        # Return only generated frames if specified
        if self.only_return_generated:
            clean_history = clean_history[:,-self.num_frames:]
            clean_audio_history = clean_audio_history[:,-self.num_frames:]
            extended_mouse = extended_mouse[:,-self.num_frames:]
            extended_btn = extended_btn[:,-self.num_frames:]

        # Decode if decoders provided
        if decode_fn is not None:
            clean_history = clean_history * image_scale
            clean_history = decode_fn(clean_history)

        if audio_decode_fn is not None:
            clean_audio_history = clean_audio_history * audio_scale
            clean_audio_history = audio_decode_fn(clean_audio_history)

        return clean_history, clean_audio_history, extended_mouse, extended_btn
