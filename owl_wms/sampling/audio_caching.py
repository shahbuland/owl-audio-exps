import torch
from tqdm import tqdm
import gc

from ..nn.kv_cache import KVCache
from .schedulers import get_sd3_euler


def get_deltas(custom_schedule):
    if custom_schedule[-1] != 0.0:
        custom_schedule.append(0.0)

    deltas = []
    crnt = custom_schedule[0]
    for nxt in custom_schedule[1:]:
        deltas.append(abs(nxt - crnt))
        crnt = nxt

    return deltas


class AudioCachingSampler:
    """
    Audio sampler that generates one token at a time using KV caching.
    
    Parameters
    ----------
    :param n_steps: Number of diffusion steps for each token
    :param num_tokens: Number of new audio tokens to generate
    :param noise_prev: Noise level for previous tokens (default 0.2)
    :param custom_schedule: Custom noise schedule, if None uses SD3 Euler
    :param max_window: Maximum context window for caching
    """
    def __init__(
        self, 
        n_steps: int = 16, 
        num_tokens: int = 120, 
        noise_prev: float = 0.2, 
        custom_schedule = None,
        max_window = None
    ) -> None:
        self.n_steps = n_steps
        self.num_tokens = num_tokens
        self.noise_prev = noise_prev
        self.custom_schedule = custom_schedule
        self.max_window = max_window

    @staticmethod
    def zlerp(x, alpha):
        """Linear interpolation with noise"""
        z = torch.randn_like(x)
        return x * (1. - alpha) + z * alpha

    @torch.no_grad()
    def __call__(self, model, x, decode_fn=None, vae_scale=1.0, compile_on_decode=False):
        """
        Generate audio tokens autoregressively
        
        Args:
            model: AudioRFT model core
            x: Initial audio latents [b, init_len, latent_channels]
            decode_fn: Function to decode latents to waveforms
            vae_scale: Scale factor for VAE
        
        Returns:
            Generated audio latents [b, init_len + num_tokens, latent_channels]
        """
        batch_size, init_len, latent_channels = x.shape

        # Get noise schedule
        if self.custom_schedule is None:
            dt = get_sd3_euler(self.n_steps).to(device=x.device, dtype=x.dtype)
        else:
            dt = get_deltas(self.custom_schedule)

        # Initialize KV cache
        kv_cache = KVCache(model.config)
        kv_cache.reset(batch_size)

        # Store generated latents
        latents = [x.clone()]
        prev_x = x

        # ==== STEP 1: Cache initial context ====
        prev_x_noisy = self.zlerp(prev_x, self.noise_prev)
        prev_t = prev_x.new_full((batch_size, prev_x.size(1)), self.noise_prev)

        kv_cache.enable_cache_updates()
        _ = model(
            prev_x_noisy,
            prev_t,
            doc_id = None,
            kv_cache=kv_cache
        )
        kv_cache.disable_cache_updates()

        def new_token():
            """Generate random initial token and timestep"""
            return torch.randn(batch_size, 1, latent_channels, device=x.device, dtype=x.dtype), \
                   prev_t.new_ones(batch_size, 1)

        # Enable decoding mode for transformer
        model.transformer.enable_decoding()
        if compile_on_decode:
            model = torch.compile(model)
        
        # ==== STEP 2: Generate tokens one by one ====
        for idx in tqdm(range(self.num_tokens), desc="Sampling Audio Tokens..."):
            curr_x, curr_t = new_token()

            # Denoise the new token
            for t_idx in range(self.n_steps):
                pred_v = model(
                    curr_x,
                    curr_t,
                    doc_id = None,
                    kv_cache=kv_cache
                ).clone()

                # Apply noise schedule (no CFG for unconditional generation)
                curr_x = curr_x - dt[t_idx] * pred_v   
                curr_t = curr_t - dt[t_idx]

            # ==== STEP 3: Token generated, append and cache ====
            latents.append(curr_x.clone())
            
            # Add noise to generated token for next iteration
            curr_x_noisy = self.zlerp(curr_x, self.noise_prev)
            curr_t_noisy = torch.ones_like(curr_t) * self.noise_prev

            # Cache the new token
            kv_cache.enable_cache_updates()
            _ = model(
                curr_x_noisy,
                curr_t_noisy,
                kv_cache=kv_cache
            )
            kv_cache.disable_cache_updates()
            
            # Manage cache window
            if self.max_window is not None and len(latents) > self.max_window:
                kv_cache.truncate(1, front=False)  # Remove oldest cached token

            # Memory cleanup
            gc.collect()
            torch.cuda.empty_cache()

        model.transformer.disable_decoding()

        # Concatenate all generated latents
        full_latents = torch.cat(latents, dim=1)  # [b, init_len + num_tokens, latent_channels]
        
        # Optionally decode to waveforms
        if decode_fn is not None:
            # Scale latents and decode to audio
            scaled_latents = full_latents * vae_scale
            waveforms = decode_fn(scaled_latents)
            return full_latents, waveforms
        
        return full_latents