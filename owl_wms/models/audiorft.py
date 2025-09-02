import torch
from torch import nn
import torch.nn.functional as F

from ..nn.embeddings import TimestepEmbedding
from ..nn.attn import DiT, FinalLayer

import einops as eo
from einops._torch_specific import allow_ops_in_compiled_graph
allow_ops_in_compiled_graph()


class AudioRFTCore(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config

        assert config.backbone == "dit"
        self.transformer = DiT(config)

        # Only timestep embedding for unconditional generation
        self.t_embed = TimestepEmbedding(config.d_model)

        # Input/output projections for audio latents
        self.proj_in = nn.Linear(config.channels, config.d_model, bias=False)
        
        # Reinterpret sample_size as latent sequence length 
        # tokens_per_frame becomes 1 for pure sequential data
        self.proj_out = FinalLayer(1, config.d_model, config.channels)

        # For audio: tokens_per_frame = 1 (each latent is one "token")
        assert config.tokens_per_frame == 1

    def forward(self, x, t, doc_id=None, kv_cache=None, local_block_mask=None, global_block_mask=None):
        """
        x: [b, n_latents, latent_channels] - audio latents from VAE
        t: [b, n_latents] - timesteps per latent
        """
        b, n_latents, latent_channels = x.shape

        # Timestep conditioning
        t_cond = self.t_embed(t)  # [b, n_latents, d_model]

        # Project latents to model dimension
        x = self.proj_in(x)  # [b, n_latents, d_model]

        # Pass through transformer
        x = self.transformer(x, t_cond, doc_id, kv_cache, local_block_mask, global_block_mask)
        
        # Project back to latent space
        x = self.proj_out(x, t_cond)  # [b, n_latents, latent_channels]

        return x


class AudioRFT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.core = AudioRFTCore(config)

    def noise(self, tensor, ts):
        """Apply noise to audio latents"""
        z = torch.randn_like(tensor)
        lerp = tensor * (1 - ts) + z * ts
        return lerp, z - tensor, z

    def forward(self, x, doc_id=None, return_dict=False):
        """
        x: [b, n_latents, latent_channels] - clean audio latents from VAE
        """
        B, n_latents, _ = x.shape

        # Generate random timesteps for each latent in the sequence
        with torch.no_grad():
            ts = torch.randn(B, n_latents, device=x.device, dtype=x.dtype).sigmoid()
            lerpd_audio, target_audio, z_audio = self.noise(x, ts[:, :, None])

        # Predict the noise
        pred_audio = self.core(lerpd_audio, ts, doc_id)
        loss = F.mse_loss(pred_audio, target_audio)

        if not return_dict:
            return loss
        else:
            return {
                'diffusion_loss': loss,
                'audio_loss': loss,
                'lerpd_audio': lerpd_audio,
                'pred_audio': pred_audio,
                'ts': ts,
                'z_audio': z_audio,
            }