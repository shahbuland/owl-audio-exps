"""
Variants of RoPE were becoming heavy for embeddings so
I made a unique script for all of them here
"""

from rotary_embedding_torch import RotaryEmbedding, apply_rotary_emb
import torch
from torch import nn
from torch.cuda.amp import autocast

import einops as eo
from einops._torch_specific import allow_ops_in_compiled_graph  # requires einops>=0.6.1
allow_ops_in_compiled_graph()


class RoPE(nn.Module):
    def __init__(self, config):
        super().__init__()
        freqs = self.get_freqs(config)
        self.cos = nn.Buffer(freqs.cos().contiguous(), persistent=False)
        self.sin = nn.Buffer(freqs.sin().contiguous(), persistent=False)

    @autocast(enabled=False)
    def forward(self, x, offset: int = 0):
        assert self.cos.dtype == torch.float32
        cos = self.cos[..., offset:offset + x.size(2), :]
        sin = self.sin[..., offset:offset + x.size(2), :]
        x0, x1 = x.float().unfold(-1, 2, 2).unbind(-1)
        y0 = x0 * cos - x1 * sin
        y1 = x1 * cos + x0 * sin
        return torch.cat((y0, y1), dim=-1).type_as(x)

    def get_freqs(self, config):
        raise NotImplementedError


class FlatVideoRoPE(RoPE):
    """
    RoPE on video + audio assuming each frame flat'd to [n_frame_toks+n_audio_toks]
    Rotate non-pad portion of feature-dims of q/k. Video 1/4th padded, audio 1/2 padded
    """
    def get_freqs(self, config):
        d_head = config.d_model // config.n_heads
        p = config.sample_size  # video is PxP pixels

        # Video freqs. Rot features: (L, P, P, <pad>)
        vid_freqs = RotaryEmbedding(d_head // 4, freqs_for="pixel", max_freq=256)\
            .get_axial_freqs(config.n_frames, p, p, 1, offsets=(0, 0, 0, 1))\
            .view(config.n_frames, p**2, -1)

        # Audio freqs. Rot features: (L, <pad>)
        aud_freqs = RotaryEmbedding(d_head // 2)\
            .get_axial_freqs(config.n_frames, 1, offsets=(0, 1))\
            .view(config.n_frames, 1, -1)

        # unified video / audio freqs. Shape: [n_frames, P^2 + 1, H]
        freqs = torch.cat([vid_freqs, aud_freqs], dim=1).flatten(0, 1)
        return freqs[..., ::2]  # subsampling


class AVRoPE(RoPE):
    """
    RoPE variant that treats audio as R+1,C+1 part of a frame
    """
    def get_freqs(self, config):
        p = config.sample_size
        head_dim = config.d_model // config.n_heads
        assert p**2 + 1 == config.tokens_per_frame

        pos_emb = RotaryEmbedding(
            dim=head_dim // 4,  # Using half dimension since we only need 1D rotation
            freqs_for='pixel',
            max_freq=256
        )
        # Rot features: (L, P+1, P+1, <pad>)
        freqs = pos_emb.get_axial_freqs(
            config.n_frames, p + 1, p + 1, 1, offsets=(0, 0, 0, 1)
        ).view(config.n_frames, p + 1, p + 1, -1)

        vid_freqs = freqs[:, :p, :p].reshape(config.n_frames, p**2, -1)  # top left square
        aud_freqs = freqs[:, -1, -1].unsqueeze(1)  # bottom right item

        freqs = torch.cat([vid_freqs, aud_freqs], dim=1).flatten(0, 1)[..., ::2]
        return freqs[..., ::2]  # subsampling


class VideoRoPE(RoPE):
    """https://arxiv.org/pdf/2502.05173"""
    def get_freqs(self, config):
        H, W = config.sample_size, config.sample_size
        F = config.n_frames
        d_head = config.d_model // config.n_heads

        dims = {
            't': getattr(config, 'rope_dim_t', d_head * 2 // 8),
            'x': getattr(config, 'rope_dim_x', d_head * 3 // 8),
            'y': getattr(config, 'rope_dim_y', d_head * 3 // 8)
        }
        theta = getattr(config, 'rope_base', 10000.0)
        ats_delta = getattr(config, 'rope_ats_delta', 2.0)

        base_freqs = RotaryEmbedding(dim=sum(dims.values()), freqs_for='lang', theta=theta).freqs.float()

        freqs_spatial, freqs_t = torch.split(base_freqs, [(dims['x'] + dims['y']) // 2, dims['t'] // 2])
        freqs_x, freqs_y = freqs_spatial[::2], freqs_spatial[1::2]

        x_pos, y_pos, t_pos = self._create_positions(F, H, W, ats_delta)

        angles_x = x_pos[:, None] * freqs_x[None, :]
        angles_y = y_pos[:, None] * freqs_y[None, :]
        angles_t = t_pos[:, None] * freqs_t[None, :]

        interleaved_spatial = eo.rearrange(
            torch.stack([angles_x, angles_y], dim=-1),
            'b n two -> b (n two)'
        )

        return torch.cat([interleaved_spatial, angles_t], dim=-1)

    def _create_positions(self, n_frames, height, width, ats_delta):
        # Create base grids
        t_grid = torch.arange(n_frames, dtype=torch.float32) * ats_delta
        h_grid = torch.arange(height, dtype=torch.float32) - (height - 1) / 2.0
        w_grid = torch.arange(width, dtype=torch.float32) - (width - 1) / 2.0

        # Video positions: broadcast and flatten
        t_video = eo.repeat(t_grid, 'f -> (f h w)', h=height, w=width)
        x_video = t_video + eo.repeat(w_grid, 'w -> (f h w)', f=n_frames, h=height)
        y_video = t_video + eo.repeat(h_grid, 'h -> (f h w)', f=n_frames, w=width)

        # Audio positions: simple repetition
        t_audio = eo.repeat(t_grid, 'f -> f')
        x_audio = t_audio  # audio_x_offset = 0
        y_audio = t_audio + (height - 1) / 2.0 + 1.0  # audio_y_offset

        # Concatenate video and audio
        return (
            torch.cat([x_video, x_audio]),
            torch.cat([y_video, y_audio]),
            torch.cat([t_video, t_audio])
        )


def visaulize_rope_freqs():
    pos_emb = RotaryEmbedding(
        dim = dim_head//6, # Using half dimension since we only need 1D rotation
        freqs_for='pixel',
        max_freq=256
    )
    freqs = pos_emb.get_axial_freqs(16, 5, 5)
