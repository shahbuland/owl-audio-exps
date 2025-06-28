"""
Variants of RoPE were becoming heavy for embeddings so 
I made a unique script for all of them here
"""

from rotary_embedding_torch import (
    RotaryEmbedding,
    apply_rotary_emb
)
import torch
from torch import nn

class VideoRoPE(nn.Module):
    """
    Video RoPE embedding for when latents are 3D [n,h,w]
    """
    def __init__(self, config : 'TransformerConfig'):
        super().__init__()

        dim_head = config.d_model // config.n_heads
        self.pos_emb = RotaryEmbedding(
            dim = dim_head//8,
            freqs_for = 'pixel',
            max_freq = 256
        )
        n_patches = config.sample_size // config.patch_size
        self.tokens_per_frame = n_patches**2

        self.rearrange_in = lambda x: x.view(x.shape[0], x.shape[1], -1, n_patches, n_patches, x.shape[3])
        self.rearrange_out = lambda x: x.view(x.shape[0], x.shape[1], -1, x.shape[-1])
        self.get_freqs = lambda n_t: self.pos_emb.get_axial_freqs(n_t, n_patches, n_patches)

    def forward(self, q, k):
        # q k both [b,h,n,d]
        q = self.rearrange_in(q)
        k = self.rearrange_in(k)

        n_t = q.shape[2]
        freqs = self.get_freqs(n_t)

        q = apply_rotary_emb(freqs.float(), q.float()).to(q.dtype)
        k = apply_rotary_emb(freqs.float(), k.float()).to(k.dtype)

        q = self.rearrange_out(q)
        k = self.rearrange_out(k)
        
        return q, k

class _FlatVideoRoPE(nn.Module):
    """
    Half-flat of RoPE that treats [n_frames, tokens_per_frame] as [n_frames, tokens_per_frame] image
    """
    def __init__(self, config):
        super().__init__()

        dim_head = config.d_model // config.n_heads
        self.pos_emb = RotaryEmbedding(
            dim = dim_head//4,
            freqs_for='pixel',
            max_freq=256
        )

        self.m = config.tokens_per_frame

    def pad_q(self, q, k):
        # Pad Q when it's needed for kv caching
        q_len = q.shape[2]
        k_len = k.shape[2]

    def forward(self, q, k):
        # q|k is [b,h,n_frames*tokens_per_frame,d]
        n = k.shape[2]//self.m
        m = self.m

        truncate = n
        if q.shape[2] < n * m:
            truncate = q.shape[2]//m # How many frames is q?

        q = q.view(q.shape[0], q.shape[1], q.shape[2]//m, m, q.shape[3])
        k = k.view(k.shape[0], k.shape[1], n, m, k.shape[3])

        with torch.no_grad():
            freqs = self.pos_emb.get_axial_freqs(n,m)
        q = apply_rotary_emb(freqs[-truncate:].detach(), q)
        k = apply_rotary_emb(freqs.detach(), k)

        q = q.view(q.shape[0], q.shape[1], -1, q.shape[4])
        k = k.view(k.shape[0], k.shape[1], -1, k.shape[4])

        if truncate is not None:
            q = q[:,:,-truncate*m:]

        return q,k


class FlatVideoRoPE(nn.Module):
    """
    RoPE that only rotates based on frame index, ignoring position within frames
    """
    def __init__(self, config):
        super().__init__()

        dim_head = config.d_model // config.n_heads
        self.pos_emb = RotaryEmbedding(
            dim = dim_head//2, # Using half dimension since we only need 1D rotation
            freqs_for='pixel',
            max_freq=256
        )

        self.m = config.tokens_per_frame

    def forward(self, q, k):
        # q|k is [b,h,n_frames*tokens_per_frame,d]
        n = k.shape[2]//self.m  # Number of frames
        n_q = q.shape[2]//self.m
        m = self.m             # Tokens per frame

        # Reshape to [b,h,n,m*d]
        q = q.reshape(q.shape[0], q.shape[1], n_q, m * q.shape[3])
        k = k.reshape(k.shape[0], k.shape[1], n, m * k.shape[3])

        # Apply rotary embeddings

        if n_q == n:
            q = self.pos_emb.rotate_queries_or_keys(q)
            k = self.pos_emb.rotate_queries_or_keys(k)
        else:
            q,k = self.pos_emb.rotate_queries_with_cached_keys(q,k)

        # Reshape back
        q = q.reshape(q.shape[0], q.shape[1], -1, q.shape[3] // m)
        k = k.reshape(k.shape[0], k.shape[1], -1, k.shape[3] // m)

        return q, k
