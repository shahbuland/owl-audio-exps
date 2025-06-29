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

class FlatVideoRoPE(nn.Module):
    """
    RoPE on video + audio assuming each frame flat'd to [n_frame_toks+n_audio_toks]
    """
    def __init__(self, config):
        super().__init__()

        dim_head = config.d_model // config.n_heads
        self.pos_emb_video = RotaryEmbedding(
            dim = dim_head//4, # Using half dimension since we only need 1D rotation
            freqs_for='pixel',
            max_freq=256
        )
        self.pos_emb_audio = RotaryEmbedding(
            dim = dim_head//2
        )

        self.m = config.tokens_per_frame
        self.p = config.sample_size
        self.p2 = self.p**2

    def forward(self, q, k):
        b,h,_,d = k.shape

        # q|k is [b,h,n_frames*tokens_per_frame,d]
        n = k.shape[2]//self.m  # Number of frames
        n_q = q.shape[2]//self.m
        m = self.m             # Tokens per frame

        # Reshape to [b,h,n,m,d]
        q = q.reshape(q.shape[0], q.shape[1], n_q, m, q.shape[3])
        k = k.reshape(k.shape[0], k.shape[1], n, m, k.shape[3])

        # Split out the video and audio
        # bhnmd-> bhn(16)d, bhnd
        q_video = q[:,:,:,:self.p2]
        q_video = q_video.view(
            b,
            h,
            n_q, self.p, self.p,
            d
        )
        k_video = k[:,:,:,:self.p2]
        k_video = k_video.view(
            b,
            h,
            n, self.p, self.p,
            d
        )
        q_audio = q[:,:,:,-1]
        k_audio = k[:,:,:,-1] # bhnd

        with torch.no_grad():
            vid_freqs = self.pos_emb_video.get_axial_freqs(n, self.p, self.p)
        
        q_video = apply_rotary_emb(vid_freqs.detach(), q_video)
        k_video = apply_rotary_emb(vid_freqs.detach(), k_video)
        q_audio, k_audio = self.pos_emb_audio.rotate_queries_with_cached_keys(q_audio, k_audio)

        q_video = q_video.reshape(
            b,
            h,
            n_q, self.p2, 
            d
        ) # bhn(16)d
        q = torch.cat([q_video, q_audio.unsqueeze(-2)], dim = -2)

        k_video = k_video.reshape(
            b,
            h,
            n, self.p2, 
            d
        )
        k = torch.cat([k_video, k_audio.unsqueeze(-2)], dim = -2)

        q = q.view(b,h,n_q*m,d)
        k = k.view(b,h,n_q*m,d)

        return q, k
