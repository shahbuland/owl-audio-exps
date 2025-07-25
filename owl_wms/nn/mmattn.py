import torch
from torch import nn
import torch.nn.functional as F

from .normalization import LayerNorm, RMSNorm, QKNorm
from .mlp import MLP

import einops as eo

from .modulation import AdaLN, Gate
from .rope import AVRoPE
from .attn import create_causal_block_mask

from torch.nn.attention.flex_attention import flex_attention

from einops._torch_specific import allow_ops_in_compiled_graph
allow_ops_in_compiled_graph()


flex_attention = torch.compile(flex_attention)

"""
This code makes the assumption that there are some
tokens from another modality that must always be attended to
"""


class MMAttn(nn.Module):
    """
    MMDiT style attention
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.n_heads = config.n_heads

        self.qkv_projs = nn.ModuleList([nn.Linear(config.d_model, 3 * config.d_model) for _ in range(2)])
        self.out_projs = nn.ModuleList([nn.Linear(config.d_model, config.d_model)for _ in range(2)])
        self.qk_norms = nn.ModuleList([QKNorm(config.d_model // self.n_heads) for _ in range(2)])
        self.rope = AVRoPE(config)

    def split(self, qkv):
        return eo.rearrange(qkv, 'b n (three h d) -> three b h n d', three=3, h=self.n_heads)

    def merge(self, x):
        return eo.rearrange(x, 'b h n d -> b n (h d)')

    def forward(self, x_1, x_2, block_mask=None, kv_cache=None):
        """
        For MMDiT we assume kv_cache is a tuple of two caches
        """
        n1 = x_1.shape[1]

        # calculate qs, ks, vs for each modality, and update kv cache
        qs, ks, vs = [], [], []
        for i, x in enumerate([x_1, x_2]):
            q, k, v = self.split(self.qkv_projs[i](x))
            q, k = self.qk_norms[i](q, k)
            q, k = q.type_as(v), k.type_as(v)

            # prepend cached values
            offset = kv_cache[i].length_at(self.layer_ind) if kv_cache is not None else 0
            if offset > 0:
                old_k, old_v = kv_cache[i].get(self.layer_ind)
                k = torch.cat([old_k, k], dim=2)
                v = torch.cat([old_v, v], dim=2)

            # update cache
            if kv_cache is not None and kv_cache.should_update:
                kv_cache[i].update(k.clone(), v.clone(), self.layer_ind)

            qs.append(q)
            ks.append(k)
            vs.append(v)

        qs, ks, vs = torch.cat(qs, dim=-2), torch.cat(ks, dim=-2), torch.cat(vs, dim=-2)

        qs, ks = self.rope(qs, offset=offset), self.rope(ks, offset=offset)

        attn_out = flex_attention(qs, ks, vs, block_mask=block_mask)
        attn_out = self.merge(attn_out)

        x_1, x_2 = attn_out[:, :n1], attn_out[:, n1:]
        x_1, x_2 = self.out_projs[0](x_1).contiguous(), self.outs_projs[1](x_2).contiguous()
        return x_1, x_2


class MMDiTBlock(nn.Module):
    def __init__(self, config):
        super().__init__()

        dim = config.d_model

        self.attn = MMAttn(config)

        self.mlp_1 = MLP(config)
        self.mlp_2 = MLP(config)

        # Stream 1 - AdaLN and gating
        self.adaln1_1 = AdaLN(dim)
        self.gate1_1 = Gate(dim)
        self.adaln2_1 = AdaLN(dim)
        self.gate2_1 = Gate(dim)

        # Stream 2 - AdaLN and gating (was LayerNorm)
        self.adaln1_2 = AdaLN(dim)
        self.gate1_2 = Gate(dim)
        self.adaln2_2 = AdaLN(dim)
        self.gate2_2 = Gate(dim)

    @torch.compile
    def forward(self, x, y, cond, block_mask = None, kv_cache = None):
        res1_x = x.clone()
        res1_y = y.clone()

        # First attention block
        x = self.adaln1_1(x, cond)
        y = self.adaln1_2(y, cond)

        x, y = self.attn(x, y, block_mask, kv_cache)

        x = self.gate1_1(x, cond)
        y = self.gate1_2(y, cond)

        x = res1_x + x
        y = res1_y + y

        # Second MLP block
        res2_x = x.clone()
        res2_y = y.clone()

        x = self.adaln2_1(x, cond)
        y = self.adaln2_2(y, cond)

        x = self.mlp_1(x)
        y = self.mlp_2(y)

        x = self.gate2_1(x, cond)
        y = self.gate2_2(y, cond)

        x = res2_x + x
        y = res2_y + y

        return x, y

class MMDIT(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config
        self.blocks = nn.ModuleList([MMDiTBlock(config) for _ in range(config.n_layers)])

        for i in range(config.n_layers):
            self.blocks[i].attn.layer_ind = i

    def get_block_mask(self, x, y, kv_cache):
        if not self.config.causal:
            return None
        seq_len = x.shape[1] + y.shape[1]
        offset = kv_cache.length_at(0) if kv_cache is not None else 0
        return create_causal_block_mask(
            n_tokens=seq_len + offset,
            tokens_per_frame=self.config.tokens_per_frame,
            n_cached_tokens=offset,
            device=x.device
        )

    def forward(self, x, y, cond, kv_cache = None):
        block_mask = self.get_block_mask(x, y, kv_cache)
        for block in self.blocks:
            x,y = block(x, y, cond, block_mask, kv_cache)
        return x,y

def test_fwd_with_cache():
    from ..configs import TransformerConfig
    from .kv_cache import KVCache

    import matplotlib.pyplot as plt

    cfg = TransformerConfig(
        None,
        6,
        6,
        384,
        1,
        128,
        4,
        0.1,
        8,
        16,
        True
    )

    model = MMUViT(cfg).bfloat16().cuda()

    NUM_FRAMES = 10
    x = torch.randn(1,16*NUM_FRAMES,384).bfloat16().cuda()
    y = torch.randn(1,16,384).bfloat16().cuda()
    cond=torch.randn(1,16,384).bfloat16().cuda()

    cache = KVCache(cfg).to(device='cuda',dtype=torch.bfloat16)
    cache.reset(1)

    with torch.no_grad():
        cache.enable_cache_updates()
        out = model(x,y,cond,cache)

        new_x = torch.randn(1,16,384).bfloat16().cuda()
        cond = torch.randn(1,1,384).bfloat16().cuda()

        print(len(cache))
        print(cache.cache[0][0].shape)
        new_out = model(new_x, y, cond, cache)

        print(len(cache))
        print(cache.cache[0][0].shape)

def test_mask():
    import matplotlib.pyplot as plt

    n_frames = 10
    n_tok_per_frame = 17
    n_audio_tokens = 1
    total_tokens = n_frames * n_tok_per_frame

    # TODO !!!!
    mask = create_block_causal_mask_with_mm(total_tokens, n_tok_per_frame, n_audio_tokens)

    # Convert to visualization format: 1 = allowed (white), 0 = blocked (black)
    mask_vis = (mask != -float('inf')).float()

    plt.figure(figsize=(12, 10))
    plt.imshow(mask_vis.cpu().numpy(), cmap='gray', interpolation='nearest')
    plt.colorbar(label='Attention Allowed (1=Yes, 0=No)')
    plt.title(f'Block Causal Mask with MM\n({total_tokens} total tokens, {n_tok_per_frame} per frame, {n_audio_tokens} audio per frame)')
    plt.xlabel('Key Position')
    plt.ylabel('Query Position')

    # Add grid lines to show frame boundaries
    for i in range(1, n_frames):
        video_boundary = i * (n_tok_per_frame - n_audio_tokens)
        audio_boundary = n_frames * (n_tok_per_frame - n_audio_tokens) + i * n_audio_tokens
        plt.axhline(y=video_boundary - 0.5, color='red', linestyle='--', alpha=0.5)
        plt.axvline(x=video_boundary - 0.5, color='red', linestyle='--', alpha=0.5)
        plt.axhline(y=audio_boundary - 0.5, color='blue', linestyle='--', alpha=0.5)
        plt.axvline(x=audio_boundary - 0.5, color='blue', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig('test_mm_mask.png', dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Mask shape: {mask.shape}")
    print(f"Number of blocked positions: {(mask == -float('inf')).sum().item()}")
    print(f"Number of allowed positions: {(mask != -float('inf')).sum().item()}")


if __name__ == "__main__":
    test_mask()
