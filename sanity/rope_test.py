from owl_wms.nn.rope import FlatVideoRoPE, FlatVideoRoPE_v2
from owl_wms.configs import TransformerConfig
import torch
import time


def benchmark_rope():
    # Set device (use cuda if available for fair benchmarking)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    q = torch.randn(1, 24, 1020, 64, device=device)
    k = torch.randn(1, 24, 1020, 64, device=device)

    config = TransformerConfig(
        n_layers = 24,
        n_heads = 24,
        d_model = 1536,
        tokens_per_frame = 17,
        sample_size = 4,
        n_frames = 60
    )

    # Run both RoPE implementations
    rope1 = FlatVideoRoPE(config).to(device).eval()
    rope2 = FlatVideoRoPE_v2(config).to(device).eval()

    # Warmup
    for _ in range(3):
        with torch.no_grad():
            _ = rope1(q, k)
            _ = rope2(q, k)
    if device.type == "cuda":
        torch.cuda.synchronize()

    # Benchmark rope1
    n_trials = 100
    with torch.no_grad():
        start = time.time()
        for _ in range(n_trials):
            q1, k1 = rope1(q, k)
        if device.type == "cuda":
            torch.cuda.synchronize()
        end = time.time()
    rope1_latency = (end - start) / n_trials * 1000  # ms

    # Benchmark rope2
    with torch.no_grad():
        start = time.time()
        for _ in range(n_trials):
            q2, k2 = rope2(q, k)
        if device.type == "cuda":
            torch.cuda.synchronize()
        end = time.time()
    rope2_latency = (end - start) / n_trials * 1000  # ms

    # Compute error between outputs
    q_err = (q1 - q2).abs().mean().item()
    k_err = (k1 - k2).abs().mean().item()

    print("q error (FlatVideoRoPE vs FlatVideoRoPE_v2):", q_err)
    print("k error (FlatVideoRoPE vs FlatVideoRoPE_v2):", k_err)

    # As a reference, compare two random tensors of same shape
    q_rand1 = torch.randn_like(q)
    q_rand2 = torch.randn_like(q)
    rand_err = (q_rand1 - q_rand2).abs().mean().item()
    print("Reference random error (randn vs randn):", rand_err)

    print("q1 shape:", q1.shape, "k1 shape:", k1.shape)
    print(f"FlatVideoRoPE latency: {rope1_latency:.3f} ms")
    print(f"FlatVideoRoPE_v2 latency: {rope2_latency:.3f} ms")


@torch.no_grad()
def test_flat_video_rope_integrity():
    from types import SimpleNamespace
    import numpy as np

    cfg = SimpleNamespace(d_model=512, n_heads=8, tokens_per_frame=17, sample_size=4, n_frames=100)
    rope = FlatVideoRoPE(cfg)

    # use numpy rng - reproducable across machines
    rng = np.random.default_rng(seed=0)
    shape = (2, cfg.n_heads, cfg.n_frames * cfg.tokens_per_frame, cfg.d_model // cfg.n_heads)
    q = torch.from_numpy(rng.standard_normal(size=shape).astype(np.float32)).bfloat16()
    k = torch.from_numpy(rng.standard_normal(size=shape).astype(np.float32)).bfloat16()

    q_out, k_out = rope(q), rope(k)

    checksum = q_out.float().sum() + k_out.float().sum()
    ref_impl_checksum = torch.tensor(1678.122802734375)
    assert torch.allclose(checksum, ref_impl_checksum, rtol=1e-6), checksum.item()
    print("FlatVideoRoPE implementation consistent")


@torch.no_grad()
def test_flat_video_rope_dot_product_invariance():
    from types import SimpleNamespace
    import numpy as np

    cfg = SimpleNamespace(d_model=512, n_heads=8, tokens_per_frame=17, sample_size=4, n_frames=100)
    rope = FlatVideoRoPE(cfg)

    B, H, L, Dh = 1, cfg.n_heads, cfg.tokens_per_frame * cfg.n_frames, cfg.d_model // cfg.n_heads

    rng = np.random.default_rng(seed=0)

    for dtype, rtol in zip([torch.float32, torch.bfloat16], [1e-4, 1e-2]):
        q = torch.from_numpy(rng.standard_normal((B, H, 1, Dh))).repeat(1, 1, L, 1).to(dtype)
        k = torch.from_numpy(rng.standard_normal((B, H, 1, Dh))).repeat(1, 1, L, 1).to(dtype)

        q, k = rope(q), rope(k)

        # q/k idxs, equidistant (53 frames away from one another)
        q0, q1, k0, k1 = [0, 30, 53, 83]
        q0, q1, k0, k1 = [idx * cfg.tokens_per_frame for idx in (q0, q1, k0, k1)]

        # per-head dot product
        q, k = q.float(), k.float()  # ignore dot product accumulation precision, only testing RoPE
        dp1 = (q[0, :, q0, :] * k[0, :, k0, :]).sum(-1)
        dp2 = (q[0, :, q1, :] * k[0, :, k1, :]).sum(-1)

        # they should be equal up to numerical tolerance
        if not torch.allclose(dp1, dp2, rtol=rtol):
            raise ValueError(f"dot product not equivalent. Per head output:\ndp1={dp1}\ndp2={dp2}")
        print("RoPE dot-product check passed")


if __name__ == "__main__":
    benchmark_rope()
    test_flat_video_rope_integrity()
    test_flat_video_rope_dot_product_invariance()
