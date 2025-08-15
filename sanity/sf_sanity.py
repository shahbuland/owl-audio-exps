import torch
from owl_wms.configs import Config
from owl_wms.models import get_model_cls
from owl_wms.nn.kv_cache import KVCache

# Load config
cfg = Config.from_yaml("configs/dit_v4_sf.yml").model

# Instantiate student model
student = get_model_cls(cfg.model_id)(cfg).cuda().train().core

# Make empty kv cache (no context frames cached)
kv_cache = KVCache(cfg)
kv_cache.reset(batch_size=1)  # batch size 1

kv_cache.enable_cache_updates()
with torch.no_grad():
    # Build context of length 60 for cache
    context_n = 60
    context_video = torch.randn(1, context_n, 128, 8, 8, device='cuda')
    context_mouse = torch.randn(1, context_n, 2, device='cuda')
    context_btn = torch.randn(1, context_n, 11, device='cuda')
    context_ts = torch.zeros(1, context_n, device='cuda')
    # Fill cache with context
    student(context_video, context_ts, context_mouse, context_btn, kv_cache=kv_cache)
kv_cache.disable_cache_updates()

student.transformer.enable_decoding()
# Make random video, mouse, btn
video = torch.randn(1, 1, 128, 8, 8, device='cuda', requires_grad=True)
mouse = torch.randn(1, 1, 2, device='cuda')
btn = torch.randn(1, 1, 11, device='cuda')
ts = torch.zeros(1, 1, device='cuda')  # dummy timestep

# Forward pass
out = student(video, ts, mouse, btn, kv_cache=kv_cache)
print(out.requires_grad)

# Try backward on mean
out.mean().backward()
print("Backward successful, video.grad is None?", video.grad is None)
