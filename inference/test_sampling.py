from owl_wms.data import get_loader
from owl_wms.configs import Config
from owl_wms import from_pretrained

from owl_wms.utils import batch_permute_to_length
from owl_wms.utils.owl_vae_bridge import make_batched_decode_fn
from owl_wms.utils.logging import LogHelper, to_wandb_av
from owl_wms.nn.rope import RoPE

import torch

cfg_path = "configs/dit_v4.yml"
ckpt_path = "/mnt/data/lapp0/checkpoints/89220499/checkpoints/step_120000.pt"

model, decoder = from_pretrained(cfg_path, ckpt_path, return_decoder=True)
model = model.core.eval().cuda().bfloat16()

# Find any RoPE modules in the model and cast their cos and sin back to fp32
def cast_rope_buffers_to_fp32(module):
    for submodule in module.modules():
        if isinstance(submodule, RoPE):
            if hasattr(submodule, "cos"):
                submodule.cos = submodule.cos.float()
            if hasattr(submodule, "sin"):
                submodule.sin = submodule.sin.float()

cast_rope_buffers_to_fp32(model)

decoder = decoder.eval().cuda().bfloat16()

cfg = Config.from_yaml(cfg_path)
train_cfg = cfg.train

print("Done Model Loading")

decode_fn = make_batched_decode_fn(decoder, 8)

print("Done Decoder Loading")

import wandb

wandb.init(
    project="video_models",
    entity="shahbuland",
    name="video_dit_v4"
)

from owl_wms.sampling import get_sampler_cls

only_return_generated = train_cfg.sampler_kwargs.pop("only_return_generated")
import os

sampler = get_sampler_cls(train_cfg.sampler_id)(**train_cfg.sampler_kwargs)

cache_path = "test_sampling_cache.pt"

if os.path.exists(cache_path):
    print(f"Loading cached data from {cache_path}")
    cache = torch.load(cache_path)
    vid = cache["vid"]
    mouse = cache["mouse"]
    btn = cache["btn"]
else:
    loader = get_loader(
        train_cfg.data_id,
        1,  # batch size must be 1 for the loader
        **train_cfg.sample_data_kwargs
    )

    loader = iter(loader)
    vids, mouses, btns, doc_ids = [], [], [], []
    for _ in range(16):
        vid, mouse, btn, doc_id = [t.bfloat16().cuda() for t in next(loader)]
        vids.append(vid)
        mouses.append(mouse)
        btns.append(btn)
        doc_ids.append(doc_id)
    # Stack along batch dimension
    vids = torch.cat(vids, dim=0)
    mouses = torch.cat(mouses, dim=0)
    btns = torch.cat(btns, dim=0)
    # Only use the first video, but all mouse/btn for batch_permute_to_length
    vid = vids[:1]
    mouse, btn = batch_permute_to_length(mouses, btns, sampler.num_frames + vid.size(1))
    mouse = mouse[:1]
    btn = btn[:1]
    # Save to cache
    torch.save({"vid": vid, "mouse": mouse, "btn": btn}, cache_path)
    print(f"Saved data to cache at {cache_path}")

with torch.no_grad():

    latent_vid = sampler(model, vid, mouse, btn, compile_on_decode = True)

    latent_vid = latent_vid[:, vid.size(1):]
    mouse = mouse[:, vid.size(1):]
    btn = btn[:, vid.size(1):]

    del model

    video = decode_fn(latent_vid * train_cfg.vae_scale)

wandb_av_out = to_wandb_av(video, None, mouse, btn)

if len(wandb_av_out) == 3:
    video, depth_gif, flow_gif = wandb_av_out
    eval_wandb_dict = dict(samples=video, depth_gif=depth_gif, flow_gif=flow_gif)
elif len(wandb_av_out) == 2:
    video, depth_gif = wandb_av_out
    eval_wandb_dict = dict(samples=video, depth_gif=depth_gif)
else:
    eval_wandb_dict = dict(samples=wandb_av_out)

wandb.log(eval_wandb_dict)

print("Done")