import torch
import os
from owl_wms.configs import Config
from owl_wms.data import get_loader
from tqdm import tqdm

def build_cache(n_samples=100, cfg_path="configs/causvid.yml", cache_dir="data_cache"):
    """
    Build a cache of random samples from the data loader.
    
    Args:
        n_samples: Number of samples to cache
        cfg_path: Path to config file
        cache_dir: Directory to save cached samples
    """
    # Create cache directory if it doesn't exist
    os.makedirs(cache_dir, exist_ok=True)
    
    # Load config and create data loader
    cfg = Config.from_yaml(cfg_path)
    train_cfg = cfg.train
    
    loader = get_loader(
        train_cfg.data_id,
        1,
        **train_cfg.data_kwargs
    )
    loader = iter(loader)

    # Sample and save n_samples
    for i in tqdm(range(n_samples)):
        history_buffer, audio_buffer, mouse_buffer, button_buffer = next(loader)
        
        # Convert to cuda and bfloat16
        history_buffer = history_buffer.cuda().bfloat16()
        audio_buffer = audio_buffer.cuda().bfloat16() 
        mouse_buffer = mouse_buffer.cuda().bfloat16()
        button_buffer = button_buffer.cuda().bfloat16()

        # Scale buffers
        history_buffer = history_buffer / train_cfg.vae_scale
        audio_buffer = audio_buffer / train_cfg.audio_vae_scale

        # Save tensors
        torch.save(history_buffer, os.path.join(cache_dir, f"history_buffer_{i}.pt"))
        torch.save(audio_buffer, os.path.join(cache_dir, f"audio_buffer_{i}.pt"))
        torch.save(mouse_buffer, os.path.join(cache_dir, f"mouse_buffer_{i}.pt"))
        torch.save(button_buffer, os.path.join(cache_dir, f"button_buffer_{i}.pt"))

if __name__ == "__main__":
    build_cache()
