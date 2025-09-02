import os
import random
from functools import partial
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler, Dataset


class AutoEpochDistributedSampler(DistributedSampler):
    """Ensure we shuffle every epoch"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._auto_epoch = 0

    def __iter__(self):
        self.set_epoch(self._auto_epoch)
        self._auto_epoch += 1
        return super().__iter__()


class LocalWaveFormDataset(Dataset):
    """
    Dataset for loading local waveform files with infinite iteration.
    Discovers *_wf.pt files in nested directories and randomly samples windows.
    """
    def __init__(self, root_dir: str, window_length: int):
        self.root_dir = root_dir
        self.window_length = window_length
        
        # Discover all *_wf.pt files
        self.waveform_paths = []
        for root, dirs, files in os.walk(root_dir):
            for file in files:
                if file.endswith('_wf.pt'):
                    self.waveform_paths.append(os.path.join(root, file))
        
        if not self.waveform_paths:
            raise ValueError(f"No *_wf.pt files found in {root_dir}")
        
        print(f"Found {len(self.waveform_paths)} waveform files in {root_dir}")

    def __len__(self):
        # Return a large number for infinite iteration
        return 1000000

    def __getitem__(self, idx):
        # Randomly select a waveform file
        path = random.choice(self.waveform_paths)
        
        # Load tensor with mmap and CPU mapping
        tensor = torch.load(path, mmap=True, map_location='cpu')
        
        # Tensor is [N, 2] where N is number of samples
        n_samples = tensor.shape[0]
        
        # Get random window
        if n_samples <= self.window_length:
            # If file is shorter than window, pad with zeros
            padded = torch.zeros(self.window_length, 2)
            padded[:n_samples] = tensor
            window = padded
        else:
            # Random start position
            start_idx = random.randint(0, n_samples - self.window_length)
            window = tensor[start_idx:start_idx + self.window_length]
        
        return {"waveform": window}


def collate_fn(batch):
    """Simple collate function that stacks waveforms"""
    return torch.stack([item["waveform"] for item in batch])


def get_loader(batch_size, root_dir, window_length):
    """Create DataLoader for LocalWaveFormDataset"""
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    rank = dist.get_rank() if dist.is_initialized() else 0

    ds = LocalWaveFormDataset(root_dir, window_length)

    if world_size > 1:
        sampler = AutoEpochDistributedSampler(ds, num_replicas=world_size, rank=rank, shuffle=True)
        loader_kwargs = dict(sampler=sampler, shuffle=False)
    else:
        loader_kwargs = dict(shuffle=True)

    return DataLoader(
        ds,
        batch_size=batch_size,
        collate_fn=collate_fn,
        num_workers=2,
        drop_last=True,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True,
        **loader_kwargs
    )