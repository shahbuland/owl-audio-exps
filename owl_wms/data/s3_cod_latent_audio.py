from npy_table import NpyTable

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler, Dataset

from dotenv import load_dotenv
load_dotenv()


class AutoEpochDistributedSampler(DistributedSampler):
    """Ensure we shuffle every epoch"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._auto_epoch = 0

    def __iter__(self):
        self.set_epoch(self._auto_epoch)
        self._auto_epoch += 1
        return super().__iter__()


class WindowedViewDataset(Dataset):
    """
    A sliding-window view over an NpyTable.
    Indexes into (row_idx, start_offset) pairs.
    """
    def __init__(
        self,
        table_dir: str,
        window_length: int,
        include_missing_features: bool = False,
        include_truncated: bool = True,
        meta_cols: tuple = ("tarball", "pt_idx", "missing", "truncated", "seq_len"),
    ):
        self.window_length = window_length
        self.table = NpyTable(table_dir)

        seq_len, missing, truncated = self.table[["seq_len", "missing", "truncated"]]
        self.columns = [c for c in self.table.columns if c not in meta_cols]

        index = []
        for i, (L, miss, trunc) in enumerate(zip(seq_len, missing, truncated)):
            if not include_missing_features and miss:
                continue
            if not include_truncated and trunc:
                continue
            for start in range(0, L, window_length):
                index.append((i, start))

        print(f"{len(index)} samples qualified out of {len(seq_len)} total videos")

        self._index = index

    def __len__(self):
        return len(self._index)

    def __getitem__(self, idx):
        row, start = self._index[idx]
        return {
            col: torch.from_numpy(self.table[col][row][start: start + self.window_length])
            for col in self.columns
        }


def collate_fn(batch):
    stacked = {
        k: torch.nan_to_num(torch.stack([item[k] for item in batch]), nan=0.0)  # TODO: preprocessing step instead
        for k in batch[0]
    }
    return [stacked[k] for k in ("video", "audio", "mouse", "buttons")]


def get_loader(batch_size, dataset_path, window_length):
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    rank = dist.get_rank() if dist.is_initialized() else 0

    ds = WindowedViewDataset(dataset_path, window_length)

    if world_size > 1:
        sampler = AutoEpochDistributedSampler(ds, num_replicas=world_size, rank=rank, shuffle=True)
        loader_kwargs = dict(sampler=sampler, shuffle=False)  # shuffle in sampler
    else:
        loader_kwargs = dict(shuffle=True)  # no sampler, shuffle in dataloader

    return DataLoader(
        ds,
        batch_size=batch_size,
        collate_fn=collate_fn,
        num_workers=8,
        drop_last=True,
        pin_memory=True,
        prefetch_factor=8,
        **loader_kwargs
    )
