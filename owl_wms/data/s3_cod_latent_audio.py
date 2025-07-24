import datasets
import pyarrow.dataset as pds

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
        if (not dist.is_initialized()) or dist.get_rank() == 0:
            print(f"Epoch: {self._auto_epoch}")
        return super().__iter__()


class WindowedViewDataset(Dataset):
    """Construct (row_idx, start) index over a hf dataset"""
    def __init__(
            self,
            dataset_path: str,
            window_length: int,
            split="train",
            include_missing_features: bool = False,
            include_truncated: bool = True,
            meta_cols=("tarball", "pt_idx", "missing_feature", "truncated", "seq_len"),

    ):
        self.window_length = window_length

        # load the dataset and convert feature columns to torch
        pa_ds = pds.dataset(dataset_path, format="ipc")
        table = pa_ds.to_table()
        self.table = table
        self.columns = [c for c in table.column_names if c not in meta_cols]

        # calculate list of unique sample keys (dataset_row_idx, window_start_offset)
        seq_len = table["seq_len"].to_pylist()
        missing_feature = table["missing_feature"].to_pylist()
        truncated = table["truncated"].to_pylist()

        pairs = []
        for i, (L, is_missing, is_truncated) in enumerate(zip(seq_len, missing_feature, truncated)):
            if (not include_missing_features) and is_missing:
                continue
            if (not include_truncated) and is_truncated:
                continue
            pairs.extend((i, j * window_length) for j in range(int(L) // window_length))
        print(f"{len(pairs)} samples qualified out of {len(seq_len)} total videos")

        self._index = pairs

    def __len__(self):
        return len(self._index)

    def __getitem__(self, idx):
        row, start = self._index[idx]
        end = start + self.window_length

        res = {}
        for k in self.columns:
            cell = self.table[k][row]
            t = torch.stack([torch.from_numpy(f) for f in cell[start:end]])
            res[k] = t
        return res


def collate_fn(batch):
    stacked = {k: torch.stack([item[k] for item in batch]) for k in batch[0]}
    return [stacked[k] for k in ("latent", "audio", "mouse", "buttons")]


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
        num_workers=4,
        drop_last=True,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=8,
        **loader_kwargs
    )
