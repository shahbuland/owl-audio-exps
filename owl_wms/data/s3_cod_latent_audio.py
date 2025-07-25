from datasets import load_dataset

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
            meta_cols=("tarball", "pt_idx", "missing", "truncated", "seq_len"),

    ):
        self.window_length = window_length

        # load the dataset and convert feature columns to torch
        self.dataset = load_dataset(
            "parquet",
            data_files=f"{dataset_path}/*.parquet",
            split=split,
            keep_in_memory=False
        )

        seq_len, missing, truncated = self.dataset["seq_len"].to_pylist(), self.dataset["missing"].to_pylist(), self.dataset["truncated"].to_pylist()

        self.columns = [c for c in self.dataset.column_names if c not in meta_cols]
        self.dataset.set_format(type="numpy", columns=self.columns)

        # calculate list of unique sample keys (dataset_row_idx, window_start_offset)
        index = []
        for i, (L, is_missing, is_truncated) in enumerate(zip(seq_len, missing, truncated)):
            if (not include_missing_features) and is_missing:
                continue
            if (not include_truncated) and is_truncated:
                continue
            index.extend((i, j * window_length) for j in range(int(L) // window_length))
        print(f"{len(index)} samples qualified out of {len(seq_len)} total videos")

        self._index = index

    def __len__(self):
        return len(self._index)

    def __getitem__(self, idx):
        row, start = self._index[idx]
        item = self.dataset[row]
        return {
            col: torch.from_numpy(item[col][start: start + self.window_length])
            for col in self.columns
        }


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
        num_workers=1,
        drop_last=True,
        pin_memory=True,
        prefetch_factor=4,
        **loader_kwargs
    )
