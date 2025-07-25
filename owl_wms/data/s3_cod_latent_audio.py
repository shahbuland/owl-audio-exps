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

        seq_len = self.dataset.data["seq_len"].to_pylist()
        missing = self.dataset.data["missing"].to_pylist()
        truncated = self.dataset.data["truncated"].to_pylist()

        self.columns = [c for c in self.dataset.column_names if c not in meta_cols]
        self.dataset.set_format(type="torch", columns=self.columns)

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
        res = {
            col: item[col][start: start + self.window_length]
            for col in self.columns
        }
        return res


def collate_fn(batch):
    stacked = {
        k: torch.nan_to_num(torch.stack([item[k] for item in batch]), nan=0.0)  # TODO: preprocessing step instead
        for k in batch[0]
    }

    # TODO: do this as a dataset preprocessing step instead
    stacked["latent"] = torch.nan_to_num(stacked["latent"], nan=0.0)
    stacked["latent"] = torch.clamp(stacked["latent"], -8, 8)

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
        num_workers=8,
        drop_last=True,
        pin_memory=True,
        prefetch_factor=8,
        **loader_kwargs
    )
