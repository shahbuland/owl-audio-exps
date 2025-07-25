from datasets import load_dataset

import numpy as np
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
    """Construct (row_idx, start) index over a hf dataset."""

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

        dataset = load_dataset(
            "parquet",
            data_files=f"{dataset_path}/*.parquet",
            split=split,
            keep_in_memory=False,
        )
        self.arrow_table = dataset.data

        seq_len = dataset.data["seq_len"].to_pylist()
        missing = dataset.data["missing"].to_pylist()
        truncated = dataset.data["truncated"].to_pylist()

        self.columns = [c for c in dataset.column_names if c not in meta_cols]

        index = []
        for i, (L, is_missing, is_truncated) in enumerate(zip(seq_len, missing, truncated)):
            if (not include_missing_features) and is_missing:
                continue
            if (not include_truncated) and is_truncated:
                continue
            index.extend((i, j * window_length) for j in range(int(L) // window_length))
        print(f"{len(index)} samples qualified out of {len(seq_len)} total videos")

        self._index = index

        # Pre-calculate inner shapes once to avoid redundant work in __getitem__
        self.inner_shapes = {}
        if self._index:
            first_valid_row_idx = self._index[0][0]
            for col in self.columns:
                example_scalar = self.arrow_table[col][first_valid_row_idx]
                if len(example_scalar.values) > 0:
                    self.inner_shapes[col] = np.array(example_scalar.values[0].as_py()).shape


    def __len__(self):
        return len(self._index)

    def __getitem__(self, idx):
        import time
        tstart = time.time()

        row_idx, start_frame = self._index[idx]
        res = {}
        for col in self.columns:
            list_scalar = self.arrow_table[col][row_idx]
            window_view = list_scalar.values.slice(start_frame, self.window_length)

            # Flatten to a 1D buffer, convert to numpy, then reshape to the correct
            # multi-dimensional shape. This avoids creating an array of dtype=object.
            numpy_flat = window_view.flatten().to_numpy(zero_copy_only=False)

            inner_shape = self.inner_shapes[col]
            target_shape = (self.window_length, *inner_shape)
            numpy_slice = numpy_flat.reshape(target_shape)

            res[col] = torch.from_numpy(numpy_slice)
        print("get sample time", time.time() - tstart)
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
        num_workers=8,
        drop_last=True,
        pin_memory=True,
        prefetch_factor=4,
        **loader_kwargs
    )
