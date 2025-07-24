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
            seq_key: str = "seq_len",
            include_missing_features: bool = False,
            include_truncated: bool = True,
            meta_cols=("tarball", "pt_idx", "missing_feature", "truncated", "seq_len"),

    ):
        self.window_length = window_length

        # load the dataset and convert feature columns to torch
        pa_ds = pds.dataset(dataset_path, format="ipc")
        pa_table = pa_ds.to_table()
        ds = datasets.Dataset(pa_table)   # keep HF-Dataset for convenient indexing

        self.columns = [c for c in self.ds.column_names if c not in meta_cols]
        ds.set_format(type="python", columns=self.columns + list(meta_cols))
        self.ds = ds

        # calculate list of unique sample keys (dataset_row_idx, window_start_offset)
        pairs = []
        for i, L in enumerate(ds[seq_key]):
            if (not include_missing_features) and ds[i]["missing_feature"]:
                continue
            if (not include_truncated) and ds[i]["truncated"]:
                continue
            pairs.extend((i, j * window_length) for j in range(int(L) // window_length))

        print(f"{pairs} samples in epoch")

        self._index = pairs

    def __len__(self):
        return len(self._index)

    def __getitem__(self, idx):
        row, start = self._index[idx]
        end = start + self.window_length
        ex = self.ds[row]
        seq_len = self.ds[row]["seq_len"]
        return {
            k: torch.frombuffer(memoryview(ex[k]), dtype=torch.float32).view(seq_len, -1)[start:end]
            for k in self.columns
        }


def collate_fn(batch):
    stacked = {k: torch.stack([item[k] for item in batch]) for k in batch[0]}
    return [stacked[k] for k in ("latent", "audio", "mouse", "buttons")]


def get_loader(batch_size, dataset_path, window_length, **_):  # TODO: no extra kwargs
    print(_)
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
