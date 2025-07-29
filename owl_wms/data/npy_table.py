import json
import numpy as np
from pathlib import Path
from typing import List, Any


class NpyTable:
    # required fields per row
    columns = [
        "video", "audio", "mouse", "buttons",
        "tarball", "pt_idx", "missing", "truncated", "seq_len"
    ]
    # ndarray blobs
    array_columns = {"video", "audio", "mouse", "buttons"}

    def __init__(self, directory: str):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.directory / "manifest.json"
        if self.manifest_path.exists():
            self.manifest = json.loads(self.manifest_path.read_text())
        else:
            self.manifest = []

    def __len__(self):
        return len(self.manifest)

    def append(self, **row: Any) -> int:
        # must be exact keys
        if set(row) != set(self.columns):
            raise ValueError(f"Expected columns {self.columns}, got {list(row)}")

        idx = len(self.manifest)
        entry = {}
        for key, val in row.items():
            if key in self.array_columns:
                path = self.directory / f"{key}_{idx}.npy"
                np.save(path, val)
                entry[key] = f"{key}_{idx}.npy"
            else:
                entry[key] = val

        self.manifest.append(entry)
        self.manifest_path.write_text(json.dumps(self.manifest))
        return idx

    def __getitem__(self, key):
        if isinstance(key, str):
            return self.get(columns=[key])[0]
        elif isinstance(key, (list, tuple)):
            return self.get(columns=list(key))
        else:
            raise KeyError(f"Invalid key: {key!r}")

    def get(self, columns: List[str], rows: List[int] | None = None) -> List[List[Any]]:
        invalid = set(columns) - set(self.columns)
        if invalid:
            raise KeyError(f"Unknown columns requested: {invalid}")

        rows = range(len(self.manifest)) if rows is None else rows

        return [
            [
                np.load(self.directory / self.manifest[r][col], mmap_mode="r")
                if col in self.array_columns
                else self.manifest[r][col]
                for r in rows
            ]
            for col in columns
        ]
