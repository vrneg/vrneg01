"""Torch adapters over the fixed-grid representation used by ROCKET models."""

from __future__ import annotations

from typing import Any

import torch
from torch.utils.data import Dataset

try:
    from minirocket.data import (
        DataBundle,
        TimeSeriesSplit,
        channel_names,
        prepare_data,
        window_to_array,
    )
except ModuleNotFoundError as error:
    if error.name != "minirocket":
        raise
    from ..minirocket.data import (
        DataBundle,
        TimeSeriesSplit,
        channel_names,
        prepare_data,
        window_to_array,
    )


class FixedGridTorchDataset(Dataset[dict[str, Any]]):
    """Zero-copy tensor view of one in-memory fixed-grid split."""

    def __init__(self, split: TimeSeriesSplit):
        self.values = torch.from_numpy(split.values)
        self.labels = torch.from_numpy(split.labels).float()
        self.sample_ids = list(split.sample_ids)

    def __len__(self) -> int:
        return int(self.labels.shape[0])

    def __getitem__(self, index: int) -> dict[str, Any]:
        return {
            "values": self.values[index],
            "label": self.labels[index],
            "sample_id": self.sample_ids[index],
        }


__all__ = [
    "DataBundle",
    "FixedGridTorchDataset",
    "TimeSeriesSplit",
    "channel_names",
    "prepare_data",
    "window_to_array",
]
