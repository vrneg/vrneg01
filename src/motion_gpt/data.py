"""Fold loading for MotionGPT, plus loaders over instruction-formatted task examples.

Stage one is identical to the T2M-GPT package, so the fixed-grid conversion and the
tokenizer datasets are re-exported rather than reimplemented: both models therefore read
exactly the same channels, grid, and folds. Only the batching of instruction sequences is
specific to MotionGPT.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

try:
    from t2m_gpt.data import (
        DataBundle,
        FixedGridSplit,
        MotionTokenDataset,
        MotionWindowDataset,
        channel_names,
        load_fold_dataset,
        modality_channel_slices,
        prepare_data,
        selected_channel_indices,
    )
except ModuleNotFoundError as error:
    if error.name != "t2m_gpt":
        raise
    from ..t2m_gpt.data import (
        DataBundle,
        FixedGridSplit,
        MotionTokenDataset,
        MotionWindowDataset,
        channel_names,
        load_fold_dataset,
        modality_channel_slices,
        prepare_data,
        selected_channel_indices,
    )

from .tasks import TaskBatchCollator


def _seed_worker(worker_id: int) -> None:
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)


def make_task_loader(
    dataset: Dataset[Any],
    collator: TaskBatchCollator,
    batch_size: int,
    shuffle: bool,
    seed: int,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> DataLoader:
    """Create a deterministic loader that pads instruction sequences per batch.

    ``num_workers`` stays at zero by default: the self-supervised datasets resample their
    corruption from an epoch counter held on the dataset object, which worker processes
    would not see updated.
    """

    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        collate_fn=collator,
        generator=generator if shuffle else None,
        worker_init_fn=_seed_worker if num_workers > 0 else None,
        persistent_workers=num_workers > 0,
    )


__all__ = [
    "DataBundle",
    "FixedGridSplit",
    "MotionTokenDataset",
    "MotionWindowDataset",
    "channel_names",
    "load_fold_dataset",
    "make_task_loader",
    "modality_channel_slices",
    "prepare_data",
    "selected_channel_indices",
]
