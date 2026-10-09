"""Data loading and clip chopping for the GHTT-derived cascade.

Everything except clip chopping and the pose block's flat clip pool is reused directly
from ``t2m_gpt.data``/``t2m_gpt_v2.data`` -- fold loading and fixed-grid conversion do
not depend on the hierarchical cascade, and the action block's per-window sequence of
mid-level features is exactly ``t2m_gpt_v2.data.MotionLatentDataset``'s shape
(``[windows, n_clips, mid_dim]`` plus labels and sample ids), so it is reused rather
than reimplemented.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

try:
    from t2m_gpt.data import (
        DataBundle,
        FixedGridSplit,
        MotionWindowDataset,
        channel_names,
        load_fold_dataset,
        make_loader,
        modality_channel_slices,
        prepare_data,
        selected_channel_indices,
    )
    from t2m_gpt_v2.data import MotionLatentDataset
except ModuleNotFoundError as error:  # pragma: no cover - import-path fallback
    if error.name not in {"t2m_gpt", "t2m_gpt_v2"}:
        raise
    from ..t2m_gpt.data import (
        DataBundle,
        FixedGridSplit,
        MotionWindowDataset,
        channel_names,
        load_fold_dataset,
        make_loader,
        modality_channel_slices,
        prepare_data,
        selected_channel_indices,
    )
    from ..t2m_gpt_v2.data import MotionLatentDataset


def split_into_clips(values: np.ndarray, clip_length: int) -> np.ndarray:
    """``[windows, channels, frames]`` -> ``[windows, n_clips, clip_length, channels]``.

    Channel-last on the clip axis because the pose block's attention operates over
    ``[batch, clip_length, input_dim]``, matching ``t2m_gpt.model.TransformerBlock``'s
    expected layout, unlike the channel-first conv layout ``t2m_gpt.vqvae`` uses.
    """

    windows, channels, frames = values.shape
    if frames % clip_length != 0:
        raise ValueError(f"{frames} frames is not divisible by clip_length {clip_length}")
    n_clips = frames // clip_length
    reshaped = values.reshape(windows, channels, n_clips, clip_length)
    return np.ascontiguousarray(reshaped.transpose(0, 2, 3, 1))


class ClipDataset(Dataset[torch.Tensor]):
    """A flat pool of clips for pose-block training, independent of window grouping."""

    def __init__(self, clips: torch.Tensor) -> None:
        if clips.ndim != 3:
            raise ValueError("clips must have shape [num_clips, clip_length, channels]")
        self.clips = clips.float()

    def __len__(self) -> int:
        return int(self.clips.shape[0])

    def __getitem__(self, index: int) -> torch.Tensor:
        return self.clips[index]


def flatten_clips_for_pose_training(split: FixedGridSplit, clip_length: int) -> ClipDataset:
    """Every clip from every window in one split, as an unordered pool."""

    clips = split_into_clips(split.values, clip_length)
    windows, n_clips, actual_clip_length, channels = clips.shape
    flat = clips.reshape(windows * n_clips, actual_clip_length, channels)
    return ClipDataset(torch.from_numpy(flat))


__all__ = [
    "ClipDataset",
    "DataBundle",
    "FixedGridSplit",
    "MotionLatentDataset",
    "MotionWindowDataset",
    "channel_names",
    "flatten_clips_for_pose_training",
    "load_fold_dataset",
    "make_loader",
    "modality_channel_slices",
    "prepare_data",
    "selected_channel_indices",
    "split_into_clips",
]
