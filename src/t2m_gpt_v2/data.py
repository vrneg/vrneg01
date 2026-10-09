"""Data loading for the continuous-latent VAE variant of T2M-GPT.

Everything except the tokenized-dataset container is reused directly from
``t2m_gpt.data`` -- fold loading, fixed-grid conversion, and channel bookkeeping do not
depend on whether stage one is discrete or continuous.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

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
except ModuleNotFoundError as error:  # pragma: no cover - import-path fallback
    if error.name != "t2m_gpt":
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


class MotionLatentDataset(Dataset[dict[str, Any]]):
    """Continuous motion-latent sequences produced by a frozen VAE tokenizer.

    The float-valued analogue of ``t2m_gpt.data.MotionTokenDataset``.
    """

    def __init__(
        self,
        latents: torch.Tensor,
        labels: torch.Tensor,
        sample_ids: Sequence[str],
    ) -> None:
        if latents.ndim != 3:
            raise ValueError("latents must have shape [windows, tokens, latent_dim]")
        if latents.shape[0] != labels.shape[0]:
            raise ValueError("latents and labels must describe the same windows")
        if len(sample_ids) != latents.shape[0]:
            raise ValueError("sample_ids must describe the same windows as latents")
        self.latents = latents.float()
        self.labels = labels.float()
        self.sample_ids = list(sample_ids)

    def __len__(self) -> int:
        return int(self.latents.shape[0])

    @property
    def num_tokens(self) -> int:
        return int(self.latents.shape[1])

    @property
    def latent_dim(self) -> int:
        return int(self.latents.shape[2])

    def __getitem__(self, index: int) -> dict[str, Any]:
        return {
            "latents": self.latents[index],
            "label": self.labels[index],
            "sample_id": self.sample_ids[index],
        }


__all__ = [
    "DataBundle",
    "FixedGridSplit",
    "MotionLatentDataset",
    "MotionWindowDataset",
    "channel_names",
    "load_fold_dataset",
    "make_loader",
    "modality_channel_slices",
    "prepare_data",
    "selected_channel_indices",
]
