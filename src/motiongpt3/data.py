"""Data loading for the MotionGPT3-derived diffusion-head classifier.

Nothing here is genuinely new: fold loading and windowing are reused directly from
``t2m_gpt.data``, the discrete tokenized-sequence dataset from ``t2m_gpt.data``, and the
continuous latent-sequence dataset from ``t2m_gpt_v2.data`` -- both already have exactly
the shapes ``MotionSummarizer`` expects (``[windows, tokens]`` / ``[windows, tokens, dim]``).
"""

from __future__ import annotations

try:
    from t2m_gpt.data import (
        DataBundle,
        FixedGridSplit,
        MotionTokenDataset,
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
        MotionTokenDataset,
        MotionWindowDataset,
        channel_names,
        load_fold_dataset,
        make_loader,
        modality_channel_slices,
        prepare_data,
        selected_channel_indices,
    )
    from ..t2m_gpt_v2.data import MotionLatentDataset


__all__ = [
    "DataBundle",
    "FixedGridSplit",
    "MotionLatentDataset",
    "MotionTokenDataset",
    "MotionWindowDataset",
    "channel_names",
    "load_fold_dataset",
    "make_loader",
    "modality_channel_slices",
    "prepare_data",
    "selected_channel_indices",
]
