"""Fold loading and the fixed-grid motion representation consumed by the VQ-VAE.

Folds are read with the datasets loader rather than from a materialized copy inside the
repository: :func:`load_fold_dataset` accepts a Hugging Face Hub repository id, a local
directory of parquet files, or a local ``save_to_disk`` directory.

Irregular sensor observations become a dense ``[channels, frames]`` array using the same
per-event feature schema, actor scope, presence channels, and linear interpolation as the
ROCKET and TCN baselines, so the tokenizer sees the representation those models saw.
"""

from __future__ import annotations

import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from datasets import DatasetDict, load_dataset, load_from_disk
from torch.utils.data import DataLoader, Dataset

try:
    from event_transformer.data import (
        EventSequenceEncoder,
        EventWindowDataset,
        fit_feature_normalizer,
    )
    from event_transformer.features import (
        MODALITY_NAMES,
        EventFeatureExtractor,
        FeatureNormalizer,
    )
    from minirocket.data import channel_names as base_channel_names
    from minirocket.data import window_to_array
except ModuleNotFoundError as error:
    if error.name not in {"event_transformer", "minirocket"}:
        raise
    from ..event_transformer.data import (
        EventSequenceEncoder,
        EventWindowDataset,
        fit_feature_normalizer,
    )
    from ..event_transformer.features import (
        MODALITY_NAMES,
        EventFeatureExtractor,
        FeatureNormalizer,
    )
    from ..minirocket.data import channel_names as base_channel_names
    from ..minirocket.data import window_to_array

try:
    from representation import (
        apply_mirror,
        build_mirror_transform,
        channel_normalization_arrays,
        fit_representation,
        grid_delta_seconds,
    )
except ModuleNotFoundError as error:  # pragma: no cover - import-path fallback
    if error.name != "representation":
        raise
    from ..representation import (
        apply_mirror,
        build_mirror_transform,
        channel_normalization_arrays,
        fit_representation,
        grid_delta_seconds,
    )

from .config import DataConfig


REQUIRED_SPLITS: tuple[str, ...] = ("train", "validation", "test")
REQUIRED_COLUMNS: frozenset[str] = frozenset({"context", "word", "label"})


@dataclass(slots=True)
class FixedGridSplit:
    """One split as dense motion windows with labels and grouping metadata."""

    values: np.ndarray
    labels: np.ndarray
    sample_ids: list[str]
    group_ids: list[str]

    def __post_init__(self) -> None:
        if self.values.ndim != 3:
            raise ValueError("values must have shape [windows, channels, frames]")
        if self.values.shape[0] != self.labels.shape[0]:
            raise ValueError("values and labels must describe the same windows")
        if len(self.sample_ids) != self.values.shape[0]:
            raise ValueError("sample_ids must describe the same windows as values")
        if len(self.group_ids) != self.values.shape[0]:
            raise ValueError("group_ids must describe the same windows as values")

    @property
    def num_channels(self) -> int:
        return int(self.values.shape[1])

    @property
    def num_frames(self) -> int:
        return int(self.values.shape[2])


@dataclass(slots=True)
class DataBundle:
    """Prepared splits plus the preprocessing state needed to reproduce them."""

    train: FixedGridSplit
    validation: FixedGridSplit
    test: FixedGridSplit
    normalizer: FeatureNormalizer | None
    time_grid: np.ndarray
    channel_names: tuple[str, ...]

    @property
    def num_channels(self) -> int:
        return len(self.channel_names)

    @property
    def num_frames(self) -> int:
        return int(self.time_grid.shape[0])


def _looks_like_saved_dataset(path: Path) -> bool:
    return (path / "dataset_dict.json").exists() or (path / "dataset_info.json").exists()


def load_fold_dataset(dataset: str | Path) -> DatasetDict:
    """Load one fold from the Hub, a parquet directory, or a saved dataset directory."""

    path = Path(str(dataset))
    if path.exists():
        if _looks_like_saved_dataset(path):
            loaded = load_from_disk(str(path))
        else:
            parquet_files = sorted(path.rglob("*.parquet"))
            if not parquet_files:
                raise ValueError(
                    f"{path} is neither a saved dataset nor a parquet directory"
                )
            loaded = load_dataset("parquet", data_dir=str(path))
    else:
        loaded = load_dataset(str(dataset))

    if not isinstance(loaded, DatasetDict):
        raise TypeError(
            f"Expected a DatasetDict for {dataset!r}, got {type(loaded).__name__}"
        )
    missing_splits = set(REQUIRED_SPLITS) - set(loaded)
    if missing_splits:
        raise ValueError(f"Dataset is missing splits: {sorted(missing_splits)}")
    for split_name in REQUIRED_SPLITS:
        missing_columns = REQUIRED_COLUMNS - set(loaded[split_name].column_names)
        if missing_columns:
            raise ValueError(
                f"Split {split_name!r} is missing columns: {sorted(missing_columns)}"
            )
    return loaded


def channel_names(config: DataConfig) -> tuple[str, ...]:
    """Return the channel order after applying the modality selection."""

    names = base_channel_names(config)
    if config.modalities is None:
        return names
    selected = set(config.modalities)
    return tuple(name for name in names if name.split(".", 1)[0] in selected)


def selected_channel_indices(config: DataConfig) -> np.ndarray:
    """Return positions of the selected modalities within the full channel order."""

    names = base_channel_names(config)
    if config.modalities is None:
        return np.arange(len(names), dtype=np.int64)
    selected = set(config.modalities)
    return np.asarray(
        [
            index
            for index, name in enumerate(names)
            if name.split(".", 1)[0] in selected
        ],
        dtype=np.int64,
    )


def modality_channel_slices(config: DataConfig) -> dict[str, tuple[int, int]]:
    """Return ``[start, stop)`` channel bounds per retained modality.

    The bounds index the selected channel order and are contiguous because
    :func:`channel_names` preserves the modality-major layout.
    """

    names = channel_names(config)
    bounds: dict[str, tuple[int, int]] = {}
    for modality in MODALITY_NAMES:
        if config.modalities is not None and modality not in config.modalities:
            continue
        positions = [
            index
            for index, name in enumerate(names)
            if name.split(".", 1)[0] == modality
        ]
        if positions:
            bounds[modality] = (positions[0], positions[-1] + 1)
    return bounds


def _record_id(value: Any) -> str:
    if isinstance(value, Mapping):
        return f"{value.get('table_name', 'record')}:{value.get('id')!r}"
    return str(value)


def _convert_split(
    source: Sequence[Mapping[str, Any]],
    dataset: EventWindowDataset,
    config: DataConfig,
    time_grid: np.ndarray,
    channel_indices: np.ndarray,
) -> FixedGridSplit:
    windows = [dataset[index] for index in range(len(dataset))]
    values = np.stack(
        [window_to_array(window, config, time_grid) for window in windows]
    ).astype(np.float32, copy=False)
    values = np.ascontiguousarray(values[:, channel_indices, :])
    return FixedGridSplit(
        values=values,
        labels=np.asarray([int(window.label) for window in windows], dtype=np.int64),
        sample_ids=[window.sample_id for window in windows],
        group_ids=[
            _record_id(source[index]["word"].get("experiment"))
            for index in range(len(source))
        ],
    )


def prepare_data(config: DataConfig) -> DataBundle:
    """Load one fold and fit every data-dependent step on the training split only."""

    config.validate()
    fold = load_fold_dataset(config.dataset)
    encoder = EventSequenceEncoder(
        EventFeatureExtractor(),
        max_events_per_modality=config.max_events_per_modality,
        time_clip_seconds=None,
        positive_label=config.positive_label,
        negative_label=config.negative_label,
    )

    train_dataset = EventWindowDataset(
        fold["train"], encoder=encoder, cache_in_memory=config.cache_in_memory
    )
    normalizer = (
        fit_feature_normalizer(train_dataset) if config.normalize_features else None
    )
    if normalizer is not None:
        train_dataset.apply_normalizer(normalizer)
    evaluation_datasets = {
        split: EventWindowDataset(
            fold[split],
            encoder=encoder,
            normalizer=normalizer,
            cache_in_memory=config.cache_in_memory,
        )
        for split in ("validation", "test")
    }

    time_grid = np.linspace(
        config.window_start_seconds,
        config.window_end_seconds,
        config.num_time_points,
        dtype=np.float32,
    )
    channel_indices = selected_channel_indices(config)
    splits = {
        "train": _convert_split(
            fold["train"], train_dataset, config, time_grid, channel_indices
        ),
        "validation": _convert_split(
            fold["validation"],
            evaluation_datasets["validation"],
            config,
            time_grid,
            channel_indices,
        ),
        "test": _convert_split(
            fold["test"],
            evaluation_datasets["test"],
            config,
            time_grid,
            channel_indices,
        ),
    }
    names = channel_names(config)

    if config.mirror_augment_train:
        # Applied before fit_representation so derived-channel statistics (accel
        # means/scales, AU pooling, ...) are fit on the doubled training set too, and
        # so every representation variant sees a mirror-consistent input regardless of
        # which channels it derives or drops.
        means, scales = channel_normalization_arrays(names, normalizer)
        mirror = build_mirror_transform(names)
        train_split = splits["train"]
        mirrored_values = apply_mirror(train_split.values, mirror, means, scales)
        splits["train"] = FixedGridSplit(
            values=np.concatenate([train_split.values, mirrored_values], axis=0),
            labels=np.concatenate([train_split.labels, train_split.labels], axis=0),
            sample_ids=[
                *train_split.sample_ids,
                *(f"{sample_id}-mirror" for sample_id in train_split.sample_ids),
            ],
            group_ids=[*train_split.group_ids, *train_split.group_ids],
        )

    if not config.representation.is_identity:
        # Fitted on the training split alone, then applied unchanged to the evaluation
        # splits, so an alternative representation cannot leak held-out information.
        transform = fit_representation(
            splits["train"].values,
            names,
            normalizer,
            config.representation,
            grid_delta_seconds(time_grid),
        )
        for split in splits.values():
            split.values = transform.apply(split.values)
        names = transform.channel_names

    return DataBundle(
        train=splits["train"],
        validation=splits["validation"],
        test=splits["test"],
        normalizer=normalizer,
        time_grid=time_grid,
        channel_names=names,
    )


class MotionWindowDataset(Dataset[dict[str, Any]]):
    """Zero-copy tensor view of one in-memory fixed-grid split."""

    def __init__(self, split: FixedGridSplit) -> None:
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


class MotionTokenDataset(Dataset[dict[str, Any]]):
    """Discrete motion-token sequences produced by a frozen tokenizer."""

    def __init__(
        self,
        indices: torch.Tensor,
        labels: torch.Tensor,
        sample_ids: Sequence[str],
    ) -> None:
        if indices.ndim != 2:
            raise ValueError("indices must have shape [windows, tokens]")
        if indices.shape[0] != labels.shape[0]:
            raise ValueError("indices and labels must describe the same windows")
        if len(sample_ids) != indices.shape[0]:
            raise ValueError("sample_ids must describe the same windows as indices")
        self.indices = indices.long()
        self.labels = labels.float()
        self.sample_ids = list(sample_ids)

    def __len__(self) -> int:
        return int(self.indices.shape[0])

    @property
    def num_tokens(self) -> int:
        return int(self.indices.shape[1])

    def __getitem__(self, index: int) -> dict[str, Any]:
        return {
            "indices": self.indices[index],
            "label": self.labels[index],
            "sample_id": self.sample_ids[index],
        }


def _seed_worker(worker_id: int) -> None:
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_loader(
    dataset: Dataset[dict[str, Any]],
    batch_size: int,
    shuffle: bool,
    seed: int,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> DataLoader:
    """Create a deterministic loader over one prepared split."""

    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        generator=generator if shuffle else None,
        worker_init_fn=_seed_worker if num_workers > 0 else None,
        persistent_workers=num_workers > 0,
    )
