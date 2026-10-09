"""Dataset adaptation, chronological event encoding, normalization, and batching."""

from __future__ import annotations

import random
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from datasets import DatasetDict, load_from_disk
from torch.utils.data import DataLoader, Dataset

from .config import DataConfig
from .features import (
    ACTOR_DIFFERENT_FROM_ANCHOR,
    ACTOR_SAME_AS_ANCHOR,
    ACTOR_UNKNOWN,
    FINGER_FLAG_DIM,
    FINGER_MODALITY_IDS,
    MODALITY_DIMS,
    MODALITY_TO_ID,
    MODALITY_NAMES,
    EventFeatureExtractor,
    FeatureNormalizer,
    augment_with_motion,
    extract_finger_flags,
)


TIME_FEATURE_DIM = 3


@dataclass(slots=True)
class EncodedWindow:
    """One window encoded as modality observations mapped to temporal tokens."""

    modality_features: dict[int, torch.Tensor]
    modality_time_step_indices: dict[int, torch.Tensor]
    modality_actor_relation_ids: dict[int, torch.Tensor]
    time_features: torch.Tensor
    finger_flags: dict[int, torch.Tensor]
    label: float
    sample_id: str

    @property
    def num_events(self) -> int:
        return sum(features.shape[0] for features in self.modality_features.values())

    @property
    def num_time_steps(self) -> int:
        return int(self.time_features.shape[0])

    def normalized(self, normalizer: FeatureNormalizer) -> "EncodedWindow":
        return EncodedWindow(
            modality_features={
                modality_id: normalizer.normalize(modality_id, features)
                for modality_id, features in self.modality_features.items()
            },
            modality_time_step_indices=self.modality_time_step_indices,
            modality_actor_relation_ids=self.modality_actor_relation_ids,
            time_features=self.time_features,
            finger_flags=self.finger_flags,
            label=self.label,
            sample_id=self.sample_id,
        )

    def with_masked_channels(
        self, masked_indices_by_modality: Mapping[int, tuple[int, ...]]
    ) -> "EncodedWindow":
        modality_features = {}
        for modality_id, features in self.modality_features.items():
            indices = masked_indices_by_modality.get(modality_id, ())
            if indices and features.numel():
                features = features.clone()
                features[:, list(indices)] = 0.0
            modality_features[modality_id] = features
        return EncodedWindow(
            modality_features=modality_features,
            modality_time_step_indices=self.modality_time_step_indices,
            modality_actor_relation_ids=self.modality_actor_relation_ids,
            time_features=self.time_features,
            finger_flags=self.finger_flags,
            label=self.label,
            sample_id=self.sample_id,
        )


def _masked_channel_indices(
    masked_channels: Sequence[str] | None,
) -> dict[int, tuple[int, ...]]:
    if masked_channels is None:
        return {}
    grouped: dict[int, list[int]] = {}
    for channel in masked_channels:
        modality_name, separator, suffix = channel.partition(".feature_")
        if not separator or modality_name not in MODALITY_TO_ID:
            raise ValueError(f"Unsupported masked channel name: {channel!r}")
        try:
            feature_index = int(suffix)
        except ValueError as error:
            raise ValueError(
                f"Unsupported masked channel name: {channel!r}"
            ) from error
        modality_id = MODALITY_TO_ID[modality_name]
        if not 0 <= feature_index < MODALITY_DIMS[modality_id]:
            raise ValueError(f"Unknown masked channel: {channel!r}")
        grouped.setdefault(modality_id, []).append(feature_index)
    return {
        modality_id: tuple(indices)
        for modality_id, indices in grouped.items()
    }


def _timestamp_ms(event: Mapping[str, Any]) -> float:
    value = event.get("timeMs", event.get("timestamp"))
    try:
        return float(value)
    except (TypeError, ValueError) as error:
        raise ValueError("Every event must have a numeric timeMs or timestamp") from error


def _record_id(value: Any) -> str:
    """Create a deterministic printable ID from a decoded SurrealDB record."""

    if isinstance(value, Mapping):
        table = value.get("table_name", "record")
        return f"{table}:{value.get('id')!r}"
    return str(value)


def _canonical_event_key(event: Mapping[str, Any]) -> tuple[float, int, str, int]:
    modality_id = MODALITY_TO_ID.get(str(event.get("event_type")), len(MODALITY_TO_ID))
    try:
        counter = int(event.get("counter", 0))
    except (TypeError, ValueError):
        counter = 0
    return (_timestamp_ms(event), modality_id, str(event.get("playerId", "")), counter)


def _uniform_indices(length: int, limit: int) -> list[int]:
    if length <= limit:
        return list(range(length))
    if limit == 1:
        return [length // 2]
    return [round(index * (length - 1) / (limit - 1)) for index in range(limit)]


def _subsample_by_modality(
    events: Sequence[Mapping[str, Any]],
    max_events_per_modality: int | None,
) -> list[Mapping[str, Any]]:
    """Uniformly limit high-rate streams without dropping entire modalities."""

    sorted_events = sorted(events, key=_canonical_event_key)
    if max_events_per_modality is None:
        return sorted_events

    by_modality: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for event in sorted_events:
        by_modality[str(event.get("event_type"))].append(event)

    selected: list[Mapping[str, Any]] = []
    for modality_events in by_modality.values():
        selected.extend(
            modality_events[index]
            for index in _uniform_indices(len(modality_events), max_events_per_modality)
        )
    return sorted(selected, key=_canonical_event_key)


def _clipped_seconds(milliseconds: float, clip_seconds: float | None) -> float:
    seconds = milliseconds / 1000.0
    if clip_seconds is None:
        return seconds
    return max(-clip_seconds, min(clip_seconds, seconds))


class EventSequenceEncoder:
    """Convert a raw Hugging Face row into a chronological event window."""

    def __init__(
        self,
        feature_extractor: EventFeatureExtractor,
        max_events_per_modality: int | None = 128,
        time_clip_seconds: float | None = 10.0,
        positive_label: str = "neg",
        negative_label: str = "none",
        included_modalities: Sequence[str] | None = None,
    ) -> None:
        self.feature_extractor = feature_extractor
        self.max_events_per_modality = max_events_per_modality
        self.time_clip_seconds = time_clip_seconds
        self.positive_label = positive_label
        self.negative_label = negative_label
        self.included_modalities = (
            None if included_modalities is None else frozenset(included_modalities)
        )
        if self.included_modalities is not None:
            unknown = self.included_modalities - set(MODALITY_NAMES)
            if unknown:
                raise ValueError(f"Unknown included modalities: {sorted(unknown)}")

    def encode(self, sample: Mapping[str, Any]) -> EncodedWindow:
        word = sample.get("word")
        if not isinstance(word, Mapping):
            raise ValueError("Each sample must contain a word mapping")
        context = sample.get("context")
        if not isinstance(context, Sequence) or isinstance(context, (str, bytes)):
            raise ValueError("Each sample must contain a context event sequence")

        anchor_time_ms = float(word["timeMs"])
        anchor_player_id = str(word.get("playerId", ""))
        observed_events = [
            event
            for event in context
            if self.feature_extractor.should_keep(event)
            and (
                self.included_modalities is None
                or str(event.get("event_type")) in self.included_modalities
            )
        ]
        events = _subsample_by_modality(observed_events, self.max_events_per_modality)

        time_step_keys = sorted(
            {
                (_timestamp_ms(event), str(event.get("playerId", "")))
                for event in events
            }
        )
        time_step_to_index = {
            time_step: index for index, time_step in enumerate(time_step_keys)
        }
        time_features = []
        for index, (timestamp, _) in enumerate(time_step_keys):
            previous_delta = (
                0.0 if index == 0 else timestamp - time_step_keys[index - 1][0]
            )
            next_delta = (
                0.0
                if index + 1 == len(time_step_keys)
                else time_step_keys[index + 1][0] - timestamp
            )
            time_features.append(
                [
                    _clipped_seconds(timestamp - anchor_time_ms, self.time_clip_seconds),
                    _clipped_seconds(previous_delta, self.time_clip_seconds),
                    _clipped_seconds(next_delta, self.time_clip_seconds),
                ]
            )

        previous_stream_values: dict[tuple[int, str], tuple[float, torch.Tensor]] = {}
        features_by_modality: dict[int, list[torch.Tensor]] = defaultdict(list)
        time_steps_by_modality: dict[int, list[int]] = defaultdict(list)
        actors_by_modality: dict[int, list[int]] = defaultdict(list)
        finger_flags_by_modality: dict[int, list[torch.Tensor]] = defaultdict(list)

        for event in events:
            modality_id, base_features = self.feature_extractor.extract(event)
            timestamp = _timestamp_ms(event)
            event_player_id = str(event.get("playerId", ""))

            if not event_player_id or not anchor_player_id:
                actor_relation_id = ACTOR_UNKNOWN
            elif event_player_id == anchor_player_id:
                actor_relation_id = ACTOR_SAME_AS_ANCHOR
            else:
                actor_relation_id = ACTOR_DIFFERENT_FROM_ANCHOR

            stream_key = (modality_id, event_player_id)
            previous_stream = previous_stream_values.get(stream_key)
            delta_seconds = (
                0.0 if previous_stream is None else (timestamp - previous_stream[0]) / 1000.0
            )
            previous_features = None if previous_stream is None else previous_stream[1]
            features = augment_with_motion(
                modality_id,
                current=base_features,
                previous=previous_features,
                delta_seconds=delta_seconds,
            )
            previous_stream_values[stream_key] = (timestamp, base_features)

            features_by_modality[modality_id].append(features)
            time_steps_by_modality[modality_id].append(
                time_step_to_index[(timestamp, event_player_id)]
            )
            actors_by_modality[modality_id].append(actor_relation_id)
            if modality_id in FINGER_MODALITY_IDS:
                finger_flags_by_modality[modality_id].append(
                    extract_finger_flags(event)
                )

        label_name = str(sample.get("label"))
        if label_name == self.positive_label:
            label = 1.0
        elif label_name == self.negative_label:
            label = 0.0
        else:
            raise ValueError(
                f"Unknown label {label_name!r}; expected "
                f"{self.negative_label!r} or {self.positive_label!r}"
            )

        stacked_features = {
            modality_id: (
                torch.stack(features_by_modality[modality_id])
                if features_by_modality[modality_id]
                else torch.empty((0, dimension), dtype=torch.float32)
            )
            for modality_id, dimension in MODALITY_DIMS.items()
        }

        return EncodedWindow(
            modality_features=stacked_features,
            modality_time_step_indices={
                modality_id: torch.tensor(
                    time_steps_by_modality[modality_id], dtype=torch.long
                )
                for modality_id in MODALITY_DIMS
            },
            modality_actor_relation_ids={
                modality_id: torch.tensor(
                    actors_by_modality[modality_id], dtype=torch.long
                )
                for modality_id in MODALITY_DIMS
            },
            time_features=torch.tensor(time_features, dtype=torch.float32).reshape(
                len(time_step_keys), TIME_FEATURE_DIM
            ),
            finger_flags={
                modality_id: (
                    torch.stack(finger_flags_by_modality[modality_id])
                    if finger_flags_by_modality[modality_id]
                    else torch.empty((0, FINGER_FLAG_DIM), dtype=torch.float32)
                )
                for modality_id in FINGER_MODALITY_IDS
            },
            label=label,
            sample_id=_record_id(word.get("id")),
        )


class EventWindowDataset(Dataset[EncodedWindow]):
    """Torch adapter for a Hugging Face split containing raw JSON contexts."""

    def __init__(
        self,
        source: Sequence[Mapping[str, Any]],
        encoder: EventSequenceEncoder,
        normalizer: FeatureNormalizer | None = None,
        masked_channels: Sequence[str] | None = None,
        cache_in_memory: bool = True,
    ) -> None:
        self.source = source
        self.encoder = encoder
        self.normalizer = normalizer
        self.masked_indices_by_modality = _masked_channel_indices(masked_channels)
        self._cache: list[EncodedWindow] | None = None
        if cache_in_memory:
            self._cache = [self._encode(index) for index in range(len(source))]

    def __len__(self) -> int:
        return len(self.source)

    def _encode(self, index: int) -> EncodedWindow:
        window = self.encoder.encode(self.source[index])
        if self.normalizer is not None:
            window = window.normalized(self.normalizer)
        if self.masked_indices_by_modality:
            window = window.with_masked_channels(self.masked_indices_by_modality)
        return window

    def __getitem__(self, index: int) -> EncodedWindow:
        if self._cache is not None:
            return self._cache[index]
        return self._encode(index)

    def apply_normalizer(self, normalizer: FeatureNormalizer) -> None:
        if self.normalizer is not None:
            raise RuntimeError("A normalizer has already been applied to this dataset")
        self.normalizer = normalizer
        if self._cache is not None:
            self._cache = [
                (
                    window.normalized(normalizer).with_masked_channels(
                        self.masked_indices_by_modality
                    )
                    if self.masked_indices_by_modality
                    else window.normalized(normalizer)
                )
                for window in self._cache
            ]


def fit_feature_normalizer(
    dataset: Dataset[EncodedWindow],
    modality_dims: Mapping[int, int] = MODALITY_DIMS,
    minimum_standard_deviation: float = 1e-6,
) -> FeatureNormalizer:
    """Fit population mean/std using only the supplied training dataset."""

    sums = {
        modality_id: torch.zeros(dimension, dtype=torch.float64)
        for modality_id, dimension in modality_dims.items()
    }
    squared_sums = {key: torch.zeros_like(value) for key, value in sums.items()}
    counts = {modality_id: 0 for modality_id in modality_dims}

    for window in dataset:
        for modality_id, features in window.modality_features.items():
            if features.numel() == 0:
                continue
            values = features.double()
            sums[modality_id] += values.sum(dim=0)
            squared_sums[modality_id] += values.square().sum(dim=0)
            counts[modality_id] += values.shape[0]

    means: dict[int, torch.Tensor] = {}
    standard_deviations: dict[int, torch.Tensor] = {}
    for modality_id, dimension in modality_dims.items():
        count = counts[modality_id]
        if count == 0:
            means[modality_id] = torch.zeros(dimension)
            standard_deviations[modality_id] = torch.ones(dimension)
            continue
        mean = sums[modality_id] / count
        variance = squared_sums[modality_id] / count - mean.square()
        standard_deviation = variance.clamp_min(0.0).sqrt()
        standard_deviation[standard_deviation < minimum_standard_deviation] = 1.0
        means[modality_id] = mean.float()
        standard_deviations[modality_id] = standard_deviation.float()

    return FeatureNormalizer(means, standard_deviations)


class EventBatchCollator:
    """Pad temporal tokens and map synchronous modalities into shared time steps."""

    def __init__(self, modality_dims: Mapping[int, int] = MODALITY_DIMS) -> None:
        self.modality_dims = dict(modality_dims)

    def __call__(self, windows: Sequence[EncodedWindow]) -> dict[str, Any]:
        if not windows:
            raise ValueError("Cannot collate an empty batch")

        batch_size = len(windows)
        sequence_lengths = [window.num_time_steps + 1 for window in windows]
        max_length = max(sequence_lengths)
        time_features = torch.zeros(
            (batch_size, max_length, TIME_FEATURE_DIM), dtype=torch.float32
        )
        event_mask = torch.zeros((batch_size, max_length), dtype=torch.bool)
        anchor_mask = torch.zeros((batch_size, max_length), dtype=torch.bool)
        padding_mask = torch.ones((batch_size, max_length), dtype=torch.bool)

        anchor_indices: list[int] = []
        for batch_index, window in enumerate(windows):
            relative_times = window.time_features[:, 0].contiguous()
            anchor_index = int(torch.searchsorted(relative_times, torch.tensor(0.0)).item())
            anchor_indices.append(anchor_index)
            total_length = window.num_time_steps + 1

            padding_mask[batch_index, :total_length] = False
            anchor_mask[batch_index, anchor_index] = True
            if anchor_index:
                time_features[batch_index, :anchor_index] = window.time_features[:anchor_index]
                event_mask[batch_index, :anchor_index] = True
            if window.num_time_steps > anchor_index:
                destination = slice(anchor_index + 1, total_length)
                time_features[batch_index, destination] = window.time_features[anchor_index:]
                event_mask[batch_index, destination] = True

        modality_features = {
            modality_id: torch.cat(
                [window.modality_features[modality_id] for window in windows], dim=0
            )
            for modality_id in self.modality_dims
        }
        modality_token_indices: dict[int, torch.Tensor] = {}
        modality_actor_relation_ids: dict[int, torch.Tensor] = {}
        for modality_id in self.modality_dims:
            token_indices = []
            actor_ids = []
            for batch_index, (window, anchor_index) in enumerate(
                zip(windows, anchor_indices, strict=True)
            ):
                source_indices = window.modality_time_step_indices[modality_id]
                padded_indices = source_indices + (source_indices >= anchor_index).long()
                token_indices.append(batch_index * max_length + padded_indices)
                actor_ids.append(window.modality_actor_relation_ids[modality_id])
            modality_token_indices[modality_id] = torch.cat(token_indices)
            modality_actor_relation_ids[modality_id] = torch.cat(actor_ids)

        finger_flags = {
            modality_id: torch.cat(
                [window.finger_flags[modality_id] for window in windows], dim=0
            )
            for modality_id in FINGER_MODALITY_IDS
        }

        for modality_id, features in modality_features.items():
            expected_events = modality_token_indices[modality_id].shape[0]
            expected_shape = (expected_events, self.modality_dims[modality_id])
            if features.shape != expected_shape:
                raise RuntimeError(
                    f"Modality {modality_id} has feature shape {tuple(features.shape)}, "
                    f"expected {expected_shape}"
                )
            if modality_actor_relation_ids[modality_id].shape != (expected_events,):
                raise RuntimeError("Actor-relation IDs do not align with modality events")
        for modality_id in FINGER_MODALITY_IDS:
            expected_finger_events = modality_token_indices[modality_id].shape[0]
            expected_shape = (expected_finger_events, FINGER_FLAG_DIM)
            if finger_flags[modality_id].shape != expected_shape:
                raise RuntimeError(
                    f"Finger flag shape is {tuple(finger_flags[modality_id].shape)}; "
                    f"expected {expected_shape}"
                )

        return {
            "modality_features": modality_features,
            "finger_flags": finger_flags,
            "modality_token_indices": modality_token_indices,
            "modality_actor_relation_ids": modality_actor_relation_ids,
            "time_features": time_features,
            "event_mask": event_mask,
            "anchor_mask": anchor_mask,
            "padding_mask": padding_mask,
            "labels": torch.tensor([window.label for window in windows], dtype=torch.float32),
            "sample_ids": [window.sample_id for window in windows],
            "sequence_lengths": torch.tensor(sequence_lengths, dtype=torch.long),
        }


@dataclass(slots=True)
class DataBundle:
    train_loader: DataLoader
    validation_loader: DataLoader
    test_loader: DataLoader
    normalizer: FeatureNormalizer | None
    datasets: dict[str, EventWindowDataset]


def _seed_worker(worker_id: int) -> None:
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def load_fold(path: str | Path) -> DatasetDict:
    fold = load_from_disk(str(path))
    if not isinstance(fold, DatasetDict):
        raise TypeError(f"Expected a DatasetDict at {path}, got {type(fold).__name__}")
    required_splits = {"train", "validation", "test"}
    missing_splits = required_splits - set(fold)
    if missing_splits:
        raise ValueError(f"Dataset is missing splits: {sorted(missing_splits)}")
    for split_name in required_splits:
        missing_columns = {"context", "word", "label"} - set(fold[split_name].column_names)
        if missing_columns:
            raise ValueError(
                f"Split {split_name!r} is missing columns: {sorted(missing_columns)}"
            )
    return fold


def prepare_data(config: DataConfig, seed: int = 42) -> DataBundle:
    """Load a saved fold and create normalized, padded PyTorch loaders."""

    config.validate()
    fold = load_fold(config.dataset_path)
    extractor = EventFeatureExtractor()
    encoder = EventSequenceEncoder(
        feature_extractor=extractor,
        max_events_per_modality=config.max_events_per_modality,
        time_clip_seconds=config.time_clip_seconds,
        positive_label=config.positive_label,
        negative_label=config.negative_label,
        included_modalities=config.included_modalities,
    )

    train_dataset = EventWindowDataset(
        fold["train"],
        encoder=encoder,
        masked_channels=config.masked_channels,
        cache_in_memory=config.cache_in_memory,
    )
    normalizer = fit_feature_normalizer(train_dataset) if config.normalize_features else None
    if normalizer is not None:
        train_dataset.apply_normalizer(normalizer)

    validation_dataset = EventWindowDataset(
        fold["validation"],
        encoder=encoder,
        normalizer=normalizer,
        masked_channels=config.masked_channels,
        cache_in_memory=config.cache_in_memory,
    )
    test_dataset = EventWindowDataset(
        fold["test"],
        encoder=encoder,
        normalizer=normalizer,
        masked_channels=config.masked_channels,
        cache_in_memory=config.cache_in_memory,
    )
    datasets = {
        "train": train_dataset,
        "validation": validation_dataset,
        "test": test_dataset,
    }

    collator = EventBatchCollator(MODALITY_DIMS)
    generator = torch.Generator().manual_seed(seed)
    evaluation_batch_size = config.evaluation_batch_size or config.batch_size
    common_loader_options = {
        "num_workers": config.num_workers,
        "collate_fn": collator,
        "worker_init_fn": _seed_worker,
        "persistent_workers": config.num_workers > 0,
    }

    return DataBundle(
        train_loader=DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            generator=generator,
            **common_loader_options,
        ),
        validation_loader=DataLoader(
            validation_dataset,
            batch_size=evaluation_batch_size,
            shuffle=False,
            **common_loader_options,
        ),
        test_loader=DataLoader(
            test_dataset,
            batch_size=evaluation_batch_size,
            shuffle=False,
            **common_loader_options,
        ),
        normalizer=normalizer,
        datasets=datasets,
    )
