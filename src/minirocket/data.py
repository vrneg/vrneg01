"""Convert irregular modality observations to fixed multivariate time series."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

try:
    from event_transformer.data import (
        EncodedWindow,
        EventSequenceEncoder,
        EventWindowDataset,
        fit_feature_normalizer,
        load_fold,
    )
    from event_transformer.features import (
        ACTOR_DIFFERENT_FROM_ANCHOR,
        ACTOR_SAME_AS_ANCHOR,
        FINGER_FLAG_DIM,
        FINGER_MODALITY_IDS,
        MODALITY_DIMS,
        MODALITY_NAMES,
        EventFeatureExtractor,
        FeatureNormalizer,
    )
except ModuleNotFoundError as error:
    if error.name != "event_transformer":
        raise
    from ..event_transformer.data import (
        EncodedWindow,
        EventSequenceEncoder,
        EventWindowDataset,
        fit_feature_normalizer,
        load_fold,
    )
    from ..event_transformer.features import (
        ACTOR_DIFFERENT_FROM_ANCHOR,
        ACTOR_SAME_AS_ANCHOR,
        FINGER_FLAG_DIM,
        FINGER_MODALITY_IDS,
        MODALITY_DIMS,
        MODALITY_NAMES,
        EventFeatureExtractor,
        FeatureNormalizer,
    )

from .config import DataConfig


# aeon rejects a case/channel pair when it has a non-zero range but a standard
# deviation at or below this threshold. Such traces are floating-point residue,
# not meaningful temporal variation. Exactly constant traces are valid input.
MINIMUM_TEMPORAL_STD = 1e-7


@dataclass(slots=True)
class TimeSeriesSplit:
    """One fixed-grid split accepted by aeon's collection estimators."""

    values: np.ndarray
    labels: np.ndarray
    sample_ids: list[str]
    group_ids: list[str]


@dataclass(slots=True)
class DataBundle:
    """Prepared train, validation, and test collections for one fold."""

    train: TimeSeriesSplit
    validation: TimeSeriesSplit
    test: TimeSeriesSplit
    normalizer: FeatureNormalizer | None
    time_grid: np.ndarray
    channel_names: tuple[str, ...]


def channel_names(config: DataConfig) -> tuple[str, ...]:
    """Return the deterministic channel order used by fixed-grid arrays."""

    names: list[str] = []
    for modality_id, modality_name in enumerate(MODALITY_NAMES):
        names.extend(
            f"{modality_name}.feature_{index}"
            for index in range(MODALITY_DIMS[modality_id])
        )
        if modality_id in FINGER_MODALITY_IDS:
            names.extend(
                f"{modality_name}.flag_{index}" for index in range(FINGER_FLAG_DIM)
            )
        if config.include_presence_channels:
            names.append(f"{modality_name}.present")
    return tuple(names)


def _actor_selection_mask(
    actor_relation_ids: np.ndarray,
    actor_scope: str,
) -> np.ndarray:
    if actor_scope == "anchor":
        return actor_relation_ids == ACTOR_SAME_AS_ANCHOR
    if actor_scope == "other":
        return actor_relation_ids == ACTOR_DIFFERENT_FROM_ANCHOR
    if actor_scope == "all":
        return np.ones(actor_relation_ids.shape, dtype=np.bool_)
    raise ValueError(f"Unknown actor_scope: {actor_scope!r}")


def _average_duplicate_timestamps(
    times: np.ndarray,
    values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Average records that map to the same modality timestamp."""

    unique_times, inverse = np.unique(times, return_inverse=True)
    if unique_times.shape[0] == times.shape[0]:
        order = np.argsort(times, kind="stable")
        return times[order], values[order]

    sums = np.zeros((unique_times.shape[0], values.shape[1]), dtype=np.float64)
    counts = np.zeros(unique_times.shape[0], dtype=np.int64)
    np.add.at(sums, inverse, values)
    np.add.at(counts, inverse, 1)
    return unique_times, (sums / counts[:, None]).astype(np.float32)


def _interpolate_stream(
    times: np.ndarray,
    values: np.ndarray,
    grid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Linearly interpolate inside the observed range and expose its presence."""

    output = np.zeros((values.shape[1], grid.shape[0]), dtype=np.float32)
    presence = np.zeros(grid.shape[0], dtype=np.float32)
    if not times.size:
        return output, presence

    times, values = _average_duplicate_timestamps(times, values)
    if times.size == 1:
        closest = int(np.abs(grid - times[0]).argmin())
        output[:, closest] = values[0]
        presence[closest] = 1.0
        return output, presence

    inside = (grid >= times[0]) & (grid <= times[-1])
    for feature_index in range(values.shape[1]):
        output[feature_index, inside] = np.interp(
            grid[inside], times, values[:, feature_index]
        )
    presence[inside] = 1.0
    return output, presence


def window_to_array(
    window: EncodedWindow,
    config: DataConfig,
    time_grid: np.ndarray | None = None,
) -> np.ndarray:
    """Convert one encoded window to ``[channels, time]`` format."""

    config.validate()
    grid = (
        np.linspace(
            config.window_start_seconds,
            config.window_end_seconds,
            config.num_time_points,
            dtype=np.float32,
        )
        if time_grid is None
        else np.asarray(time_grid, dtype=np.float32)
    )
    if grid.shape != (config.num_time_points,):
        raise ValueError("time_grid shape does not match num_time_points")

    channels: list[np.ndarray] = []
    for modality_id in MODALITY_DIMS:
        time_step_indices = window.modality_time_step_indices[modality_id].numpy()
        actor_ids = window.modality_actor_relation_ids[modality_id].numpy()
        selected = _actor_selection_mask(actor_ids, config.actor_scope)
        times = window.time_features[time_step_indices, 0].numpy()[selected]
        values = window.modality_features[modality_id].numpy()[selected]
        if modality_id in FINGER_MODALITY_IDS:
            flags = window.finger_flags[modality_id].numpy()[selected]
            values = np.concatenate((values, flags), axis=1)

        interpolated, presence = _interpolate_stream(times, values, grid)
        channels.append(interpolated)
        if config.include_presence_channels:
            channels.append(presence[None, :])

    result = np.concatenate(channels, axis=0)
    expected_shape = (len(channel_names(config)), config.num_time_points)
    if result.shape != expected_shape:
        raise RuntimeError(
            f"Fixed-grid window has shape {result.shape}; expected {expected_shape}"
        )
    return result


def _record_id(value: Any) -> str:
    if isinstance(value, dict):
        return f"{value.get('table_name', 'record')}:{value.get('id')!r}"
    return str(value)


def _convert_split(
    source: Any,
    dataset: EventWindowDataset,
    config: DataConfig,
    time_grid: np.ndarray,
) -> TimeSeriesSplit:
    windows = [dataset[index] for index in range(len(dataset))]
    values = np.stack(
        [window_to_array(window, config, time_grid) for window in windows]
    ).astype(np.float32, copy=False)
    if config.included_modalities is not None:
        included = set(config.included_modalities)
        names = channel_names(config)
        excluded_indices = [
            index
            for index, name in enumerate(names)
            if name.split(".", 1)[0] not in included
        ]
        if excluded_indices:
            values[:, excluded_indices, :] = 0.0
    if config.masked_channels is not None:
        names = channel_names(config)
        name_to_index = {name: index for index, name in enumerate(names)}
        unknown = set(config.masked_channels) - set(name_to_index)
        if unknown:
            raise ValueError(f"Unknown masked channels: {sorted(unknown)}")
        masked_indices = [name_to_index[name] for name in config.masked_channels]
        if masked_indices:
            values[:, masked_indices, :] = 0.0
    values = collapse_near_constant_traces(values)
    labels = np.asarray([int(window.label) for window in windows])
    sample_ids = [window.sample_id for window in windows]
    group_ids = [
        _record_id(source[index]["word"].get("experiment"))
        for index in range(len(source))
    ]
    return TimeSeriesSplit(
        values=values,
        labels=labels,
        sample_ids=sample_ids,
        group_ids=group_ids,
    )


def collapse_near_constant_traces(values: np.ndarray) -> np.ndarray:
    """Remove sub-threshold numerical residue from individual time series.

    MiniRocket accepts exactly constant channels, but aeon's collection validator
    rejects non-constant traces whose temporal standard deviation is at most
    ``1e-7``. Interpolation and float32 normalization can create differences of a
    few ULPs around zero. Replacing only those traces with their temporal mean
    prevents a validation failure without amplifying noise or adding synthetic
    variation.
    """

    if values.ndim != 3:
        raise ValueError(
            "MiniRocket split values must have shape [cases, channels, time]"
        )
    # Always measure in float64. A float32 reduction can round a sub-threshold
    # trace just above the cutoff; aeon may then recompute it in float64 after a
    # model-specific cast or with a different memory layout and reject it.
    temporal_ranges = np.ptp(values, axis=2)
    temporal_stds = np.std(values, axis=2, ddof=0, dtype=np.float64)
    near_constant = (temporal_stds <= MINIMUM_TEMPORAL_STD) & (
        temporal_ranges != 0
    )
    if not np.any(near_constant):
        return values

    stabilized = values.copy()
    case_indices, channel_indices = np.nonzero(near_constant)
    levels = np.mean(
        values[case_indices, channel_indices, :],
        axis=1,
        dtype=np.float64,
    ).astype(stabilized.dtype, copy=False)
    stabilized[case_indices, channel_indices, :] = levels[:, None]
    return stabilized


# Backwards-compatible private name retained for existing imports.
_collapse_near_constant_traces = collapse_near_constant_traces


def prepare_data(config: DataConfig) -> DataBundle:
    """Load one fold and fit all data-dependent preprocessing on training only."""

    config.validate()
    fold = load_fold(config.dataset_path)
    encoder = EventSequenceEncoder(
        EventFeatureExtractor(),
        max_events_per_modality=config.max_events_per_modality,
        time_clip_seconds=None,
        positive_label=config.positive_label,
        negative_label=config.negative_label,
        included_modalities=config.included_modalities,
    )
    train_dataset = EventWindowDataset(
        fold["train"],
        encoder=encoder,
        cache_in_memory=config.cache_in_memory,
    )
    normalizer = (
        fit_feature_normalizer(train_dataset) if config.normalize_features else None
    )
    if normalizer is not None:
        train_dataset.apply_normalizer(normalizer)

    validation_dataset = EventWindowDataset(
        fold["validation"],
        encoder=encoder,
        normalizer=normalizer,
        cache_in_memory=config.cache_in_memory,
    )
    test_dataset = EventWindowDataset(
        fold["test"],
        encoder=encoder,
        normalizer=normalizer,
        cache_in_memory=config.cache_in_memory,
    )
    time_grid = np.linspace(
        config.window_start_seconds,
        config.window_end_seconds,
        config.num_time_points,
        dtype=np.float32,
    )
    names = channel_names(config)
    return DataBundle(
        train=_convert_split(fold["train"], train_dataset, config, time_grid),
        validation=_convert_split(
            fold["validation"], validation_dataset, config, time_grid
        ),
        test=_convert_split(fold["test"], test_dataset, config, time_grid),
        normalizer=normalizer,
        time_grid=time_grid,
        channel_names=names,
    )


def apply_fixed_grid_masks(data: DataBundle, config: DataConfig) -> DataBundle:
    """Derive a subset-training bundle from one fully prepared fold.

    Fixed-grid normalization is fitted independently per modality, and interpolation
    uses each observation's anchor-relative time. Consequently, preparing every
    modality once and zeroing excluded channels is equivalent to preparing the same
    fold with ``included_modalities``/``masked_channels`` directly, while avoiding
    repeated event decoding for attribution variants.
    """

    config.validate()
    expected_names = channel_names(config)
    if data.channel_names != expected_names:
        raise ValueError(
            "Prepared data and subset configuration use different channels"
        )

    masked_names = set(config.masked_channels or ())
    if config.included_modalities is not None:
        included = set(config.included_modalities)
        masked_names.update(
            name
            for name in data.channel_names
            if name.split(".", 1)[0] not in included
        )
    unknown = masked_names - set(data.channel_names)
    if unknown:
        raise ValueError(f"Unknown masked channels: {sorted(unknown)}")
    masked_indices = np.asarray(
        [
            index
            for index, name in enumerate(data.channel_names)
            if name in masked_names
        ],
        dtype=np.int64,
    )

    def apply(split: TimeSeriesSplit) -> TimeSeriesSplit:
        values = np.array(split.values, copy=True)
        if masked_indices.size:
            values[:, masked_indices, :] = 0.0
        return TimeSeriesSplit(
            values=values,
            labels=split.labels,
            sample_ids=split.sample_ids,
            group_ids=split.group_ids,
        )

    return DataBundle(
        train=apply(data.train),
        validation=apply(data.validation),
        test=apply(data.test),
        normalizer=data.normalizer,
        time_grid=data.time_grid,
        channel_names=data.channel_names,
    )


def prepare_fixed_grid_split(
    config: DataConfig,
    split_name: str,
    normalizer: FeatureNormalizer | None,
) -> TimeSeriesSplit:
    """Prepare one split using already-fitted checkpoint normalization statistics."""

    config.validate()
    if split_name not in {"train", "validation", "test"}:
        raise ValueError("split_name must be 'train', 'validation', or 'test'")
    fold = load_fold(config.dataset_path)
    encoder = EventSequenceEncoder(
        EventFeatureExtractor(),
        max_events_per_modality=config.max_events_per_modality,
        time_clip_seconds=None,
        positive_label=config.positive_label,
        negative_label=config.negative_label,
        included_modalities=config.included_modalities,
    )
    dataset = EventWindowDataset(
        fold[split_name],
        encoder=encoder,
        normalizer=normalizer,
        cache_in_memory=config.cache_in_memory,
    )
    time_grid = np.linspace(
        config.window_start_seconds,
        config.window_end_seconds,
        config.num_time_points,
        dtype=np.float32,
    )
    return _convert_split(fold[split_name], dataset, config, time_grid)
