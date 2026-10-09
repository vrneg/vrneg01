"""Leakage-safe fixed-grid trajectories reconstructed from fold checkpoints."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import torch

try:
    from event_transformer.features import MODALITY_NAMES
    from minirocket.config import DataConfig
    from minirocket.data import DataBundle, TimeSeriesSplit, prepare_data
    from modality_attribution.configuration import dataclass_from_dict
    from modality_attribution.grouping import build_channel_groups
except ModuleNotFoundError as error:
    if error.name not in {"event_transformer", "minirocket", "modality_attribution"}:
        raise
    from ..event_transformer.features import MODALITY_NAMES
    from ..minirocket.config import DataConfig
    from ..minirocket.data import DataBundle, TimeSeriesSplit, prepare_data
    from ..modality_attribution.configuration import dataclass_from_dict
    from ..modality_attribution.grouping import build_channel_groups

from .config import DTWAnalysisConfig


@dataclass(frozen=True, slots=True)
class Trajectory:
    """One observed modality trace in time-major format."""

    values: np.ndarray
    time_indices: np.ndarray


@dataclass(frozen=True, slots=True)
class TrajectorySplit:
    trajectories: dict[str, list[Trajectory | None]]
    presence: dict[str, np.ndarray]
    labels: np.ndarray
    sample_ids: list[str]
    group_ids: list[str]


@dataclass(frozen=True, slots=True)
class DTWFoldData:
    checkpoint_path: Path
    checkpoint_type: str
    fold_name: str
    data_config: DataConfig
    time_grid: np.ndarray
    train: TrajectorySplit
    validation: TrajectorySplit
    test: TrajectorySplit


def load_checkpoint_payload(path: str | Path) -> dict[str, Any]:
    checkpoint_path = Path(path)
    payload = (
        torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if checkpoint_path.suffix == ".pt"
        else joblib.load(checkpoint_path)
    )
    if not isinstance(payload, dict) or "experiment_config" not in payload:
        raise ValueError(f"Unsupported repository checkpoint: {checkpoint_path}")
    return payload


def checkpoint_type(payload: dict[str, Any]) -> str:
    return str(payload.get("artifact_type", payload.get("checkpoint_type", "unknown")))


def _fixed_grid_config(
    payload: dict[str, Any], analysis: DTWAnalysisConfig
) -> DataConfig:
    saved = dict(payload["experiment_config"]["data"])
    if checkpoint_type(payload) == "fine_tuned_classifier":
        return DataConfig(
            dataset_path=Path(saved["dataset_path"]),
            num_time_points=analysis.event_num_time_points,
            window_start_seconds=analysis.event_window_start_seconds,
            window_end_seconds=analysis.event_window_end_seconds,
            max_events_per_modality=saved.get("max_events_per_modality", 128),
            normalize_features=bool(saved.get("normalize_features", True)),
            cache_in_memory=bool(saved.get("cache_in_memory", True)),
            actor_scope="anchor",
            include_presence_channels=True,
            positive_label=str(saved.get("positive_label", "neg")),
            negative_label=str(saved.get("negative_label", "none")),
        )
    config = dataclass_from_dict(DataConfig, saved)
    return replace(
        config,
        include_presence_channels=True,
        included_modalities=None,
    )


def _extract_split(
    split: TimeSeriesSplit,
    channel_names: tuple[str, ...],
) -> TrajectorySplit:
    groups = build_channel_groups(channel_names)
    trajectories: dict[str, list[Trajectory | None]] = {}
    presence_by_modality: dict[str, np.ndarray] = {}
    for group in groups:
        presence_name = f"{group.name}.present"
        if presence_name not in channel_names:
            raise ValueError(
                f"DTW extraction requires a presence channel for {group.name}"
            )
        presence_index = channel_names.index(presence_name)
        feature_indices = tuple(
            index for index in group.indices if index != presence_index
        )
        observed_steps = split.values[:, presence_index, :] > 0.5
        present = observed_steps.sum(axis=1) >= 2
        presence_by_modality[group.name] = present
        modality_trajectories: list[Trajectory | None] = []
        for sample_index, usable in enumerate(present):
            if not usable:
                modality_trajectories.append(None)
                continue
            time_indices = np.flatnonzero(observed_steps[sample_index])
            values = split.values[
                sample_index, feature_indices, :
            ][:, time_indices].T.astype(np.float64, copy=False)
            if not np.all(np.isfinite(values)):
                raise ValueError(
                    f"Non-finite {group.name} trajectory for "
                    f"{split.sample_ids[sample_index]}"
                )
            modality_trajectories.append(Trajectory(values, time_indices))
        trajectories[group.name] = modality_trajectories
    return TrajectorySplit(
        trajectories=trajectories,
        presence=presence_by_modality,
        labels=np.asarray(split.labels, dtype=np.int64),
        sample_ids=list(split.sample_ids),
        group_ids=list(split.group_ids),
    )


def load_dtw_fold(
    checkpoint_path: str | Path,
    analysis: DTWAnalysisConfig,
) -> DTWFoldData:
    """Load one fold's data configuration without running its fitted classifier."""

    path = Path(checkpoint_path)
    payload = load_checkpoint_payload(path)
    data_config = _fixed_grid_config(payload, analysis)
    bundle: DataBundle = prepare_data(data_config)
    if tuple(bundle.channel_names) == ():
        raise ValueError("Prepared DTW fold has no channels")
    return DTWFoldData(
        checkpoint_path=path,
        checkpoint_type=checkpoint_type(payload),
        fold_name=path.parent.name,
        data_config=data_config,
        time_grid=np.asarray(bundle.time_grid, dtype=np.float64),
        train=_extract_split(bundle.train, bundle.channel_names),
        validation=_extract_split(bundle.validation, bundle.channel_names),
        test=_extract_split(bundle.test, bundle.channel_names),
    )


def validate_dtw_folds(
    folds: list[DTWFoldData], *, allow_duplicate_samples: bool
) -> int:
    if len(folds) < 2:
        raise ValueError("Cross-validation DTW analysis requires at least two folds")
    if len({fold.fold_name for fold in folds}) != len(folds):
        raise ValueError("Fold checkpoint directory names must be unique")
    signatures = {
        (
            fold.data_config.num_time_points,
            fold.data_config.window_start_seconds,
            fold.data_config.window_end_seconds,
            fold.data_config.actor_scope,
            fold.data_config.positive_label,
            fold.data_config.negative_label,
        )
        for fold in folds
    }
    if len(signatures) != 1:
        raise ValueError("Fold checkpoints resolve to incompatible DTW data settings")
    test_groups: set[str] = set()
    outer_test_sample_ids: list[str] = []
    outer_test_folds: dict[str, set[str]] = {}
    duplicate_issues: set[tuple[str, ...]] = set()
    for fold in folds:
        split_groups = {
            "train": set(fold.train.group_ids),
            "validation": set(fold.validation.group_ids),
            "test": set(fold.test.group_ids),
        }
        if (
            split_groups["train"] & split_groups["validation"]
            or split_groups["train"] & split_groups["test"]
            or split_groups["validation"] & split_groups["test"]
        ):
            raise ValueError(f"Session leakage detected inside {fold.fold_name}")
        if test_groups & split_groups["test"]:
            raise ValueError("A recording/session occurs in multiple outer test folds")
        test_groups.update(split_groups["test"])
        for split_name, split in (
            ("train", fold.train),
            ("validation", fold.validation),
            ("test", fold.test),
        ):
            counts: dict[str, int] = {}
            for sample_id in split.sample_ids:
                counts[sample_id] = counts.get(sample_id, 0) + 1
            duplicate_issues.update(
                ("within_split", fold.fold_name, split_name, sample_id)
                for sample_id, count in counts.items()
                if count > 1
            )
        outer_test_sample_ids.extend(fold.test.sample_ids)
        for sample_id in fold.test.sample_ids:
            outer_test_folds.setdefault(sample_id, set()).add(fold.fold_name)
    outer_test_counts: dict[str, int] = {}
    for sample_id in outer_test_sample_ids:
        outer_test_counts[sample_id] = outer_test_counts.get(sample_id, 0) + 1
    duplicate_issues.update(
        ("across_outer_test", sample_id)
        for sample_id, count in outer_test_counts.items()
        if count > 1 and len(outer_test_folds[sample_id]) > 1
    )
    duplicate_count = len(duplicate_issues)
    if duplicate_count and not allow_duplicate_samples:
        raise ValueError(
            f"Fold data contain {duplicate_count} duplicate sample-ID issues; "
            "audit them or enable the explicit duplicate override"
        )
    return duplicate_count


def duplicate_sample_rows(folds: list[DTWFoldData]) -> list[dict[str, Any]]:
    """Return metadata for repeated IDs without flagging expected CV overlap."""

    outer_test_counts: dict[str, int] = {}
    outer_test_folds: dict[str, set[str]] = {}
    for fold in folds:
        for sample_id in fold.test.sample_ids:
            outer_test_counts[sample_id] = outer_test_counts.get(sample_id, 0) + 1
            outer_test_folds.setdefault(sample_id, set()).add(fold.fold_name)
    rows: list[dict[str, Any]] = []
    for fold in folds:
        for split_name, split in (
            ("train", fold.train),
            ("validation", fold.validation),
            ("test", fold.test),
        ):
            local_counts: dict[str, int] = {}
            for sample_id in split.sample_ids:
                local_counts[sample_id] = local_counts.get(sample_id, 0) + 1
            for index, sample_id in enumerate(split.sample_ids):
                reasons = []
                if local_counts[sample_id] > 1:
                    reasons.append("repeated_within_split")
                if (
                    split_name == "test"
                    and len(outer_test_folds[sample_id]) > 1
                ):
                    reasons.append("repeated_across_outer_test_rows")
                if not reasons:
                    continue
                rows.append(
                    {
                        "sample_id": sample_id,
                        "within_split_occurrences": local_counts[sample_id],
                        "outer_test_occurrences": (
                            outer_test_counts[sample_id]
                            if split_name == "test"
                            else None
                        ),
                        "reason": ";".join(reasons),
                        "fold": fold.fold_name,
                        "split": split_name,
                        "sample_index": index,
                        "sample_key": f"{fold.fold_name}:{split_name}:{index}",
                        "group_id": split.group_ids[index],
                        "label": int(split.labels[index]),
                    }
                )
    return rows


def observed(
    split: TrajectorySplit, modality: str
) -> tuple[list[Trajectory], np.ndarray]:
    indices = np.flatnonzero(split.presence[modality])
    values = [split.trajectories[modality][index] for index in indices]
    if any(value is None for value in values):
        raise RuntimeError("Presence and trajectory extraction disagree")
    return [value for value in values if value is not None], indices


__all__ = [
    "DTWFoldData",
    "Trajectory",
    "TrajectorySplit",
    "duplicate_sample_rows",
    "load_dtw_fold",
    "observed",
    "validate_dtw_folds",
    "MODALITY_NAMES",
]
