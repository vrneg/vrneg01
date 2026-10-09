"""Checkpoint adapters exposing modality-coalition logits through one API."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import MISSING, fields, is_dataclass, replace
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import torch
from torch.utils.data import DataLoader

try:
    from event_transformer.config import DataConfig as EventDataConfig
    from event_transformer.data import (
        EventBatchCollator,
        EventSequenceEncoder,
        EventWindowDataset,
        load_fold,
    )
    from event_transformer.features import (
        MODALITY_DIMS,
        MODALITY_NAMES,
        FINGER_FLAG_DIM,
        FINGER_MODALITY_IDS,
        EventFeatureExtractor,
        FeatureNormalizer,
    )
    from minirocket.config import DataConfig as FixedGridDataConfig
    from minirocket.data import prepare_fixed_grid_split
except ModuleNotFoundError as error:
    if error.name not in {"event_transformer", "minirocket"}:
        raise
    from ..event_transformer.config import DataConfig as EventDataConfig
    from ..event_transformer.data import (
        EventBatchCollator,
        EventSequenceEncoder,
        EventWindowDataset,
        load_fold,
    )
    from ..event_transformer.features import (
        MODALITY_DIMS,
        MODALITY_NAMES,
        FINGER_FLAG_DIM,
        FINGER_MODALITY_IDS,
        EventFeatureExtractor,
        FeatureNormalizer,
    )
    from ..minirocket.config import DataConfig as FixedGridDataConfig
    from ..minirocket.data import prepare_fixed_grid_split

from .configuration import dataclass_from_dict, relocate_checkpoint_paths
from .grouping import build_channel_groups, mask_fixed_grid, normalize_modalities


JOBLIB_ESTIMATOR_KEYS = ("pipeline", "classifier", "model")
FIXED_GRID_TORCH_TYPES = {
    "modality_aware_inception_tcn_classifier",
    "compact_cue_aware_fusion_tcn_classifier",
}


def _replace_legacy_dataclass(instance: Any, /, **changes: Any) -> Any:
    """Replace fields after restoring defaults absent from an older pickle.

    Pickle restores slotted dataclasses without invoking their generated initializer.
    Consequently, a checkpoint written before a defaulted field was added can load as
    the current class while still lacking that slot's value.  ``dataclasses.replace``
    reads every declared field and otherwise fails before it can apply ``changes``.
    """

    if not is_dataclass(instance) or isinstance(instance, type):
        raise TypeError(f"Expected a dataclass instance, got {type(instance).__name__}")
    for descriptor in fields(instance):
        if hasattr(instance, descriptor.name):
            continue
        if descriptor.default is not MISSING:
            default = descriptor.default
        elif descriptor.default_factory is not MISSING:
            default = descriptor.default_factory()
        else:
            raise AttributeError(
                f"Legacy {type(instance).__name__} is missing required field "
                f"{descriptor.name!r}"
            )
        object.__setattr__(instance, descriptor.name, default)
    return replace(instance, **changes)


def _probabilities_to_logits(probabilities: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(probabilities, dtype=np.float64), 1e-7, 1.0 - 1e-7)
    return np.log(clipped) - np.log1p(-clipped)


def _positive_probability(estimator: Any, probabilities: np.ndarray) -> np.ndarray:
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if probabilities.ndim == 1:
        return probabilities
    if probabilities.ndim != 2 or probabilities.shape[1] < 2:
        raise ValueError(f"predict_proba returned unsupported shape {probabilities.shape}")
    classes = getattr(estimator, "classes_", None)
    if classes is None:
        return probabilities[:, 1]
    matches = np.flatnonzero(np.asarray(classes) == 1)
    if matches.size != 1:
        raise ValueError(f"Could not identify positive class 1 in classes_={classes!r}")
    return probabilities[:, int(matches[0])]


def _estimator_logits(estimator: Any, values: np.ndarray) -> np.ndarray:
    if hasattr(estimator, "decision_function"):
        scores = np.asarray(estimator.decision_function(values), dtype=np.float64)
        if scores.ndim == 2 and scores.shape[1] == 2:
            scores = scores[:, 1] - scores[:, 0]
        if scores.ndim != 1:
            raise ValueError(f"decision_function returned unsupported shape {scores.shape}")
        return scores
    if hasattr(estimator, "predict_proba"):
        return _probabilities_to_logits(
            _positive_probability(estimator, estimator.predict_proba(values))
        )
    raise TypeError(f"{type(estimator).__name__} exposes neither scores nor probabilities")


class CheckpointAdapter(ABC):
    """A fitted fold model evaluated under arbitrary modality coalitions."""

    def __init__(self, checkpoint_path: Path, payload: dict[str, Any]) -> None:
        self.checkpoint_path = checkpoint_path
        self.payload = payload
        self.checkpoint_type = str(
            payload.get("artifact_type", payload.get("checkpoint_type", "unknown"))
        )
        self.config_payload = dict(payload["experiment_config"])
        self.threshold = float(payload.get("decision_threshold", 0.5))
        self._cache: dict[tuple[str, ...], np.ndarray] = {}

    @property
    def fold_name(self) -> str:
        return self.checkpoint_path.parent.name

    @property
    def modality_names(self) -> tuple[str, ...]:
        return MODALITY_NAMES

    @property
    def modality_sizes(self) -> dict[str, int]:
        return {
            name: int(MODALITY_DIMS[index])
            + (FINGER_FLAG_DIM if index in FINGER_MODALITY_IDS else 0)
            for index, name in enumerate(MODALITY_NAMES)
        }

    def predict_logits(self, included_modalities: Sequence[str]) -> np.ndarray:
        coalition = normalize_modalities(included_modalities)
        if coalition not in self._cache:
            scores = np.asarray(self._predict_logits(coalition), dtype=np.float64)
            if scores.shape != self.labels.shape:
                raise RuntimeError(
                    f"Expected logits with shape {self.labels.shape}, got {scores.shape}"
                )
            self._cache[coalition] = scores
        return self._cache[coalition]

    @property
    @abstractmethod
    def sample_ids(self) -> list[str]: ...

    @property
    @abstractmethod
    def labels(self) -> np.ndarray: ...

    @property
    @abstractmethod
    def group_ids(self) -> list[str]: ...

    def predict_timing_perturbation(
        self,
        modalities: Sequence[str],
        *,
        kind: str,
        magnitude: float,
        event_step_milliseconds: float,
    ) -> np.ndarray:
        """Predict after a label-independent timing perturbation."""

        del modalities, kind, magnitude, event_step_milliseconds
        raise NotImplementedError

    def predict_robustness_logits(
        self,
        perturbations: Sequence[tuple[Sequence[str], str, float]],
        *,
        event_step_milliseconds: float,
        include_baseline: bool,
    ) -> tuple[np.ndarray | None, list[np.ndarray]]:
        """Score a baseline and several timing perturbations.

        The default preserves the ordinary one-condition-at-a-time adapter behavior.
        Estimators with reusable inference contexts can override this method.
        """

        baseline = self.predict_logits(MODALITY_NAMES) if include_baseline else None
        logits = [
            self.predict_timing_perturbation(
                modalities,
                kind=kind,
                magnitude=magnitude,
                event_step_milliseconds=event_step_milliseconds,
            )
            for modalities, kind, magnitude in perturbations
        ]
        return baseline, logits

    def timing_step_milliseconds(self, event_fallback: float) -> float:
        """Return the physical duration represented by one shift step."""

        return event_fallback

    @abstractmethod
    def _predict_logits(self, coalition: tuple[str, ...]) -> np.ndarray: ...


class FixedGridEstimatorAdapter(CheckpointAdapter):
    def __init__(
        self, checkpoint_path: Path, payload: dict[str, Any], *, device: str
    ) -> None:
        super().__init__(checkpoint_path, payload)
        self.estimator = next(
            (payload[key] for key in JOBLIB_ESTIMATOR_KEYS if key in payload), None
        )
        if self.estimator is None:
            raise ValueError("Joblib checkpoint contains no fitted estimator")
        model_config = getattr(self.estimator, "model_config", None)
        if model_config is not None and hasattr(model_config, "tabpfn_device"):
            self.estimator.model_config = _replace_legacy_dataclass(
                model_config, tabpfn_device=device
            )
            self.estimator.model_config = relocate_checkpoint_paths(
                self.estimator.model_config
            )
        self.data_config = dataclass_from_dict(
            FixedGridDataConfig, self.config_payload["data"]
        )
        self.data_config = relocate_checkpoint_paths(self.data_config)
        normalizer_state = payload.get("normalizer")
        normalizer = (
            None
            if normalizer_state is None
            else FeatureNormalizer.from_state_dict(normalizer_state)
        )
        self.split = prepare_fixed_grid_split(self.data_config, "test", normalizer)
        channel_names = tuple(payload["channel_names"])
        if len(channel_names) != self.split.values.shape[1]:
            raise ValueError("Checkpoint channel schema does not match prepared test data")
        self.groups = build_channel_groups(channel_names)
        self.channel_names = channel_names

    @property
    def modality_sizes(self) -> dict[str, int]:
        return {group.name: len(group.indices) for group in self.groups}

    @property
    def sample_ids(self) -> list[str]:
        return self.split.sample_ids

    @property
    def labels(self) -> np.ndarray:
        return self.split.labels

    @property
    def group_ids(self) -> list[str]:
        return self.split.group_ids

    def predict_fixed_grid_logits(self, values: np.ndarray) -> np.ndarray:
        return _estimator_logits(self.estimator, values)

    def predict_fixed_grid_logits_many(
        self, values: Sequence[np.ndarray]
    ) -> list[np.ndarray]:
        predict_many = getattr(self.estimator, "predict_proba_many", None)
        if predict_many is None:
            return [self.predict_fixed_grid_logits(item) for item in values]
        probability_sets = list(predict_many(values))
        if len(probability_sets) != len(values):
            raise RuntimeError(
                "Repeated estimator inference returned the wrong number of outputs"
            )
        return [
            _probabilities_to_logits(
                _positive_probability(self.estimator, probabilities)
            )
            for probabilities in probability_sets
        ]

    def _predict_logits(self, coalition: tuple[str, ...]) -> np.ndarray:
        values = mask_fixed_grid(self.split.values, self.groups, coalition)
        return self.predict_fixed_grid_logits(values)

    def predict_timing_perturbation(
        self,
        modalities: Sequence[str],
        *,
        kind: str,
        magnitude: float,
        event_step_milliseconds: float,
    ) -> np.ndarray:
        del event_step_milliseconds
        try:
            from dtw_analysis.perturbations import warp_fixed_grid
        except ModuleNotFoundError as error:
            if error.name != "dtw_analysis":
                raise
            from ..dtw_analysis.perturbations import warp_fixed_grid
        values = warp_fixed_grid(
            self.split.values,
            self.channel_names,
            modalities,
            kind=kind,
            magnitude=magnitude,
            center_index=(
                -self.data_config.window_start_seconds
                / (
                    self.data_config.window_end_seconds
                    - self.data_config.window_start_seconds
                )
                * (self.data_config.num_time_points - 1)
            ),
        )
        return self.predict_fixed_grid_logits(values)

    def predict_robustness_logits(
        self,
        perturbations: Sequence[tuple[Sequence[str], str, float]],
        *,
        event_step_milliseconds: float,
        include_baseline: bool,
    ) -> tuple[np.ndarray | None, list[np.ndarray]]:
        del event_step_milliseconds
        try:
            from dtw_analysis.perturbations import warp_fixed_grid
        except ModuleNotFoundError as error:
            if error.name != "dtw_analysis":
                raise
            from ..dtw_analysis.perturbations import warp_fixed_grid

        center_index = (
            -self.data_config.window_start_seconds
            / (
                self.data_config.window_end_seconds
                - self.data_config.window_start_seconds
            )
            * (self.data_config.num_time_points - 1)
        )
        values: list[np.ndarray] = []
        if include_baseline:
            values.append(self.split.values)
        values.extend(
            warp_fixed_grid(
                self.split.values,
                self.channel_names,
                modalities,
                kind=kind,
                magnitude=magnitude,
                center_index=center_index,
            )
            for modalities, kind, magnitude in perturbations
        )
        logits = self.predict_fixed_grid_logits_many(values)
        baseline: np.ndarray | None = None
        if include_baseline:
            baseline = logits.pop(0)
            self._cache[normalize_modalities(MODALITY_NAMES)] = baseline
        return baseline, logits

    def timing_step_milliseconds(self, event_fallback: float) -> float:
        del event_fallback
        return (
            self.data_config.window_end_seconds
            - self.data_config.window_start_seconds
        ) / (self.data_config.num_time_points - 1) * 1_000.0


class FixedGridTorchAdapter(CheckpointAdapter):
    def __init__(
        self,
        checkpoint_path: Path,
        payload: dict[str, Any],
        *,
        device: str,
        batch_size: int,
    ) -> None:
        super().__init__(checkpoint_path, payload)
        if self.checkpoint_type == "modality_aware_inception_tcn_classifier":
            try:
                from inception_tcn.training import load_trained_model
            except ModuleNotFoundError as error:
                if error.name != "inception_tcn":
                    raise
                from ..inception_tcn.training import load_trained_model
        else:
            try:
                from compact_fusion_tcn.training import load_trained_model
            except ModuleNotFoundError as error:
                if error.name != "compact_fusion_tcn":
                    raise
                from ..compact_fusion_tcn.training import load_trained_model
        self.model, normalizer, restored = load_trained_model(checkpoint_path, device)
        self.payload = restored
        self.device = next(self.model.parameters()).device
        self.batch_size = batch_size
        self.data_config = dataclass_from_dict(
            FixedGridDataConfig, self.config_payload["data"]
        )
        self.data_config = relocate_checkpoint_paths(self.data_config)
        self.split = prepare_fixed_grid_split(self.data_config, "test", normalizer)
        self.channel_names = tuple(payload["channel_names"])
        self.groups = build_channel_groups(self.channel_names)

    @property
    def modality_sizes(self) -> dict[str, int]:
        return {group.name: len(group.indices) for group in self.groups}

    @property
    def sample_ids(self) -> list[str]:
        return self.split.sample_ids

    @property
    def labels(self) -> np.ndarray:
        return self.split.labels

    @property
    def group_ids(self) -> list[str]:
        return self.split.group_ids

    def predict_fixed_grid_logits(self, values: np.ndarray) -> np.ndarray:
        outputs: list[np.ndarray] = []
        self.model.eval()
        with torch.no_grad():
            for start in range(0, values.shape[0], self.batch_size):
                batch = torch.from_numpy(values[start : start + self.batch_size]).to(
                    self.device
                )
                outputs.append(self.model(batch).detach().cpu().numpy())
        return np.concatenate(outputs).astype(np.float64, copy=False)

    def _predict_logits(self, coalition: tuple[str, ...]) -> np.ndarray:
        values = mask_fixed_grid(self.split.values, self.groups, coalition)
        return self.predict_fixed_grid_logits(values)

    def predict_timing_perturbation(
        self,
        modalities: Sequence[str],
        *,
        kind: str,
        magnitude: float,
        event_step_milliseconds: float,
    ) -> np.ndarray:
        del event_step_milliseconds
        try:
            from dtw_analysis.perturbations import warp_fixed_grid
        except ModuleNotFoundError as error:
            if error.name != "dtw_analysis":
                raise
            from ..dtw_analysis.perturbations import warp_fixed_grid
        values = warp_fixed_grid(
            self.split.values,
            self.channel_names,
            modalities,
            kind=kind,
            magnitude=magnitude,
            center_index=(
                -self.data_config.window_start_seconds
                / (
                    self.data_config.window_end_seconds
                    - self.data_config.window_start_seconds
                )
                * (self.data_config.num_time_points - 1)
            ),
        )
        return self.predict_fixed_grid_logits(values)

    def timing_step_milliseconds(self, event_fallback: float) -> float:
        del event_fallback
        return (
            self.data_config.window_end_seconds
            - self.data_config.window_start_seconds
        ) / (self.data_config.num_time_points - 1) * 1_000.0


class EventTransformerAdapter(CheckpointAdapter):
    def __init__(
        self,
        checkpoint_path: Path,
        payload: dict[str, Any],
        *,
        device: str,
        batch_size: int,
    ) -> None:
        super().__init__(checkpoint_path, payload)
        try:
            from event_transformer.training import load_trained_model
        except ModuleNotFoundError as error:
            if error.name != "event_transformer":
                raise
            from ..event_transformer.training import load_trained_model

        self.model, self.normalizer, restored = load_trained_model(
            checkpoint_path, device
        )
        self.payload = restored
        self.device = next(self.model.parameters()).device
        self.data_config = dataclass_from_dict(
            EventDataConfig, self.config_payload["data"]
        )
        self.data_config = relocate_checkpoint_paths(self.data_config)
        self.source = load_fold(self.data_config.dataset_path)["test"]
        self.batch_size = batch_size
        self._sample_ids: list[str] | None = None
        self._labels: np.ndarray | None = None
        self._group_ids = [str(row["word"].get("experiment")) for row in self.source]
        # Establish deterministic row metadata once using the full checkpoint input.
        self.predict_logits(MODALITY_NAMES)

    @property
    def sample_ids(self) -> list[str]:
        assert self._sample_ids is not None
        return self._sample_ids

    @property
    def labels(self) -> np.ndarray:
        assert self._labels is not None
        return self._labels

    @property
    def group_ids(self) -> list[str]:
        return self._group_ids

    def predict_logits(self, included_modalities: Sequence[str]) -> np.ndarray:
        coalition = normalize_modalities(included_modalities)
        if coalition not in self._cache:
            self._cache[coalition] = np.asarray(
                self._predict_logits(coalition), dtype=np.float64
            )
        return self._cache[coalition]

    def _predict_logits(self, coalition: tuple[str, ...]) -> np.ndarray:
        return self._predict_source(self.source, coalition, validate_identity=True)

    def _predict_source(
        self,
        source: Sequence[Any],
        coalition: tuple[str, ...],
        *,
        validate_identity: bool,
    ) -> np.ndarray:
        encoder = EventSequenceEncoder(
            EventFeatureExtractor(),
            max_events_per_modality=self.data_config.max_events_per_modality,
            time_clip_seconds=self.data_config.time_clip_seconds,
            positive_label=self.data_config.positive_label,
            negative_label=self.data_config.negative_label,
            included_modalities=coalition,
        )
        dataset = EventWindowDataset(
            source,
            encoder=encoder,
            normalizer=self.normalizer,
            cache_in_memory=self.data_config.cache_in_memory,
        )
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=EventBatchCollator(MODALITY_DIMS),
        )
        logits: list[np.ndarray] = []
        labels: list[np.ndarray] = []
        sample_ids: list[str] = []
        self.model.eval()
        with torch.no_grad():
            for raw_batch in loader:
                sample_ids.extend(str(value) for value in raw_batch["sample_ids"])
                batch = {
                    key: (
                        {item_key: item.to(self.device) for item_key, item in value.items()}
                        if isinstance(value, dict)
                        else value.to(self.device)
                        if isinstance(value, torch.Tensor)
                        else value
                    )
                    for key, value in raw_batch.items()
                }
                output = self.model(
                    modality_features=batch["modality_features"],
                    finger_flags=batch["finger_flags"],
                    modality_token_indices=batch["modality_token_indices"],
                    modality_actor_relation_ids=batch["modality_actor_relation_ids"],
                    time_features=batch["time_features"],
                    event_mask=batch["event_mask"],
                    anchor_mask=batch["anchor_mask"],
                    padding_mask=batch["padding_mask"],
                )
                logits.append(output.detach().cpu().numpy())
                labels.append(batch["labels"].detach().cpu().numpy())
        current_labels = np.concatenate(labels).astype(np.int64)
        if self._sample_ids is None:
            self._sample_ids = sample_ids
            self._labels = current_labels
        elif validate_identity and (
            sample_ids != self._sample_ids
            or not np.array_equal(current_labels, self._labels)
        ):
            raise RuntimeError("Event coalition changed test sample order or labels")
        return np.concatenate(logits)

    def predict_timing_perturbation(
        self,
        modalities: Sequence[str],
        *,
        kind: str,
        magnitude: float,
        event_step_milliseconds: float,
    ) -> np.ndarray:
        try:
            from dtw_analysis.perturbations import warp_event_rows
        except ModuleNotFoundError as error:
            if error.name != "dtw_analysis":
                raise
            from ..dtw_analysis.perturbations import warp_event_rows
        source = warp_event_rows(
            self.source,
            modalities,
            kind=kind,
            magnitude=magnitude,
            step_milliseconds=event_step_milliseconds,
        )
        return self._predict_source(source, MODALITY_NAMES, validate_identity=True)


def load_checkpoint_adapter(
    checkpoint_path: str | Path,
    *,
    device: str = "cpu",
    batch_size: int = 64,
) -> CheckpointAdapter:
    """Detect a repository checkpoint and return its attribution adapter."""

    path = Path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.suffix == ".pt":
        payload = torch.load(path, map_location="cpu", weights_only=False)
        checkpoint_type = payload.get("checkpoint_type")
        if checkpoint_type == "fine_tuned_classifier":
            return EventTransformerAdapter(
                path, payload, device=device, batch_size=batch_size
            )
        if checkpoint_type in FIXED_GRID_TORCH_TYPES:
            return FixedGridTorchAdapter(
                path, payload, device=device, batch_size=batch_size
            )
        raise ValueError(f"Unsupported PyTorch checkpoint type {checkpoint_type!r}")
    payload = joblib.load(path)
    if not isinstance(payload, dict) or "artifact_type" not in payload:
        raise ValueError(f"{path} is not a supported repository model artifact")
    return FixedGridEstimatorAdapter(path, payload, device=device)
