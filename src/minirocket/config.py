"""Configuration objects for multivariate MiniRocket experiments."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

try:
    from event_transformer.features import MODALITY_NAMES
except ModuleNotFoundError as error:
    if error.name != "event_transformer":
        raise
    from ..event_transformer.features import MODALITY_NAMES


# Half-decade steps retain the original weak-regularization candidates while
# extending the previous upper bound from 1e3 to 1e6.
EXTENDED_RIDGE_ALPHAS: tuple[float, ...] = (
    1e-3,
    3.1622776601683795e-3,
    1e-2,
    3.1622776601683795e-2,
    1e-1,
    3.1622776601683795e-1,
    1.0,
    3.1622776601683795,
    10.0,
    31.622776601683793,
    100.0,
    316.22776601683796,
    1_000.0,
    3_162.2776601683795,
    10_000.0,
    31_622.776601683792,
    100_000.0,
    316_227.7660168379,
    1_000_000.0,
)


@dataclass(frozen=True, slots=True)
class DataConfig:
    """Fold loading and fixed-grid conversion settings."""

    dataset_path: Path
    num_time_points: int = 32
    window_start_seconds: float = -0.5
    window_end_seconds: float = 0.5
    max_events_per_modality: int | None = 128
    normalize_features: bool = True
    cache_in_memory: bool = True
    actor_scope: Literal["anchor", "other", "all"] = "anchor"
    include_presence_channels: bool = True
    positive_label: str = "neg"
    negative_label: str = "none"
    # ``None`` preserves the original all-modality representation.  Attribution
    # retraining uses an explicit subset while retaining the checkpoint-compatible
    # fixed channel layout; excluded modality channels are set to zero.
    included_modalities: tuple[str, ...] | None = None
    # Optional fixed-grid channel names to zero while preserving the channel layout.
    masked_channels: tuple[str, ...] | None = None

    def validate(self) -> None:
        # MiniRocket uses kernels of length nine.
        if self.num_time_points < 9:
            raise ValueError("num_time_points must be at least 9 for MiniRocket")
        if self.window_end_seconds <= self.window_start_seconds:
            raise ValueError("window_end_seconds must exceed window_start_seconds")
        if self.max_events_per_modality is not None and self.max_events_per_modality < 1:
            raise ValueError("max_events_per_modality must be at least 1 or None")
        if self.positive_label == self.negative_label:
            raise ValueError("positive_label and negative_label must differ")
        if self.included_modalities is not None:
            unknown = set(self.included_modalities) - set(MODALITY_NAMES)
            if unknown:
                raise ValueError(f"Unknown included modalities: {sorted(unknown)}")
            if len(set(self.included_modalities)) != len(self.included_modalities):
                raise ValueError("included_modalities must not contain duplicates")
        if self.masked_channels is not None:
            if any(not channel.strip() for channel in self.masked_channels):
                raise ValueError("masked_channels must not contain empty names")
            if len(set(self.masked_channels)) != len(self.masked_channels):
                raise ValueError("masked_channels must not contain duplicates")


@dataclass(frozen=True, slots=True)
class MiniRocketConfig:
    """MiniRocket transform and ridge-classifier settings."""

    num_kernels: int = 10_000
    max_dilations_per_kernel: int = 32
    alphas: tuple[float, ...] = EXTENDED_RIDGE_ALPHAS
    class_weight: Literal["balanced"] | None = None
    n_jobs: int = 1

    def validate(self) -> None:
        if self.num_kernels < 84:
            raise ValueError("num_kernels must be at least 84")
        if self.max_dilations_per_kernel < 1:
            raise ValueError("max_dilations_per_kernel must be at least 1")
        if not self.alphas or any(alpha <= 0 for alpha in self.alphas):
            raise ValueError("alphas must contain only positive values")
        if self.n_jobs == 0:
            raise ValueError("n_jobs cannot be zero")


@dataclass(frozen=True, slots=True)
class EvaluationConfig:
    """Decision-threshold and reporting settings."""

    threshold: float = 0.5
    calibrate_threshold_on_validation: bool = False
    threshold_metric: Literal[
        "accuracy",
        "balanced_accuracy",
        "precision",
        "recall",
        "specificity",
        "f1",
        "negative_f1",
        "macro_f1",
        "weighted_f1",
        "matthews_correlation_coefficient",
    ] = "macro_f1"

    def validate(self) -> None:
        if not 0.0 < self.threshold < 1.0:
            raise ValueError("threshold must be strictly between 0 and 1")


@dataclass(frozen=True, slots=True)
class ExperimentConfig:
    """Complete configuration for one saved dataset fold."""

    data: DataConfig
    output_dir: Path
    run_name: str
    model: MiniRocketConfig = field(default_factory=MiniRocketConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    seed: int = 42

    def validate(self) -> None:
        self.data.validate()
        self.model.validate()
        self.evaluation.validate()
        if not self.run_name.strip():
            raise ValueError("run_name cannot be empty")
