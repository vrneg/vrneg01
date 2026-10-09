"""Configuration for the compact cue-aware fusion-TCN follow-up."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

try:
    from minirocket.config import DataConfig
except ModuleNotFoundError as error:
    if error.name != "minirocket":
        raise
    from ..minirocket.config import DataConfig


@dataclass(frozen=True, slots=True)
class ModelConfig:
    """Small learned encoder with capacity scaled to each sensor modality."""

    # Eye, Facial, Head, Body, LeftHand, RightHand, LeftFinger, RightFinger.
    modality_projection_channels: tuple[int, ...] = (8, 16, 8, 8, 8, 8, 24, 24)
    fusion_channels: int = 48
    temporal_kernel_size: int = 5
    tcn_dilations: tuple[int, ...] = (1, 2, 4)
    convolutions_per_tcn_block: int = 2
    classifier_hidden_dim: int = 32
    dropout: float = 0.45
    modality_dropout: float = 0.15
    include_relative_time_channel: bool = True
    include_modality_presence_channels: bool = True
    pooling_regions: tuple[str, ...] = ("global", "pre", "post")
    pooling_statistics: tuple[str, ...] = ("mean", "maximum", "std")

    def validate(self) -> None:
        if len(self.modality_projection_channels) != 8:
            raise ValueError(
                "modality_projection_channels must contain one value for each of "
                "the eight modalities"
            )
        if any(value < 1 for value in self.modality_projection_channels):
            raise ValueError("modality projection channels must be positive")
        for name, value in (
            ("fusion_channels", self.fusion_channels),
            ("convolutions_per_tcn_block", self.convolutions_per_tcn_block),
            ("classifier_hidden_dim", self.classifier_hidden_dim),
        ):
            if value < 1:
                raise ValueError(f"{name} must be positive")
        if self.temporal_kernel_size < 1 or self.temporal_kernel_size % 2 == 0:
            raise ValueError("temporal_kernel_size must be a positive odd value")
        if not self.tcn_dilations or any(value < 1 for value in self.tcn_dilations):
            raise ValueError("tcn_dilations must contain positive values")
        for name, value in (
            ("dropout", self.dropout),
            ("modality_dropout", self.modality_dropout),
        ):
            if not 0.0 <= value < 1.0:
                raise ValueError(f"{name} must be in [0, 1)")
        allowed_regions = {"global", "pre", "post"}
        if not self.pooling_regions or len(set(self.pooling_regions)) != len(
            self.pooling_regions
        ):
            raise ValueError("pooling_regions must be non-empty and unique")
        if any(region not in allowed_regions for region in self.pooling_regions):
            raise ValueError(
                f"pooling_regions must be selected from {sorted(allowed_regions)}"
            )
        allowed_statistics = {"mean", "maximum", "std"}
        if not self.pooling_statistics or len(set(self.pooling_statistics)) != len(
            self.pooling_statistics
        ):
            raise ValueError("pooling_statistics must be non-empty and unique")
        if any(
            statistic not in allowed_statistics
            for statistic in self.pooling_statistics
        ):
            raise ValueError(
                "pooling_statistics must be selected from "
                f"{sorted(allowed_statistics)}"
            )


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    """Optimization and regularization for one seed and fold."""

    batch_size: int = 32
    evaluation_batch_size: int = 64
    num_workers: int = 0
    max_epochs: int = 100
    learning_rate: float = 3e-4
    weight_decay: float = 1e-2
    gradient_clip_norm: float | None = 1.0
    early_stopping_patience: int = 20
    early_stopping_min_delta: float = 1e-4
    selection_metric: Literal[
        "loss",
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
        "matthews_correlation_coefficient",
        "roc_auc",
        "average_precision",
    ] = "loss"
    positive_class_weight: float | None = None
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
    lr_scheduler_factor: float = 0.5
    lr_scheduler_patience: int = 6
    minimum_learning_rate: float = 1e-6
    mixed_precision: bool = True
    deterministic_algorithms: bool = False
    input_noise_std: float = 0.02
    temporal_shift_steps: int = 1
    show_progress: bool = True

    def validate(self) -> None:
        if self.batch_size < 1 or self.evaluation_batch_size < 1:
            raise ValueError("batch sizes must be positive")
        if self.num_workers < 0:
            raise ValueError("num_workers cannot be negative")
        if self.max_epochs < 1:
            raise ValueError("max_epochs must be at least 1")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.weight_decay < 0:
            raise ValueError("weight_decay cannot be negative")
        if self.gradient_clip_norm is not None and self.gradient_clip_norm <= 0:
            raise ValueError("gradient_clip_norm must be positive or None")
        if self.early_stopping_patience < 1:
            raise ValueError("early_stopping_patience must be at least 1")
        if self.early_stopping_min_delta < 0:
            raise ValueError("early_stopping_min_delta cannot be negative")
        if self.positive_class_weight is not None and self.positive_class_weight <= 0:
            raise ValueError("positive_class_weight must be positive or None")
        if not 0.0 < self.threshold < 1.0:
            raise ValueError("threshold must be strictly between 0 and 1")
        if not 0.0 < self.lr_scheduler_factor < 1.0:
            raise ValueError("lr_scheduler_factor must be in (0, 1)")
        if self.lr_scheduler_patience < 1:
            raise ValueError("lr_scheduler_patience must be at least 1")
        if not 0.0 <= self.minimum_learning_rate < self.learning_rate:
            raise ValueError("minimum_learning_rate must be below learning_rate")
        if self.input_noise_std < 0:
            raise ValueError("input_noise_std cannot be negative")
        if self.temporal_shift_steps < 0:
            raise ValueError("temporal_shift_steps cannot be negative")


@dataclass(frozen=True, slots=True)
class ExperimentConfig:
    """Complete configuration for one seed and fold."""

    data: DataConfig
    output_dir: Path
    run_name: str
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    seed: int = 42
    device: str = "auto"

    def validate(self) -> None:
        self.data.validate()
        self.model.validate()
        self.training.validate()
        if not self.run_name.strip():
            raise ValueError("run_name cannot be empty")
        if not self.device.strip():
            raise ValueError("device cannot be empty")


__all__ = ["DataConfig", "ExperimentConfig", "ModelConfig", "TrainingConfig"]
