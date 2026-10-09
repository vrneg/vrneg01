"""Configuration for modality-aware Inception-TCN experiments."""

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
    """Compact per-modality Inception and dilated-convolution architecture."""

    projection_channels: int = 16
    inception_branch_channels: int = 8
    inception_kernel_sizes: tuple[int, ...] = (3, 5, 9)
    num_inception_blocks: int = 2
    tcn_dilations: tuple[int, ...] = (1, 2, 4)
    classifier_hidden_dim: int = 64
    dropout: float = 0.30
    include_relative_time_channel: bool = True

    def validate(self) -> None:
        for name, value in (
            ("projection_channels", self.projection_channels),
            ("inception_branch_channels", self.inception_branch_channels),
            ("num_inception_blocks", self.num_inception_blocks),
            ("classifier_hidden_dim", self.classifier_hidden_dim),
        ):
            if value < 1:
                raise ValueError(f"{name} must be positive")
        if not self.inception_kernel_sizes:
            raise ValueError("inception_kernel_sizes cannot be empty")
        if any(size < 1 or size % 2 == 0 for size in self.inception_kernel_sizes):
            raise ValueError("inception_kernel_sizes must contain positive odd values")
        if not self.tcn_dilations or any(value < 1 for value in self.tcn_dilations):
            raise ValueError("tcn_dilations must contain positive values")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    """Optimization, early stopping, evaluation, and loader settings."""

    batch_size: int = 32
    evaluation_batch_size: int = 64
    num_workers: int = 0
    max_epochs: int = 100
    learning_rate: float = 3e-4
    weight_decay: float = 1e-3
    gradient_clip_norm: float | None = 1.0
    early_stopping_patience: int = 15
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
    lr_scheduler_patience: int = 5
    minimum_learning_rate: float = 1e-6
    mixed_precision: bool = True
    deterministic_algorithms: bool = False
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


@dataclass(frozen=True, slots=True)
class ExperimentConfig:
    """Complete configuration for one fold and random seed."""

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
