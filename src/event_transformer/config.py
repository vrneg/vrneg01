"""Configuration objects for event-transformer experiments.

The project intentionally uses Python configuration objects instead of a command-line
interface.  Construct these dataclasses in a script and pass the resulting
``ExperimentConfig`` directly to the training function.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from .features import MODALITY_NAMES


@dataclass(frozen=True, slots=True)
class DataConfig:
    """Dataset loading and event-sequence preparation settings."""

    dataset_path: Path
    batch_size: int = 8
    evaluation_batch_size: int | None = None
    num_workers: int = 0
    cache_in_memory: bool = True
    normalize_features: bool = True
    max_events_per_modality: int | None = 128
    time_clip_seconds: float | None = 10.0
    positive_label: str = "neg"
    negative_label: str = "none"
    included_modalities: tuple[str, ...] | None = None
    masked_channels: tuple[str, ...] | None = None

    def validate(self) -> None:
        if self.batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        if self.evaluation_batch_size is not None and self.evaluation_batch_size < 1:
            raise ValueError("evaluation_batch_size must be at least 1")
        if self.num_workers < 0:
            raise ValueError("num_workers cannot be negative")
        if self.max_events_per_modality is not None and self.max_events_per_modality < 1:
            raise ValueError("max_events_per_modality must be at least 1 or None")
        if self.time_clip_seconds is not None and self.time_clip_seconds <= 0:
            raise ValueError("time_clip_seconds must be positive or None")
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
class ModelConfig:
    """Event Transformer architecture settings."""

    d_model: int = 32
    nhead: int = 4
    num_layers: int = 1
    dim_feedforward: int = 128
    modality_hidden_dim: int = 64
    modality_fusion_hidden_dim: int = 64
    time_hidden_dim: int = 32
    classifier_hidden_dim: int = 16
    dropout: float = 0.25
    norm_first: bool = True

    def validate(self) -> None:
        if self.d_model < 1:
            raise ValueError("d_model must be positive")
        if self.nhead < 1 or self.d_model % self.nhead != 0:
            raise ValueError("nhead must be positive and divide d_model exactly")
        if self.num_layers < 1:
            raise ValueError("num_layers must be at least 1")
        for name, value in (
            ("dim_feedforward", self.dim_feedforward),
            ("modality_hidden_dim", self.modality_hidden_dim),
            ("modality_fusion_hidden_dim", self.modality_fusion_hidden_dim),
            ("time_hidden_dim", self.time_hidden_dim),
            ("classifier_hidden_dim", self.classifier_hidden_dim),
        ):
            if value < 1:
                raise ValueError(f"{name} must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    """Optimization, early-stopping, and reproducibility settings."""

    max_epochs: int = 60
    # ``learning_rate`` applies to the pretrained feature extractor and temporal
    # backbone.  A newly initialized classifier can optionally learn faster.
    learning_rate: float = 1e-4
    classifier_learning_rate: float | None = None
    # Train only the classifier for these initial epochs.  This gives the new head
    # time to adapt before fine-tuning the pretrained representation.
    freeze_backbone_epochs: int = 0
    weight_decay: float = 1e-3
    gradient_clip_norm: float | None = 1.0
    early_stopping_patience: int = 10
    early_stopping_min_delta: float = 1e-4
    selection_metric: Literal[
        "loss",
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
    lr_scheduler: Literal["none", "reduce_on_plateau"] = "reduce_on_plateau"
    lr_scheduler_factor: float = 0.5
    lr_scheduler_patience: int = 3
    minimum_learning_rate: float = 1e-6
    mixed_precision: bool = True
    deterministic_algorithms: bool = False
    show_progress: bool = True

    def validate(self) -> None:
        if self.max_epochs < 1:
            raise ValueError("max_epochs must be at least 1")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if (
            self.classifier_learning_rate is not None
            and self.classifier_learning_rate <= 0
        ):
            raise ValueError("classifier_learning_rate must be positive or None")
        if self.freeze_backbone_epochs < 0:
            raise ValueError("freeze_backbone_epochs cannot be negative")
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
        if self.minimum_learning_rate < 0:
            raise ValueError("minimum_learning_rate cannot be negative")
        if self.minimum_learning_rate >= self.learning_rate:
            raise ValueError("minimum_learning_rate must be below learning_rate")
        if (
            self.classifier_learning_rate is not None
            and self.minimum_learning_rate >= self.classifier_learning_rate
        ):
            raise ValueError(
                "minimum_learning_rate must be below classifier_learning_rate"
            )


@dataclass(frozen=True, slots=True)
class PretrainingConfig:
    """Self-supervised masked-modality reconstruction settings.

    Pretraining uses only the training split belonging to the current fold.  It masks
    complete modality observations and asks the temporal encoder to reconstruct their
    normalized continuous feature vectors from the remaining context.
    """

    num_epochs: int = 10
    mask_probability: float = 0.20
    learning_rate: float = 1e-4
    weight_decay: float = 1e-3
    gradient_clip_norm: float | None = 1.0
    lr_scheduler_factor: float = 0.5
    lr_scheduler_patience: int = 2
    minimum_learning_rate: float = 1e-6

    def validate(self) -> None:
        if self.num_epochs < 1:
            raise ValueError("pretraining num_epochs must be at least 1")
        if not 0.0 < self.mask_probability < 1.0:
            raise ValueError("pretraining mask_probability must be in (0, 1)")
        if self.learning_rate <= 0:
            raise ValueError("pretraining learning_rate must be positive")
        if self.weight_decay < 0:
            raise ValueError("pretraining weight_decay cannot be negative")
        if self.gradient_clip_norm is not None and self.gradient_clip_norm <= 0:
            raise ValueError("pretraining gradient_clip_norm must be positive or None")
        if not 0.0 < self.lr_scheduler_factor < 1.0:
            raise ValueError("pretraining lr_scheduler_factor must be in (0, 1)")
        if self.lr_scheduler_patience < 1:
            raise ValueError("pretraining lr_scheduler_patience must be at least 1")
        if self.minimum_learning_rate < 0:
            raise ValueError("pretraining minimum_learning_rate cannot be negative")
        if self.minimum_learning_rate >= self.learning_rate:
            raise ValueError(
                "pretraining minimum_learning_rate must be below learning_rate"
            )


@dataclass(frozen=True, slots=True)
class ExperimentConfig:
    """Complete configuration for one fold and one random seed."""

    data: DataConfig
    output_dir: Path
    run_name: str
    model: ModelConfig = field(default_factory=ModelConfig)
    pretraining: PretrainingConfig = field(default_factory=PretrainingConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    seed: int = 42
    device: str = "cuda"

    def validate(self) -> None:
        self.data.validate()
        self.model.validate()
        self.pretraining.validate()
        self.training.validate()
        if not self.run_name.strip():
            raise ValueError("run_name cannot be empty")
