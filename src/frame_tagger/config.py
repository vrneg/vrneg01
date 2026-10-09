"""Configuration for frame-level negation tagging experiments.

The data half is reused verbatim from :mod:`src.t2m_gpt.config` -- same folds, same fixed
grid, same optional representation toggles -- so a tagger result is comparable with the
window-level models rather than being an experiment on a different dataset.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

try:
    from t2m_gpt.config import DataConfig
except ModuleNotFoundError as error:  # pragma: no cover - import-path fallback
    if error.name != "t2m_gpt":
        raise
    from ..t2m_gpt.config import DataConfig

from .labels import DEFAULT_ASSUMED_DURATION_MS


@dataclass(frozen=True, slots=True)
class LabelConfig:
    """How per-frame BIO targets are derived from the word-level annotation."""

    #: ``None`` estimates the median anchor duration from the training split, which is
    #: preferred; a float pins it, which is useful for ablating the assumption itself.
    assumed_token_duration_ms: float | None = None
    fallback_duration_ms: float = DEFAULT_ASSUMED_DURATION_MS

    def validate(self) -> None:
        if self.assumed_token_duration_ms is not None:
            if self.assumed_token_duration_ms <= 0.0:
                raise ValueError("assumed_token_duration_ms must be positive or None")
        if self.fallback_duration_ms <= 0.0:
            raise ValueError("fallback_duration_ms must be positive")


@dataclass(frozen=True, slots=True)
class TaggerConfig:
    """Encoder architecture and CRF options."""

    encoder: Literal["bilstm", "transformer"] = "bilstm"
    d_model: int = 96
    num_layers: int = 2
    nhead: int = 4
    dim_feedforward: int = 256
    dropout: float = 0.3
    constrain_transitions: bool = True

    def validate(self) -> None:
        if self.d_model < 2 or self.d_model % 2 != 0:
            raise ValueError("d_model must be an even number of at least 2")
        if self.num_layers < 1:
            raise ValueError("num_layers must be positive")
        if self.encoder == "transformer" and self.d_model % self.nhead != 0:
            raise ValueError("d_model must be divisible by nhead for the transformer")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")


@dataclass(frozen=True, slots=True)
class TaggerTrainingConfig:
    """Optimization, early stopping, and evaluation options."""

    max_epochs: int = 80
    batch_size: int = 32
    evaluation_batch_size: int = 64
    learning_rate: float = 1e-3
    weight_decay: float = 1e-2
    gradient_clip_norm: float = 1.0
    early_stopping_patience: int = 12
    lr_scheduler_factor: float = 0.5
    lr_scheduler_patience: int = 5
    minimum_learning_rate: float = 1e-6
    #: Overlap threshold for span matching during selection and reporting.
    minimum_span_overlap: float = 0.5
    #: ``span_f1`` selects on overlap micro F1, ``loss`` on validation CRF loss.
    selection_metric: Literal["span_f1", "loss"] = "span_f1"
    show_progress: bool = True

    def validate(self) -> None:
        if self.max_epochs < 1:
            raise ValueError("max_epochs must be positive")
        if self.batch_size < 1 or self.evaluation_batch_size < 1:
            raise ValueError("batch sizes must be positive")
        if not 0.0 < self.minimum_span_overlap <= 1.0:
            raise ValueError("minimum_span_overlap must be in (0, 1]")
        if self.selection_metric not in ("span_f1", "loss"):
            raise ValueError(f"Unknown selection_metric: {self.selection_metric!r}")


@dataclass(frozen=True, slots=True)
class TaggerExperimentConfig:
    """One frame-tagging run on one fold."""

    data: DataConfig
    output_dir: str
    run_name: str
    labels: LabelConfig = field(default_factory=LabelConfig)
    tagger: TaggerConfig = field(default_factory=TaggerConfig)
    training: TaggerTrainingConfig = field(default_factory=TaggerTrainingConfig)
    seed: int = 42
    device: str = "auto"

    def validate(self) -> None:
        self.data.validate()
        self.labels.validate()
        self.tagger.validate()
        self.training.validate()
        if not str(self.run_name).strip():
            raise ValueError("run_name cannot be empty")


__all__ = [
    "DataConfig",
    "LabelConfig",
    "TaggerConfig",
    "TaggerExperimentConfig",
    "TaggerTrainingConfig",
]
