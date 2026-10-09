"""Configuration objects for multivariate MrSQM experiments."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

try:
    from minirocket.config import DataConfig, EvaluationConfig
except ModuleNotFoundError as error:
    if error.name != "minirocket":
        raise
    from ..minirocket.config import DataConfig, EvaluationConfig


VALID_STRATEGIES = frozenset({"R", "S", "RS", "SR"})
VALID_SOLVERS = frozenset(
    {"lbfgs", "liblinear", "newton-cg", "newton-cholesky", "sag", "saga"}
)


@dataclass(frozen=True, slots=True)
class MrSQMConfig:
    """Symbolic representations, feature selection, and logistic settings."""

    strategy: str = "RS"
    features_per_representation: int = 500
    selection_per_representation: int = 2_000
    num_sax_representations: int = 0
    num_sfa_representations: int = 5
    sfa_normalize: bool = True
    use_first_difference: bool = True

    # Each MrSQM representation is repeated for every input channel. Screen the
    # 698 channels on training data to keep its dense selected-feature matrix bounded.
    max_channels: int = 8
    channel_score_epsilon: float = 1e-8

    logistic_c: float = 1.0
    logistic_solver: str = "newton-cg"
    logistic_max_iterations: int = 1_000
    class_weight: Literal["balanced"] | None = "balanced"

    def validate(self) -> None:
        if self.strategy not in VALID_STRATEGIES:
            raise ValueError(
                f"strategy must be selected from {sorted(VALID_STRATEGIES)}"
            )
        if self.features_per_representation < 1:
            raise ValueError("features_per_representation must be positive")
        if self.selection_per_representation < 1:
            raise ValueError("selection_per_representation must be positive")
        if self.strategy in {"RS", "SR"} and (
            self.selection_per_representation < self.features_per_representation
        ):
            raise ValueError(
                "selection_per_representation must be at least "
                "features_per_representation for two-stage selection"
            )
        if self.num_sax_representations < 0 or self.num_sfa_representations < 0:
            raise ValueError("representation counts cannot be negative")
        if self.num_sax_representations + self.num_sfa_representations < 1:
            raise ValueError("at least one SAX or SFA representation is required")
        if self.max_channels < 1:
            raise ValueError("max_channels must be positive")
        if self.channel_score_epsilon <= 0:
            raise ValueError("channel_score_epsilon must be positive")
        if self.logistic_c <= 0:
            raise ValueError("logistic_c must be positive")
        if self.logistic_solver not in VALID_SOLVERS:
            raise ValueError(
                f"logistic_solver must be selected from {sorted(VALID_SOLVERS)}"
            )
        if self.logistic_max_iterations < 1:
            raise ValueError("logistic_max_iterations must be positive")


@dataclass(frozen=True, slots=True)
class ExperimentConfig:
    """Complete configuration for one saved MrSQM dataset fold."""

    data: DataConfig
    output_dir: Path
    run_name: str
    model: MrSQMConfig = field(default_factory=MrSQMConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    seed: int = 42

    def validate(self) -> None:
        self.data.validate()
        self.model.validate()
        if self.data.num_time_points < 16:
            raise ValueError("num_time_points must be at least 16 for MrSQM")
        self.evaluation.validate()
        if not self.run_name.strip():
            raise ValueError("run_name cannot be empty")


__all__ = [
    "VALID_SOLVERS",
    "VALID_STRATEGIES",
    "DataConfig",
    "EvaluationConfig",
    "ExperimentConfig",
    "MrSQMConfig",
]
