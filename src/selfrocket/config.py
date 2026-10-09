"""Configuration objects for multivariate SelF-Rocket experiments."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

try:
    from minirocket.config import (
        EXTENDED_RIDGE_ALPHAS,
        DataConfig,
        EvaluationConfig,
    )
except ModuleNotFoundError as error:
    if error.name != "minirocket":
        raise
    from ..minirocket.config import (
        EXTENDED_RIDGE_ALPHAS,
        DataConfig,
        EvaluationConfig,
    )


# The reference implementation uses ten logarithmically spaced values from
# 1e-3 through 1e3 for the small classifiers in the wrapper-selection stage.
SELECTION_RIDGE_ALPHAS: tuple[float, ...] = (
    1e-3,
    4.641588833612777e-3,
    2.1544346900318843e-2,
    1e-1,
    4.641588833612777e-1,
    2.154434690031882,
    10.0,
    46.41588833612773,
    215.44346900318823,
    1_000.0,
)


@dataclass(frozen=True, slots=True)
class SelFRocketConfig:
    """Feature generation, wrapper selection, and ridge settings."""

    num_kernels: int = 10_000
    max_dilations_per_kernel: int = 32
    normalise_per_instance: bool = False
    only_mix: bool = False

    selection_num_folds: int = 2
    selection_num_runs: int = 10
    selection_num_features: int = 2_500
    selection_max_samples: int = 500
    selection_alphas: tuple[float, ...] = SELECTION_RIDGE_ALPHAS

    vote_top: int = 5
    vote_threshold: float = 0.9
    length_threshold: int = 512

    alphas: tuple[float, ...] = EXTENDED_RIDGE_ALPHAS
    class_weight: Literal["balanced"] | None = None
    n_jobs: int = 1

    def validate(self) -> None:
        if self.num_kernels < 84:
            raise ValueError("num_kernels must be at least 84")
        if self.max_dilations_per_kernel < 1:
            raise ValueError("max_dilations_per_kernel must be at least 1")
        if self.selection_num_folds < 2:
            raise ValueError("selection_num_folds must be at least 2")
        if self.selection_num_runs < 1:
            raise ValueError("selection_num_runs must be at least 1")
        if self.selection_num_features < 1:
            raise ValueError("selection_num_features must be at least 1")
        if self.selection_max_samples < 4:
            raise ValueError("selection_max_samples must be at least 4")
        if not self.selection_alphas or any(
            alpha <= 0 for alpha in self.selection_alphas
        ):
            raise ValueError("selection_alphas must contain only positive values")
        num_candidates = 5 if self.only_mix else 15
        if not 1 <= self.vote_top <= num_candidates:
            raise ValueError(
                f"vote_top must be between 1 and {num_candidates} for this mode"
            )
        if not 0.0 < self.vote_threshold <= 1.0:
            raise ValueError("vote_threshold must be in (0, 1]")
        if self.length_threshold < 1:
            raise ValueError("length_threshold must be at least 1")
        if not self.alphas or any(alpha <= 0 for alpha in self.alphas):
            raise ValueError("alphas must contain only positive values")
        if self.n_jobs == 0:
            raise ValueError("n_jobs cannot be zero")


@dataclass(frozen=True, slots=True)
class ExperimentConfig:
    """Complete configuration for one saved SelF-Rocket dataset fold."""

    data: DataConfig
    output_dir: Path
    run_name: str
    model: SelFRocketConfig = field(default_factory=SelFRocketConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    seed: int = 42

    def validate(self) -> None:
        self.data.validate()
        # The difference representation must retain at least nine time points.
        if self.data.num_time_points < 10:
            raise ValueError("num_time_points must be at least 10 for SelF-Rocket")
        self.model.validate()
        self.evaluation.validate()
        if not self.run_name.strip():
            raise ValueError("run_name cannot be empty")


__all__ = [
    "DataConfig",
    "EvaluationConfig",
    "ExperimentConfig",
    "EXTENDED_RIDGE_ALPHAS",
    "SELECTION_RIDGE_ALPHAS",
    "SelFRocketConfig",
]
