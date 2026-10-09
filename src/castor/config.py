"""Configuration objects for multivariate CASTOR experiments."""

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


# The CASTOR paper evaluates RidgeClassifierCV with this leave-one-out grid.
CASTOR_RIDGE_ALPHAS: tuple[float, ...] = (0.01, 1.0, 10.0)


@dataclass(frozen=True, slots=True)
class CastorConfig:
    """Competing-dilated-shapelet and ridge-classifier settings."""

    # Paper defaults: g=128 groups and k=16 shapelets per group.
    n_groups: int = 128
    n_shapelets: int = 16
    shapelet_size: int = 9
    metric: str = "euclidean"
    normalize_prob: float = 0.5
    lower: float = 0.01
    upper: float = 0.2
    soft_min: bool = True
    soft_max: bool = False
    soft_threshold: bool = True
    ignore_y: bool = False
    use_first_difference: bool = True

    # CASTOR's reference classifier applies square-root sparse scaling.
    sparse_scaler_exp: float = 4.0
    alphas: tuple[float, ...] = CASTOR_RIDGE_ALPHAS
    class_weight: Literal["balanced"] | None = None
    n_jobs: int = 1

    def validate(self) -> None:
        if self.n_groups < 1:
            raise ValueError("n_groups must be positive")
        if self.use_first_difference and (
            self.n_groups < 2 or self.n_groups % 2 != 0
        ):
            raise ValueError(
                "n_groups must be an even integer of at least 2 when first "
                "differences are enabled"
            )
        if self.n_shapelets < 1:
            raise ValueError("n_shapelets must be positive")
        if self.shapelet_size < 3 or self.shapelet_size % 2 == 0:
            raise ValueError("shapelet_size must be an odd integer of at least 3")
        if not self.metric.strip():
            raise ValueError("metric cannot be empty")
        if not 0.0 <= self.normalize_prob <= 1.0:
            raise ValueError("normalize_prob must be in [0, 1]")
        if not 0.0 <= self.lower < self.upper <= 1.0:
            raise ValueError("lower and upper must satisfy 0 <= lower < upper <= 1")
        if self.sparse_scaler_exp < 0:
            raise ValueError("sparse_scaler_exp must be non-negative")
        if not self.alphas or any(alpha <= 0 for alpha in self.alphas):
            raise ValueError("alphas must contain only positive values")
        if self.n_jobs == 0:
            raise ValueError("n_jobs cannot be zero")


@dataclass(frozen=True, slots=True)
class ExperimentConfig:
    """Complete configuration for one saved CASTOR dataset fold."""

    data: DataConfig
    output_dir: Path
    run_name: str
    model: CastorConfig = field(default_factory=CastorConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    seed: int = 42

    def validate(self) -> None:
        self.data.validate()
        self.model.validate()
        shortest_representation = self.data.num_time_points - int(
            self.model.use_first_difference
        )
        if self.model.shapelet_size > shortest_representation:
            raise ValueError(
                "shapelet_size cannot exceed the number of time points in the "
                "shortest CASTOR representation"
            )
        self.evaluation.validate()
        if not self.run_name.strip():
            raise ValueError("run_name cannot be empty")


__all__ = [
    "CASTOR_RIDGE_ALPHAS",
    "CastorConfig",
    "DataConfig",
    "EvaluationConfig",
    "ExperimentConfig",
]
