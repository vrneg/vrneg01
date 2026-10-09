"""Configuration objects for MultiRocket and optional Hydra experiments."""

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


@dataclass(frozen=True, slots=True)
class MultiRocketConfig:
    """MultiRocket, optional Hydra, and ridge-classifier settings."""

    num_kernels: int = 6_250
    max_dilations_per_kernel: int = 32
    num_features_per_kernel: int = 4
    normalise_per_instance: bool = False
    use_hydra: bool = False
    hydra_num_kernels: int = 8
    hydra_num_groups: int = 64
    hydra_max_num_channels: int = 8
    alphas: tuple[float, ...] = EXTENDED_RIDGE_ALPHAS
    class_weight: Literal["balanced"] | None = None
    n_jobs: int = 1

    def validate(self) -> None:
        if self.num_kernels < 84:
            raise ValueError("num_kernels must be at least 84")
        if self.max_dilations_per_kernel < 1:
            raise ValueError("max_dilations_per_kernel must be at least 1")
        # aeon 1.x's compiled transform writes all four pooling features even when
        # configured with a smaller value, which can corrupt native memory.
        if self.num_features_per_kernel != 4:
            raise ValueError("num_features_per_kernel must be 4 for aeon 1.x")
        if self.hydra_num_kernels < 2:
            raise ValueError("hydra_num_kernels must be at least 2")
        if self.hydra_num_groups < 1:
            raise ValueError("hydra_num_groups must be at least 1")
        if self.hydra_max_num_channels < 2:
            raise ValueError("hydra_max_num_channels must be at least 2")
        if not self.alphas or any(alpha <= 0 for alpha in self.alphas):
            raise ValueError("alphas must contain only positive values")
        if self.n_jobs == 0:
            raise ValueError("n_jobs cannot be zero")


@dataclass(frozen=True, slots=True)
class ExperimentConfig:
    """Complete configuration for one saved MultiRocket dataset fold."""

    data: DataConfig
    output_dir: Path
    run_name: str
    model: MultiRocketConfig = field(default_factory=MultiRocketConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    seed: int = 42

    def validate(self) -> None:
        self.data.validate()
        self.model.validate()
        self.evaluation.validate()
        if not self.run_name.strip():
            raise ValueError("run_name cannot be empty")


__all__ = [
    "EXTENDED_RIDGE_ALPHAS",
    "DataConfig",
    "EvaluationConfig",
    "ExperimentConfig",
    "MultiRocketConfig",
]
