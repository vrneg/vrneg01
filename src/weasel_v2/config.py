"""Configuration objects for multivariate WEASEL 2.0 experiments."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np

try:
    from minirocket.config import DataConfig, EvaluationConfig
except ModuleNotFoundError as error:
    if error.name != "minirocket":
        raise
    from ..minirocket.config import DataConfig, EvaluationConfig


WEASEL_RIDGE_ALPHAS: tuple[float, ...] = tuple(
    float(value) for value in np.logspace(-1, 5, 10)
)
VALID_FEATURE_SELECTION = frozenset({"chi2_top_k", "none", "random"})


@dataclass(frozen=True, slots=True)
class WeaselV2Config:
    """Randomized symbolic-transform and ridge-classifier settings."""

    min_window: int = 4
    norm_options: tuple[bool, ...] = (False,)
    word_lengths: tuple[int, ...] = (7, 8)
    use_first_differences: tuple[bool, ...] = (True, False)
    feature_selection: str = "chi2_top_k"
    max_feature_count: int = 30_000
    ensemble_size: int | None = None

    # Native WEASEL 2.0 is univariate. The multivariate adapter screens channels
    # using training labels, then shares the fixed ensemble across those channels.
    max_channels: int = 32
    channel_score_epsilon: float = 1e-8

    alphas: tuple[float, ...] = WEASEL_RIDGE_ALPHAS
    class_weight: Literal["balanced"] | None = None
    n_jobs: int = 1

    def validate(self) -> None:
        if self.min_window < 4:
            raise ValueError("min_window must be at least 4")
        if not self.norm_options or any(
            not isinstance(value, bool) for value in self.norm_options
        ):
            raise ValueError("norm_options must contain one or more booleans")
        if not self.word_lengths or any(value < 2 for value in self.word_lengths):
            raise ValueError("word_lengths must contain integers of at least 2")
        if not self.use_first_differences or any(
            not isinstance(value, bool) for value in self.use_first_differences
        ):
            raise ValueError(
                "use_first_differences must contain one or more booleans"
            )
        if self.feature_selection not in VALID_FEATURE_SELECTION:
            raise ValueError(
                "feature_selection must be selected from "
                f"{sorted(VALID_FEATURE_SELECTION)}"
            )
        if self.max_feature_count < 1:
            raise ValueError("max_feature_count must be positive")
        if self.ensemble_size is not None and self.ensemble_size < 1:
            raise ValueError("ensemble_size must be positive or None")
        if self.max_channels < 1:
            raise ValueError("max_channels must be positive")
        if self.channel_score_epsilon <= 0:
            raise ValueError("channel_score_epsilon must be positive")
        if not self.alphas or any(alpha <= 0 for alpha in self.alphas):
            raise ValueError("alphas must contain only positive values")
        if self.n_jobs == 0:
            raise ValueError("n_jobs cannot be zero")


@dataclass(frozen=True, slots=True)
class ExperimentConfig:
    """Complete configuration for one saved WEASEL 2.0 dataset fold."""

    data: DataConfig
    output_dir: Path
    run_name: str
    model: WeaselV2Config = field(default_factory=WeaselV2Config)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    seed: int = 42

    def validate(self) -> None:
        self.data.validate()
        self.model.validate()
        if self.model.min_window > self.data.num_time_points:
            raise ValueError("min_window cannot exceed num_time_points")
        self.evaluation.validate()
        if not self.run_name.strip():
            raise ValueError("run_name cannot be empty")


__all__ = [
    "VALID_FEATURE_SELECTION",
    "WEASEL_RIDGE_ALPHAS",
    "DataConfig",
    "EvaluationConfig",
    "ExperimentConfig",
    "WeaselV2Config",
]
