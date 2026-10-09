"""Configuration for Adaptive Multi-Representation RocketPFN."""

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


ViewName = Literal[
    "raw",
    "first_diff",
    "second_diff",
    "smoothed",
    "highpass",
    "local_norm",
]
TabPFNFitMode = Literal["low_memory", "fit_preprocessors", "fit_with_cache"]

DEFAULT_VIEWS: tuple[ViewName, ...] = (
    "raw",
    "first_diff",
    "second_diff",
    "smoothed",
    "highpass",
    "local_norm",
)


@dataclass(frozen=True, slots=True)
class AdaptiveRocketPFNConfig:
    """Candidate-bank, stable-selection, and semantic-ensemble settings."""

    views: tuple[ViewName, ...] = DEFAULT_VIEWS
    smoothing_window: int = 5
    local_normalization_epsilon: float = 1e-6

    # Two banks approximate a learnable dilation prior: a short-scale-only bank
    # and the full MultiRocket dilation schedule. With 32 points these differ.
    dilation_regimes: tuple[int, ...] = (1, 32)
    multirocket_num_kernels_per_bank: int = 625
    multirocket_num_features_per_kernel: int = 4
    multirocket_normalise_per_instance: bool = False

    hydra_num_kernels: int = 8
    hydra_num_groups: int = 128
    hydra_max_num_channels: int = 8

    prototype_max_shapelets: int = 512
    prototype_shapelet_lengths: tuple[int, ...] = (7, 9, 11)
    prototype_normalization_probability: float = 0.8
    prototype_similarity: float = 0.5

    selected_features_per_expert: int = 500
    active_families_per_expert: int = 12
    minimum_features_per_active_family: int = 8
    selection_inner_folds: int = 3
    selection_repeats: int = 2
    selection_pool_multiplier: int = 3
    allocation_temperature: float = 0.5
    redundancy_correlation_threshold: float = 0.98
    redundancy_sample_count: int = 256

    oof_folds: int = 3
    oof_tabpfn_n_estimators: int = 2
    ensemble_weight_l2: float = 0.05

    tabpfn_model_path: Path | None = None
    tabpfn_n_estimators: int = 8
    tabpfn_auto_scale_n_estimators: bool = True
    tabpfn_device: str = "cuda"
    tabpfn_fit_mode: TabPFNFitMode = "low_memory"
    tabpfn_memory_saving_mode: Literal["auto"] | bool = "auto"
    tabpfn_inference_precision: Literal["auto", "autocast"] = "auto"
    tabpfn_preprocessing_jobs: int = 8
    tabpfn_show_progress_bar: bool = False
    tabpfn_ignore_pretraining_limits: bool = True
    tabpfn_balance_probabilities: bool = False

    transform_n_jobs: int = 4

    def validate(self) -> None:
        valid_views = set(DEFAULT_VIEWS)
        if not self.views:
            raise ValueError("views must contain at least one representation")
        if len(set(self.views)) != len(self.views):
            raise ValueError("views must not contain duplicates")
        unknown_views = set(self.views) - valid_views
        if unknown_views:
            raise ValueError(f"Unknown views: {sorted(unknown_views)}")
        if self.smoothing_window < 3 or self.smoothing_window % 2 == 0:
            raise ValueError("smoothing_window must be an odd integer of at least 3")
        if self.local_normalization_epsilon <= 0:
            raise ValueError("local_normalization_epsilon must be positive")

        if not self.dilation_regimes or any(
            value < 1 for value in self.dilation_regimes
        ):
            raise ValueError("dilation_regimes must contain positive integers")
        if len(set(self.dilation_regimes)) != len(self.dilation_regimes):
            raise ValueError("dilation_regimes must not contain duplicates")
        if self.multirocket_num_kernels_per_bank < 84:
            raise ValueError("multirocket_num_kernels_per_bank must be at least 84")
        if self.multirocket_num_features_per_kernel != 4:
            raise ValueError(
                "multirocket_num_features_per_kernel must be 4 for aeon 1.x"
            )
        if self.multirocket_normalise_per_instance:
            raise ValueError(
                "Per-view normalization is explicit; MultiRocket normalise must be False"
            )
        if self.hydra_num_kernels < 2 or self.hydra_num_groups < 1:
            raise ValueError("HYDRA requires at least two kernels and one group")
        if self.hydra_max_num_channels < 2:
            raise ValueError("hydra_max_num_channels must be at least 2")
        if self.prototype_max_shapelets < 1:
            raise ValueError("prototype_max_shapelets must be positive")
        if not self.prototype_shapelet_lengths or any(
            value < 3 for value in self.prototype_shapelet_lengths
        ):
            raise ValueError("prototype_shapelet_lengths must be at least 3")
        if not 0.0 <= self.prototype_normalization_probability <= 1.0:
            raise ValueError("prototype_normalization_probability must be in [0, 1]")
        if not 0.0 <= self.prototype_similarity <= 1.0:
            raise ValueError("prototype_similarity must be in [0, 1]")

        if self.selected_features_per_expert < 1:
            raise ValueError("selected_features_per_expert must be positive")
        if self.active_families_per_expert < 1:
            raise ValueError("active_families_per_expert must be positive")
        if self.minimum_features_per_active_family < 1:
            raise ValueError("minimum_features_per_active_family must be positive")
        if self.selection_inner_folds < 2 or self.selection_repeats < 1:
            raise ValueError("selection requires at least two folds and one repeat")
        if self.selection_pool_multiplier < 1:
            raise ValueError("selection_pool_multiplier must be positive")
        if self.allocation_temperature <= 0:
            raise ValueError("allocation_temperature must be positive")
        if not 0.0 < self.redundancy_correlation_threshold <= 1.0:
            raise ValueError("redundancy_correlation_threshold must be in (0, 1]")
        if self.redundancy_sample_count < 2:
            raise ValueError("redundancy_sample_count must be at least 2")
        if self.oof_folds < 2:
            raise ValueError("oof_folds must be at least 2")
        if self.oof_tabpfn_n_estimators < 1 or self.tabpfn_n_estimators < 1:
            raise ValueError("TabPFN estimator counts must be positive")
        if self.ensemble_weight_l2 < 0:
            raise ValueError("ensemble_weight_l2 cannot be negative")

        if self.tabpfn_model_path is not None and not self.tabpfn_model_path.name:
            raise ValueError("tabpfn_model_path must identify a checkpoint file")
        if not self.tabpfn_device.strip():
            raise ValueError("tabpfn_device cannot be empty")
        if self.tabpfn_fit_mode not in {
            "low_memory",
            "fit_preprocessors",
            "fit_with_cache",
        }:
            raise ValueError("Unsupported tabpfn_fit_mode")
        if self.tabpfn_memory_saving_mode not in {"auto", True, False}:
            raise ValueError("Unsupported tabpfn_memory_saving_mode")
        if self.tabpfn_inference_precision not in {"auto", "autocast"}:
            raise ValueError("Unsupported tabpfn_inference_precision")
        if self.tabpfn_preprocessing_jobs < 1:
            raise ValueError("tabpfn_preprocessing_jobs must be positive")
        if self.transform_n_jobs == 0:
            raise ValueError("transform_n_jobs cannot be zero")


@dataclass(frozen=True, slots=True)
class ExperimentConfig:
    """Complete configuration for one saved adaptive RocketPFN fold."""

    data: DataConfig
    output_dir: Path
    run_name: str
    model: AdaptiveRocketPFNConfig = field(
        default_factory=AdaptiveRocketPFNConfig
    )
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    seed: int = 42

    def validate(self) -> None:
        self.data.validate()
        self.model.validate()
        self.evaluation.validate()
        shortest_view = self.data.num_time_points - (
            2 if "second_diff" in self.model.views else 1
            if "first_diff" in self.model.views
            else 0
        )
        if shortest_view < 9:
            raise ValueError("Every adaptive ROCKET view must retain 9 time points")
        smoothed_views = {"smoothed", "highpass", "local_norm"}
        if (
            set(self.model.views) & smoothed_views
            and self.model.smoothing_window > self.data.num_time_points
        ):
            raise ValueError("smoothing_window exceeds the time grid")
        if max(self.model.prototype_shapelet_lengths) > self.data.num_time_points:
            raise ValueError("Prototype shapelet lengths exceed the time grid")
        if not self.run_name.strip():
            raise ValueError("run_name cannot be empty")


__all__ = [
    "AdaptiveRocketPFNConfig",
    "DEFAULT_VIEWS",
    "DataConfig",
    "EvaluationConfig",
    "ExperimentConfig",
    "TabPFNFitMode",
    "ViewName",
]
