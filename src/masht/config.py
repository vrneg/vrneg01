"""Configuration for MASHT: MultiRocket + HYDRA features classified by TabPFN-3."""

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


FeatureBudgetScope = Literal["train", "all_splits"]
TabPFNFitMode = Literal["low_memory", "fit_preprocessors", "fit_with_cache"]


@dataclass(frozen=True, slots=True)
class MASHTConfig:
    """Paper-aligned MASHT feature and TabPFN-3 settings.

    ``max_features=None`` enables MASHT's adaptive nominal feature budget. The
    benchmark selects that budget from train plus test sample count; this project
    uses train, validation, and test counts because its held-out data has two splits.
    Only split sizes are used, never validation or test labels.
    """

    max_features: int | None = None
    feature_budget_scope: FeatureBudgetScope = "all_splits"

    multirocket_max_dilations_per_kernel: int = 32
    multirocket_num_features_per_kernel: int = 4
    multirocket_normalise_per_instance: bool = False
    hydra_num_kernels: int = 8
    hydra_max_num_channels: int = 8
    transform_n_jobs: int = 4

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

    def validate(self) -> None:
        if self.max_features is not None and self.max_features < 1:
            raise ValueError("max_features must be positive or None")
        if self.feature_budget_scope not in {"train", "all_splits"}:
            raise ValueError("feature_budget_scope must be 'train' or 'all_splits'")
        if self.multirocket_max_dilations_per_kernel < 1:
            raise ValueError(
                "multirocket_max_dilations_per_kernel must be at least 1"
            )
        if self.multirocket_num_features_per_kernel != 4:
            raise ValueError(
                "multirocket_num_features_per_kernel must be 4 for aeon 1.x"
            )
        if self.multirocket_normalise_per_instance:
            raise ValueError(
                "MASHT uses unnormalised MultiRocket features; set "
                "multirocket_normalise_per_instance=False"
            )
        if self.hydra_num_kernels != 8:
            raise ValueError("MASHT requires exactly 8 competing HYDRA kernels")
        if self.hydra_max_num_channels < 2:
            raise ValueError("hydra_max_num_channels must be at least 2")
        if self.transform_n_jobs == 0:
            raise ValueError("transform_n_jobs cannot be zero")

        if self.tabpfn_model_path is not None and not self.tabpfn_model_path.name:
            raise ValueError("tabpfn_model_path must identify a checkpoint file")
        if self.tabpfn_n_estimators < 1:
            raise ValueError("tabpfn_n_estimators must be at least 1")
        if not self.tabpfn_device.strip():
            raise ValueError("tabpfn_device cannot be empty")
        if self.tabpfn_fit_mode not in {
            "low_memory",
            "fit_preprocessors",
            "fit_with_cache",
        }:
            raise ValueError("Unsupported tabpfn_fit_mode")
        if self.tabpfn_memory_saving_mode not in {"auto", True, False}:
            raise ValueError(
                "tabpfn_memory_saving_mode must be 'auto', True, or False"
            )
        if self.tabpfn_inference_precision not in {"auto", "autocast"}:
            raise ValueError(
                "tabpfn_inference_precision must be 'auto' or 'autocast'"
            )
        if self.tabpfn_preprocessing_jobs < 1:
            raise ValueError("tabpfn_preprocessing_jobs must be at least 1")


@dataclass(frozen=True, slots=True)
class ExperimentConfig:
    """Complete configuration for one saved MASHT dataset fold."""

    data: DataConfig
    output_dir: Path
    run_name: str
    model: MASHTConfig = field(default_factory=MASHTConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    seed: int = 42

    def validate(self) -> None:
        self.data.validate()
        self.model.validate()
        self.evaluation.validate()
        if self.data.num_time_points < 10:
            raise ValueError("MASHT requires at least 10 time points")
        if not self.run_name.strip():
            raise ValueError("run_name cannot be empty")


__all__ = [
    "DataConfig",
    "EvaluationConfig",
    "ExperimentConfig",
    "FeatureBudgetScope",
    "MASHTConfig",
    "TabPFNFitMode",
]
