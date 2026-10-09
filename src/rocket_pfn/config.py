"""Configuration for ROCKET features classified by TabPFN."""

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


FeatureRepresentation = Literal["rocket", "multirocket_hydra"]
TabPFNVersion = Literal["2.5", "3"]
TabPFNFitMode = Literal["low_memory", "fit_preprocessors", "fit_with_cache"]


@dataclass(frozen=True, slots=True)
class RocketPFNConfig:
    """Random convolutional representation and TabPFN inference settings.

    The defaults reproduce the representation budget in RocketPFN: ten independent
    groups of 1,000 ROCKET kernels, 2,000 features per group, and TabPFN v2.5 with
    eight internal estimators. ``multirocket_hydra`` is a project-specific extension.
    """

    feature_representation: FeatureRepresentation = "rocket"

    rocket_num_groups: int = 10
    rocket_kernels_per_group: int = 1_000
    rocket_normalise_per_instance: bool = True

    multirocket_num_kernels: int = 6_250
    multirocket_max_dilations_per_kernel: int = 32
    multirocket_num_features_per_kernel: int = 4
    multirocket_normalise_per_instance: bool = False
    hydra_num_kernels: int = 8
    hydra_num_groups: int = 64
    hydra_max_num_channels: int = 8
    reduced_feature_count: int = 10_000
    max_features_per_group: int = 2_000

    tabpfn_version: TabPFNVersion = "2.5"
    tabpfn_model_path: Path | None = None
    tabpfn_n_estimators: int = 8
    tabpfn_device: str = "auto"
    tabpfn_fit_mode: TabPFNFitMode = "fit_preprocessors"
    tabpfn_memory_saving_mode: Literal["auto"] | bool = "auto"
    tabpfn_inference_precision: Literal["auto", "autocast"] = "auto"
    tabpfn_show_progress_bar: bool = True
    tabpfn_balance_probabilities: bool = False
    tabpfn_preprocessing_jobs: int = 1

    transform_n_jobs: int = 4
    # Keep new fields after all artifact-v1 fields: slots dataclasses are pickled
    # positionally, so inserting one earlier corrupts old saved model configs.
    tabpfn_batch_groups: bool = False

    def validate(self) -> None:
        if self.feature_representation not in {"rocket", "multirocket_hydra"}:
            raise ValueError(
                "feature_representation must be 'rocket' or 'multirocket_hydra'"
            )
        if self.rocket_num_groups < 1:
            raise ValueError("rocket_num_groups must be at least 1")
        if self.rocket_kernels_per_group < 1:
            raise ValueError("rocket_kernels_per_group must be at least 1")
        rocket_features = 2 * self.rocket_kernels_per_group
        if self.tabpfn_version == "2.5" and rocket_features > 2_000:
            raise ValueError(
                "TabPFN v2.5 supports at most 2,000 features per group; "
                "rocket_kernels_per_group must not exceed 1,000"
            )

        if self.multirocket_num_kernels < 84:
            raise ValueError("multirocket_num_kernels must be at least 84")
        if self.multirocket_max_dilations_per_kernel < 1:
            raise ValueError(
                "multirocket_max_dilations_per_kernel must be at least 1"
            )
        if self.multirocket_num_features_per_kernel != 4:
            raise ValueError(
                "multirocket_num_features_per_kernel must be 4 for aeon 1.x"
            )
        if self.hydra_num_kernels < 2:
            raise ValueError("hydra_num_kernels must be at least 2")
        if self.hydra_num_groups < 1:
            raise ValueError("hydra_num_groups must be at least 1")
        if self.hydra_max_num_channels < 2:
            raise ValueError("hydra_max_num_channels must be at least 2")
        if self.reduced_feature_count < 1:
            raise ValueError("reduced_feature_count must be at least 1")
        if not 1 <= self.max_features_per_group <= 2_000:
            raise ValueError(
                "max_features_per_group must be between 1 and 2,000"
            )

        if self.tabpfn_version not in {"2.5", "3"}:
            raise ValueError("tabpfn_version must be '2.5' or '3'")
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
        if not isinstance(self.tabpfn_batch_groups, bool):
            raise ValueError("tabpfn_batch_groups must be a boolean")
        if self.transform_n_jobs == 0:
            raise ValueError("transform_n_jobs cannot be zero")


@dataclass(frozen=True, slots=True)
class ExperimentConfig:
    """Complete configuration for one saved RocketPFN dataset fold."""

    data: DataConfig
    output_dir: Path
    run_name: str
    model: RocketPFNConfig = field(default_factory=RocketPFNConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    # Optional fold-matched baseline used only by the MultiRocket+HYDRA extension.
    feature_artifact_path: Path | None = None
    seed: int = 42

    def validate(self) -> None:
        self.data.validate()
        self.model.validate()
        self.evaluation.validate()
        if not self.run_name.strip():
            raise ValueError("run_name cannot be empty")
        if (
            self.feature_artifact_path is not None
            and self.model.feature_representation != "multirocket_hydra"
        ):
            raise ValueError(
                "feature_artifact_path is only valid for multirocket_hydra"
            )


__all__ = [
    "DataConfig",
    "EvaluationConfig",
    "ExperimentConfig",
    "FeatureRepresentation",
    "RocketPFNConfig",
    "TabPFNFitMode",
    "TabPFNVersion",
]
