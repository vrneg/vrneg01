"""Configuration for sparse MultiRocket-HYDRA experiments."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, TypeAlias

try:
    from minirocket.config import DataConfig, EvaluationConfig
except ModuleNotFoundError as error:
    if error.name != "minirocket":
        raise
    from ..minirocket.config import DataConfig, EvaluationConfig


ClassifierName: TypeAlias = Literal[
    "ridge",
    "logistic_l2",
    "elastic_net",
    "linear_svm",
]
InnerScoring: TypeAlias = Literal[
    "accuracy",
    "balanced_accuracy",
    "macro_f1",
    "roc_auc",
    "average_precision",
]

ALL_CLASSIFIERS: tuple[ClassifierName, ...] = (
    "ridge",
    "logistic_l2",
    "elastic_net",
    "linear_svm",
)


@dataclass(frozen=True, slots=True)
class SparseMultiRocketHydraConfig:
    """Transform, sparse-selection, and inner model-search settings."""

    num_kernels: int = 6_250
    max_dilations_per_kernel: int = 32
    num_features_per_kernel: int = 4
    normalise_per_instance: bool = False
    hydra_num_kernels: int = 8
    hydra_num_groups: int = 64
    hydra_max_num_channels: int = 8

    # The selector is fitted separately in every inner-CV training split.
    feature_counts: tuple[int, ...] = (5_000, 10_000, 15_000)
    classifier_candidates: tuple[ClassifierName, ...] = ALL_CLASSIFIERS
    ridge_alphas: tuple[float, ...] = (1.0, 10.0, 100.0, 1_000.0, 10_000.0)
    logistic_l2_cs: tuple[float, ...] = (1e-3, 1e-2, 1e-1, 1.0, 10.0)
    # Elastic-net logistic regression uses the stochastic log-loss solver. Its
    # alpha is the direct L1/L2 penalty strength (larger means more regularized).
    elastic_net_alphas: tuple[float, ...] = (1e-3, 1e-2, 1e-1)
    elastic_net_l1_ratios: tuple[float, ...] = (0.2, 0.5, 0.8)
    linear_svm_cs: tuple[float, ...] = (1e-3, 1e-2, 1e-1, 1.0)
    inner_cv_folds: int = 3
    inner_scoring: InnerScoring = "balanced_accuracy"
    class_weight: Literal["balanced"] | None = None
    max_iter: int = 2_000
    tolerance: float = 1e-3

    # Transform threads and inner-search workers are separate to make memory and
    # oversubscription explicit. Pre-dispatch bounds concurrent p >> n fits.
    n_jobs: int = 1
    search_n_jobs: int = 3
    search_pre_dispatch: int = 3
    search_verbose: int = 2

    def validate(self) -> None:
        if self.num_kernels < 84:
            raise ValueError("num_kernels must be at least 84")
        if self.max_dilations_per_kernel < 1:
            raise ValueError("max_dilations_per_kernel must be at least 1")
        if self.num_features_per_kernel != 4:
            raise ValueError("num_features_per_kernel must be 4 for aeon 1.x")
        if self.hydra_num_kernels < 2:
            raise ValueError("hydra_num_kernels must be at least 2")
        if self.hydra_num_groups < 1:
            raise ValueError("hydra_num_groups must be at least 1")
        if self.hydra_max_num_channels < 2:
            raise ValueError("hydra_max_num_channels must be at least 2")
        if not self.feature_counts or any(count < 1 for count in self.feature_counts):
            raise ValueError("feature_counts must contain only positive values")
        if len(set(self.feature_counts)) != len(self.feature_counts):
            raise ValueError("feature_counts must not contain duplicates")
        if not self.classifier_candidates:
            raise ValueError("classifier_candidates cannot be empty")
        if len(set(self.classifier_candidates)) != len(self.classifier_candidates):
            raise ValueError("classifier_candidates must not contain duplicates")
        unknown = set(self.classifier_candidates).difference(ALL_CLASSIFIERS)
        if unknown:
            raise ValueError(f"Unknown classifier candidates: {sorted(unknown)}")
        if "ridge" in self.classifier_candidates:
            self._validate_positive_grid("ridge_alphas", self.ridge_alphas)
        if "logistic_l2" in self.classifier_candidates:
            self._validate_positive_grid("logistic_l2_cs", self.logistic_l2_cs)
        if "elastic_net" in self.classifier_candidates:
            self._validate_positive_grid(
                "elastic_net_alphas", self.elastic_net_alphas
            )
            if not self.elastic_net_l1_ratios or any(
                not 0.0 < ratio < 1.0 for ratio in self.elastic_net_l1_ratios
            ):
                raise ValueError(
                    "elastic_net_l1_ratios must be strictly between zero and one"
                )
        if "linear_svm" in self.classifier_candidates:
            self._validate_positive_grid("linear_svm_cs", self.linear_svm_cs)
        if self.inner_cv_folds < 2:
            raise ValueError("inner_cv_folds must be at least 2")
        if self.max_iter < 1:
            raise ValueError("max_iter must be positive")
        if self.tolerance <= 0.0:
            raise ValueError("tolerance must be positive")
        if self.n_jobs == 0 or self.search_n_jobs == 0:
            raise ValueError("n_jobs and search_n_jobs cannot be zero")
        if self.search_pre_dispatch < 1:
            raise ValueError("search_pre_dispatch must be positive")
        if self.search_verbose < 0:
            raise ValueError("search_verbose cannot be negative")

    @staticmethod
    def _validate_positive_grid(name: str, values: tuple[float, ...]) -> None:
        if not values or any(value <= 0.0 for value in values):
            raise ValueError(f"{name} must contain only positive values")


@dataclass(frozen=True, slots=True)
class ExperimentConfig:
    """Complete configuration for one sparse MultiRocket-HYDRA fold."""

    data: DataConfig
    output_dir: Path
    run_name: str
    model: SparseMultiRocketHydraConfig = field(
        default_factory=SparseMultiRocketHydraConfig
    )
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    # Optional fold-matched artifact from the existing MultiRocket+HYDRA run.
    # Reusing its fitted random transforms avoids paying that cost a second time.
    feature_artifact_path: Path | None = None
    seed: int = 42

    def validate(self) -> None:
        self.data.validate()
        self.model.validate()
        self.evaluation.validate()
        if not self.run_name.strip():
            raise ValueError("run_name cannot be empty")


__all__ = [
    "ALL_CLASSIFIERS",
    "ClassifierName",
    "DataConfig",
    "EvaluationConfig",
    "ExperimentConfig",
    "InnerScoring",
    "SparseMultiRocketHydraConfig",
]
