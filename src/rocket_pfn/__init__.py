"""Public API for RocketPFN and the MultiRocket+HYDRA extension."""

from .aggregation import (
    SCORE_METRICS,
    CrossValidationArtifacts,
    aggregate_cross_validation_results,
)
from .config import (
    DataConfig,
    EvaluationConfig,
    ExperimentConfig,
    FeatureRepresentation,
    RocketPFNConfig,
    TabPFNFitMode,
    TabPFNVersion,
)
from .features import FittedRocketPFN, TabPFNAccessError, ranked_feature_groups
from .training import (
    ARTIFACT_VERSION,
    TrainingResult,
    load_trained_model,
    load_training_result,
    train_rocket_pfn,
)

__all__ = [
    "ARTIFACT_VERSION",
    "SCORE_METRICS",
    "CrossValidationArtifacts",
    "DataConfig",
    "EvaluationConfig",
    "ExperimentConfig",
    "FeatureRepresentation",
    "FittedRocketPFN",
    "RocketPFNConfig",
    "TabPFNFitMode",
    "TabPFNAccessError",
    "TabPFNVersion",
    "TrainingResult",
    "aggregate_cross_validation_results",
    "load_trained_model",
    "load_training_result",
    "ranked_feature_groups",
    "train_rocket_pfn",
]
