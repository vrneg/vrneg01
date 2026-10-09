"""Public API for Adaptive Multi-Representation RocketPFN."""

from .aggregation import (
    SCORE_METRICS,
    CrossValidationArtifacts,
    aggregate_cross_validation_results,
)
from .config import (
    DEFAULT_VIEWS,
    AdaptiveRocketPFNConfig,
    DataConfig,
    EvaluationConfig,
    ExperimentConfig,
    TabPFNFitMode,
    ViewName,
)
from .ensemble import (
    FittedAdaptiveRocketPFN,
    TabPFNAccessError,
    learn_simplex_weights,
)
from .features import (
    EXPERT_NAMES,
    AdaptiveCandidateBank,
    FeatureFamily,
)
from .selection import ExpertSelection
from .training import (
    ARTIFACT_VERSION,
    TrainingResult,
    load_trained_model,
    load_training_result,
    train_adaptive_rocket_pfn,
)

__all__ = [
    "ARTIFACT_VERSION",
    "DEFAULT_VIEWS",
    "EXPERT_NAMES",
    "SCORE_METRICS",
    "AdaptiveCandidateBank",
    "AdaptiveRocketPFNConfig",
    "CrossValidationArtifacts",
    "DataConfig",
    "EvaluationConfig",
    "ExperimentConfig",
    "ExpertSelection",
    "FeatureFamily",
    "FittedAdaptiveRocketPFN",
    "TabPFNAccessError",
    "TabPFNFitMode",
    "TrainingResult",
    "ViewName",
    "aggregate_cross_validation_results",
    "learn_simplex_weights",
    "load_trained_model",
    "load_training_result",
    "train_adaptive_rocket_pfn",
]
