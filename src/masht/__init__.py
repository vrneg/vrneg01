"""Public API for MASHT with explicitly versioned TabPFN-3 inference."""

from .aggregation import (
    SCORE_METRICS,
    CrossValidationArtifacts,
    aggregate_cross_validation_results,
)
from .config import (
    DataConfig,
    EvaluationConfig,
    ExperimentConfig,
    FeatureBudgetScope,
    MASHTConfig,
    TabPFNFitMode,
)
from .features import (
    FeaturePlan,
    FittedMASHT,
    MASHTFeatureTransformer,
    TabPFNAccessError,
    adaptive_feature_budget,
    resolve_feature_plan,
)
from .training import (
    ARTIFACT_VERSION,
    TrainingResult,
    load_trained_model,
    load_training_result,
    train_masht,
)

__all__ = [
    "ARTIFACT_VERSION",
    "SCORE_METRICS",
    "CrossValidationArtifacts",
    "DataConfig",
    "EvaluationConfig",
    "ExperimentConfig",
    "FeatureBudgetScope",
    "FeaturePlan",
    "FittedMASHT",
    "MASHTConfig",
    "MASHTFeatureTransformer",
    "TabPFNAccessError",
    "TabPFNFitMode",
    "TrainingResult",
    "adaptive_feature_budget",
    "aggregate_cross_validation_results",
    "load_trained_model",
    "load_training_result",
    "resolve_feature_plan",
    "train_masht",
]
