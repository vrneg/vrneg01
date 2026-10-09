"""Public API for sparse MultiRocket-HYDRA experiments."""

from .aggregation import (
    SCORE_METRICS,
    CrossValidationArtifacts,
    aggregate_cross_validation_results,
)
from .config import (
    ALL_CLASSIFIERS,
    ClassifierName,
    DataConfig,
    EvaluationConfig,
    ExperimentConfig,
    InnerScoring,
    SparseMultiRocketHydraConfig,
)
from .data import DataBundle, TimeSeriesSplit, channel_names, prepare_data, window_to_array
from .features import MultiRocketHydraRawFeatures, MultiRocketHydraStandardizer
from .training import (
    FittedSparseMultiRocketHydra,
    TrainingResult,
    load_trained_model,
    load_training_result,
    train_sparse_multirocket_hydra,
)

__all__ = [
    "ALL_CLASSIFIERS",
    "SCORE_METRICS",
    "ClassifierName",
    "CrossValidationArtifacts",
    "DataBundle",
    "DataConfig",
    "EvaluationConfig",
    "ExperimentConfig",
    "FittedSparseMultiRocketHydra",
    "InnerScoring",
    "MultiRocketHydraRawFeatures",
    "MultiRocketHydraStandardizer",
    "SparseMultiRocketHydraConfig",
    "TimeSeriesSplit",
    "TrainingResult",
    "aggregate_cross_validation_results",
    "channel_names",
    "load_trained_model",
    "load_training_result",
    "prepare_data",
    "train_sparse_multirocket_hydra",
    "window_to_array",
]
