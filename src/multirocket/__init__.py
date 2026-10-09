"""Public API for MultiRocket and optional Hydra experiments."""

from .aggregation import (
    SCORE_METRICS,
    CrossValidationArtifacts,
    aggregate_cross_validation_results,
)
from .config import (
    EXTENDED_RIDGE_ALPHAS,
    DataConfig,
    EvaluationConfig,
    ExperimentConfig,
    MultiRocketConfig,
)
from .data import DataBundle, TimeSeriesSplit, channel_names, prepare_data, window_to_array
from .training import (
    MultiRocketHydraFeatures,
    TrainingResult,
    load_trained_model,
    load_training_result,
    train_multirocket,
)

__all__ = [
    "SCORE_METRICS",
    "CrossValidationArtifacts",
    "DataBundle",
    "DataConfig",
    "EvaluationConfig",
    "ExperimentConfig",
    "EXTENDED_RIDGE_ALPHAS",
    "MultiRocketConfig",
    "MultiRocketHydraFeatures",
    "TimeSeriesSplit",
    "TrainingResult",
    "aggregate_cross_validation_results",
    "channel_names",
    "load_trained_model",
    "load_training_result",
    "prepare_data",
    "train_multirocket",
    "window_to_array",
]
