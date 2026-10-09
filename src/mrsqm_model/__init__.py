"""Public API for MrSQM experiments."""

from .aggregation import (
    SCORE_METRICS,
    CrossValidationArtifacts,
    aggregate_cross_validation_results,
)
from .config import (
    DataConfig,
    EvaluationConfig,
    ExperimentConfig,
    MrSQMConfig,
)
from .data import (
    DataBundle,
    TimeSeriesSplit,
    channel_names,
    prepare_data,
    window_to_array,
)
from .model import MultivariateMrSQMClassifier, build_classifier
from .training import (
    TrainingResult,
    load_trained_model,
    load_training_result,
    train_mrsqm,
)

__all__ = [
    "SCORE_METRICS",
    "CrossValidationArtifacts",
    "DataBundle",
    "DataConfig",
    "EvaluationConfig",
    "ExperimentConfig",
    "MrSQMConfig",
    "MultivariateMrSQMClassifier",
    "TimeSeriesSplit",
    "TrainingResult",
    "aggregate_cross_validation_results",
    "build_classifier",
    "channel_names",
    "load_trained_model",
    "load_training_result",
    "prepare_data",
    "train_mrsqm",
    "window_to_array",
]
