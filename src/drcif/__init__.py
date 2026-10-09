"""Public API for Diverse Representation CIF experiments."""

from .aggregation import (
    SCORE_METRICS,
    CrossValidationArtifacts,
    aggregate_cross_validation_results,
)
from .config import DataConfig, DrCIFConfig, EvaluationConfig, ExperimentConfig
from .data import (
    DataBundle,
    TimeSeriesSplit,
    channel_names,
    prepare_data,
    window_to_array,
)
from .training import (
    PredictionOutput,
    TrainingResult,
    build_classifier,
    load_trained_model,
    load_training_result,
    train_drcif,
)

__all__ = [
    "SCORE_METRICS",
    "CrossValidationArtifacts",
    "DataBundle",
    "DataConfig",
    "DrCIFConfig",
    "EvaluationConfig",
    "ExperimentConfig",
    "PredictionOutput",
    "TimeSeriesSplit",
    "TrainingResult",
    "aggregate_cross_validation_results",
    "build_classifier",
    "channel_names",
    "load_trained_model",
    "load_training_result",
    "prepare_data",
    "train_drcif",
    "window_to_array",
]
