"""Public API for CASTOR time-series classification experiments."""

from .aggregation import (
    SCORE_METRICS,
    CrossValidationArtifacts,
    aggregate_cross_validation_results,
)
from .config import (
    CASTOR_RIDGE_ALPHAS,
    CastorConfig,
    DataConfig,
    EvaluationConfig,
    ExperimentConfig,
)
from .data import (
    DataBundle,
    TimeSeriesSplit,
    channel_names,
    prepare_data,
    window_to_array,
)
from .features import (
    CastorSparseScaler,
    FirstDifferenceTransformer,
    build_feature_transformer,
)
from .training import (
    TrainingResult,
    build_pipeline,
    load_trained_model,
    load_training_result,
    train_castor,
)

__all__ = [
    "CASTOR_RIDGE_ALPHAS",
    "SCORE_METRICS",
    "CastorConfig",
    "CastorSparseScaler",
    "CrossValidationArtifacts",
    "DataBundle",
    "DataConfig",
    "EvaluationConfig",
    "ExperimentConfig",
    "FirstDifferenceTransformer",
    "TimeSeriesSplit",
    "TrainingResult",
    "aggregate_cross_validation_results",
    "build_feature_transformer",
    "build_pipeline",
    "channel_names",
    "load_trained_model",
    "load_training_result",
    "prepare_data",
    "train_castor",
    "window_to_array",
]
