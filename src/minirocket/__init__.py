"""Public API for fixed-grid multivariate MiniRocket experiments."""

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
    MiniRocketConfig,
)
from .data import (
    DataBundle,
    TimeSeriesSplit,
    channel_names,
    collapse_near_constant_traces,
    prepare_data,
    prepare_fixed_grid_split,
    window_to_array,
)
from .training import (
    TrainingResult,
    load_trained_model,
    load_training_result,
    train_minirocket,
)

__all__ = [
    "SCORE_METRICS",
    "CrossValidationArtifacts",
    "DataBundle",
    "DataConfig",
    "EvaluationConfig",
    "ExperimentConfig",
    "EXTENDED_RIDGE_ALPHAS",
    "MiniRocketConfig",
    "TimeSeriesSplit",
    "TrainingResult",
    "aggregate_cross_validation_results",
    "channel_names",
    "collapse_near_constant_traces",
    "load_trained_model",
    "load_training_result",
    "prepare_data",
    "prepare_fixed_grid_split",
    "train_minirocket",
    "window_to_array",
]
