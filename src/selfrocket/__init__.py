"""Public API for multivariate SelF-Rocket experiments."""

from .aggregation import (
    SCORE_METRICS,
    CrossValidationArtifacts,
    aggregate_cross_validation_results,
)
from .config import (
    EXTENDED_RIDGE_ALPHAS,
    SELECTION_RIDGE_ALPHAS,
    DataConfig,
    EvaluationConfig,
    ExperimentConfig,
    SelFRocketConfig,
)
from .data import DataBundle, TimeSeriesSplit, channel_names, prepare_data, window_to_array
from .features import (
    CANDIDATE_NAMES,
    POOLING_OPERATORS,
    SelFRocketFeatures,
    candidate_names,
    highest_median_choice,
    vote_validated_choice,
)
from .training import (
    TrainingResult,
    load_trained_model,
    load_training_result,
    train_selfrocket,
)

__all__ = [
    "CANDIDATE_NAMES",
    "POOLING_OPERATORS",
    "SCORE_METRICS",
    "CrossValidationArtifacts",
    "DataBundle",
    "DataConfig",
    "EvaluationConfig",
    "ExperimentConfig",
    "EXTENDED_RIDGE_ALPHAS",
    "SELECTION_RIDGE_ALPHAS",
    "SelFRocketConfig",
    "SelFRocketFeatures",
    "TimeSeriesSplit",
    "TrainingResult",
    "aggregate_cross_validation_results",
    "candidate_names",
    "channel_names",
    "highest_median_choice",
    "load_trained_model",
    "load_training_result",
    "prepare_data",
    "train_selfrocket",
    "vote_validated_choice",
    "window_to_array",
]
