"""Public API for WEASEL 2.0 experiments."""

from .aggregation import (
    SCORE_METRICS,
    CrossValidationArtifacts,
    aggregate_cross_validation_results,
)
from .config import (
    WEASEL_RIDGE_ALPHAS,
    DataConfig,
    EvaluationConfig,
    ExperimentConfig,
    WeaselV2Config,
)
from .data import (
    DataBundle,
    TimeSeriesSplit,
    channel_names,
    prepare_data,
    window_to_array,
)
from .features import MultivariateWEASELTransformerV2, SupervisedChannelSelector
from .training import (
    TrainingResult,
    build_pipeline,
    load_trained_model,
    load_training_result,
    train_weasel_v2,
)

__all__ = [
    "SCORE_METRICS",
    "WEASEL_RIDGE_ALPHAS",
    "CrossValidationArtifacts",
    "DataBundle",
    "DataConfig",
    "EvaluationConfig",
    "ExperimentConfig",
    "MultivariateWEASELTransformerV2",
    "SupervisedChannelSelector",
    "TimeSeriesSplit",
    "TrainingResult",
    "WeaselV2Config",
    "aggregate_cross_validation_results",
    "build_pipeline",
    "channel_names",
    "load_trained_model",
    "load_training_result",
    "prepare_data",
    "train_weasel_v2",
    "window_to_array",
]
