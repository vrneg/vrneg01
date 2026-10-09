"""Public API for HIVE-COTE 2.0 experiments."""

from .aggregation import (
    SCORE_METRICS,
    CrossValidationArtifacts,
    aggregate_cross_validation_results,
)
from .config import (
    ArsenalComponentConfig,
    DataConfig,
    DrCIFComponentConfig,
    EvaluationConfig,
    ExperimentConfig,
    HIVECOTEV2Config,
    STCComponentConfig,
    TDEComponentConfig,
)
from .data import (
    DataBundle,
    TimeSeriesSplit,
    channel_names,
    prepare_data,
    window_to_array,
)
from .estimator import ResourceAwareHIVECOTEV2
from .orchestration import resolve_fold_worker_count
from .training import (
    EnsemblePredictionOutput,
    PredictionOutput,
    TrainingResult,
    build_classifier,
    load_trained_model,
    load_training_result,
    train_hivecote_v2,
)

__all__ = [
    "SCORE_METRICS",
    "ArsenalComponentConfig",
    "CrossValidationArtifacts",
    "DataBundle",
    "DataConfig",
    "DrCIFComponentConfig",
    "EnsemblePredictionOutput",
    "EvaluationConfig",
    "ExperimentConfig",
    "HIVECOTEV2Config",
    "PredictionOutput",
    "ResourceAwareHIVECOTEV2",
    "STCComponentConfig",
    "TDEComponentConfig",
    "TimeSeriesSplit",
    "TrainingResult",
    "aggregate_cross_validation_results",
    "build_classifier",
    "channel_names",
    "load_trained_model",
    "load_training_result",
    "prepare_data",
    "resolve_fold_worker_count",
    "train_hivecote_v2",
    "window_to_array",
]
