"""Public API for modality-aware Inception-TCN experiments."""

from .aggregation import (
    SCORE_METRICS,
    CrossValidationArtifacts,
    aggregate_cross_validation_results,
)
from .config import DataConfig, ExperimentConfig, ModelConfig, TrainingConfig
from .data import (
    DataBundle,
    FixedGridTorchDataset,
    TimeSeriesSplit,
    channel_names,
    prepare_data,
    window_to_array,
)
from .model import (
    ModalityAwareInceptionTCN,
    ModalityChannelSpec,
    build_modality_channel_layout,
)
from .training import (
    EvaluationOutput,
    TrainingResult,
    evaluate_with_predictions,
    load_trained_model,
    load_training_result,
    train_inception_tcn,
)

__all__ = [
    "SCORE_METRICS",
    "CrossValidationArtifacts",
    "DataBundle",
    "DataConfig",
    "EvaluationOutput",
    "ExperimentConfig",
    "FixedGridTorchDataset",
    "ModelConfig",
    "ModalityAwareInceptionTCN",
    "ModalityChannelSpec",
    "TimeSeriesSplit",
    "TrainingConfig",
    "TrainingResult",
    "aggregate_cross_validation_results",
    "build_modality_channel_layout",
    "channel_names",
    "evaluate_with_predictions",
    "load_trained_model",
    "load_training_result",
    "prepare_data",
    "train_inception_tcn",
    "window_to_array",
]
