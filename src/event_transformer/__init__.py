"""Public API for the modality-aware event Transformer."""

from .aggregation import (
    SCORE_METRICS,
    CrossValidationArtifacts,
    aggregate_cross_validation_results,
)
from .config import (
    DataConfig,
    ExperimentConfig,
    ModelConfig,
    PretrainingConfig,
    TrainingConfig,
)
from .data import EventBatchCollator, EventSequenceEncoder, EventWindowDataset, prepare_data
from .features import MODALITY_DIMS, MODALITY_NAMES, EventFeatureExtractor
from .model import EventTransformer
from .pretraining import PretrainingOutput, pretrain_event_transformer
from .training import TrainingResult, load_trained_model, train_event_transformer

__all__ = [
    "DataConfig",
    "CrossValidationArtifacts",
    "EventBatchCollator",
    "EventFeatureExtractor",
    "EventSequenceEncoder",
    "EventTransformer",
    "EventWindowDataset",
    "ExperimentConfig",
    "MODALITY_DIMS",
    "MODALITY_NAMES",
    "ModelConfig",
    "PretrainingConfig",
    "PretrainingOutput",
    "SCORE_METRICS",
    "TrainingConfig",
    "TrainingResult",
    "aggregate_cross_validation_results",
    "load_trained_model",
    "prepare_data",
    "pretrain_event_transformer",
    "train_event_transformer",
]
