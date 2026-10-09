"""Public API for compact cue-aware fusion-TCN experiments."""

from .aggregation import (
    SCORE_METRICS,
    CrossValidationArtifacts,
    MultiSeedArtifacts,
    aggregate_cross_validation_results,
    aggregate_multi_seed_results,
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
    CompactFusionTCN,
    ModalityChannelSpec,
    ResidualTemporalBlock,
    build_modality_channel_layout,
)
from .training import (
    EvaluationOutput,
    TrainingResult,
    evaluate_with_predictions,
    load_trained_model,
    load_training_result,
    train_compact_fusion_tcn,
)

__all__ = [
    "SCORE_METRICS",
    "CompactFusionTCN",
    "CrossValidationArtifacts",
    "DataBundle",
    "DataConfig",
    "EvaluationOutput",
    "ExperimentConfig",
    "FixedGridTorchDataset",
    "ModelConfig",
    "ModalityChannelSpec",
    "MultiSeedArtifacts",
    "ResidualTemporalBlock",
    "TimeSeriesSplit",
    "TrainingConfig",
    "TrainingResult",
    "aggregate_cross_validation_results",
    "aggregate_multi_seed_results",
    "build_modality_channel_layout",
    "channel_names",
    "evaluate_with_predictions",
    "load_trained_model",
    "load_training_result",
    "prepare_data",
    "train_compact_fusion_tcn",
    "window_to_array",
]
