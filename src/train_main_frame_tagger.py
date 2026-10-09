"""Editable entry point for frame-level negation tagging experiments.

Edit the configuration block, set ``RUN_MODE``, then run::

    python src/train_main_frame_tagger.py

This model answers a different question from the window classifiers in
``train_main_t2m_gpt.py`` and ``train_main_motion_gpt.py``: instead of "does this window
contain a negation cue", it labels every frame ``O``/``B-CUE``/``I-CUE``/``B-SCOPE``/
``I-SCOPE``. It reads the same folds and the same fixed grid, and it also writes a
window-level score, so its results can sit in the same comparison table.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal

from frame_tagger import (
    LabelConfig,
    TaggerConfig,
    TaggerExperimentConfig,
    TaggerTrainingConfig,
    TaggerTrainingResult,
    train_frame_tagger,
)
from representation import RepresentationConfig
from t2m_gpt import DataConfig

# -----------------------------------------------------------------------------
# Experiment selection
# -----------------------------------------------------------------------------

RUN_MODE: Literal["single", "cross_validation"] = "single"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_NAMESPACE = "VR-Faces-Neg"
DATASET_PREFIX = "target-cue_window-500_splits-10"
OUTPUT_ROOT = PROJECT_ROOT / "outputs/frame_tagger"
EXPERIMENT_VARIANT = "bilstm-crf"

SINGLE_FOLD = 0
CROSS_VALIDATION_FOLDS: tuple[int, ...] = tuple(range(10))
SEED = 42
DEVICE = "auto"


# -----------------------------------------------------------------------------
# Fixed-grid representation
#
# Identical to the window-level experiments so a tagger result is comparable. Switch
# REPRESENTATION to a non-identity configuration to compare data representations; the
# same object is accepted by the T2M-GPT and MotionGPT controllers.
# -----------------------------------------------------------------------------

NUM_TIME_POINTS = 32
WINDOW_START_SECONDS = -0.5
WINDOW_END_SECONDS = 0.5
MAX_EVENTS_PER_MODALITY: int | None = 128
NORMALIZE_FEATURES = True
CACHE_DATASET_IN_MEMORY = True

ACTOR_SCOPE: Literal["anchor", "other", "all"] = "anchor"
INCLUDE_PRESENCE_CHANNELS = True
MODALITIES: tuple[str, ...] | None = None
POSITIVE_LABEL = "neg"
NEGATIVE_LABEL = "none"

REPRESENTATION = RepresentationConfig()


# -----------------------------------------------------------------------------
# Frame targets
#
# Token onsets are exact; most token offsets are assumed. Leaving
# assumed_token_duration_ms at None estimates the median anchor-word duration from the
# training split of each fold, which is the least arbitrary available choice.
# -----------------------------------------------------------------------------

LABEL_CONFIG = LabelConfig(assumed_token_duration_ms=None)


# -----------------------------------------------------------------------------
# Encoder and CRF
# -----------------------------------------------------------------------------

TAGGER_CONFIG = TaggerConfig(
    encoder="bilstm",
    d_model=96,
    num_layers=2,
    nhead=4,
    dim_feedforward=256,
    dropout=0.3,
    constrain_transitions=True,
)

TRAINING_CONFIG = TaggerTrainingConfig(
    max_epochs=80,
    batch_size=32,
    evaluation_batch_size=64,
    learning_rate=1e-3,
    weight_decay=1e-2,
    gradient_clip_norm=1.0,
    early_stopping_patience=12,
    lr_scheduler_factor=0.5,
    lr_scheduler_patience=5,
    minimum_span_overlap=0.5,
    selection_metric="span_f1",
    show_progress=True,
)


@dataclass(slots=True)
class CrossValidationRun:
    fold_results: dict[int, TaggerTrainingResult]


def experiment_output_dir() -> Path:
    return OUTPUT_ROOT / DATASET_PREFIX / EXPERIMENT_VARIANT


def dataset_source(fold: int) -> str:
    return f"{DATASET_NAMESPACE}/{DATASET_PREFIX}_fold-{fold}"


def build_experiment_config(
    fold: int,
    *,
    show_progress: bool = TRAINING_CONFIG.show_progress,
) -> TaggerExperimentConfig:
    return TaggerExperimentConfig(
        data=DataConfig(
            dataset=dataset_source(fold),
            num_time_points=NUM_TIME_POINTS,
            window_start_seconds=WINDOW_START_SECONDS,
            window_end_seconds=WINDOW_END_SECONDS,
            max_events_per_modality=MAX_EVENTS_PER_MODALITY,
            normalize_features=NORMALIZE_FEATURES,
            cache_in_memory=CACHE_DATASET_IN_MEMORY,
            actor_scope=ACTOR_SCOPE,
            include_presence_channels=INCLUDE_PRESENCE_CHANNELS,
            modalities=MODALITIES,
            positive_label=POSITIVE_LABEL,
            negative_label=NEGATIVE_LABEL,
            representation=REPRESENTATION,
        ),
        output_dir=str(experiment_output_dir()),
        run_name=f"fold-{fold}_seed-{SEED}",
        labels=LABEL_CONFIG,
        tagger=TAGGER_CONFIG,
        training=replace(TRAINING_CONFIG, show_progress=show_progress),
        seed=SEED,
        device=DEVICE,
    )


def train_single_model(fold: int = SINGLE_FOLD) -> TaggerTrainingResult:
    return train_frame_tagger(build_experiment_config(fold))


def train_cross_validation(
    folds: Sequence[int] = CROSS_VALIDATION_FOLDS,
) -> CrossValidationRun:
    """Train every fold sequentially and print a per-fold summary."""

    fold_results: dict[int, TaggerTrainingResult] = {}
    for fold in folds:
        print(f"Training frame tagger fold {fold} ...", flush=True)
        result = train_frame_tagger(build_experiment_config(fold))
        fold_results[fold] = result
        print(
            f"Completed fold {fold}: best_epoch={result.best_epoch}, "
            f"span_f1={result.test_metrics['span_overlap']['micro']['f1']:.4f}, "
            f"window_auc={result.test_metrics.get('window_roc_auc', float('nan')):.4f}",
            flush=True,
        )
    return CrossValidationRun(fold_results=fold_results)


def main() -> TaggerTrainingResult | CrossValidationRun:
    if RUN_MODE == "single":
        result = train_single_model()
        print(f"Best epoch: {result.best_epoch}")
        print(f"Model parameters: {result.model_parameter_count}")
        print(f"Label statistics: {result.label_stats['train']}")
        print(f"Validation metrics: {result.validation_metrics}")
        print(f"Test metrics: {result.test_metrics}")
        return result
    if RUN_MODE == "cross_validation":
        return train_cross_validation()
    raise ValueError(f"Unknown RUN_MODE: {RUN_MODE!r}")


if __name__ == "__main__":
    main()
