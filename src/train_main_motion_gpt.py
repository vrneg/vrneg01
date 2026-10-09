"""Editable entry point for MotionGPT motion-language experiments.

Edit the configuration block, set ``RUN_MODE``, then run::

    python src/train_main_motion_gpt.py
"""

from __future__ import annotations

import multiprocessing
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal

import torch

from motion_gpt import (
    SCORE_METRICS,
    CrossValidationArtifacts,
    DataConfig,
    ExperimentConfig,
    InstructionPretrainingConfig,
    MotionLanguageConfig,
    TrainingConfig,
    TrainingResult,
    VQVAEConfig,
    VQVAETrainingConfig,
    aggregate_cross_validation_results,
    load_training_result,
    train_motion_gpt,
)


# -----------------------------------------------------------------------------
# Experiment selection
# -----------------------------------------------------------------------------

RUN_MODE: Literal["single", "cross_validation"] = "cross_validation"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
# Folds are read with the datasets loader. Point DATASET_NAMESPACE at the Hub
# organization, or set it to a local directory containing "<PREFIX>_fold-<N>".
DATASET_NAMESPACE = "VR-Faces-Neg"
DATASET_PREFIX = "target-cue_window-500_splits-10"
OUTPUT_ROOT = PROJECT_ROOT / "outputs/motion_gpt"
EXPERIMENT_VARIANT = "motion-to-text"

SINGLE_FOLD = 0
CROSS_VALIDATION_FOLDS: tuple[int, ...] = tuple(range(10))
SEED = 42
DEVICE = "auto"


# -----------------------------------------------------------------------------
# Fixed-grid representation (identical to the ROCKET, TCN, and T2M-GPT experiments)
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


# -----------------------------------------------------------------------------
# Phase one: motion tokenizer
#
# Deliberately identical to src/train_main_t2m_gpt.py so both models read the same
# discrete tokens and the comparison isolates the stage-two architecture.
# -----------------------------------------------------------------------------

VQVAE_CONFIG = VQVAEConfig(
    num_codes=128,
    code_dim=64,
    width=128,
    num_downsample_layers=2,
    num_residual_blocks=2,
    dilation_growth_rate=3,
    dropout=0.0,
    codebook_decay=0.99,
    code_reset_threshold=1.0,
)

VQVAE_TRAINING_CONFIG = VQVAETrainingConfig(
    max_epochs=100,
    batch_size=32,
    evaluation_batch_size=64,
    learning_rate=2e-4,
    reconstruction_loss="smooth_l1",
    velocity_loss_weight=0.5,
    commitment_loss_weight=0.02,
    early_stopping_patience=12,
    lr_scheduler_patience=6,
)


# -----------------------------------------------------------------------------
# Phases two and three: encoder-decoder over motion tokens
#
# These are smaller than the published MotionGPT and than the dataclass defaults.
# A paper-shaped T2M-GPT stage two (roughly 700K parameters) reached its best epoch
# on the first pass over the 741 training windows of this dataset and then overfit
# monotonically; see src/t2m_gpt/README.md for the measured run. This scale, by
# contrast, generalized (best epoch 5-10 rather than 1). Scale back up when
# training on more windows, longer windows, or the cueandscope target.
# -----------------------------------------------------------------------------

MODEL_CONFIG = MotionLanguageConfig(
    head="motion_to_text",
    d_model=64,
    nhead=4,
    num_encoder_layers=2,
    num_decoder_layers=2,
    dim_feedforward=256,
    dropout=0.3,
    classifier_hidden_dim=32,
    tie_word_embeddings=True,
    max_sequence_length=64,
    num_sentinels=8,
)

USE_PRETRAINING = True

PRETRAINING_CONFIG = InstructionPretrainingConfig(
    num_epochs=25,
    batch_size=32,
    learning_rate=3e-4,
    weight_decay=1e-2,
    gradient_clip_norm=1.0,
    tasks=("denoise", "predict", "inbetween"),
    span_corruption_rate=0.25,
    mean_span_length=2.0,
    prediction_context_fraction=0.5,
    inbetween_span_fraction=0.25,
)

TRAINING_CONFIG = TrainingConfig(
    max_epochs=60,
    batch_size=32,
    evaluation_batch_size=64,
    num_workers=0,
    learning_rate=3e-4,
    classifier_learning_rate=None,
    freeze_backbone_epochs=0,
    weight_decay=1e-2,
    gradient_clip_norm=1.0,
    early_stopping_patience=10,
    selection_metric="loss",
    positive_class_weight=None,
    threshold=0.5,
    # The motion-to-text score is a log-odds between two answer tokens rather than a
    # calibrated probability, so a validation-fitted threshold is usually worth it.
    calibrate_threshold_on_validation=True,
    threshold_metric="macro_f1",
    lr_scheduler_factor=0.5,
    lr_scheduler_patience=4,
    minimum_learning_rate=1e-6,
    mixed_precision=False,
    deterministic_algorithms=False,
    show_progress=True,
)


# -----------------------------------------------------------------------------
# Fold-level execution
# -----------------------------------------------------------------------------

USE_PROCESS_POOL = False
MAX_PARALLEL_FOLDS = 1
CPU_THREADS_PER_FOLD = 4
RESUME_COMPLETED_FOLDS = True


@dataclass(frozen=True, slots=True)
class FoldJob:
    fold: int
    config: ExperimentConfig


@dataclass(slots=True)
class CrossValidationRun:
    fold_results: dict[int, TrainingResult]
    artifacts: CrossValidationArtifacts


def experiment_output_dir() -> Path:
    return OUTPUT_ROOT / DATASET_PREFIX / EXPERIMENT_VARIANT


def dataset_source(fold: int) -> str:
    return f"{DATASET_NAMESPACE}/{DATASET_PREFIX}_fold-{fold}"


def build_experiment_config(
    fold: int,
    *,
    show_progress: bool = TRAINING_CONFIG.show_progress,
) -> ExperimentConfig:
    return ExperimentConfig(
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
        ),
        output_dir=str(experiment_output_dir()),
        run_name=f"fold-{fold}_seed-{SEED}",
        vqvae=VQVAE_CONFIG,
        vqvae_training=VQVAE_TRAINING_CONFIG,
        model=MODEL_CONFIG,
        pretraining=PRETRAINING_CONFIG,
        training=replace(TRAINING_CONFIG, show_progress=show_progress),
        use_pretraining=USE_PRETRAINING,
        seed=SEED,
        device=DEVICE,
    )


def _execute_fold(job: FoldJob) -> tuple[int, TrainingResult]:
    torch.set_num_threads(CPU_THREADS_PER_FOLD)
    return job.fold, train_motion_gpt(job.config)


def train_single_model(fold: int = SINGLE_FOLD) -> TrainingResult:
    return train_motion_gpt(build_experiment_config(fold))


def _build_fold_jobs(folds: Sequence[int], *, show_progress: bool) -> list[FoldJob]:
    if not folds:
        raise ValueError("At least one fold must be selected")
    if len(set(folds)) != len(folds):
        raise ValueError("folds must not contain duplicates")
    if CPU_THREADS_PER_FOLD < 1:
        raise ValueError("CPU_THREADS_PER_FOLD must be at least 1")
    return [
        FoldJob(fold=fold, config=build_experiment_config(fold, show_progress=show_progress))
        for fold in folds
    ]


def _completion_line(prefix: str, fold: int, result: TrainingResult) -> str:
    return (
        f"{prefix} fold {fold}: best_epoch={result.best_epoch}, "
        f"test_auc={result.test_metrics['roc_auc']:.4f}, "
        f"test_macro_f1={result.test_metrics['macro_f1']:.4f}"
    )


def _run_parallel_fold_jobs(
    jobs: Sequence[FoldJob],
    worker_count: int,
) -> dict[int, TrainingResult]:
    spawn_context = multiprocessing.get_context("spawn")
    results: dict[int, TrainingResult] = {}
    with ProcessPoolExecutor(max_workers=worker_count, mp_context=spawn_context) as executor:
        futures = {executor.submit(_execute_fold, job): job.fold for job in jobs}
        for future in as_completed(futures):
            expected_fold = futures[future]
            try:
                fold, result = future.result()
            except Exception as error:
                raise RuntimeError(
                    f"MotionGPT training failed for fold {expected_fold}"
                ) from error
            results[fold] = result
            print(_completion_line("Completed", fold, result))
    return results


def train_cross_validation(
    folds: Sequence[int] = CROSS_VALIDATION_FOLDS,
    max_parallel_folds: int = MAX_PARALLEL_FOLDS,
    use_process_pool: bool = USE_PROCESS_POOL,
    resume_completed_folds: bool = RESUME_COMPLETED_FOLDS,
) -> CrossValidationRun:
    """Train folds and produce fold-wise and pooled held-out metrics."""

    if max_parallel_folds < 1:
        raise ValueError("max_parallel_folds must be at least 1")
    jobs = _build_fold_jobs(
        tuple(folds),
        show_progress=TRAINING_CONFIG.show_progress and not use_process_pool,
    )
    fold_results: dict[int, TrainingResult] = {}
    pending_jobs: list[FoldJob] = []
    for job in jobs:
        if not resume_completed_folds:
            pending_jobs.append(job)
            continue
        try:
            result = load_training_result(job.config)
        except FileNotFoundError:
            pending_jobs.append(job)
        except ValueError as error:
            print(f"Not reusing fold {job.fold}: {error}")
            pending_jobs.append(job)
        else:
            fold_results[job.fold] = result
            print(_completion_line("Reusing completed", job.fold, result))

    worker_count = min(max_parallel_folds, len(pending_jobs))
    if use_process_pool and worker_count > 1:
        fold_results.update(_run_parallel_fold_jobs(pending_jobs, worker_count))
    else:
        for job in pending_jobs:
            print(f"Training MotionGPT fold {job.fold} on {job.config.device} ...")
            fold, result = _execute_fold(job)
            fold_results[fold] = result
            print(_completion_line("Completed", fold, result))

    summary_dir = experiment_output_dir() / f"cross_validation_seed-{SEED}"
    artifacts = aggregate_cross_validation_results(
        fold_results=fold_results,
        output_dir=summary_dir,
        threshold=TRAINING_CONFIG.threshold,
        artifact_prefix=f"{DATASET_PREFIX}_{EXPERIMENT_VARIANT}",
    )
    _print_cross_validation_summary(artifacts)
    return CrossValidationRun(fold_results=fold_results, artifacts=artifacts)


def _format_statistic(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.4f}"


def _print_cross_validation_summary(artifacts: CrossValidationArtifacts) -> None:
    test_statistics = artifacts.summary["fold_statistics"]["test"]
    print("\nCross-validation test metrics (unweighted fold mean ± sample SD):")
    for metric in SCORE_METRICS:
        statistics = test_statistics[metric]
        print(
            f"  {metric:20s} "
            f"{_format_statistic(statistics['mean'])} ± "
            f"{_format_statistic(statistics['standard_deviation'])}"
        )

    pooled = artifacts.summary["pooled_out_of_fold"]["metrics"]
    print("\nPooled out-of-fold metrics:")
    for metric in SCORE_METRICS:
        print(f"  {metric:20s} {_format_statistic(pooled[metric])}")
    print(f"\nSummary: {artifacts.summary_path}")


def main() -> TrainingResult | CrossValidationRun:
    if RUN_MODE == "single":
        result = train_single_model()
        print(f"Best epoch: {result.best_epoch}")
        print(f"Model parameters: {result.model_parameter_count}")
        print(f"Decision threshold: {result.decision_threshold:.4f}")
        print(f"Tokenizer validation metrics: {result.tokenizer_metrics}")
        print(f"Validation metrics: {result.validation_metrics}")
        print(f"Test metrics: {result.test_metrics}")
        print(f"Checkpoint: {result.checkpoint_path}")
        return result
    if RUN_MODE == "cross_validation":
        return train_cross_validation()
    raise ValueError(f"Unknown RUN_MODE: {RUN_MODE!r}")


if __name__ == "__main__":
    main()
