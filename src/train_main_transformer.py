"""Editable entry point for one fold or a complete cross-validation run.

This file intentionally has no command-line parser. Change the configuration block
below, run it as a module, or import and call ``train_single_model`` /
``train_cross_validation`` directly.
"""

from __future__ import annotations

import multiprocessing
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, replace
from pathlib import Path
from queue import Empty
from typing import Any, Literal

import torch
from tqdm import tqdm

from event_transformer import (
    SCORE_METRICS,
    CrossValidationArtifacts,
    aggregate_cross_validation_results,
    DataConfig,
    ExperimentConfig,
    ModelConfig,
    PretrainingConfig,
    TrainingConfig,
    TrainingResult,
    train_event_transformer,
)

# -----------------------------------------------------------------------------
# Experiment selection
# -----------------------------------------------------------------------------

RUN_MODE: Literal["single", "cross_validation"] = "cross_validation"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = PROJECT_ROOT / "data/trainsets"
DATASET_PREFIX = "target-cue_window-500_splits-10"
OUTPUT_ROOT = PROJECT_ROOT / "outputs/event_transformer"

SINGLE_FOLD = 0
CROSS_VALIDATION_FOLDS: tuple[int, ...] = tuple(range(10))
SEED = 42


# -----------------------------------------------------------------------------
# Hardware and fold-level parallelism
# -----------------------------------------------------------------------------

# CUDA is deliberately the default. Use "cpu" only for debugging or smoke tests.
SINGLE_MODEL_DEVICE = "cuda:0"

# Fold jobs are assigned to these devices in round-robin order. Multiple entries may
# point to the same GPU, e.g. ten copies of "cuda:0", but concurrent models on one GPU
# compete for memory and are often slower than sequential training.
GPU_DEVICES: tuple[str, ...] = ("cuda:0",)
USE_PROCESS_POOL = True
MAX_PARALLEL_FOLDS = 10
CPU_THREADS_PER_FOLD = 4


# -----------------------------------------------------------------------------
# Data and batching
# -----------------------------------------------------------------------------

BATCH_SIZE = 8
EVALUATION_BATCH_SIZE: int | None = None
NUM_DATA_WORKERS = 0
CACHE_DATASET_IN_MEMORY = True
NORMALIZE_FEATURES = True
MAX_EVENTS_PER_MODALITY: int | None = 128
TIME_CLIP_SECONDS: float | None = 10.0
POSITIVE_LABEL = "neg"
NEGATIVE_LABEL = "none"


# -----------------------------------------------------------------------------
# Model
# -----------------------------------------------------------------------------

MODEL_CONFIG = ModelConfig(
    d_model=32,
    nhead=4,
    num_layers=1,
    dim_feedforward=128,
    modality_hidden_dim=64,
    modality_fusion_hidden_dim=64,
    time_hidden_dim=32,
    classifier_hidden_dim=16,
    dropout=0.35,
    norm_first=True,
)


# -----------------------------------------------------------------------------
# Self-supervised pretraining
# -----------------------------------------------------------------------------

# This is also exposed as a keyword on both public training functions below.
USE_PRETRAINING = True

PRETRAINING_CONFIG = PretrainingConfig(
    num_epochs=10,
    mask_probability=0.20,
    learning_rate=1e-4,
    weight_decay=1e-3,
    gradient_clip_norm=1.0,
    lr_scheduler_factor=0.5,
    lr_scheduler_patience=2,
    minimum_learning_rate=1e-6,
)


# -----------------------------------------------------------------------------
# Optimization and model selection
# -----------------------------------------------------------------------------

TRAINING_CONFIG = TrainingConfig(
    max_epochs=60,
    learning_rate=1e-4,
    classifier_learning_rate=3e-4,
    freeze_backbone_epochs=3,
    weight_decay=1e-2,
    gradient_clip_norm=1.0,
    early_stopping_patience=10,
    early_stopping_min_delta=1e-4,
    # Loss is substantially less noisy than AUROC on validation folds containing
    # only a few held-out sessions.
    selection_metric="loss",
    positive_class_weight=None,
    threshold=0.5,
    calibrate_threshold_on_validation=False,
    threshold_metric="macro_f1",
    lr_scheduler="reduce_on_plateau",
    lr_scheduler_factor=0.5,
    lr_scheduler_patience=3,
    minimum_learning_rate=1e-6,
    mixed_precision=True,
    deterministic_algorithms=False,
    show_progress=True,
)


@dataclass(frozen=True, slots=True)
class FoldJob:
    fold: int
    config: ExperimentConfig
    cpu_threads: int
    use_pretraining: bool


@dataclass(slots=True)
class CrossValidationRun:
    fold_results: dict[int, TrainingResult]
    artifacts: CrossValidationArtifacts


_WORKER_PROGRESS_QUEUE: Any | None = None


def _initialize_fold_worker(progress_queue: Any | None) -> None:
    """Give each spawned worker the queue inherited during process creation."""

    global _WORKER_PROGRESS_QUEUE
    _WORKER_PROGRESS_QUEUE = progress_queue


def experiment_output_dir() -> Path:
    """Keep every artifact for an experiment under its dataset-prefix name."""

    return OUTPUT_ROOT / DATASET_PREFIX


def dataset_path(fold: int) -> Path:
    return DATASET_ROOT / f"{DATASET_PREFIX}_fold-{fold}"


def build_experiment_config(
    fold: int,
    device: str,
    show_progress: bool = True,
) -> ExperimentConfig:
    """Build the complete immutable configuration for one fold."""

    data_config = DataConfig(
        dataset_path=dataset_path(fold),
        batch_size=BATCH_SIZE,
        evaluation_batch_size=EVALUATION_BATCH_SIZE,
        num_workers=NUM_DATA_WORKERS,
        cache_in_memory=CACHE_DATASET_IN_MEMORY,
        normalize_features=NORMALIZE_FEATURES,
        max_events_per_modality=MAX_EVENTS_PER_MODALITY,
        time_clip_seconds=TIME_CLIP_SECONDS,
        positive_label=POSITIVE_LABEL,
        negative_label=NEGATIVE_LABEL,
    )
    training_config = replace(TRAINING_CONFIG, show_progress=show_progress)
    return ExperimentConfig(
        data=data_config,
        model=MODEL_CONFIG,
        pretraining=PRETRAINING_CONFIG,
        training=training_config,
        output_dir=experiment_output_dir(),
        run_name=f"fold-{fold}_seed-{SEED}",
        seed=SEED,
        device=device,
    )


def _execute_fold(job: FoldJob) -> tuple[int, TrainingResult]:
    """Top-level worker function so CUDA jobs are spawn-process compatible."""

    torch.set_num_threads(job.cpu_threads)
    progress_callback = None
    progress_queue = _WORKER_PROGRESS_QUEUE
    if progress_queue is not None:
        progress_queue.put(
            {
                "event": "started",
                "fold": job.fold,
                "device": job.config.device,
            }
        )

        def report_progress(message: dict[str, Any]) -> None:
            progress_queue.put({"fold": job.fold, **message})

        progress_callback = report_progress

    result = train_event_transformer(
        job.config,
        use_pretraining=job.use_pretraining,
        progress_callback=progress_callback,
    )
    return job.fold, result


def train_single_model(
    fold: int = SINGLE_FOLD,
    device: str = SINGLE_MODEL_DEVICE,
    *,
    use_pretraining: bool = USE_PRETRAINING,
) -> TrainingResult:
    """Pretrain, fine-tune, and test one selected fold on CUDA by default."""

    config = build_experiment_config(
        fold,
        device=device,
        show_progress=TRAINING_CONFIG.show_progress,
    )
    return train_event_transformer(config, use_pretraining=use_pretraining)


def _build_fold_jobs(
    folds: Sequence[int],
    devices: Sequence[str],
    show_progress: bool,
    use_pretraining: bool,
) -> list[FoldJob]:
    if not folds:
        raise ValueError("At least one fold must be selected")
    if not devices:
        raise ValueError("At least one CUDA device must be configured")
    if CPU_THREADS_PER_FOLD < 1:
        raise ValueError("CPU_THREADS_PER_FOLD must be at least 1")

    return [
        FoldJob(
            fold=fold,
            config=build_experiment_config(
                fold,
                device=devices[index % len(devices)],
                show_progress=show_progress,
            ),
            cpu_threads=CPU_THREADS_PER_FOLD,
            use_pretraining=use_pretraining,
        )
        for index, fold in enumerate(folds)
    ]


def _create_fold_progress_bars(jobs: Sequence[FoldJob]) -> dict[int, tqdm]:
    bars: dict[int, tqdm] = {}
    for position, job in enumerate(jobs):
        bar = tqdm(
            total=(
                job.config.training.max_epochs
                + (job.config.pretraining.num_epochs if job.use_pretraining else 0)
            ),
            desc=f"fold-{job.fold}",
            unit="epoch",
            position=position,
            leave=True,
            dynamic_ncols=True,
        )
        bar.set_postfix_str("queued", refresh=True)
        bars[job.fold] = bar
    return bars


def _update_fold_progress_bar(bar: tqdm, message: dict[str, Any]) -> None:
    event = message["event"]
    if event == "started":
        bar.set_postfix_str(f"starting on {message['device']}", refresh=True)
        return
    if event == "pretraining_epoch_complete":
        completed_epoch = int(message["epoch"])
        bar.update(max(0, completed_epoch - int(bar.n)))
        bar.set_postfix(
            {
                "phase": "pretrain",
                "reconstruction": f"{message['reconstruction_loss']:.4f}",
                "masked": message["num_masked_observations"],
            },
            refresh=True,
        )
        return
    if event == "pretraining_complete":
        bar.set_postfix_str("pretraining done; starting fine-tuning", refresh=True)
        return
    if event == "epoch_complete":
        completed_epoch = int(message.get("progress_epoch", message["epoch"]))
        bar.update(max(0, completed_epoch - int(bar.n)))
        validation = message["validation_metrics"]
        bar.set_postfix(
            {
                "phase": "fine-tune",
                "train": f"{message['training_loss']:.4f}",
                "val_loss": f"{validation['loss']:.4f}",
                "val_auc": f"{validation['roc_auc']:.4f}",
                "val_macro_f1": f"{validation['macro_f1']:.4f}",
                "best": message["best_epoch"],
            },
            refresh=True,
        )
        return
    if event == "training_complete":
        bar.set_postfix_str(
            f"evaluating best epoch {message['best_epoch']}",
            refresh=True,
        )
        return
    if event == "finished":
        if bar.n < bar.total:
            bar.total = bar.n
        bar.set_postfix_str(
            f"done; best={message['best_epoch']}, "
            f"test_macro_f1={message['test_macro_f1']:.4f}, "
            f"threshold={message['decision_threshold']:.3f}",
            refresh=True,
        )


def _run_parallel_fold_jobs(
    jobs: Sequence[FoldJob],
    worker_count: int,
) -> dict[int, TrainingResult]:
    """Run folds in spawned workers while the parent renders all progress bars."""

    spawn_context = multiprocessing.get_context("spawn")
    show_progress = TRAINING_CONFIG.show_progress
    bars = _create_fold_progress_bars(jobs) if show_progress else {}
    fold_results: dict[int, TrainingResult] = {}
    progress_queue = spawn_context.Queue() if show_progress else None

    try:
        with ProcessPoolExecutor(
            max_workers=worker_count,
            mp_context=spawn_context,
            initializer=_initialize_fold_worker,
            initargs=(progress_queue,),
        ) as executor:
            future_to_fold = {
                executor.submit(_execute_fold, job): job.fold for job in jobs
            }
            pending = set(future_to_fold)
            while pending:
                if progress_queue is not None:
                    try:
                        message = progress_queue.get(timeout=0.1)
                        _update_fold_progress_bar(
                            bars[int(message["fold"])],
                            message,
                        )
                        while True:
                            message = progress_queue.get_nowait()
                            _update_fold_progress_bar(
                                bars[int(message["fold"])],
                                message,
                            )
                    except Empty:
                        pass

                completed = {future for future in pending if future.done()}
                for future in completed:
                    pending.remove(future)
                    expected_fold = future_to_fold[future]
                    try:
                        fold, result = future.result()
                    except Exception as error:
                        if show_progress:
                            bars[expected_fold].set_postfix_str(
                                "failed",
                                refresh=True,
                            )
                        raise RuntimeError(
                            f"Training failed for fold {expected_fold}"
                        ) from error
                    fold_results[fold] = result
                    if show_progress:
                        _update_fold_progress_bar(
                            bars[fold],
                            {
                                "event": "finished",
                                "fold": fold,
                                "best_epoch": result.best_epoch,
                                "test_macro_f1": result.test_metrics["macro_f1"],
                                "decision_threshold": result.decision_threshold,
                            },
                        )
    finally:
        if progress_queue is not None:
            progress_queue.close()
            progress_queue.join_thread()
        for bar in bars.values():
            bar.close()

    return fold_results


def train_cross_validation(
    folds: Sequence[int] = CROSS_VALIDATION_FOLDS,
    devices: Sequence[str] = GPU_DEVICES,
    max_parallel_folds: int = MAX_PARALLEL_FOLDS,
    use_process_pool: bool = USE_PROCESS_POOL,
    *,
    use_pretraining: bool = USE_PRETRAINING,
) -> CrossValidationRun:
    """Train selected folds and produce fold-wise plus pooled OOF statistics.

    CUDA processes must be spawned, never forked. Set ``max_parallel_folds=10`` and
    ``devices=("cuda:0",)`` to permit ten concurrent jobs on one GPU. In practice,
    start with one job per GPU and increase only after measuring memory and throughput.
    """

    folds = tuple(folds)
    devices = tuple(devices)
    if len(set(folds)) != len(folds):
        raise ValueError("folds must not contain duplicates")
    if max_parallel_folds < 1:
        raise ValueError("max_parallel_folds must be at least 1")

    worker_count = min(max_parallel_folds, len(folds))
    jobs = _build_fold_jobs(
        folds,
        devices,
        # Spawned workers report to parent-owned progress bars. They must never write
        # their own tqdm control sequences into the shared terminal.
        show_progress=TRAINING_CONFIG.show_progress and not use_process_pool,
        use_pretraining=use_pretraining,
    )
    fold_results: dict[int, TrainingResult] = {}

    if use_process_pool:
        fold_results = _run_parallel_fold_jobs(jobs, worker_count)
    else:
        for job in jobs:
            fold, result = _execute_fold(job)
            fold_results[fold] = result
            print(f"Completed fold {fold}: {result.test_metrics}")

    summary_dir = experiment_output_dir() / f"cross_validation_seed-{SEED}"
    artifacts = aggregate_cross_validation_results(
        fold_results=fold_results,
        output_dir=summary_dir,
        threshold=TRAINING_CONFIG.threshold,
        artifact_prefix=DATASET_PREFIX,
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
        print(f"Fixed decision threshold: {result.decision_threshold:.4f}")
        print(f"Validation metrics: {result.validation_metrics}")
        print(f"Test metrics: {result.test_metrics}")
        print(f"Checkpoint: {result.checkpoint_path}")
        return result
    if RUN_MODE == "cross_validation":
        return train_cross_validation()
    raise ValueError(f"Unknown RUN_MODE: {RUN_MODE!r}")


if __name__ == "__main__":
    main()
