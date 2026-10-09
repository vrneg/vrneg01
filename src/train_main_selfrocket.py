"""Editable entry point for SelF-Rocket single-fold and CV experiments.

SelF-Rocket is CPU based and generates ten large feature blocks before selecting
one input-representation/pooling-operator combination. Folds therefore run
sequentially by default.
"""

from __future__ import annotations

import multiprocessing
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from selfrocket import (
    SCORE_METRICS,
    CrossValidationArtifacts,
    DataConfig,
    EvaluationConfig,
    ExperimentConfig,
    SelFRocketConfig,
    TrainingResult,
    aggregate_cross_validation_results,
    load_training_result,
    train_selfrocket,
)


# -----------------------------------------------------------------------------
# Experiment selection
# -----------------------------------------------------------------------------

RUN_MODE: Literal["single", "cross_validation"] = "cross_validation"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = PROJECT_ROOT / "data/trainsets"
DATASET_PREFIX = "target-cue_window-500_splits-10"
OUTPUT_ROOT = PROJECT_ROOT / "outputs/selfrocket"

SINGLE_FOLD = 0
CROSS_VALIDATION_FOLDS: tuple[int, ...] = tuple(range(10))
SEED = 42


# -----------------------------------------------------------------------------
# Fixed-grid representation (identical to MiniRocket and MultiRocket)
# -----------------------------------------------------------------------------

NUM_TIME_POINTS = 32
WINDOW_START_SECONDS = -0.5
WINDOW_END_SECONDS = 0.5
MAX_EVENTS_PER_MODALITY: int | None = 128
NORMALIZE_FEATURES = True
CACHE_DATASET_IN_MEMORY = True

ACTOR_SCOPE: Literal["anchor", "other", "all"] = "anchor"
INCLUDE_PRESENCE_CHANNELS = True
POSITIVE_LABEL = "neg"
NEGATIVE_LABEL = "none"


# -----------------------------------------------------------------------------
# SelF-Rocket feature generation, selection, and ridge classifier
# -----------------------------------------------------------------------------

SELFROCKET_CONFIG = SelFRocketConfig(
    # MiniRocket's nominal 10,000 kernels produce 9,996 features per operator
    # and representation (19,992 for a MIX candidate).
    num_kernels=10_000,
    max_dilations_per_kernel=32,
    normalise_per_instance=False,
    # False evaluates all 15 BASE/DIFF/MIX x pooling candidates. True evaluates
    # only the five MIX candidates as the faster paper ablation.
    only_mix=False,
    selection_num_folds=2,
    selection_num_runs=10,
    selection_num_features=2_500,
    selection_max_samples=500,
    vote_top=5,
    vote_threshold=0.9,
    length_threshold=512,
    class_weight=None,
    n_jobs=4,
)

EVALUATION_CONFIG = EvaluationConfig(
    threshold=0.5,
    calibrate_threshold_on_validation=False,
    threshold_metric="macro_f1",
)


# -----------------------------------------------------------------------------
# Fold-level parallelism
# -----------------------------------------------------------------------------

USE_PROCESS_POOL = False
MAX_PARALLEL_FOLDS = 1
RESUME_COMPLETED_FOLDS = True


@dataclass(frozen=True, slots=True)
class FoldJob:
    fold: int
    config: ExperimentConfig


@dataclass(slots=True)
class CrossValidationRun:
    fold_results: dict[int, TrainingResult]
    artifacts: CrossValidationArtifacts


def experiment_variant() -> str:
    return "mix-only" if SELFROCKET_CONFIG.only_mix else "full"


def experiment_output_dir() -> Path:
    return OUTPUT_ROOT / DATASET_PREFIX / experiment_variant()


def dataset_path(fold: int) -> Path:
    return DATASET_ROOT / f"{DATASET_PREFIX}_fold-{fold}"


def build_experiment_config(fold: int) -> ExperimentConfig:
    return ExperimentConfig(
        data=DataConfig(
            dataset_path=dataset_path(fold),
            num_time_points=NUM_TIME_POINTS,
            window_start_seconds=WINDOW_START_SECONDS,
            window_end_seconds=WINDOW_END_SECONDS,
            max_events_per_modality=MAX_EVENTS_PER_MODALITY,
            normalize_features=NORMALIZE_FEATURES,
            cache_in_memory=CACHE_DATASET_IN_MEMORY,
            actor_scope=ACTOR_SCOPE,
            include_presence_channels=INCLUDE_PRESENCE_CHANNELS,
            positive_label=POSITIVE_LABEL,
            negative_label=NEGATIVE_LABEL,
        ),
        model=SELFROCKET_CONFIG,
        evaluation=EVALUATION_CONFIG,
        output_dir=experiment_output_dir(),
        run_name=f"fold-{fold}_seed-{SEED}",
        seed=SEED,
    )


def _execute_fold(job: FoldJob) -> tuple[int, TrainingResult]:
    result = train_selfrocket(job.config)
    return job.fold, result


def train_single_model(fold: int = SINGLE_FOLD) -> TrainingResult:
    """Train and evaluate one fold."""

    return train_selfrocket(build_experiment_config(fold))


def _build_fold_jobs(folds: Sequence[int]) -> list[FoldJob]:
    if not folds:
        raise ValueError("At least one fold must be selected")
    if len(set(folds)) != len(folds):
        raise ValueError("folds must not contain duplicates")
    return [FoldJob(fold=fold, config=build_experiment_config(fold)) for fold in folds]


def _completion_line(prefix: str, fold: int, result: TrainingResult) -> str:
    fallback = " (fallback)" if result.selection_used_fallback else ""
    return (
        f"{prefix} fold {fold}: selected={result.selected_candidate}{fallback}, "
        f"alpha={result.best_alpha:.6g}, "
        f"features={result.num_transformed_features}, "
        f"test_auc={result.test_metrics['roc_auc']:.4f}, "
        f"test_macro_f1={result.test_metrics['macro_f1']:.4f}"
    )


def _run_parallel_fold_jobs(
    jobs: Sequence[FoldJob],
    worker_count: int,
) -> dict[int, TrainingResult]:
    spawn_context = multiprocessing.get_context("spawn")
    results: dict[int, TrainingResult] = {}
    with ProcessPoolExecutor(
        max_workers=worker_count,
        mp_context=spawn_context,
    ) as executor:
        futures = {executor.submit(_execute_fold, job): job.fold for job in jobs}
        for future in as_completed(futures):
            expected_fold = futures[future]
            try:
                fold, result = future.result()
            except Exception as error:
                raise RuntimeError(
                    f"SelF-Rocket training failed for fold {expected_fold}"
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
    """Train selected folds and aggregate fold-wise plus pooled OOF metrics."""

    if max_parallel_folds < 1:
        raise ValueError("max_parallel_folds must be at least 1")
    jobs = _build_fold_jobs(tuple(folds))
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
            print(
                f"Training SelF-Rocket fold {job.fold} "
                f"(variant={experiment_variant()}) ..."
            )
            fold, result = _execute_fold(job)
            fold_results[fold] = result
            print(_completion_line("Completed", fold, result))

    summary_dir = experiment_output_dir() / f"cross_validation_seed-{SEED}"
    artifact_prefix = f"{DATASET_PREFIX}_{experiment_variant()}"
    artifacts = aggregate_cross_validation_results(
        fold_results=fold_results,
        output_dir=summary_dir,
        threshold=EVALUATION_CONFIG.threshold,
        artifact_prefix=artifact_prefix,
    )
    _print_cross_validation_summary(artifacts)
    return CrossValidationRun(fold_results=fold_results, artifacts=artifacts)


def _format_statistic(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.4f}"


def _print_cross_validation_summary(artifacts: CrossValidationArtifacts) -> None:
    test_statistics = artifacts.summary["fold_statistics"]["test"]
    print("\nCross-validation test metrics (unweighted fold mean +/- sample SD):")
    for metric in SCORE_METRICS:
        statistics = test_statistics[metric]
        print(
            f"  {metric:20s} "
            f"{_format_statistic(statistics['mean'])} +/- "
            f"{_format_statistic(statistics['standard_deviation'])}"
        )

    pooled = artifacts.summary["pooled_out_of_fold"]["metrics"]
    print("\nPooled out-of-fold metrics:")
    for metric in SCORE_METRICS:
        print(f"  {metric:20s} {_format_statistic(pooled[metric])}")
    print(f"\nSummary: {artifacts.summary_path}")


def main() -> TrainingResult | CrossValidationRun:
    print(f"SelF-Rocket variant: {experiment_variant()}")
    if RUN_MODE == "single":
        result = train_single_model()
        print(f"Selected candidate: {result.selected_candidate}")
        print(f"Candidate before vote validation: {result.selected_candidate_before_vote}")
        print(f"Vote support: {result.selection_vote_support:.3f}")
        print(f"Selected ridge alpha: {result.best_alpha:.6g}")
        print(f"Transformed features: {result.num_transformed_features}")
        print(f"Decision threshold: {result.decision_threshold:.4f}")
        print(f"Validation metrics: {result.validation_metrics}")
        print(f"Test metrics: {result.test_metrics}")
        print(f"Checkpoint: {result.checkpoint_path}")
        return result
    if RUN_MODE == "cross_validation":
        return train_cross_validation()
    raise ValueError(f"Unknown RUN_MODE: {RUN_MODE!r}")


if __name__ == "__main__":
    main()
