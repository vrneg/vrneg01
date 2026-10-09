"""Editable entry point for HIVE-COTE 2.0 upper-bound experiments.

HC2 is exceptionally CPU and memory intensive. Outer folds can run sequentially
or in separate processes, while each fold also has aeon-level internal workers.
"""

from __future__ import annotations

import multiprocessing
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from hivecote_v2 import (
    SCORE_METRICS,
    ArsenalComponentConfig,
    CrossValidationArtifacts,
    DataConfig,
    DrCIFComponentConfig,
    EvaluationConfig,
    ExperimentConfig,
    HIVECOTEV2Config,
    STCComponentConfig,
    TDEComponentConfig,
    TrainingResult,
    aggregate_cross_validation_results,
    load_training_result,
    resolve_fold_worker_count,
    train_hivecote_v2,
)


# -----------------------------------------------------------------------------
# Experiment selection
# -----------------------------------------------------------------------------

RUN_MODE: Literal["single", "cross_validation"] = "cross_validation"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = PROJECT_ROOT / "data/trainsets"
DATASET_PREFIX = "target-cue_window-500_splits-10"
OUTPUT_ROOT = PROJECT_ROOT / "outputs/hivecote_v2"
EXPERIMENT_VARIANT = "contract-6h-paper-caps"

SINGLE_FOLD = 0
CROSS_VALIDATION_FOLDS: tuple[int, ...] = tuple(range(10))
SEED = 42


# -----------------------------------------------------------------------------
# Fixed-grid representation (identical to the other time-series experiments)
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
# HIVE-COTE 2.0: paper defaults with an approximate per-fold resource contract
# -----------------------------------------------------------------------------

HIVECOTE_CONFIG = HIVECOTEV2Config(
    # Set to 0 for the complete paper-default counts without a time limit.
    # Positive values are approximate; aeon assigns value / 6 to each component.
    time_limit_in_minutes=360.0,
    stc=STCComponentConfig(
        n_shapelet_samples=10_000,
        max_shapelets=None,
        max_shapelet_length=None,
        batch_size=100,
        contract_max_n_shapelet_samples=10_000,
        rotation_forest_n_estimators=200,
        rotation_forest_contract_max_n_estimators=200,
    ),
    drcif=DrCIFComponentConfig(
        n_estimators=500,
        n_intervals=(4, "sqrt-div"),
        min_interval_length=3,
        max_interval_length=0.5,
        att_subsample_size=10,
        contract_max_n_estimators=500,
        use_pycatch22=False,
        stabilize_near_constant_intervals=True,
    ),
    arsenal=ArsenalComponentConfig(
        n_kernels=2_000,
        n_estimators=25,
        rocket_transform="rocket",
        max_dilations_per_kernel=32,
        n_features_per_kernel=4,
        contract_max_n_estimators=25,
    ),
    tde=TDEComponentConfig(
        n_parameter_samples=250,
        max_ensemble_size=50,
        max_win_len_prop=1.0,
        min_window=10,
        randomly_selected_params=50,
        bigrams=None,
        dim_threshold=0.85,
        max_dims=20,
        contract_max_n_parameter_samples=250,
    ),
    save_component_predictions=True,
    verbose=1,
    n_jobs=4,
    parallel_backend=None,
)

EVALUATION_CONFIG = EvaluationConfig(
    threshold=0.5,
    calibrate_threshold_on_validation=False,
    threshold_metric="macro_f1",
)


# -----------------------------------------------------------------------------
# Fold-level execution
# -----------------------------------------------------------------------------

# True launches the selected pending folds in separate processes. With no cap,
# all ten folds are submitted concurrently on a sufficiently powerful server.
PARALLEL_FOLDS = True
MAX_PARALLEL_FOLDS: int | None = None
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
        model=HIVECOTE_CONFIG,
        evaluation=EVALUATION_CONFIG,
        output_dir=experiment_output_dir(),
        run_name=f"fold-{fold}_seed-{SEED}",
        seed=SEED,
    )


def _execute_fold(job: FoldJob) -> tuple[int, TrainingResult]:
    return job.fold, train_hivecote_v2(job.config)


def train_single_model(fold: int = SINGLE_FOLD) -> TrainingResult:
    return train_hivecote_v2(build_experiment_config(fold))


def _build_fold_jobs(folds: Sequence[int]) -> list[FoldJob]:
    if not folds:
        raise ValueError("At least one fold must be selected")
    if len(set(folds)) != len(folds):
        raise ValueError("folds must not contain duplicates")
    return [FoldJob(fold=fold, config=build_experiment_config(fold)) for fold in folds]


def _completion_line(prefix: str, fold: int, result: TrainingResult) -> str:
    return (
        f"{prefix} fold {fold}: fit={result.fit_seconds / 60:.1f} min, "
        f"test_auc={result.test_metrics['roc_auc']:.4f}, "
        f"test_macro_f1={result.test_metrics['macro_f1']:.4f}, "
        f"weights={result.component_weights}"
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
                    f"HIVE-COTE 2.0 training failed for fold {expected_fold}"
                ) from error
            results[fold] = result
            print(_completion_line("Completed", fold, result))
    return results


def _print_pending_fold_plan(
    jobs: Sequence[FoldJob],
    worker_count: int,
) -> None:
    if not jobs:
        return
    internal_workers = jobs[0].config.model.n_jobs
    total_internal_workers = worker_count * internal_workers
    print(
        f"Launching {len(jobs)} pending HIVE-COTE 2.0 folds with "
        f"{worker_count} concurrent fold process(es)."
    )
    print(
        f"Each fold may use {internal_workers} internal workers "
        f"(~{total_internal_workers} total at peak)."
    )
    for job in jobs:
        print(
            f"Fold {job.fold} progress file: "
            f"{job.config.output_dir / job.config.run_name / 'status.json'}"
        )


def train_cross_validation(
    folds: Sequence[int] = CROSS_VALIDATION_FOLDS,
    parallel_folds: bool = PARALLEL_FOLDS,
    max_parallel_folds: int | None = MAX_PARALLEL_FOLDS,
    resume_completed_folds: bool = RESUME_COMPLETED_FOLDS,
) -> CrossValidationRun:
    """Train selected folds and aggregate out-of-fold metrics.

    When ``parallel_folds`` is true and ``max_parallel_folds`` is ``None``, one
    process is allowed for every selected fold that does not already have a
    completed result.
    """

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
            print(_completion_line("Reusing", job.fold, result))

    worker_count = resolve_fold_worker_count(
        pending_fold_count=len(pending_jobs),
        parallel_folds=parallel_folds,
        max_parallel_folds=max_parallel_folds,
    )
    _print_pending_fold_plan(pending_jobs, worker_count)
    if parallel_folds and worker_count > 1:
        fold_results.update(_run_parallel_fold_jobs(pending_jobs, worker_count))
    else:
        for job in pending_jobs:
            contract = job.config.model.time_limit_in_minutes
            contract_label = "uncontracted" if contract == 0 else f"~{contract:g} min"
            print(
                f"Training HIVE-COTE 2.0 fold {job.fold} ({contract_label}, "
                f"{job.config.model.n_jobs} internal workers) ..."
            )
            print(
                f"Progress file: "
                f"{job.config.output_dir / job.config.run_name / 'status.json'}"
            )
            fold, result = _execute_fold(job)
            fold_results[fold] = result
            print(_completion_line("Completed", fold, result))

    artifacts = aggregate_cross_validation_results(
        fold_results=fold_results,
        output_dir=experiment_output_dir() / f"cross_validation_seed-{SEED}",
        threshold=EVALUATION_CONFIG.threshold,
        artifact_prefix=f"{DATASET_PREFIX}_{EXPERIMENT_VARIANT}",
    )
    _print_cross_validation_summary(artifacts)
    return CrossValidationRun(fold_results=fold_results, artifacts=artifacts)


def _format_statistic(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.4f}"


def _print_cross_validation_summary(artifacts: CrossValidationArtifacts) -> None:
    test_statistics = artifacts.summary["fold_statistics"]["test"]
    print("\nHIVE-COTE 2.0 test metrics (fold mean ± descriptive fold SD):")
    for metric in SCORE_METRICS:
        statistics = test_statistics[metric]
        print(
            f"  {metric:36s} "
            f"{_format_statistic(statistics['mean'])} ± "
            f"{_format_statistic(statistics['standard_deviation'])}"
        )
    pooled = artifacts.summary["pooled_out_of_fold"]["metrics"]
    print("\nPooled out-of-fold metrics:")
    for metric in SCORE_METRICS:
        print(f"  {metric:36s} {_format_statistic(pooled[metric])}")
    print(f"\nSummary: {artifacts.summary_path}")


def main() -> TrainingResult | CrossValidationRun:
    print(f"HIVE-COTE 2.0 variant: {EXPERIMENT_VARIANT}")
    print("Compute backend: CPU (the official HC2 components have no GPU backend)")
    if RUN_MODE == "single":
        result = train_single_model()
        print(f"Fit time: {result.fit_seconds / 60:.2f} min")
        print(f"Component weights: {result.component_weights}")
        print(f"Validation metrics: {result.validation_metrics}")
        print(f"Test metrics: {result.test_metrics}")
        print(f"Checkpoint: {result.checkpoint_path}")
        return result
    if RUN_MODE == "cross_validation":
        return train_cross_validation()
    raise ValueError(f"Unknown RUN_MODE: {RUN_MODE!r}")


if __name__ == "__main__":
    main()
