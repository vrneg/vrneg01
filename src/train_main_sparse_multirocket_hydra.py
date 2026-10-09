"""Editable entry point for sparse MultiRocket-HYDRA experiments.

The expensive convolutional transforms are fitted once per outer fold. Scaling,
supervised feature selection, classifier comparison, and regularization tuning run
inside stratified inner cross-validation on the outer training split only.
"""

from __future__ import annotations

import multiprocessing
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from sparse_multirocket_hydra import (
    SCORE_METRICS,
    CrossValidationArtifacts,
    DataConfig,
    EvaluationConfig,
    ExperimentConfig,
    SparseMultiRocketHydraConfig,
    TrainingResult,
    aggregate_cross_validation_results,
    load_training_result,
    train_sparse_multirocket_hydra,
)


# -----------------------------------------------------------------------------
# Experiment selection
# -----------------------------------------------------------------------------

RUN_MODE: Literal["single", "cross_validation"] = "cross_validation"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = PROJECT_ROOT / "data/trainsets"
DATASET_PREFIX = "target-cue_window-500_splits-10"
OUTPUT_ROOT = PROJECT_ROOT / "outputs/sparse_multirocket_hydra"

SINGLE_FOLD = 0
CROSS_VALIDATION_FOLDS: tuple[int, ...] = tuple(range(10))
SEED = 42


# -----------------------------------------------------------------------------
# Fixed-grid representation (identical to the MultiRocket benchmark)
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
# MultiRocket + HYDRA transform
# -----------------------------------------------------------------------------

NUM_KERNELS = 6_250
MAX_DILATIONS_PER_KERNEL = 32
NUM_FEATURES_PER_KERNEL = 4
NORMALISE_PER_INSTANCE = False
HYDRA_NUM_KERNELS = 8
HYDRA_NUM_GROUPS = 64
HYDRA_MAX_NUM_CHANNELS = 8
TRANSFORM_N_JOBS = 4


# -----------------------------------------------------------------------------
# Sparse selection and inner-CV classifier comparison
# -----------------------------------------------------------------------------

FEATURE_COUNTS: tuple[int, ...] = (5_000, 10_000, 15_000)
CLASSIFIER_CANDIDATES = (
    "ridge",
    "logistic_l2",
    "elastic_net",
    "linear_svm",
)

# These grids emphasize regularized models while retaining one weaker endpoint.
RIDGE_ALPHAS = (1.0, 10.0, 100.0, 1_000.0, 10_000.0)
LOGISTIC_L2_CS = (1e-3, 1e-2, 1e-1, 1.0, 10.0)
ELASTIC_NET_ALPHAS = (1e-3, 1e-2, 1e-1)
ELASTIC_NET_L1_RATIOS = (0.2, 0.5, 0.8)
LINEAR_SVM_CS = (1e-3, 1e-2, 1e-1, 1.0)

INNER_CV_FOLDS = 3
INNER_SCORING: Literal[
    "accuracy",
    "balanced_accuracy",
    "macro_f1",
    "roc_auc",
    "average_precision",
] = "balanced_accuracy"
CLASS_WEIGHT: Literal["balanced"] | None = None
MAX_ITER = 2_000
TOLERANCE = 1e-3
SEARCH_N_JOBS = 3
SEARCH_PRE_DISPATCH = 3
SEARCH_VERBOSE = 2

# Reuse the fold-matched random transforms from the completed benchmark. This
# changes neither inputs nor features; it avoids refitting identical transforms.
REUSE_FITTED_MULTIROCKET_HYDRA = True

SPARSE_MODEL_CONFIG = SparseMultiRocketHydraConfig(
    num_kernels=NUM_KERNELS,
    max_dilations_per_kernel=MAX_DILATIONS_PER_KERNEL,
    num_features_per_kernel=NUM_FEATURES_PER_KERNEL,
    normalise_per_instance=NORMALISE_PER_INSTANCE,
    hydra_num_kernels=HYDRA_NUM_KERNELS,
    hydra_num_groups=HYDRA_NUM_GROUPS,
    hydra_max_num_channels=HYDRA_MAX_NUM_CHANNELS,
    feature_counts=FEATURE_COUNTS,
    classifier_candidates=CLASSIFIER_CANDIDATES,
    ridge_alphas=RIDGE_ALPHAS,
    logistic_l2_cs=LOGISTIC_L2_CS,
    elastic_net_alphas=ELASTIC_NET_ALPHAS,
    elastic_net_l1_ratios=ELASTIC_NET_L1_RATIOS,
    linear_svm_cs=LINEAR_SVM_CS,
    inner_cv_folds=INNER_CV_FOLDS,
    inner_scoring=INNER_SCORING,
    class_weight=CLASS_WEIGHT,
    max_iter=MAX_ITER,
    tolerance=TOLERANCE,
    n_jobs=TRANSFORM_N_JOBS,
    search_n_jobs=SEARCH_N_JOBS,
    search_pre_dispatch=SEARCH_PRE_DISPATCH,
    search_verbose=SEARCH_VERBOSE,
)

EVALUATION_CONFIG = EvaluationConfig(
    threshold=0.5,
    calibrate_threshold_on_validation=False,
    threshold_metric="macro_f1",
)


# -----------------------------------------------------------------------------
# Outer-fold parallelism
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


def _feature_count_tag(count: int) -> str:
    return f"{count // 1_000}k" if count % 1_000 == 0 else str(count)


def experiment_variant() -> str:
    counts = "-".join(_feature_count_tag(count) for count in FEATURE_COUNTS)
    classifiers = "-".join(CLASSIFIER_CANDIDATES)
    return f"topk-{counts}_{classifiers}"


def experiment_output_dir() -> Path:
    return OUTPUT_ROOT / DATASET_PREFIX / experiment_variant()


def dataset_path(fold: int) -> Path:
    return DATASET_ROOT / f"{DATASET_PREFIX}_fold-{fold}"


def feature_artifact_path(fold: int) -> Path | None:
    if not REUSE_FITTED_MULTIROCKET_HYDRA:
        return None
    # ``train_main_all`` replaces DATASET_PREFIX after importing this module.
    # Derive the baseline root here so it follows that runtime selection instead
    # of retaining the module's default dataset family.
    baseline_root = (
        PROJECT_ROOT / "outputs/multirocket" / DATASET_PREFIX / "with-hydra"
    )
    return baseline_root / f"fold-{fold}_seed-{SEED}/model.joblib"


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
        model=SPARSE_MODEL_CONFIG,
        evaluation=EVALUATION_CONFIG,
        feature_artifact_path=feature_artifact_path(fold),
        output_dir=experiment_output_dir(),
        run_name=f"fold-{fold}_seed-{SEED}",
        seed=SEED,
    )


def _execute_fold(job: FoldJob) -> tuple[int, TrainingResult]:
    result = train_sparse_multirocket_hydra(job.config)
    return job.fold, result


def train_single_model(fold: int = SINGLE_FOLD) -> TrainingResult:
    """Train and evaluate one outer fold."""

    return train_sparse_multirocket_hydra(build_experiment_config(fold))


def _build_fold_jobs(folds: Sequence[int]) -> list[FoldJob]:
    if not folds:
        raise ValueError("At least one fold must be selected")
    if len(set(folds)) != len(folds):
        raise ValueError("folds must not contain duplicates")
    return [FoldJob(fold=fold, config=build_experiment_config(fold)) for fold in folds]


def _completion_line(prefix: str, fold: int, result: TrainingResult) -> str:
    return (
        f"{prefix} fold {fold}: classifier={result.best_classifier}, "
        f"selected={result.num_selected_features}, "
        f"nonzero={result.num_nonzero_coefficients}, "
        f"inner_{INNER_SCORING}={result.best_cv_score:.4f}, "
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
                    f"Sparse MultiRocket-HYDRA training failed for fold {expected_fold}"
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
    """Train selected outer folds and aggregate metrics plus model selection."""

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
                f"Training sparse MultiRocket-HYDRA fold {job.fold} "
                f"with {INNER_CV_FOLDS}-fold inner CV ..."
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

    selection = artifacts.summary["sparse_multirocket_hydra_selection"]
    print(f"\nWinning classifiers: {selection['best_classifier_counts']}")
    print(f"Summary: {artifacts.summary_path}")


def main() -> TrainingResult | CrossValidationRun:
    print(f"Sparse MultiRocket-HYDRA variant: {experiment_variant()}")
    if RUN_MODE == "single":
        result = train_single_model()
        print(f"Best classifier: {result.best_classifier}")
        print(f"Best inner-CV score: {result.best_cv_score:.4f}")
        print(f"Best hyperparameters: {result.best_hyperparameters}")
        print(f"Classifier leaderboard: {result.classifier_leaderboard}")
        print(
            f"Selected features: {result.num_selected_features} / "
            f"{result.num_transformed_features}"
        )
        print(f"Non-zero coefficients: {result.num_nonzero_coefficients}")
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
