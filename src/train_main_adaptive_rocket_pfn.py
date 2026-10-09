"""Editable entry point for Adaptive Multi-Representation RocketPFN."""

from __future__ import annotations

import multiprocessing
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv

from adaptive_rocket_pfn import (
    SCORE_METRICS,
    AdaptiveRocketPFNConfig,
    CrossValidationArtifacts,
    DataConfig,
    EvaluationConfig,
    ExperimentConfig,
    TabPFNAccessError,
    TrainingResult,
    aggregate_cross_validation_results,
    load_training_result,
    train_adaptive_rocket_pfn,
)


# -----------------------------------------------------------------------------
# Experiment selection
# -----------------------------------------------------------------------------

RUN_MODE: Literal["single", "cross_validation"] = "cross_validation"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")
DATASET_ROOT = PROJECT_ROOT / "data/trainsets"
DATASET_PREFIX = "target-cue_window-500_splits-10"
OUTPUT_ROOT = PROJECT_ROOT / "outputs/adaptive_rocket_pfn"

SINGLE_FOLD = 0
CROSS_VALIDATION_FOLDS: tuple[int, ...] = tuple(range(10))
SEED = 42


# -----------------------------------------------------------------------------
# Fixed-grid representation
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
# Structured candidate bank
# -----------------------------------------------------------------------------

TEMPORAL_VIEWS = (
    "raw",
    "first_diff",
    "second_diff",
    "smoothed",
    "highpass",
    "local_norm",
)
SMOOTHING_WINDOW = 5
LOCAL_NORMALIZATION_EPSILON = 1e-6

DILATION_REGIMES = (1, 32)
MULTIROCKET_NUM_KERNELS_PER_BANK = 625
MULTIROCKET_NUM_FEATURES_PER_KERNEL = 4
MULTIROCKET_NORMALISE_PER_INSTANCE = False

HYDRA_NUM_KERNELS = 8
HYDRA_NUM_GROUPS = 128
HYDRA_MAX_NUM_CHANNELS = 8

PROTOTYPE_MAX_SHAPELETS = 512
PROTOTYPE_SHAPELET_LENGTHS = (7, 9, 11)
PROTOTYPE_NORMALIZATION_PROBABILITY = 0.8
PROTOTYPE_SIMILARITY = 0.5
TRANSFORM_N_JOBS = 4


# -----------------------------------------------------------------------------
# Nested adaptive selection and semantic ensemble
# -----------------------------------------------------------------------------

SELECTED_FEATURES_PER_EXPERT = 500
ACTIVE_FAMILIES_PER_EXPERT = 12
MINIMUM_FEATURES_PER_ACTIVE_FAMILY = 8
SELECTION_INNER_FOLDS = 3
SELECTION_REPEATS = 2
SELECTION_POOL_MULTIPLIER = 3
ALLOCATION_TEMPERATURE = 0.5
REDUNDANCY_CORRELATION_THRESHOLD = 0.98
REDUNDANCY_SAMPLE_COUNT = 256

OOF_FOLDS = 3
OOF_TABPFN_N_ESTIMATORS = 2
ENSEMBLE_WEIGHT_L2 = 0.05


# -----------------------------------------------------------------------------
# TabPFN-3
# -----------------------------------------------------------------------------

TABPFN_MODEL_PATH = (
    PROJECT_ROOT / "data/tabpfn/tabpfn-v3-classifier-v3_default.ckpt"
)
TABPFN_N_ESTIMATORS = 8
TABPFN_AUTO_SCALE_N_ESTIMATORS = True
TABPFN_DEVICE = "cuda"
TABPFN_FIT_MODE: Literal[
    "low_memory", "fit_preprocessors", "fit_with_cache"
] = "low_memory"
TABPFN_MEMORY_SAVING_MODE: Literal["auto"] | bool = "auto"
TABPFN_INFERENCE_PRECISION: Literal["auto", "autocast"] = "auto"
TABPFN_PREPROCESSING_JOBS = 8
TABPFN_SHOW_PROGRESS_BAR = False
TABPFN_IGNORE_PRETRAINING_LIMITS = True
TABPFN_BALANCE_PROBABILITIES = False

ADAPTIVE_MODEL_CONFIG = AdaptiveRocketPFNConfig(
    views=TEMPORAL_VIEWS,
    smoothing_window=SMOOTHING_WINDOW,
    local_normalization_epsilon=LOCAL_NORMALIZATION_EPSILON,
    dilation_regimes=DILATION_REGIMES,
    multirocket_num_kernels_per_bank=MULTIROCKET_NUM_KERNELS_PER_BANK,
    multirocket_num_features_per_kernel=MULTIROCKET_NUM_FEATURES_PER_KERNEL,
    multirocket_normalise_per_instance=MULTIROCKET_NORMALISE_PER_INSTANCE,
    hydra_num_kernels=HYDRA_NUM_KERNELS,
    hydra_num_groups=HYDRA_NUM_GROUPS,
    hydra_max_num_channels=HYDRA_MAX_NUM_CHANNELS,
    prototype_max_shapelets=PROTOTYPE_MAX_SHAPELETS,
    prototype_shapelet_lengths=PROTOTYPE_SHAPELET_LENGTHS,
    prototype_normalization_probability=PROTOTYPE_NORMALIZATION_PROBABILITY,
    prototype_similarity=PROTOTYPE_SIMILARITY,
    selected_features_per_expert=SELECTED_FEATURES_PER_EXPERT,
    active_families_per_expert=ACTIVE_FAMILIES_PER_EXPERT,
    minimum_features_per_active_family=MINIMUM_FEATURES_PER_ACTIVE_FAMILY,
    selection_inner_folds=SELECTION_INNER_FOLDS,
    selection_repeats=SELECTION_REPEATS,
    selection_pool_multiplier=SELECTION_POOL_MULTIPLIER,
    allocation_temperature=ALLOCATION_TEMPERATURE,
    redundancy_correlation_threshold=REDUNDANCY_CORRELATION_THRESHOLD,
    redundancy_sample_count=REDUNDANCY_SAMPLE_COUNT,
    oof_folds=OOF_FOLDS,
    oof_tabpfn_n_estimators=OOF_TABPFN_N_ESTIMATORS,
    ensemble_weight_l2=ENSEMBLE_WEIGHT_L2,
    tabpfn_model_path=TABPFN_MODEL_PATH,
    tabpfn_n_estimators=TABPFN_N_ESTIMATORS,
    tabpfn_auto_scale_n_estimators=TABPFN_AUTO_SCALE_N_ESTIMATORS,
    tabpfn_device=TABPFN_DEVICE,
    tabpfn_fit_mode=TABPFN_FIT_MODE,
    tabpfn_memory_saving_mode=TABPFN_MEMORY_SAVING_MODE,
    tabpfn_inference_precision=TABPFN_INFERENCE_PRECISION,
    tabpfn_preprocessing_jobs=TABPFN_PREPROCESSING_JOBS,
    tabpfn_show_progress_bar=TABPFN_SHOW_PROGRESS_BAR,
    tabpfn_ignore_pretraining_limits=TABPFN_IGNORE_PRETRAINING_LIMITS,
    tabpfn_balance_probabilities=TABPFN_BALANCE_PROBABILITIES,
    transform_n_jobs=TRANSFORM_N_JOBS,
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


def experiment_variant() -> str:
    return (
        f"adaptive-multirep-k{MULTIROCKET_NUM_KERNELS_PER_BANK}"
        f"-top{SELECTED_FEATURES_PER_EXPERT}-oof{OOF_FOLDS}"
        f"-tabpfn-v3-e{TABPFN_N_ESTIMATORS}"
    )


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
        model=ADAPTIVE_MODEL_CONFIG,
        evaluation=EVALUATION_CONFIG,
        output_dir=experiment_output_dir(),
        run_name=f"fold-{fold}_seed-{SEED}",
        seed=SEED,
    )


def _execute_fold(job: FoldJob) -> tuple[int, TrainingResult]:
    return job.fold, train_adaptive_rocket_pfn(job.config)


def train_single_model(fold: int = SINGLE_FOLD) -> TrainingResult:
    return train_adaptive_rocket_pfn(build_experiment_config(fold))


def _completion_line(prefix: str, fold: int, result: TrainingResult) -> str:
    return (
        f"{prefix} fold {fold}: candidates={result.num_candidate_features:,}, "
        f"selected={result.num_selected_features:,}, "
        f"test_auc={result.test_metrics['roc_auc']:.4f}, "
        f"test_macro_f1={result.test_metrics['macro_f1']:.4f}"
    )


def _run_parallel_fold_jobs(
    jobs: Sequence[FoldJob], worker_count: int
) -> dict[int, TrainingResult]:
    context = multiprocessing.get_context("spawn")
    results: dict[int, TrainingResult] = {}
    with ProcessPoolExecutor(max_workers=worker_count, mp_context=context) as executor:
        futures = {executor.submit(_execute_fold, job): job.fold for job in jobs}
        for future in as_completed(futures):
            expected_fold = futures[future]
            try:
                fold, result = future.result()
            except Exception as error:
                raise RuntimeError(
                    f"Adaptive RocketPFN failed for fold {expected_fold}"
                ) from error
            results[fold] = result
            print(_completion_line("Completed", fold, result), flush=True)
    return results


def train_cross_validation(
    folds: Sequence[int] = CROSS_VALIDATION_FOLDS,
    max_parallel_folds: int = MAX_PARALLEL_FOLDS,
    use_process_pool: bool = USE_PROCESS_POOL,
    resume_completed_folds: bool = RESUME_COMPLETED_FOLDS,
) -> CrossValidationRun:
    if not folds:
        raise ValueError("At least one fold must be selected")
    if len(set(folds)) != len(folds):
        raise ValueError("folds must not contain duplicates")
    if max_parallel_folds < 1:
        raise ValueError("max_parallel_folds must be at least 1")

    jobs = [FoldJob(fold, build_experiment_config(fold)) for fold in folds]
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
            print(f"Not reusing fold {job.fold}: {error}", flush=True)
            pending_jobs.append(job)
        else:
            fold_results[job.fold] = result
            print(_completion_line("Reusing completed", job.fold, result), flush=True)

    worker_count = min(max_parallel_folds, len(pending_jobs))
    if use_process_pool and worker_count > 1:
        fold_results.update(_run_parallel_fold_jobs(pending_jobs, worker_count))
    else:
        for job in pending_jobs:
            print(f"Training Adaptive RocketPFN fold {job.fold} ...", flush=True)
            fold, result = _execute_fold(job)
            fold_results[fold] = result
            print(_completion_line("Completed", fold, result), flush=True)

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
    statistics = artifacts.summary["fold_statistics"]["test"]
    print("\nCross-validation test metrics (fold mean ± sample SD):")
    for metric in SCORE_METRICS:
        values = statistics[metric]
        print(
            f"  {metric:20s} {_format_statistic(values['mean'])} ± "
            f"{_format_statistic(values['standard_deviation'])}"
        )
    print(f"\nSummary: {artifacts.summary_path}")


def main() -> TrainingResult | CrossValidationRun:
    print(f"Adaptive RocketPFN variant: {experiment_variant()}", flush=True)
    if RUN_MODE == "single":
        result = train_single_model()
        print(f"Candidate features: {result.num_candidate_features:,}")
        print(f"Selected features: {result.selected_features_by_expert}")
        print(f"OOF ensemble weights: {result.ensemble_weights}")
        print(f"Validation metrics: {result.validation_metrics}")
        print(f"Test metrics: {result.test_metrics}")
        print(f"Checkpoint: {result.checkpoint_path}")
        return result
    if RUN_MODE == "cross_validation":
        return train_cross_validation()
    raise ValueError(f"Unknown RUN_MODE: {RUN_MODE!r}")


if __name__ == "__main__":
    try:
        main()
    except TabPFNAccessError as error:
        raise SystemExit(f"\nAdaptive RocketPFN setup error:\n{error}") from None
