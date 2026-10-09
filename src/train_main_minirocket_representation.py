"""Editable entry point: MiniRocket on top of the representation toggles.

Runs the same MiniRocket transform+ridge classifier and cross-validation aggregation as
`train_main_minirocket.py`, but on the `RepresentationConfig` channel variants explored
for T2M-GPT/MotionGPT (see `src/representation/README.md`) instead of the identity
representation -- so their effect can be measured on the strongest baseline in this
repository, not only on the two motion-token architectures.

`src/minirocket/` (including `data.py`) is not modified. This script drives MiniRocket
through `src/minirocket_representation.py`, which substitutes MiniRocket's data-loading
step for the shared, representation-aware `t2m_gpt.data.prepare_data` and leaves
everything else -- the transform, the ridge classifier, threshold calibration, metrics,
artifact format -- untouched.

Edit ``VARIANTS`` or the fold/grid settings below, then run::

    python src/train_main_minirocket_representation.py
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from minirocket import (
    DataConfig,
    EvaluationConfig,
    ExperimentConfig,
    MiniRocketConfig,
    SCORE_METRICS,
    TrainingResult,
    aggregate_cross_validation_results,
)

from minirocket_representation import load_matching_result, train_minirocket_with_representation
from representation import RepresentationConfig

# -----------------------------------------------------------------------------
# Experiment selection
# -----------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = PROJECT_ROOT / "data/trainsets"
DATASET_PREFIX = "target-cue_window-500_splits-10"
OUTPUT_ROOT = PROJECT_ROOT / "outputs/minirocket_representation"

CROSS_VALIDATION_FOLDS: tuple[int, ...] = tuple(range(10))
SEED = 42
RESUME_COMPLETED_FOLDS = True


# -----------------------------------------------------------------------------
# Fixed-grid representation
#
# Identical to train_main_minirocket.py's identity baseline (ridge-alpha-1e-3-to-1e6,
# pooled AUROC 0.6862) except for the RepresentationConfig applied per variant below.
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

MINIROCKET_CONFIG = MiniRocketConfig(
    num_kernels=10_000,
    max_dilations_per_kernel=32,
    class_weight=None,
    n_jobs=4,
)
EVALUATION_CONFIG = EvaluationConfig(
    threshold=0.5,
    calibrate_threshold_on_validation=False,
    threshold_metric="macro_f1",
)


# -----------------------------------------------------------------------------
# Representation variants
#
# Same six non-identity variants screened against MotionGPT, so results sit next to
# that comparison. "identity" is included for an in-pipeline parity check against the
# existing native-MiniRocket baseline rather than as a variant to interpret on its own.
# -----------------------------------------------------------------------------

VARIANTS: dict[str, RepresentationConfig] = {
    "identity": RepresentationConfig(),
    "rootrel": RepresentationConfig(root_relative_positions=True),
    "rootrel_drop": RepresentationConfig(
        root_relative_positions=True, drop_absolute_positions=True
    ),
    "au_all": RepresentationConfig(append_action_units=True),
    "au_neg_drop": RepresentationConfig(
        append_action_units=True, action_unit_subset="negation", drop_raw_blendshapes=True
    ),
    "accel_head_hands": RepresentationConfig(
        append_acceleration=True,
        acceleration_modalities=("Head", "LeftHand", "RightHand"),
    ),
    "combo": RepresentationConfig(
        root_relative_positions=True, drop_absolute_positions=True,
        append_action_units=True, append_acceleration=True,
        acceleration_modalities=("Head", "LeftHand", "RightHand", "Facial"),
    ),
}


def dataset_path(fold: int) -> Path:
    return DATASET_ROOT / f"{DATASET_PREFIX}_fold-{fold}"


def experiment_output_dir(variant: str) -> Path:
    return OUTPUT_ROOT / DATASET_PREFIX / variant


def build_minirocket_config(fold: int, variant: str) -> ExperimentConfig:
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
        model=MINIROCKET_CONFIG,
        evaluation=EVALUATION_CONFIG,
        output_dir=experiment_output_dir(variant),
        run_name=f"fold-{fold}_seed-{SEED}",
        seed=SEED,
    )


def train_variant_fold(variant: str, fold: int) -> TrainingResult:
    config = build_minirocket_config(fold, variant)
    representation = VARIANTS[variant]
    if RESUME_COMPLETED_FOLDS:
        try:
            result = load_matching_result(config, representation)
        except (FileNotFoundError, ValueError):
            # ValueError also covers a base-config mismatch unrelated to the
            # representation sidecar (e.g. a differently-tuned MiniRocketConfig);
            # either way, the cache cannot be trusted, so retrain rather than crash.
            pass
        else:
            print(
                f"[{variant}] reusing completed fold {fold}: "
                f"test_auc={result.test_metrics['roc_auc']:.4f}",
                flush=True,
            )
            return result
    print(f"[{variant}] training fold {fold} ...", flush=True)
    result = train_minirocket_with_representation(
        config, str(dataset_path(fold)), representation
    )
    print(
        f"[{variant}] fold {fold}: alpha={result.best_alpha:.6g} "
        f"test_auc={result.test_metrics['roc_auc']:.4f} "
        f"test_macro_f1={result.test_metrics['macro_f1']:.4f}",
        flush=True,
    )
    return result


def train_variant_cross_validation(
    variant: str, folds: tuple[int, ...] = CROSS_VALIDATION_FOLDS
) -> tuple[dict[int, TrainingResult], dict]:
    fold_results = {fold: train_variant_fold(variant, fold) for fold in folds}
    summary_dir = experiment_output_dir(variant) / f"cross_validation_seed-{SEED}"
    artifacts = aggregate_cross_validation_results(
        fold_results=fold_results,
        output_dir=summary_dir,
        threshold=EVALUATION_CONFIG.threshold,
        artifact_prefix=f"{DATASET_PREFIX}_{variant}",
    )
    pooled = artifacts.summary["pooled_out_of_fold"]["metrics"]
    fold_stats = artifacts.summary["fold_statistics"]["test"]["roc_auc"]
    print(
        f"=== {variant}: pooled_auc={pooled['roc_auc']:.4f} "
        f"fold_mean={fold_stats['mean']:.4f} fold_sd={fold_stats['standard_deviation']:.4f} ===",
        flush=True,
    )
    return fold_results, artifacts


def main() -> dict[str, tuple[dict[int, TrainingResult], dict]]:
    return {variant: train_variant_cross_validation(variant) for variant in VARIANTS}


if __name__ == "__main__":
    main()
