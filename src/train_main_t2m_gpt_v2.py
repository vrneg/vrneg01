"""Editable entry point for the continuous-latent VAE variant of T2M-GPT.

Same fold data, actor scope, window, and stage-two transformer as
``train_main_t2m_gpt.py`` (imported directly, so the two cannot drift apart) -- the only
variable this isolates is the stage-one bottleneck: a continuous VAE latent
(``t2m_gpt_v2``) in place of the discrete VQ-VAE codebook (``t2m_gpt``). See
``src/t2m_gpt_v2/README.md`` and the approved plan in this session for why this
comparison is the direct test of "does hard quantization lose subtle-motion signal."

``VAE_CONFIG``'s width/downsampling/residual-block/dilation settings mirror
``train_main_t2m_gpt.VQVAE_CONFIG`` exactly, and ``latent_dim=64`` matches its
``code_dim``, so the encoder/decoder capacity is held constant and only the bottleneck
type differs. ``kl_weight`` has no discrete analogue and is not yet validated against
this dataset -- treat the smoke test's ``active_latent_dim_fraction`` diagnostic as the
first signal of whether it needs adjusting (too low: posterior collapse, raise
``free_bits`` or lower ``kl_weight``; stays near 1.0 with poor reconstruction: raise it).

Edit the configuration block, set ``RUN_MODE``, then run::

    python src/train_main_t2m_gpt_v2.py
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal

import train_main_t2m_gpt as baseline
from t2m_gpt_v2 import (
    ExperimentConfig,
    TrainingResult,
    VAEConfig,
    VAETrainingConfig,
    load_training_result_v2,
    train_t2m_gpt_v2,
)
from t2m_gpt_v2.config import DataConfig
from t2m_gpt.aggregation import SCORE_METRICS, CrossValidationArtifacts, aggregate_cross_validation_results


RUN_MODE: Literal["single", "cross_validation"] = "cross_validation"

DATASET_NAMESPACE = "VR-Faces-Neg"
DATASET_PREFIX = baseline.DATASET_PREFIX
WINDOW_START_SECONDS = baseline.WINDOW_START_SECONDS
WINDOW_END_SECONDS = baseline.WINDOW_END_SECONDS
PROJECT_ROOT = baseline.PROJECT_ROOT
OUTPUT_ROOT = PROJECT_ROOT / "outputs/t2m_gpt_v2"
EXPERIMENT_VARIANT = "discriminative"

SINGLE_FOLD = 0
CROSS_VALIDATION_FOLDS: tuple[int, ...] = tuple(range(10))
SEED = baseline.SEED
DEVICE = baseline.DEVICE


def dataset_source(fold: int) -> str:
    return f"{DATASET_NAMESPACE}/{DATASET_PREFIX}_fold-{fold}"


# -----------------------------------------------------------------------------
# Stage one: continuous motion VAE (architecturally matched to VQVAE_CONFIG so the
# comparison isolates discrete-vs-continuous, not capacity)
# -----------------------------------------------------------------------------

VAE_CONFIG = VAEConfig(
    latent_dim=64,
    width=128,
    num_downsample_layers=2,
    num_residual_blocks=2,
    dilation_growth_rate=3,
    dropout=0.0,
)

VAE_TRAINING_CONFIG = VAETrainingConfig(
    max_epochs=100,
    batch_size=32,
    evaluation_batch_size=64,
    learning_rate=2e-4,
    reconstruction_loss="smooth_l1",
    velocity_loss_weight=0.5,
    kl_weight=1e-4,
    free_bits=0.0,
    early_stopping_patience=12,
    lr_scheduler_patience=6,
)


# -----------------------------------------------------------------------------
# Fold-level execution
# -----------------------------------------------------------------------------

RESUME_COMPLETED_FOLDS = True


@dataclass(slots=True)
class CrossValidationRun:
    fold_results: dict[int, TrainingResult]
    artifacts: CrossValidationArtifacts


def experiment_output_dir() -> Path:
    return OUTPUT_ROOT / DATASET_PREFIX / EXPERIMENT_VARIANT


def build_experiment_config(
    fold: int,
    *,
    show_progress: bool = baseline.TRAINING_CONFIG.show_progress,
) -> ExperimentConfig:
    return ExperimentConfig(
        data=DataConfig(
            dataset=dataset_source(fold),
            num_time_points=baseline.NUM_TIME_POINTS,
            window_start_seconds=WINDOW_START_SECONDS,
            window_end_seconds=WINDOW_END_SECONDS,
            max_events_per_modality=baseline.MAX_EVENTS_PER_MODALITY,
            normalize_features=baseline.NORMALIZE_FEATURES,
            cache_in_memory=baseline.CACHE_DATASET_IN_MEMORY,
            actor_scope=baseline.ACTOR_SCOPE,
            include_presence_channels=baseline.INCLUDE_PRESENCE_CHANNELS,
            modalities=baseline.MODALITIES,
            positive_label=baseline.POSITIVE_LABEL,
            negative_label=baseline.NEGATIVE_LABEL,
        ),
        output_dir=str(experiment_output_dir()),
        run_name=f"fold-{fold}_seed-{SEED}",
        vae=VAE_CONFIG,
        vae_training=VAE_TRAINING_CONFIG,
        gpt=baseline.GPT_CONFIG,
        pretraining=baseline.PRETRAINING_CONFIG,
        training=replace(baseline.TRAINING_CONFIG, show_progress=show_progress),
        use_pretraining=baseline.USE_PRETRAINING,
        seed=SEED,
        device=DEVICE,
    )


def train_single_model(fold: int = SINGLE_FOLD) -> TrainingResult:
    return train_t2m_gpt_v2(build_experiment_config(fold))


def _completion_line(prefix: str, fold: int, result: TrainingResult) -> str:
    return (
        f"{prefix} fold {fold}: best_epoch={result.best_epoch}, "
        f"test_auc={result.test_metrics['roc_auc']:.4f}, "
        f"test_macro_f1={result.test_metrics['macro_f1']:.4f}"
    )


def train_cross_validation(
    folds: Sequence[int] = CROSS_VALIDATION_FOLDS,
    resume_completed_folds: bool = RESUME_COMPLETED_FOLDS,
) -> CrossValidationRun:
    """Train folds and produce fold-wise and pooled held-out metrics."""

    fold_results: dict[int, TrainingResult] = {}
    for fold in folds:
        config = build_experiment_config(fold)
        if resume_completed_folds:
            try:
                result = load_training_result_v2(config)
            except FileNotFoundError:
                pass
            except ValueError as error:
                print(f"Not reusing fold {fold}: {error}")
            else:
                fold_results[fold] = result
                print(_completion_line("Reusing completed", fold, result))
                continue

        print(f"Training T2M-GPT-v2 fold {fold} on {config.device} ...")
        result = train_t2m_gpt_v2(config)
        fold_results[fold] = result
        print(_completion_line("Completed", fold, result))

    summary_dir = experiment_output_dir() / f"cross_validation_seed-{SEED}"
    artifacts = aggregate_cross_validation_results(
        fold_results=fold_results,
        output_dir=summary_dir,
        threshold=baseline.TRAINING_CONFIG.threshold,
        artifact_prefix=f"{DATASET_PREFIX}_{EXPERIMENT_VARIANT}_v2",
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
        print(f"VAE validation metrics: {result.tokenizer_metrics}")
        print(f"Validation metrics: {result.validation_metrics}")
        print(f"Test metrics: {result.test_metrics}")
        print(f"Checkpoint: {result.checkpoint_path}")
        return result
    if RUN_MODE == "cross_validation":
        return train_cross_validation()
    raise ValueError(f"Unknown RUN_MODE: {RUN_MODE!r}")


if __name__ == "__main__":
    main()
