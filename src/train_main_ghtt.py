"""Editable entry point for the GHTT-derived hierarchical short/long motion model.

Same fold data, window, and stage-two optimization settings as ``train_main_t2m_gpt.py``
(imported directly). Set ``POSE_BLOCK_MODE`` to switch between the two-axis comparison
this package exists for: ``"vae"`` is GHTT's own native continuous per-clip bottleneck;
``"vqvae"`` replaces it with this project's existing discrete VQ-VAE applied per clip,
feeding the identical action-block hierarchy, to test whether the short/long cascade
itself helps independent of the pose block's bottleneck type. See ``src/ghtt/config.py``
and the approved plan in this session for the full adaptation rationale.

``CLIP_LENGTH=8`` with ``NUM_TIME_POINTS=32`` gives 4 clips per window (matching T2M-
GPT's own 8-motion-token sequence length closely enough for a fair action-block capacity
comparison) and, in "vqvae" mode, ``num_downsample_layers=3`` on ``POSE_VQVAE_CONFIG``
(temporal_downsample_factor=8=CLIP_LENGTH) so each clip yields exactly one token.

Edit the configuration block, set ``RUN_MODE`` and ``POSE_BLOCK_MODE``, then run::

    python src/train_main_ghtt.py
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal

import train_main_t2m_gpt as baseline
from ghtt import (
    ActionBlockConfig,
    ExperimentConfig,
    PoseBlockConfig,
    PoseTrainingConfig,
    TrainingResult,
    VQVAEConfig,
    load_training_result_ghtt,
    train_ghtt,
)
from ghtt.config import DataConfig
from t2m_gpt.aggregation import SCORE_METRICS, CrossValidationArtifacts, aggregate_cross_validation_results


RUN_MODE: Literal["single", "cross_validation"] = "cross_validation"
# Overridable via GHTT_POSE_BLOCK_MODE so run_all_cue.sh can drive both variants
# without duplicating this file.
POSE_BLOCK_MODE: Literal["vae", "vqvae"] = os.environ.get("GHTT_POSE_BLOCK_MODE", "vae")  # type: ignore[assignment]

DATASET_NAMESPACE = baseline.DATASET_NAMESPACE
DATASET_PREFIX = baseline.DATASET_PREFIX
WINDOW_START_SECONDS = baseline.WINDOW_START_SECONDS
WINDOW_END_SECONDS = baseline.WINDOW_END_SECONDS
PROJECT_ROOT = baseline.PROJECT_ROOT
OUTPUT_ROOT = PROJECT_ROOT / "outputs/ghtt"

SINGLE_FOLD = 0
CROSS_VALIDATION_FOLDS: tuple[int, ...] = tuple(range(10))
SEED = baseline.SEED
DEVICE = baseline.DEVICE


def dataset_source(fold: int) -> str:
    return f"{DATASET_NAMESPACE}/{DATASET_PREFIX}_fold-{fold}"


CLIP_LENGTH = 8

POSE_BLOCK_CONFIG = PoseBlockConfig(
    clip_length=CLIP_LENGTH,
    latent_dim=32,
    d_model=64,
    nhead=4,
    num_layers=2,
    dim_feedforward=256,
    dropout=0.1,
    component_loss_weight=1.0,
    trajectory_loss_weight=1.0,
    kl_weight=1e-5,
)

POSE_VQVAE_CONFIG = VQVAEConfig(
    num_codes=128,
    code_dim=32,
    width=128,
    num_downsample_layers=3,  # 2**3 = 8 = CLIP_LENGTH: exactly one token per clip
    num_residual_blocks=2,
    dilation_growth_rate=3,
    dropout=0.0,
)

POSE_TRAINING_CONFIG = PoseTrainingConfig(
    max_epochs=100,
    batch_size=32,
    evaluation_batch_size=64,
    learning_rate=2e-4,
    early_stopping_patience=12,
    lr_scheduler_patience=6,
)

ACTION_BLOCK_CONFIG = ActionBlockConfig(
    d_model=64,
    nhead=4,
    num_layers=2,
    dim_feedforward=256,
    dropout=0.3,
    latent_dim=32,
    classifier_hidden_dim=32,
    mid_reconstruction_weight=1.0,
    classification_weight=0.1,
    kl_weight=1e-5,
)

RESUME_COMPLETED_FOLDS = True


@dataclass(slots=True)
class CrossValidationRun:
    fold_results: dict[int, TrainingResult]
    artifacts: CrossValidationArtifacts


def experiment_output_dir() -> Path:
    return OUTPUT_ROOT / DATASET_PREFIX / POSE_BLOCK_MODE


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
        pose_block_mode=POSE_BLOCK_MODE,
        pose_block=POSE_BLOCK_CONFIG,
        pose_vqvae=POSE_VQVAE_CONFIG,
        pose_training=POSE_TRAINING_CONFIG,
        action_block=ACTION_BLOCK_CONFIG,
        training=replace(baseline.TRAINING_CONFIG, show_progress=show_progress),
        seed=SEED,
        device=DEVICE,
    )


def train_single_model(fold: int = SINGLE_FOLD) -> TrainingResult:
    return train_ghtt(build_experiment_config(fold))


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
    fold_results: dict[int, TrainingResult] = {}
    for fold in folds:
        config = build_experiment_config(fold)
        if resume_completed_folds:
            try:
                result = load_training_result_ghtt(config)
            except FileNotFoundError:
                pass
            except ValueError as error:
                print(f"Not reusing fold {fold}: {error}")
            else:
                fold_results[fold] = result
                print(_completion_line("Reusing completed", fold, result))
                continue

        print(f"Training GHTT ({POSE_BLOCK_MODE}) fold {fold} on {config.device} ...")
        result = train_ghtt(config)
        fold_results[fold] = result
        print(_completion_line("Completed", fold, result))

    summary_dir = experiment_output_dir() / f"cross_validation_seed-{SEED}"
    artifacts = aggregate_cross_validation_results(
        fold_results=fold_results,
        output_dir=summary_dir,
        threshold=baseline.TRAINING_CONFIG.threshold,
        artifact_prefix=f"{DATASET_PREFIX}_ghtt_{POSE_BLOCK_MODE}",
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
        print(f"Pose validation metrics: {result.tokenizer_metrics}")
        print(f"Validation metrics: {result.validation_metrics}")
        print(f"Test metrics: {result.test_metrics}")
        print(f"Checkpoint: {result.checkpoint_path}")
        return result
    if RUN_MODE == "cross_validation":
        return train_cross_validation()
    raise ValueError(f"Unknown RUN_MODE: {RUN_MODE!r}")


if __name__ == "__main__":
    main()
