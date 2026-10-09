"""Editable entry point for the MotionGPT3-derived diffusion-head classifier.

Same fold data, window, and stage-two optimization settings as ``train_main_t2m_gpt.py``
(imported directly). Set ``TOKENIZATION_MODE`` to switch between the two-axis comparison
this package exists for: ``"continuous"`` embeds ``t2m_gpt_v2``'s VAE latents before
pooling (MotionGPT3's own native continuous representation feeding its diffusion head);
``"discrete"`` embeds this project's existing VQ-VAE tokens instead, feeding the identical
diffusion-head classifier, to test whether the diffusion-based classification mechanism
itself helps independent of which tokenizer produced its input. See
``src/motiongpt3/config.py`` and the approved plan in this session for the full
adaptation rationale (the paper's text branch and cross-modal attention are dropped
entirely -- no captions are available -- leaving the diffusion head as the faithfully
portable mechanism).

``VQVAE_CONFIG``/``VAE_CONFIG`` mirror ``train_main_t2m_gpt.VQVAE_CONFIG`` and
``train_main_t2m_gpt_v2.VAE_CONFIG`` exactly, so tokenizer capacity matches the existing
baselines and only the downstream classification mechanism differs. ``SUMMARIZER_CONFIG``
matches ``train_main_t2m_gpt.GPT_CONFIG``'s ``d_model``/``nhead``/``num_layers`` for the
same reason, but is always non-causal (mean-pooling summarizer, not a generator) and has
no ``head``/``pooling``/``token_corruption_rate`` role. ``DIFFUSION_CONFIG`` is scaled down
from the paper's hidden_dim=1024/1000-step diffusion head to a size proportionate to this
project's small-model scale -- unvalidated starting point, tune from the smoke test.

Edit the configuration block, set ``RUN_MODE`` and ``TOKENIZATION_MODE``, then run::

    python src/train_main_motiongpt3.py
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal

import train_main_t2m_gpt as baseline
from motiongpt3 import (
    DiffusionHeadConfig,
    ExperimentConfig,
    GPTConfig,
    TrainingResult,
    VAEConfig,
    VAETrainingConfig,
    VQVAEConfig,
    VQVAETrainingConfig,
    load_training_result_motiongpt3,
    train_motiongpt3,
)
from motiongpt3.config import DataConfig
from t2m_gpt.aggregation import SCORE_METRICS, CrossValidationArtifacts, aggregate_cross_validation_results


RUN_MODE: Literal["single", "cross_validation"] = "cross_validation"
# Overridable via MOTIONGPT3_TOKENIZATION_MODE so run_all_cue.sh can drive both variants
# without duplicating this file.
TOKENIZATION_MODE: Literal["discrete", "continuous"] = os.environ.get(  # type: ignore[assignment]
    "MOTIONGPT3_TOKENIZATION_MODE", "continuous"
)

DATASET_NAMESPACE = baseline.DATASET_NAMESPACE
DATASET_PREFIX = baseline.DATASET_PREFIX
WINDOW_START_SECONDS = baseline.WINDOW_START_SECONDS
WINDOW_END_SECONDS = baseline.WINDOW_END_SECONDS
PROJECT_ROOT = baseline.PROJECT_ROOT
OUTPUT_ROOT = PROJECT_ROOT / "outputs/motiongpt3"

SINGLE_FOLD = 0
CROSS_VALIDATION_FOLDS: tuple[int, ...] = tuple(range(10))
SEED = baseline.SEED
DEVICE = baseline.DEVICE


def dataset_source(fold: int) -> str:
    return f"{DATASET_NAMESPACE}/{DATASET_PREFIX}_fold-{fold}"


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

SUMMARIZER_CONFIG = GPTConfig(
    causal=False,
    d_model=64,
    nhead=4,
    num_layers=2,
    dim_feedforward=256,
    dropout=0.3,
)

DIFFUSION_CONFIG = DiffusionHeadConfig(
    hidden_dim=256,
    num_resblocks=2,
    num_train_timesteps=100,
    beta_start=1e-4,
    beta_end=0.02,
    dropout=0.1,
    loss_samples=4,
)

RESUME_COMPLETED_FOLDS = True


@dataclass(slots=True)
class CrossValidationRun:
    fold_results: dict[int, TrainingResult]
    artifacts: CrossValidationArtifacts


def experiment_output_dir() -> Path:
    return OUTPUT_ROOT / DATASET_PREFIX / TOKENIZATION_MODE


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
        tokenization_mode=TOKENIZATION_MODE,
        vqvae=VQVAE_CONFIG,
        vqvae_training=VQVAE_TRAINING_CONFIG,
        vae=VAE_CONFIG,
        vae_training=VAE_TRAINING_CONFIG,
        summarizer=SUMMARIZER_CONFIG,
        diffusion=DIFFUSION_CONFIG,
        training=replace(baseline.TRAINING_CONFIG, show_progress=show_progress),
        seed=SEED,
        device=DEVICE,
    )


def train_single_model(fold: int = SINGLE_FOLD) -> TrainingResult:
    return train_motiongpt3(build_experiment_config(fold))


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
                result = load_training_result_motiongpt3(config)
            except FileNotFoundError:
                pass
            except ValueError as error:
                print(f"Not reusing fold {fold}: {error}")
            else:
                fold_results[fold] = result
                print(_completion_line("Reusing completed", fold, result))
                continue

        print(f"Training MotionGPT3 ({TOKENIZATION_MODE}) fold {fold} on {config.device} ...")
        result = train_motiongpt3(config)
        fold_results[fold] = result
        print(_completion_line("Completed", fold, result))

    summary_dir = experiment_output_dir() / f"cross_validation_seed-{SEED}"
    artifacts = aggregate_cross_validation_results(
        fold_results=fold_results,
        output_dir=summary_dir,
        threshold=baseline.TRAINING_CONFIG.threshold,
        artifact_prefix=f"{DATASET_PREFIX}_motiongpt3_{TOKENIZATION_MODE}",
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
