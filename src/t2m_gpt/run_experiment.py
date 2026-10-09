"""An editable direct-call training script for one or all T2M-GPT folds.

Run it as a module from the repository root::

    python -m src.t2m_gpt.run_experiment

or import :func:`run_training` and :func:`run_cross_validation` from a notebook.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path

from .aggregation import CrossValidationArtifacts, aggregate_cross_validation_results
from .config import (
    DataConfig,
    ExperimentConfig,
    GPTConfig,
    PretrainingConfig,
    TrainingConfig,
    VQVAEConfig,
    VQVAETrainingConfig,
)
from .training import TrainingResult, train_t2m_gpt


DATASET_NAMESPACE = "VR-Faces-Neg"
DEFAULT_DATASET_PREFIX = "target-cue_window-500_splits-10"
DEFAULT_OUTPUT_ROOT = Path("outputs/t2m_gpt")


def dataset_source(dataset_prefix: str, fold: int, namespace: str = DATASET_NAMESPACE) -> str:
    """Return the fold identifier passed to the datasets loader."""

    return f"{namespace}/{dataset_prefix}_fold-{fold}"


def build_config(
    dataset_prefix: str = DEFAULT_DATASET_PREFIX,
    fold: int = 0,
    run_name: str | None = None,
    seed: int = 42,
    head: str = "discriminative",
    use_pretraining: bool = True,
    num_time_points: int = 32,
    window_seconds: float = 0.5,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    modalities: tuple[str, ...] | None = None,
    device: str = "auto",
    show_progress: bool = True,
) -> ExperimentConfig:
    """Assemble one fold's configuration with the package defaults."""

    return ExperimentConfig(
        data=DataConfig(
            dataset=dataset_source(dataset_prefix, fold),
            num_time_points=num_time_points,
            window_start_seconds=-window_seconds,
            window_end_seconds=window_seconds,
            modalities=modalities,
        ),
        output_dir=str(output_root / dataset_prefix / head),
        run_name=run_name or f"fold-{fold}_seed-{seed}",
        vqvae=VQVAEConfig(),
        vqvae_training=VQVAETrainingConfig(),
        gpt=GPTConfig(head=head),  # type: ignore[arg-type]
        pretraining=PretrainingConfig(),
        training=TrainingConfig(show_progress=show_progress),
        use_pretraining=use_pretraining,
        seed=seed,
        device=device,
    )


def run_training(
    dataset_prefix: str = DEFAULT_DATASET_PREFIX,
    fold: int = 0,
    seed: int = 42,
    head: str = "discriminative",
    use_pretraining: bool = True,
) -> TrainingResult:
    """Train one fold and return its metrics and artifact paths."""

    config = build_config(
        dataset_prefix=dataset_prefix,
        fold=fold,
        seed=seed,
        head=head,
        use_pretraining=use_pretraining,
    )
    return train_t2m_gpt(config)


def run_cross_validation(
    dataset_prefix: str = DEFAULT_DATASET_PREFIX,
    fold_numbers: Iterable[int] = range(10),
    seed: int = 42,
    head: str = "discriminative",
    use_pretraining: bool = True,
) -> tuple[dict[int, TrainingResult], CrossValidationArtifacts]:
    """Train every requested fold sequentially and aggregate held-out metrics."""

    folds: Sequence[int] = tuple(fold_numbers)
    if not folds:
        raise ValueError("At least one fold must be selected")

    results: dict[int, TrainingResult] = {}
    config: ExperimentConfig | None = None
    for fold in folds:
        config = build_config(
            dataset_prefix=dataset_prefix,
            fold=fold,
            seed=seed,
            head=head,
            use_pretraining=use_pretraining,
        )
        print(f"Training T2M-GPT fold {fold} ...")
        results[fold] = train_t2m_gpt(config)

    assert config is not None
    artifacts = aggregate_cross_validation_results(
        fold_results=results,
        output_dir=Path(config.output_dir) / f"cross_validation_seed-{seed}",
        threshold=config.training.threshold,
        artifact_prefix=f"{dataset_prefix}_{head}",
    )
    return results, artifacts


def main() -> TrainingResult:
    result = run_training()
    print(f"Best epoch: {result.best_epoch}")
    print(f"Model parameters: {result.model_parameter_count}")
    print(f"Tokenizer validation metrics: {result.tokenizer_metrics}")
    print(f"Validation metrics: {result.validation_metrics}")
    print(f"Test metrics: {result.test_metrics}")
    print(f"Checkpoint: {result.checkpoint_path}")
    return result


if __name__ == "__main__":
    main()
