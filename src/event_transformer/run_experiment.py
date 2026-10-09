"""Direct-call experiment script; edit the configuration and run this file.

There is intentionally no argparse interface.  This module can also be imported and
``run_training`` called directly from a notebook or another Python script.
"""

from collections.abc import Sequence
from pathlib import Path

from .config import (
    DataConfig,
    ExperimentConfig,
    ModelConfig,
    PretrainingConfig,
    TrainingConfig,
)
from .training import TrainingResult, train_event_transformer


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def run_training(
    dataset_path: Path = PROJECT_ROOT
    / "data/trainsets/target-cue_window-1000_splits-10_fold-0",
    output_dir: Path = PROJECT_ROOT / "outputs/event_transformer",
    run_name: str = "cue-window1000-fold0-seed42",
    seed: int = 42,
    *,
    use_pretraining: bool = True,
) -> TrainingResult:
    """Configure and train one fold with ordinary Python function arguments."""

    experiment = ExperimentConfig(
        data=DataConfig(
            dataset_path=dataset_path,
            batch_size=8,
            max_events_per_modality=128,
            normalize_features=True,
            cache_in_memory=True,
        ),
        model=ModelConfig(
            d_model=32,
            nhead=4,
            num_layers=1,
            dim_feedforward=128,
            modality_hidden_dim=64,
            modality_fusion_hidden_dim=64,
            time_hidden_dim=32,
            classifier_hidden_dim=16,
            dropout=0.35,
        ),
        pretraining=PretrainingConfig(
            num_epochs=10,
            mask_probability=0.20,
            learning_rate=1e-4,
        ),
        training=TrainingConfig(
            max_epochs=60,
            learning_rate=1e-4,
            classifier_learning_rate=3e-4,
            freeze_backbone_epochs=3,
            weight_decay=1e-2,
            early_stopping_patience=10,
            selection_metric="loss",
            calibrate_threshold_on_validation=False,
            threshold_metric="macro_f1",
        ),
        output_dir=output_dir,
        run_name=run_name,
        seed=seed,
        device="cuda",
    )
    return train_event_transformer(
        experiment,
        use_pretraining=use_pretraining,
    )


def run_cross_validation(
    dataset_prefix: str = "target-cue_window-1000_splits-10",
    fold_numbers: Sequence[int] = tuple(range(10)),
    output_dir: Path = PROJECT_ROOT / "outputs/event_transformer",
    seed: int = 42,
    *,
    use_pretraining: bool = True,
) -> list[TrainingResult]:
    """Train the same configuration on several predefined folds."""

    results = []
    for fold_number in fold_numbers:
        results.append(
            run_training(
                dataset_path=PROJECT_ROOT
                / f"data/trainsets/{dataset_prefix}_fold-{fold_number}",
                output_dir=output_dir,
                run_name=f"{dataset_prefix}-fold{fold_number}-seed{seed}",
                seed=seed,
                use_pretraining=use_pretraining,
            )
        )
    return results


if __name__ == "__main__":
    result = run_training()
    print(f"Best epoch: {result.best_epoch}")
    print(f"Fixed decision threshold: {result.decision_threshold:.4f}")
    print(f"Validation metrics: {result.validation_metrics}")
    print(f"Test metrics: {result.test_metrics}")
    print(f"Checkpoint: {result.checkpoint_path}")
