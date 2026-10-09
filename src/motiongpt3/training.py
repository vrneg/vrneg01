"""Training for the MotionGPT3-derived diffusion-head classifier.

Two variants share this one orchestration, selected by ``config.tokenization_mode``:
``"discrete"`` tokenizes with ``t2m_gpt``'s existing VQ-VAE (stage one reuses
``t2m_gpt.training.train_motion_vqvae``/``tokenize_split`` directly, unmodified);
``"continuous"`` uses ``t2m_gpt_v2``'s VAE instead (stage one reuses
``t2m_gpt_v2.training.train_motion_vae``/``tokenize_split_continuous`` directly). Either
way stage two is identical: embed the frozen sequence, pool it with ``MotionSummarizer``,
and train ``MotionGPT3Classifier`` with the diffusion-head objective -- this isolates
what the diffusion-based classification mechanism itself contributes, independent of
which tokenizer produced its input, mirroring the discrete/continuous ablation already
used by ``ghtt``.

The classifier's training loop follows the same "train on true class only, score by
comparing both classes at eval time" split already used by ``t2m_gpt.model``'s
``head="generative"`` (see ``t2m_gpt.training._classification_loss``): training calls
``MotionGPT3Classifier.training_loss`` (one diffusion forward pass, on the true class),
evaluation additionally calls ``classification_score`` (two forward passes, one per
class) to produce the log-likelihood-ratio-like decision score.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
from torch import nn

try:
    from event_transformer.metrics import binary_classification_metrics, optimize_binary_threshold
    from t2m_gpt.data import MotionTokenDataset
    from t2m_gpt.training import (
        CHECKPOINT_VERSION,
        SPLIT_NAMES,
        EvaluationOutput,
        ProgressCallback,
        TrainingResult,
        _config_signature,
        _detached_state,
        _initial_best_value,
        _is_improvement,
        _json_ready,
        _load_checkpoint,
        _prediction_rows,
        _report,
        _weighted_mean,
        _write_lines,
        resolve_device,
        set_reproducible_seed,
        tokenize_split as tokenize_split_discrete,
        token_statistics,
        train_motion_vqvae,
    )
    from t2m_gpt_v2.data import MotionLatentDataset
    from t2m_gpt_v2.training import (
        latent_statistics,
        tokenize_split_continuous,
        train_motion_vae,
    )
except ModuleNotFoundError as error:  # pragma: no cover - import-path fallback
    if error.name not in {"event_transformer", "t2m_gpt", "t2m_gpt_v2"}:
        raise
    from ..event_transformer.metrics import binary_classification_metrics, optimize_binary_threshold
    from ..t2m_gpt.data import MotionTokenDataset
    from ..t2m_gpt.training import (
        CHECKPOINT_VERSION,
        SPLIT_NAMES,
        EvaluationOutput,
        ProgressCallback,
        TrainingResult,
        _config_signature,
        _detached_state,
        _initial_best_value,
        _is_improvement,
        _json_ready,
        _load_checkpoint,
        _prediction_rows,
        _report,
        _weighted_mean,
        _write_lines,
        resolve_device,
        set_reproducible_seed,
        tokenize_split as tokenize_split_discrete,
        token_statistics,
        train_motion_vqvae,
    )
    from ..t2m_gpt_v2.data import MotionLatentDataset
    from ..t2m_gpt_v2.training import (
        latent_statistics,
        tokenize_split_continuous,
        train_motion_vae,
    )

from .config import ExperimentConfig
from .data import DataBundle, make_loader, prepare_data
from .model import DiffusionHead, MotionGPT3Classifier, MotionSummarizer, count_parameters

# -----------------------------------------------------------------------------
# Stage one: tokenizer (dispatches on tokenization_mode)
# -----------------------------------------------------------------------------


def train_tokenizer_stage(
    data: DataBundle,
    config: ExperimentConfig,
    device: torch.device,
    progress_callback: ProgressCallback | None = None,
) -> tuple[nn.Module, list[dict[str, Any]], dict[str, float]]:
    if config.tokenization_mode == "discrete":
        return train_motion_vqvae(data, config, device, progress_callback)
    return train_motion_vae(data, config, device, progress_callback)


def tokenize_all_splits(
    tokenizer: nn.Module,
    data: DataBundle,
    config: ExperimentConfig,
    device: torch.device,
) -> dict[str, MotionTokenDataset | MotionLatentDataset]:
    batch_size = (
        config.vqvae_training.evaluation_batch_size
        if config.tokenization_mode == "discrete"
        else config.vae_training.evaluation_batch_size
    )
    tokenize = tokenize_split_discrete if config.tokenization_mode == "discrete" else tokenize_split_continuous
    return {split: tokenize(tokenizer, getattr(data, split), batch_size, device) for split in SPLIT_NAMES}


def tokenizer_statistics(
    datasets: dict[str, MotionTokenDataset | MotionLatentDataset], config: ExperimentConfig
) -> dict[str, dict[str, float]]:
    if config.tokenization_mode == "discrete":
        return {
            split: token_statistics(dataset, config.vqvae.num_codes) for split, dataset in datasets.items()
        }
    return {split: latent_statistics(dataset) for split, dataset in datasets.items()}


def build_embedder(config: ExperimentConfig, embedded_dim: int) -> tuple[nn.Module, int]:
    """Returns ``(embedder, source_dim)`` -- an ``nn.Embedding`` for discrete tokens (source_dim
    is unused, embedding dim is fixed at construction) or ``nn.Linear`` for continuous latents."""

    if config.tokenization_mode == "discrete":
        return nn.Embedding(config.vqvae.num_codes, embedded_dim), embedded_dim
    return nn.Linear(config.vae.latent_dim, embedded_dim), embedded_dim


# -----------------------------------------------------------------------------
# Stage two: pooling transformer + diffusion-head classifier
# -----------------------------------------------------------------------------


def _sequence_batch(batch: dict[str, Any], config: ExperimentConfig, device: torch.device) -> torch.Tensor:
    key = "indices" if config.tokenization_mode == "discrete" else "latents"
    return batch[key].to(device, non_blocking=True)


def _build_optimizer(model: MotionGPT3Classifier, config: ExperimentConfig) -> torch.optim.Optimizer:
    settings = config.training
    return torch.optim.AdamW(model.parameters(), lr=settings.learning_rate, weight_decay=settings.weight_decay)


@torch.no_grad()
def evaluate_with_predictions(
    model: MotionGPT3Classifier,
    loader: torch.utils.data.DataLoader,
    config: ExperimentConfig,
    device: torch.device,
    threshold: float | None = None,
) -> EvaluationOutput:
    model.eval()
    all_scores: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []
    sample_ids: list[str] = []
    losses: list[tuple[float, int]] = []

    for batch in loader:
        sequence = _sequence_batch(batch, config, device)
        labels = batch["label"].to(device, non_blocking=True)
        loss = model.training_loss(sequence, labels)
        score = model.classification_score(sequence)
        losses.append((float(loss.item()), int(labels.shape[0])))
        all_scores.append(score.detach().float().cpu())
        all_labels.append(labels.detach().float().cpu())
        sample_ids.extend(batch["sample_id"])

    scores = torch.cat(all_scores) if all_scores else torch.empty(0, dtype=torch.float32)
    labels = torch.cat(all_labels) if all_labels else torch.empty(0, dtype=torch.float32)
    loss = _weighted_mean(losses)
    metrics = binary_classification_metrics(
        scores, labels, loss, config.training.threshold if threshold is None else threshold
    )
    return EvaluationOutput(metrics=metrics, sample_ids=sample_ids, labels=labels, logits=scores)


def train_classifier(
    model: MotionGPT3Classifier,
    sequence_datasets: dict[str, MotionTokenDataset | MotionLatentDataset],
    config: ExperimentConfig,
    device: torch.device,
    progress_callback: ProgressCallback | None = None,
) -> tuple[list[dict[str, Any]], dict[str, torch.Tensor], int]:
    settings = config.training
    pin_memory = device.type == "cuda"
    train_loader = make_loader(
        sequence_datasets["train"],
        batch_size=settings.batch_size,
        shuffle=True,
        seed=config.seed,
        num_workers=settings.num_workers,
        pin_memory=pin_memory,
    )
    validation_loader = make_loader(
        sequence_datasets["validation"],
        batch_size=settings.evaluation_batch_size,
        shuffle=False,
        seed=config.seed,
        num_workers=settings.num_workers,
        pin_memory=pin_memory,
    )

    optimizer = _build_optimizer(model, config)
    scheduler = (
        torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min" if settings.selection_metric == "loss" else "max",
            factor=settings.lr_scheduler_factor,
            patience=settings.lr_scheduler_patience,
            min_lr=settings.minimum_learning_rate,
        )
        if settings.lr_scheduler == "reduce_on_plateau"
        else None
    )

    history: list[dict[str, Any]] = []
    best_value = _initial_best_value(settings.selection_metric)
    best_state = _detached_state(model)
    best_epoch = 0
    epochs_without_improvement = 0

    for epoch in range(1, settings.max_epochs + 1):
        model.train()
        train_losses: list[tuple[float, int]] = []
        for batch in train_loader:
            sequence = _sequence_batch(batch, config, device)
            labels = batch["label"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = model.training_loss(sequence, labels)
            loss.backward()
            if settings.gradient_clip_norm is not None:
                nn.utils.clip_grad_norm_(model.parameters(), settings.gradient_clip_norm)
            optimizer.step()
            train_losses.append((float(loss.item()), int(labels.shape[0])))

        validation_output = evaluate_with_predictions(model, validation_loader, config, device)
        selection_value = validation_output.metrics[settings.selection_metric]
        if scheduler is not None:
            scheduler.step(selection_value)

        record = {
            "stage": "classification",
            "epoch": epoch,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "train_loss": _weighted_mean(train_losses),
            **{f"validation_{key}": value for key, value in validation_output.metrics.items()},
        }

        if _is_improvement(
            settings.selection_metric, selection_value, best_value, settings.early_stopping_min_delta
        ):
            best_value = selection_value
            best_state = _detached_state(model)
            best_epoch = epoch
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        record["best_epoch"] = best_epoch
        history.append(record)
        _report(
            progress_callback,
            settings.show_progress,
            record,
            f"[motiongpt3-{config.tokenization_mode}] epoch {epoch:3d}/{settings.max_epochs} "
            f"train_loss={record['train_loss']:.4f} "
            f"val_loss={validation_output.metrics['loss']:.4f} "
            f"val_auc={validation_output.metrics['roc_auc']:.4f} "
            f"val_macro_f1={validation_output.metrics['macro_f1']:.4f} "
            f"best={best_epoch}",
        )

        if epochs_without_improvement >= settings.early_stopping_patience:
            break

    if settings.show_progress and progress_callback is None:
        print()
    return history, best_state, best_epoch


# -----------------------------------------------------------------------------
# Orchestration
# -----------------------------------------------------------------------------


def _tokenizer_checkpoint(
    tokenizer: nn.Module,
    config: ExperimentConfig,
    data: DataBundle,
    history: list[dict[str, Any]],
    metrics: dict[str, float],
) -> dict[str, Any]:
    tokenizer_config = asdict(config.vqvae) if config.tokenization_mode == "discrete" else asdict(config.vae)
    return {
        "checkpoint_version": CHECKPOINT_VERSION,
        "checkpoint_type": f"motiongpt3_tokenizer_{config.tokenization_mode}",
        "model_state_dict": tokenizer.state_dict(),
        "tokenizer_config": _json_ready(tokenizer_config),
        "experiment_config": _config_signature(config),
        "normalizer": None if data.normalizer is None else data.normalizer.state_dict(),
        "time_grid": data.time_grid,
        "channel_names": data.channel_names,
        "input_dim": data.num_channels,
        "num_frames": data.num_frames,
        "history": history,
        "validation_metrics": metrics,
    }


def _checkpoint(
    model_state: dict[str, torch.Tensor],
    config: ExperimentConfig,
    data: DataBundle,
    num_tokens: int,
    best_epoch: int,
    decision_threshold: float,
    validation_metrics: dict[str, float],
    test_metrics: dict[str, float],
    model_parameter_count: int,
    history: list[dict[str, Any]],
    tokenizer_history: list[dict[str, Any]],
    tokenizer_metrics: dict[str, float],
    tokenizer_statistics_by_split: dict[str, dict[str, float]],
) -> dict[str, Any]:
    return {
        "checkpoint_version": CHECKPOINT_VERSION,
        "checkpoint_type": "motiongpt3_diffusion_classifier",
        "model_state_dict": model_state,
        "experiment_config": _config_signature(config),
        "normalizer": None if data.normalizer is None else data.normalizer.state_dict(),
        "time_grid": data.time_grid,
        "channel_names": data.channel_names,
        "num_tokens": num_tokens,
        "tokenization_mode": config.tokenization_mode,
        "label_convention": {config.data.negative_label: 0, config.data.positive_label: 1},
        "best_epoch": best_epoch,
        "decision_threshold": decision_threshold,
        "validation_metrics": validation_metrics,
        "test_metrics": test_metrics,
        "model_parameter_count": model_parameter_count,
        "history": history,
        "tokenizer_history": tokenizer_history,
        "tokenizer_metrics": tokenizer_metrics,
        "tokenizer_statistics": tokenizer_statistics_by_split,
    }


def train_motiongpt3(
    config: ExperimentConfig,
    progress_callback: ProgressCallback | None = None,
) -> TrainingResult:
    """Run both stages for one fold and evaluate the test split exactly once."""

    config.validate()
    set_reproducible_seed(config.seed, config.training.deterministic_algorithms)
    device = resolve_device(config.device)

    run_dir = Path(config.output_dir) / config.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    tokenizer_path = run_dir / "tokenizer.pt"
    checkpoint_path = run_dir / "best_model.pt"
    history_path = run_dir / "metrics.json"
    validation_predictions_path = run_dir / "validation_predictions.jsonl"
    test_predictions_path = run_dir / "test_predictions.jsonl"

    data = prepare_data(config.data)

    tokenizer, tokenizer_history, tokenizer_metrics = train_tokenizer_stage(
        data, config, device, progress_callback
    )
    torch.save(_tokenizer_checkpoint(tokenizer, config, data, tokenizer_history, tokenizer_metrics), tokenizer_path)

    sequence_datasets = tokenize_all_splits(tokenizer, data, config, device)
    tokenizer_statistics_by_split = tokenizer_statistics(sequence_datasets, config)
    num_tokens = sequence_datasets["train"].num_tokens

    set_reproducible_seed(config.seed, config.training.deterministic_algorithms)
    embedder, embedded_dim = build_embedder(config, config.summarizer.d_model)
    summarizer = MotionSummarizer(embedder, embedded_dim, num_tokens, config.summarizer)
    model = MotionGPT3Classifier(summarizer, config.summarizer.d_model, config.diffusion).to(device)

    history, best_state, best_epoch = train_classifier(
        model, sequence_datasets, config, device, progress_callback
    )
    model.load_state_dict(best_state)

    pin_memory = device.type == "cuda"
    validation_loader = make_loader(
        sequence_datasets["validation"],
        batch_size=config.training.evaluation_batch_size,
        shuffle=False,
        seed=config.seed,
        num_workers=config.training.num_workers,
        pin_memory=pin_memory,
    )
    test_loader = make_loader(
        sequence_datasets["test"],
        batch_size=config.training.evaluation_batch_size,
        shuffle=False,
        seed=config.seed,
        num_workers=config.training.num_workers,
        pin_memory=pin_memory,
    )

    validation_output = evaluate_with_predictions(model, validation_loader, config, device)
    decision_threshold = config.training.threshold
    if config.training.calibrate_threshold_on_validation:
        decision_threshold = optimize_binary_threshold(
            validation_output.logits, validation_output.labels, config.training.threshold_metric
        )
        validation_output = EvaluationOutput(
            metrics=binary_classification_metrics(
                validation_output.logits,
                validation_output.labels,
                validation_output.metrics["loss"],
                decision_threshold,
            ),
            sample_ids=validation_output.sample_ids,
            labels=validation_output.labels,
            logits=validation_output.logits,
        )
    test_output = evaluate_with_predictions(model, test_loader, config, device, threshold=decision_threshold)

    _write_lines(validation_predictions_path, _prediction_rows(validation_output, decision_threshold))
    _write_lines(test_predictions_path, _prediction_rows(test_output, decision_threshold))

    model_parameter_count = count_parameters(model, trainable_only=False)
    torch.save(
        _checkpoint(
            model_state=best_state,
            config=config,
            data=data,
            num_tokens=num_tokens,
            best_epoch=best_epoch,
            decision_threshold=decision_threshold,
            validation_metrics=validation_output.metrics,
            test_metrics=test_output.metrics,
            model_parameter_count=model_parameter_count,
            history=history,
            tokenizer_history=tokenizer_history,
            tokenizer_metrics=tokenizer_metrics,
            tokenizer_statistics_by_split=tokenizer_statistics_by_split,
        ),
        checkpoint_path,
    )

    history_path.write_text(
        json.dumps(
            _json_ready(
                {
                    "run_name": config.run_name,
                    "experiment_config": _config_signature(config),
                    "num_channels": data.num_channels,
                    "num_frames": data.num_frames,
                    "num_tokens": num_tokens,
                    "tokenization_mode": config.tokenization_mode,
                    "split_sizes": {split: len(sequence_datasets[split]) for split in SPLIT_NAMES},
                    "best_epoch": best_epoch,
                    "decision_threshold": decision_threshold,
                    "model_parameter_count": model_parameter_count,
                    "tokenizer_validation_metrics": tokenizer_metrics,
                    "tokenizer_statistics": tokenizer_statistics_by_split,
                    "validation_metrics": validation_output.metrics,
                    "test_metrics": test_output.metrics,
                    "tokenizer_history": tokenizer_history,
                    "history": history,
                }
            ),
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    return TrainingResult(
        checkpoint_path=checkpoint_path,
        history_path=history_path,
        validation_predictions_path=validation_predictions_path,
        predictions_path=test_predictions_path,
        run_name=config.run_name,
        best_epoch=best_epoch,
        decision_threshold=decision_threshold,
        validation_metrics=validation_output.metrics,
        test_metrics=test_output.metrics,
        model_parameter_count=model_parameter_count,
        history=history,
        tokenizer_checkpoint_path=tokenizer_path,
        tokenizer_history=tokenizer_history,
        tokenizer_metrics=tokenizer_metrics,
    )


def load_training_result_motiongpt3(config: ExperimentConfig) -> TrainingResult:
    """Reload a finished fold, refusing checkpoints from a different configuration."""

    config.validate()
    run_dir = Path(config.output_dir) / config.run_name
    checkpoint = _load_checkpoint(run_dir / "best_model.pt")
    if checkpoint.get("experiment_config") != _config_signature(config):
        raise ValueError(f"Checkpoint in {run_dir} was produced by a different configuration")
    for required in ("validation_predictions.jsonl", "test_predictions.jsonl", "metrics.json"):
        if not (run_dir / required).exists():
            raise FileNotFoundError(f"Missing {required} in {run_dir}")

    return TrainingResult(
        checkpoint_path=run_dir / "best_model.pt",
        history_path=run_dir / "metrics.json",
        validation_predictions_path=run_dir / "validation_predictions.jsonl",
        predictions_path=run_dir / "test_predictions.jsonl",
        run_name=config.run_name,
        best_epoch=int(checkpoint["best_epoch"]),
        decision_threshold=float(checkpoint["decision_threshold"]),
        validation_metrics=dict(checkpoint["validation_metrics"]),
        test_metrics=dict(checkpoint["test_metrics"]),
        model_parameter_count=int(checkpoint["model_parameter_count"]),
        history=list(checkpoint["history"]),
        tokenizer_checkpoint_path=run_dir / "tokenizer.pt",
        tokenizer_history=list(checkpoint.get("tokenizer_history", [])),
        tokenizer_metrics=dict(checkpoint.get("tokenizer_metrics", {})),
    )


__all__ = [
    "DiffusionHead",
    "EvaluationOutput",
    "MotionGPT3Classifier",
    "MotionSummarizer",
    "TrainingResult",
    "build_embedder",
    "evaluate_with_predictions",
    "load_training_result_motiongpt3",
    "resolve_device",
    "set_reproducible_seed",
    "tokenize_all_splits",
    "tokenizer_statistics",
    "train_classifier",
    "train_motiongpt3",
    "train_tokenizer_stage",
]
