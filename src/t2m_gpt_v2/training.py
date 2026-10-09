"""Two-stage training for the continuous-latent variant: VAE, then transformer over z.

Mirrors ``t2m_gpt.training``'s two-stage structure. Genuinely generic helpers that never
branch on discreteness (seeding, device resolution, dict/JSON plumbing, optimizer
param-group construction, prediction-row formatting, and the ``TrainingResult``/
``EvaluationOutput`` containers themselves) are imported directly from ``t2m_gpt.training``
rather than duplicated. Only the pieces that are actually different because the
representation is continuous -- the stage-one loss (KL instead of commitment),
tokenization (a deterministic latent instead of a code index), and the stage-two model's
forward signature -- are new.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional as functional

try:
    from event_transformer.metrics import (
        binary_classification_metrics,
        optimize_binary_threshold,
    )
    from t2m_gpt.vqvae import reconstruction_losses
    from t2m_gpt.training import (
        CHECKPOINT_VERSION,
        SPLIT_NAMES,
        EvaluationOutput,
        ProgressCallback,
        TrainingResult,
        _autocast,
        _build_optimizer,
        _config_signature,
        _detached_state,
        _initial_best_value,
        _is_improvement,
        _json_ready,
        _load_checkpoint,
        _prediction_rows,
        _report,
        _set_backbone_requires_grad,
        _weighted_mean,
        _write_lines,
        resolve_device,
        set_reproducible_seed,
    )
except ModuleNotFoundError as error:  # pragma: no cover - import-path fallback
    if error.name not in {"event_transformer", "t2m_gpt"}:
        raise
    from ..event_transformer.metrics import (
        binary_classification_metrics,
        optimize_binary_threshold,
    )
    from ..t2m_gpt.vqvae import reconstruction_losses
    from ..t2m_gpt.training import (
        CHECKPOINT_VERSION,
        SPLIT_NAMES,
        EvaluationOutput,
        ProgressCallback,
        TrainingResult,
        _autocast,
        _build_optimizer,
        _config_signature,
        _detached_state,
        _initial_best_value,
        _is_improvement,
        _json_ready,
        _load_checkpoint,
        _prediction_rows,
        _report,
        _set_backbone_requires_grad,
        _weighted_mean,
        _write_lines,
        resolve_device,
        set_reproducible_seed,
    )

from .config import ExperimentConfig
from .data import DataBundle, FixedGridSplit, MotionLatentDataset, MotionWindowDataset, make_loader, prepare_data
from .model import MotionLatentGPT, count_parameters
from .vae import MotionVAE, kl_divergence_loss

# -----------------------------------------------------------------------------
# Stage one: motion VAE
# -----------------------------------------------------------------------------


def _vae_batch_losses(
    model: MotionVAE,
    values: torch.Tensor,
    config: ExperimentConfig,
) -> tuple[torch.Tensor, dict[str, float]]:
    settings = config.vae_training
    output = model(values)
    frame_loss, velocity_loss = reconstruction_losses(
        output.reconstruction, values, settings.reconstruction_loss
    )
    kl = kl_divergence_loss(output.mean, output.log_variance, settings.free_bits)
    total = frame_loss + settings.velocity_loss_weight * velocity_loss + settings.kl_weight * kl
    statistics = {
        "loss": float(total.item()),
        "frame_loss": float(frame_loss.item()),
        "velocity_loss": float(velocity_loss.item()),
        "kl_divergence": float(kl.item()),
    }
    return total, statistics


@torch.no_grad()
def evaluate_vae(
    model: MotionVAE,
    loader: torch.utils.data.DataLoader,
    config: ExperimentConfig,
    device: torch.device,
) -> dict[str, float]:
    """Reconstruction/KL quality on one split, plus a posterior-collapse diagnostic.

    ``active_latent_dim_fraction`` is the continuous analogue of the discrete
    tokenizer's ``codebook_usage_fraction``: the fraction of latent dimensions whose
    posterior mean actually varies across the split, rather than the fraction of
    codebook entries in use.
    """

    model.eval()
    accumulated: dict[str, list[tuple[float, int]]] = {}
    all_means: list[torch.Tensor] = []
    for batch in loader:
        values = batch["values"].to(device, non_blocking=True)
        _, statistics = _vae_batch_losses(model, values, config)
        weight = int(values.shape[0])
        for key, value in statistics.items():
            accumulated.setdefault(key, []).append((value, weight))
        all_means.append(
            model.encode_latents(values).permute(0, 2, 1).reshape(-1, model.config.latent_dim).cpu()
        )

    metrics = {key: _weighted_mean(value) for key, value in accumulated.items()}
    if all_means:
        stacked = torch.cat(all_means, dim=0)
        per_dimension_std = stacked.std(dim=0)
        metrics["active_latent_dim_fraction"] = float(
            (per_dimension_std > 1e-2).float().mean().item()
        )
    else:
        metrics["active_latent_dim_fraction"] = 0.0
    return metrics


def train_motion_vae(
    data: DataBundle,
    config: ExperimentConfig,
    device: torch.device,
    progress_callback: ProgressCallback | None = None,
) -> tuple[MotionVAE, list[dict[str, Any]], dict[str, float]]:
    """Fit the VAE on training windows and select on validation ELBO."""

    settings = config.vae_training
    model = MotionVAE(data.num_channels, config.vae).to(device)
    pin_memory = device.type == "cuda"
    train_loader = make_loader(
        MotionWindowDataset(data.train),
        batch_size=settings.batch_size,
        shuffle=True,
        seed=config.seed,
        num_workers=settings.num_workers,
        pin_memory=pin_memory,
    )
    validation_loader = make_loader(
        MotionWindowDataset(data.validation),
        batch_size=settings.evaluation_batch_size,
        shuffle=False,
        seed=config.seed,
        num_workers=settings.num_workers,
        pin_memory=pin_memory,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=settings.learning_rate, weight_decay=settings.weight_decay
    )
    scheduler = (
        torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=settings.lr_scheduler_factor,
            patience=settings.lr_scheduler_patience,
            min_lr=settings.minimum_learning_rate,
        )
        if settings.lr_scheduler == "reduce_on_plateau"
        else None
    )

    history: list[dict[str, Any]] = []
    best_value = math.inf
    best_state = _detached_state(model)
    best_epoch = 0
    best_metrics: dict[str, float] = {}
    epochs_without_improvement = 0

    for epoch in range(1, settings.max_epochs + 1):
        model.train()
        batch_statistics: dict[str, list[tuple[float, int]]] = {}
        for batch in train_loader:
            values = batch["values"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            total, statistics = _vae_batch_losses(model, values, config)
            total.backward()
            if settings.gradient_clip_norm is not None:
                nn.utils.clip_grad_norm_(model.parameters(), settings.gradient_clip_norm)
            optimizer.step()
            weight = int(values.shape[0])
            for key, value in statistics.items():
                batch_statistics.setdefault(key, []).append((value, weight))

        train_metrics = {key: _weighted_mean(value) for key, value in batch_statistics.items()}
        validation_metrics = evaluate_vae(model, validation_loader, config, device)
        if scheduler is not None:
            scheduler.step(validation_metrics["loss"])

        record = {
            "stage": "vae",
            "epoch": epoch,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            **{f"train_{key}": value for key, value in train_metrics.items()},
            **{f"validation_{key}": value for key, value in validation_metrics.items()},
        }

        if _is_improvement(
            "loss", validation_metrics["loss"], best_value, settings.early_stopping_min_delta
        ):
            best_value = validation_metrics["loss"]
            best_state = _detached_state(model)
            best_epoch = epoch
            best_metrics = dict(validation_metrics)
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        record["best_epoch"] = best_epoch
        history.append(record)
        _report(
            progress_callback,
            config.training.show_progress,
            record,
            f"[vae] epoch {epoch:3d}/{settings.max_epochs} "
            f"train_loss={train_metrics['loss']:.4f} "
            f"val_loss={validation_metrics['loss']:.4f} "
            f"kl={validation_metrics['kl_divergence']:.4f} "
            f"active_dims={validation_metrics['active_latent_dim_fraction']:.2f} "
            f"best={best_epoch}",
        )

        if epochs_without_improvement >= settings.early_stopping_patience:
            break

    if config.training.show_progress and progress_callback is None:
        print()
    model.load_state_dict(best_state)
    best_metrics["best_epoch"] = float(best_epoch)
    return model, history, best_metrics


# -----------------------------------------------------------------------------
# Tokenization (latentization)
# -----------------------------------------------------------------------------


@torch.no_grad()
def tokenize_split_continuous(
    model: MotionVAE,
    split: FixedGridSplit,
    batch_size: int,
    device: torch.device,
) -> MotionLatentDataset:
    """Convert one fixed-grid split into frozen, deterministic latent sequences."""

    model.eval()
    values = torch.from_numpy(split.values)
    latent_batches = [
        model.encode_latents(values[start : start + batch_size].to(device))
        .permute(0, 2, 1)
        .cpu()
        for start in range(0, values.shape[0], batch_size)
    ]
    latents = (
        torch.cat(latent_batches, dim=0)
        if latent_batches
        else torch.empty((0, 0, model.config.latent_dim))
    )
    return MotionLatentDataset(
        latents=latents,
        labels=torch.from_numpy(split.labels).float(),
        sample_ids=split.sample_ids,
    )


def latent_statistics(dataset: MotionLatentDataset) -> dict[str, float]:
    """Latent-magnitude diagnostics for one split, the continuous analogue of
    ``t2m_gpt.training.token_statistics``'s codebook-occupancy report."""

    if len(dataset) == 0:
        return {"mean_latent_norm": 0.0, "latent_std": 0.0}
    flat = dataset.latents.reshape(-1, dataset.latent_dim)
    return {
        "mean_latent_norm": float(flat.norm(dim=1).mean().item()),
        "latent_std": float(flat.std().item()),
    }


# -----------------------------------------------------------------------------
# Stage two: transformer over continuous latents
# -----------------------------------------------------------------------------


def _classification_loss(
    model: MotionLatentGPT,
    latents: torch.Tensor,
    labels: torch.Tensor,
    config: ExperimentConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    corruption = config.gpt.token_corruption_rate if model.training else 0.0
    score = model.forward_classifier(latents, corruption_rate=corruption)
    weight = (
        None
        if config.training.positive_class_weight is None
        else torch.tensor(
            config.training.positive_class_weight, device=score.device, dtype=score.dtype
        )
    )
    loss = functional.binary_cross_entropy_with_logits(score, labels, pos_weight=weight)
    return loss, score


@torch.no_grad()
def evaluate_with_predictions(
    model: MotionLatentGPT,
    loader: torch.utils.data.DataLoader,
    config: ExperimentConfig,
    device: torch.device,
    threshold: float | None = None,
) -> EvaluationOutput:
    """Score one split and compute metrics at the supplied decision threshold."""

    model.eval()
    all_scores: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []
    sample_ids: list[str] = []
    losses: list[tuple[float, int]] = []

    for batch in loader:
        latents = batch["latents"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)
        loss, score = _classification_loss(model, latents, labels, config)
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


def pretrain_motion_latent_gpt(
    model: MotionLatentGPT,
    train_dataset: MotionLatentDataset,
    validation_dataset: MotionLatentDataset,
    config: ExperimentConfig,
    device: torch.device,
    progress_callback: ProgressCallback | None = None,
) -> list[dict[str, Any]]:
    """Unconditional next-latent regression pretraining on the training split only."""

    settings = config.pretraining
    train_loader = make_loader(
        train_dataset,
        batch_size=settings.batch_size,
        shuffle=True,
        seed=config.seed,
        num_workers=config.training.num_workers,
        pin_memory=device.type == "cuda",
    )
    validation_loader = make_loader(
        validation_dataset,
        batch_size=config.training.evaluation_batch_size,
        shuffle=False,
        seed=config.seed,
        num_workers=config.training.num_workers,
        pin_memory=device.type == "cuda",
    )
    optimizer = torch.optim.AdamW(
        model.backbone_parameters(), lr=settings.learning_rate, weight_decay=settings.weight_decay
    )

    history: list[dict[str, Any]] = []
    for epoch in range(1, settings.num_epochs + 1):
        model.train()
        train_losses: list[tuple[float, int]] = []
        for batch in train_loader:
            latents = batch["latents"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = model.next_latent_loss(latents, corruption_rate=settings.token_corruption_rate)
            loss.backward()
            if settings.gradient_clip_norm is not None:
                nn.utils.clip_grad_norm_(model.parameters(), settings.gradient_clip_norm)
            optimizer.step()
            train_losses.append((float(loss.item()), int(latents.shape[0])))

        model.eval()
        validation_losses: list[tuple[float, int]] = []
        with torch.no_grad():
            for batch in validation_loader:
                latents = batch["latents"].to(device, non_blocking=True)
                loss = model.next_latent_loss(latents)
                validation_losses.append((float(loss.item()), int(latents.shape[0])))

        record = {
            "stage": "pretraining",
            "epoch": epoch,
            "train_latent_loss": _weighted_mean(train_losses),
            "validation_latent_loss": _weighted_mean(validation_losses),
        }
        history.append(record)
        _report(
            progress_callback,
            config.training.show_progress,
            record,
            f"[pretraining] epoch {epoch:3d}/{settings.num_epochs} "
            f"train_latent_loss={record['train_latent_loss']:.4f} "
            f"val_latent_loss={record['validation_latent_loss']:.4f}",
        )

    if config.training.show_progress and progress_callback is None:
        print()
    return history


def train_motion_latent_gpt(
    model: MotionLatentGPT,
    latent_datasets: dict[str, MotionLatentDataset],
    config: ExperimentConfig,
    device: torch.device,
    progress_callback: ProgressCallback | None = None,
) -> tuple[list[dict[str, Any]], dict[str, torch.Tensor], int]:
    """Train the classification head, selecting the epoch on validation only."""

    settings = config.training
    pin_memory = device.type == "cuda"
    train_loader = make_loader(
        latent_datasets["train"],
        batch_size=settings.batch_size,
        shuffle=True,
        seed=config.seed,
        num_workers=settings.num_workers,
        pin_memory=pin_memory,
    )
    validation_loader = make_loader(
        latent_datasets["validation"],
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
    scaler = torch.amp.GradScaler(
        device.type, enabled=settings.mixed_precision and device.type == "cuda"
    )

    history: list[dict[str, Any]] = []
    best_value = _initial_best_value(settings.selection_metric)
    best_state = _detached_state(model)
    best_epoch = 0
    epochs_without_improvement = 0

    for epoch in range(1, settings.max_epochs + 1):
        backbone_frozen = epoch <= settings.freeze_backbone_epochs
        _set_backbone_requires_grad(model, not backbone_frozen)

        model.train()
        train_losses: list[tuple[float, int]] = []
        for batch in train_loader:
            latents = batch["latents"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with _autocast(device, settings.mixed_precision):
                loss, _ = _classification_loss(model, latents, labels, config)
            scaler.scale(loss).backward()
            if settings.gradient_clip_norm is not None:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), settings.gradient_clip_norm)
            scaler.step(optimizer)
            scaler.update()
            train_losses.append((float(loss.item()), int(labels.shape[0])))

        validation_output = evaluate_with_predictions(model, validation_loader, config, device)
        selection_value = validation_output.metrics[settings.selection_metric]
        if scheduler is not None:
            scheduler.step(selection_value)

        record = {
            "stage": "classification",
            "epoch": epoch,
            "backbone_frozen": backbone_frozen,
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
            f"[discriminative-v2] epoch {epoch:3d}/{settings.max_epochs} "
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
    _set_backbone_requires_grad(model, True)
    return history, best_state, best_epoch


# -----------------------------------------------------------------------------
# Orchestration
# -----------------------------------------------------------------------------


def _vae_checkpoint(
    model: MotionVAE,
    config: ExperimentConfig,
    data: DataBundle,
    history: list[dict[str, Any]],
    metrics: dict[str, float],
) -> dict[str, Any]:
    return {
        "checkpoint_version": CHECKPOINT_VERSION,
        "checkpoint_type": "t2m_gpt_v2_motion_vae",
        "model_state_dict": model.state_dict(),
        "vae_config": _json_ready(asdict(config.vae)),
        "experiment_config": _config_signature(config),
        "normalizer": None if data.normalizer is None else data.normalizer.state_dict(),
        "time_grid": data.time_grid,
        "channel_names": data.channel_names,
        "input_dim": data.num_channels,
        "num_frames": data.num_frames,
        "num_tokens": data.num_frames // config.vae.temporal_downsample_factor,
        "history": history,
        "validation_metrics": metrics,
        "model_parameter_count": count_parameters(model, trainable_only=False),
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
    pretraining_history: list[dict[str, Any]],
    vae_history: list[dict[str, Any]],
    vae_metrics: dict[str, float],
    latent_statistics_by_split: dict[str, dict[str, float]],
) -> dict[str, Any]:
    return {
        "checkpoint_version": CHECKPOINT_VERSION,
        "checkpoint_type": "t2m_gpt_v2_discriminative_classifier",
        "model_state_dict": model_state,
        "gpt_config": _json_ready(asdict(config.gpt)),
        "vae_config": _json_ready(asdict(config.vae)),
        "experiment_config": _config_signature(config),
        "normalizer": None if data.normalizer is None else data.normalizer.state_dict(),
        "time_grid": data.time_grid,
        "channel_names": data.channel_names,
        "latent_dim": config.vae.latent_dim,
        "num_tokens": num_tokens,
        "label_convention": {config.data.negative_label: 0, config.data.positive_label: 1},
        "best_epoch": best_epoch,
        "decision_threshold": decision_threshold,
        "validation_metrics": validation_metrics,
        "test_metrics": test_metrics,
        "model_parameter_count": model_parameter_count,
        "history": history,
        "pretraining_history": pretraining_history,
        "vae_history": vae_history,
        "vae_metrics": vae_metrics,
        "latent_statistics": latent_statistics_by_split,
    }


def train_t2m_gpt_v2(
    config: ExperimentConfig,
    progress_callback: ProgressCallback | None = None,
) -> TrainingResult:
    """Run both stages for one fold and evaluate the test split exactly once."""

    config.validate()
    set_reproducible_seed(config.seed, config.training.deterministic_algorithms)
    device = resolve_device(config.device)

    run_dir = Path(config.output_dir) / config.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    vae_path = run_dir / "motion_vae.pt"
    pretrained_path = run_dir / "pretrained_backbone.pt"
    checkpoint_path = run_dir / "best_model.pt"
    history_path = run_dir / "metrics.json"
    validation_predictions_path = run_dir / "validation_predictions.jsonl"
    test_predictions_path = run_dir / "test_predictions.jsonl"

    data = prepare_data(config.data)

    vae, vae_history, vae_metrics = train_motion_vae(data, config, device, progress_callback)
    torch.save(_vae_checkpoint(vae, config, data, vae_history, vae_metrics), vae_path)

    latent_datasets = {
        split: tokenize_split_continuous(
            vae, getattr(data, split), config.vae_training.evaluation_batch_size, device
        )
        for split in SPLIT_NAMES
    }
    latent_statistics_by_split = {
        split: latent_statistics(dataset) for split, dataset in latent_datasets.items()
    }
    num_tokens = latent_datasets["train"].num_tokens

    set_reproducible_seed(config.seed, config.training.deterministic_algorithms)
    model = MotionLatentGPT(config.vae.latent_dim, num_tokens, config.gpt).to(device)

    pretraining_history: list[dict[str, Any]] = []
    if config.use_pretraining:
        pretraining_history = pretrain_motion_latent_gpt(
            model, latent_datasets["train"], latent_datasets["validation"], config, device, progress_callback
        )
        torch.save(
            {
                "checkpoint_version": CHECKPOINT_VERSION,
                "checkpoint_type": "t2m_gpt_v2_pretrained_backbone",
                "model_state_dict": model.state_dict(),
                "gpt_config": _json_ready(asdict(config.gpt)),
                "latent_dim": config.vae.latent_dim,
                "num_tokens": num_tokens,
                "pretraining_history": pretraining_history,
            },
            pretrained_path,
        )

    history, best_state, best_epoch = train_motion_latent_gpt(
        model, latent_datasets, config, device, progress_callback
    )
    model.load_state_dict(best_state)

    pin_memory = device.type == "cuda"
    validation_loader = make_loader(
        latent_datasets["validation"],
        batch_size=config.training.evaluation_batch_size,
        shuffle=False,
        seed=config.seed,
        num_workers=config.training.num_workers,
        pin_memory=pin_memory,
    )
    test_loader = make_loader(
        latent_datasets["test"],
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
    test_output = evaluate_with_predictions(
        model, test_loader, config, device, threshold=decision_threshold
    )

    _write_lines(
        validation_predictions_path, _prediction_rows(validation_output, decision_threshold)
    )
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
            pretraining_history=pretraining_history,
            vae_history=vae_history,
            vae_metrics=vae_metrics,
            latent_statistics_by_split=latent_statistics_by_split,
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
                    "split_sizes": {split: len(latent_datasets[split]) for split in SPLIT_NAMES},
                    "best_epoch": best_epoch,
                    "decision_threshold": decision_threshold,
                    "model_parameter_count": model_parameter_count,
                    "vae_validation_metrics": vae_metrics,
                    "latent_statistics": latent_statistics_by_split,
                    "validation_metrics": validation_output.metrics,
                    "test_metrics": test_output.metrics,
                    "vae_history": vae_history,
                    "pretraining_history": pretraining_history,
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
        tokenizer_checkpoint_path=vae_path,
        tokenizer_history=vae_history,
        tokenizer_metrics=vae_metrics,
        pretrained_checkpoint_path=pretrained_path if config.use_pretraining else None,
        pretraining_history=pretraining_history,
    )


def load_trained_model_v2(
    config: ExperimentConfig,
) -> tuple[MotionLatentGPT, MotionVAE, Any, dict[str, Any]]:
    """Restore the VAE, the transformer, and the fitted normalizer."""

    try:
        from event_transformer.features import FeatureNormalizer
    except ModuleNotFoundError as error:  # pragma: no cover - import-path fallback
        if error.name != "event_transformer":
            raise
        from ..event_transformer.features import FeatureNormalizer

    config.validate()
    run_dir = Path(config.output_dir) / config.run_name
    classifier_checkpoint = _load_checkpoint(run_dir / "best_model.pt")
    vae_checkpoint = _load_checkpoint(run_dir / "motion_vae.pt")

    vae = MotionVAE(vae_checkpoint["input_dim"], config.vae)
    vae.load_state_dict(vae_checkpoint["model_state_dict"])
    vae.eval()

    model = MotionLatentGPT(
        classifier_checkpoint["latent_dim"], classifier_checkpoint["num_tokens"], config.gpt
    )
    model.load_state_dict(classifier_checkpoint["model_state_dict"])
    model.eval()

    normalizer_state = classifier_checkpoint.get("normalizer")
    normalizer = (
        None if normalizer_state is None else FeatureNormalizer.from_state_dict(normalizer_state)
    )
    return model, vae, normalizer, classifier_checkpoint


def load_training_result_v2(config: ExperimentConfig) -> TrainingResult:
    """Reload a finished fold, refusing checkpoints from a different configuration."""

    config.validate()
    run_dir = Path(config.output_dir) / config.run_name
    checkpoint = _load_checkpoint(run_dir / "best_model.pt")
    if checkpoint.get("experiment_config") != _config_signature(config):
        raise ValueError(f"Checkpoint in {run_dir} was produced by a different configuration")
    for required in ("validation_predictions.jsonl", "test_predictions.jsonl", "metrics.json"):
        if not (run_dir / required).exists():
            raise FileNotFoundError(f"Missing {required} in {run_dir}")

    pretrained_path = run_dir / "pretrained_backbone.pt"
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
        tokenizer_checkpoint_path=run_dir / "motion_vae.pt",
        tokenizer_history=list(checkpoint.get("vae_history", [])),
        tokenizer_metrics=dict(checkpoint.get("vae_metrics", {})),
        pretrained_checkpoint_path=pretrained_path if pretrained_path.exists() else None,
        pretraining_history=list(checkpoint.get("pretraining_history", [])),
    )


__all__ = [
    "EvaluationOutput",
    "TrainingResult",
    "evaluate_vae",
    "evaluate_with_predictions",
    "latent_statistics",
    "load_trained_model_v2",
    "load_training_result_v2",
    "pretrain_motion_latent_gpt",
    "resolve_device",
    "set_reproducible_seed",
    "tokenize_split_continuous",
    "train_motion_latent_gpt",
    "train_motion_vae",
    "train_t2m_gpt_v2",
]
