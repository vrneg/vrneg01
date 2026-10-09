"""Two-stage T2M-GPT training: motion tokenizer, then transformer over its codes.

Stage one fits the VQ-VAE on the current fold's training windows only.  Stage two
optionally pretrains the transformer with unconditional next-token prediction and then
trains the configured head.  Validation drives every selection decision; the test split
is evaluated once, after the selected checkpoint is restored.
"""

from __future__ import annotations

import json
import math
import random
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as functional
from torch.utils.data import DataLoader

try:
    from event_transformer.features import FeatureNormalizer
    from event_transformer.metrics import (
        binary_classification_metrics,
        binary_classification_metrics_from_predictions,
        optimize_binary_threshold,
    )
except ModuleNotFoundError as error:
    if error.name != "event_transformer":
        raise
    from ..event_transformer.features import FeatureNormalizer
    from ..event_transformer.metrics import (
        binary_classification_metrics,
        binary_classification_metrics_from_predictions,
        optimize_binary_threshold,
    )

from .config import ExperimentConfig
from .data import (
    DataBundle,
    FixedGridSplit,
    MotionTokenDataset,
    MotionWindowDataset,
    make_loader,
    prepare_data,
)
from .model import MotionTokenGPT, count_parameters
from .vqvae import MotionVQVAE, reconstruction_losses

ProgressCallback = Callable[[dict[str, Any]], None]
CHECKPOINT_VERSION = 1
SPLIT_NAMES: tuple[str, ...] = ("train", "validation", "test")


@dataclass(slots=True)
class TrainingResult:
    """Artifacts and final metrics for one trained fold."""

    checkpoint_path: Path
    history_path: Path
    validation_predictions_path: Path
    predictions_path: Path
    run_name: str
    best_epoch: int
    decision_threshold: float
    validation_metrics: dict[str, float]
    test_metrics: dict[str, float]
    model_parameter_count: int
    history: list[dict[str, Any]]
    tokenizer_checkpoint_path: Path | None = None
    tokenizer_history: list[dict[str, Any]] = field(default_factory=list)
    tokenizer_metrics: dict[str, float] = field(default_factory=dict)
    # Compatibility fields for the shared cross-validation aggregator.
    pretrained_checkpoint_path: Path | None = None
    pretraining_history: list[dict[str, Any]] = field(default_factory=list)


@dataclass(slots=True)
class EvaluationOutput:
    """Metrics and row-level outputs from one evaluation pass."""

    metrics: dict[str, float]
    sample_ids: list[str]
    labels: torch.Tensor
    logits: torch.Tensor


def set_reproducible_seed(seed: int, deterministic_algorithms: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(deterministic_algorithms)


def resolve_device(requested_device: str) -> torch.device:
    if requested_device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    device = torch.device(requested_device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    if device.type == "mps" and not (
        hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    ):
        raise RuntimeError("MPS was requested but is not available")
    return device


def _is_improvement(
    metric_name: str,
    candidate: float,
    best: float,
    minimum_delta: float,
) -> bool:
    if not math.isfinite(candidate):
        return False
    if metric_name == "loss":
        return candidate < best - minimum_delta
    return candidate > best + minimum_delta


def _initial_best_value(metric_name: str) -> float:
    return math.inf if metric_name == "loss" else -math.inf


def _json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _config_signature(config: ExperimentConfig) -> dict[str, Any]:
    """Return the result-affecting config, excluding progress rendering."""

    signature = _json_ready(asdict(config))
    signature["training"]["show_progress"] = False
    return signature


def _detached_state(module: nn.Module) -> dict[str, torch.Tensor]:
    return {
        key: value.detach().clone().cpu() for key, value in module.state_dict().items()
    }


def _report(
    progress_callback: ProgressCallback | None,
    show_progress: bool,
    record: dict[str, Any],
    line: str,
) -> None:
    if progress_callback is not None:
        progress_callback(record)
    elif show_progress:
        print(f"\r{line}", end="", flush=True)


# -----------------------------------------------------------------------------
# Stage one: motion tokenizer
# -----------------------------------------------------------------------------


def _vqvae_batch_losses(
    model: MotionVQVAE,
    values: torch.Tensor,
    config: ExperimentConfig,
) -> tuple[torch.Tensor, dict[str, float], torch.Tensor]:
    settings = config.vqvae_training
    output = model(values)
    frame_loss, velocity_loss = reconstruction_losses(
        output.reconstruction, values, settings.reconstruction_loss
    )
    total = (
        frame_loss
        + settings.velocity_loss_weight * velocity_loss
        + settings.commitment_loss_weight * output.commitment_loss
    )
    statistics = {
        "loss": float(total.item()),
        "frame_loss": float(frame_loss.item()),
        "velocity_loss": float(velocity_loss.item()),
        "commitment_loss": float(output.commitment_loss.item()),
        "perplexity": float(output.perplexity.item()),
        "num_used_codes": float(output.num_used_codes),
    }
    return total, statistics, output.indices


def _weighted_mean(values: list[tuple[float, int]]) -> float:
    total_weight = sum(weight for _, weight in values)
    if total_weight == 0:
        return math.nan
    return sum(value * weight for value, weight in values) / total_weight


@torch.no_grad()
def evaluate_vqvae(
    model: MotionVQVAE,
    loader: DataLoader,
    config: ExperimentConfig,
    device: torch.device,
) -> dict[str, float]:
    """Reconstruction quality on one split, with the codebook held fixed."""

    model.eval()
    accumulated: dict[str, list[tuple[float, int]]] = {}
    used_codes: set[int] = set()
    for batch in loader:
        values = batch["values"].to(device, non_blocking=True)
        _, statistics, indices = _vqvae_batch_losses(model, values, config)
        weight = int(values.shape[0])
        for key, value in statistics.items():
            accumulated.setdefault(key, []).append((value, weight))
        used_codes.update(indices.unique().tolist())

    metrics = {key: _weighted_mean(value) for key, value in accumulated.items()}
    metrics["num_used_codes"] = float(len(used_codes))
    metrics["codebook_usage_fraction"] = len(used_codes) / float(model.config.num_codes)
    return metrics


def train_motion_vqvae(
    data: DataBundle,
    config: ExperimentConfig,
    device: torch.device,
    progress_callback: ProgressCallback | None = None,
) -> tuple[MotionVQVAE, list[dict[str, Any]], dict[str, float]]:
    """Fit the tokenizer on training windows and select on validation reconstruction."""

    settings = config.vqvae_training
    model = MotionVQVAE(data.num_channels, config.vqvae).to(device)
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
        model.parameters(),
        lr=settings.learning_rate,
        weight_decay=settings.weight_decay,
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
            total, statistics, _ = _vqvae_batch_losses(model, values, config)
            total.backward()
            if settings.gradient_clip_norm is not None:
                nn.utils.clip_grad_norm_(model.parameters(), settings.gradient_clip_norm)
            optimizer.step()
            weight = int(values.shape[0])
            for key, value in statistics.items():
                batch_statistics.setdefault(key, []).append((value, weight))

        train_metrics = {
            key: _weighted_mean(value) for key, value in batch_statistics.items()
        }
        validation_metrics = evaluate_vqvae(model, validation_loader, config, device)
        if scheduler is not None:
            scheduler.step(validation_metrics["loss"])

        record = {
            "stage": "tokenizer",
            "epoch": epoch,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            **{f"train_{key}": value for key, value in train_metrics.items()},
            **{f"validation_{key}": value for key, value in validation_metrics.items()},
        }

        if _is_improvement(
            "loss",
            validation_metrics["loss"],
            best_value,
            settings.early_stopping_min_delta,
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
            f"[tokenizer] epoch {epoch:3d}/{settings.max_epochs} "
            f"train_loss={train_metrics['loss']:.4f} "
            f"val_loss={validation_metrics['loss']:.4f} "
            f"codes={validation_metrics['num_used_codes']:.0f}"
            f"/{config.vqvae.num_codes} best={best_epoch}",
        )

        if epochs_without_improvement >= settings.early_stopping_patience:
            break

    if config.training.show_progress and progress_callback is None:
        print()
    model.load_state_dict(best_state)
    best_metrics["best_epoch"] = float(best_epoch)
    return model, history, best_metrics


# -----------------------------------------------------------------------------
# Tokenization
# -----------------------------------------------------------------------------


@torch.no_grad()
def tokenize_split(
    model: MotionVQVAE,
    split: FixedGridSplit,
    batch_size: int,
    device: torch.device,
) -> MotionTokenDataset:
    """Convert one fixed-grid split into frozen motion-token sequences."""

    model.eval()
    values = torch.from_numpy(split.values)
    token_batches = [
        model.encode_indices(values[start : start + batch_size].to(device)).cpu()
        for start in range(0, values.shape[0], batch_size)
    ]
    indices = (
        torch.cat(token_batches, dim=0)
        if token_batches
        else torch.empty((0, 0), dtype=torch.long)
    )
    return MotionTokenDataset(
        indices=indices,
        labels=torch.from_numpy(split.labels).float(),
        sample_ids=split.sample_ids,
    )


def token_statistics(dataset: MotionTokenDataset, num_codes: int) -> dict[str, float]:
    """Codebook occupancy and per-window token diversity for one split."""

    if len(dataset) == 0:
        return {"num_used_codes": 0.0, "codebook_usage_fraction": 0.0}
    counts = torch.bincount(dataset.indices.reshape(-1), minlength=num_codes).float()
    probabilities = counts / counts.sum()
    nonzero = probabilities[probabilities > 0]
    entropy = float(-(nonzero * nonzero.log()).sum().item())
    unique_per_window = [float(row.unique().numel()) for row in dataset.indices]
    num_used_codes = float((counts > 0).sum().item())
    return {
        "num_used_codes": num_used_codes,
        "codebook_usage_fraction": num_used_codes / float(num_codes),
        "token_entropy_nats": entropy,
        "token_perplexity": float(math.exp(entropy)),
        "mean_unique_codes_per_window": sum(unique_per_window) / len(unique_per_window),
    }


# -----------------------------------------------------------------------------
# Stage two: transformer over motion tokens
# -----------------------------------------------------------------------------


def _autocast(device: torch.device, enabled: bool):
    return torch.autocast(device_type=device.type, enabled=enabled and device.type == "cuda")


def _classification_loss(
    model: MotionTokenGPT,
    indices: torch.Tensor,
    labels: torch.Tensor,
    config: ExperimentConfig,
    compute_score: bool = True,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Return the training loss and, when requested, the decision score.

    The generative head needs two extra forward passes to score a batch, so training
    skips scoring and reports only the class-conditional token loss it optimizes.
    """

    corruption = config.gpt.token_corruption_rate if model.training else 0.0
    if config.gpt.head == "generative":
        # Class-conditional autoregressive modelling: fit p(tokens | true class).
        logits = model.forward_language_model(
            indices, prefix_ids=labels.long(), corruption_rate=corruption
        )
        loss = functional.cross_entropy(
            logits.reshape(-1, model.num_codes), indices.reshape(-1)
        )
        if not compute_score:
            return loss, None
        with torch.no_grad():
            score = model.log_likelihood_ratio(indices)
        return loss, score

    score = model.forward_classifier(indices, corruption_rate=corruption)
    weight = (
        None
        if config.training.positive_class_weight is None
        else torch.tensor(
            config.training.positive_class_weight,
            device=score.device,
            dtype=score.dtype,
        )
    )
    loss = functional.binary_cross_entropy_with_logits(
        score, labels, pos_weight=weight
    )
    return loss, score


@torch.no_grad()
def evaluate_with_predictions(
    model: MotionTokenGPT,
    loader: DataLoader,
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
        indices = batch["indices"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)
        loss, score = _classification_loss(model, indices, labels, config)
        if score is None:
            raise RuntimeError("Evaluation requires a decision score")
        losses.append((float(loss.item()), int(labels.shape[0])))
        all_scores.append(score.detach().float().cpu())
        all_labels.append(labels.detach().float().cpu())
        sample_ids.extend(batch["sample_id"])

    scores = (
        torch.cat(all_scores) if all_scores else torch.empty(0, dtype=torch.float32)
    )
    labels = (
        torch.cat(all_labels) if all_labels else torch.empty(0, dtype=torch.float32)
    )
    loss = _weighted_mean(losses)
    metrics = binary_classification_metrics(
        scores, labels, loss, config.training.threshold if threshold is None else threshold
    )
    return EvaluationOutput(
        metrics=metrics,
        sample_ids=sample_ids,
        labels=labels,
        logits=scores,
    )


def pretrain_motion_gpt(
    model: MotionTokenGPT,
    train_dataset: MotionTokenDataset,
    validation_dataset: MotionTokenDataset,
    config: ExperimentConfig,
    device: torch.device,
    progress_callback: ProgressCallback | None = None,
) -> list[dict[str, Any]]:
    """Unconditional next-token pretraining on the training split only."""

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
        model.backbone_parameters(),
        lr=settings.learning_rate,
        weight_decay=settings.weight_decay,
    )

    history: list[dict[str, Any]] = []
    for epoch in range(1, settings.num_epochs + 1):
        model.train()
        train_losses: list[tuple[float, int]] = []
        for batch in train_loader:
            indices = batch["indices"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = model.language_model_loss(
                indices, corruption_rate=settings.token_corruption_rate
            )
            loss.backward()
            if settings.gradient_clip_norm is not None:
                nn.utils.clip_grad_norm_(
                    model.parameters(), settings.gradient_clip_norm
                )
            optimizer.step()
            train_losses.append((float(loss.item()), int(indices.shape[0])))

        model.eval()
        validation_losses: list[tuple[float, int]] = []
        with torch.no_grad():
            for batch in validation_loader:
                indices = batch["indices"].to(device, non_blocking=True)
                loss = model.language_model_loss(indices)
                validation_losses.append((float(loss.item()), int(indices.shape[0])))

        record = {
            "stage": "pretraining",
            "epoch": epoch,
            "train_token_loss": _weighted_mean(train_losses),
            "validation_token_loss": _weighted_mean(validation_losses),
        }
        history.append(record)
        _report(
            progress_callback,
            config.training.show_progress,
            record,
            f"[pretraining] epoch {epoch:3d}/{settings.num_epochs} "
            f"train_token_loss={record['train_token_loss']:.4f} "
            f"val_token_loss={record['validation_token_loss']:.4f}",
        )

    if config.training.show_progress and progress_callback is None:
        print()
    return history


def _build_optimizer(
    model: MotionTokenGPT,
    config: ExperimentConfig,
) -> torch.optim.Optimizer:
    settings = config.training
    if config.gpt.head == "generative":
        parameter_groups = [
            {"params": model.backbone_parameters(), "lr": settings.learning_rate}
        ]
    elif settings.classifier_learning_rate is None:
        parameter_groups = [
            {"params": list(model.parameters()), "lr": settings.learning_rate}
        ]
    else:
        parameter_groups = [
            {"params": model.backbone_parameters(), "lr": settings.learning_rate},
            {
                "params": model.head_parameters(),
                "lr": settings.classifier_learning_rate,
            },
        ]
    return torch.optim.AdamW(parameter_groups, weight_decay=settings.weight_decay)


def _set_backbone_requires_grad(model: MotionTokenGPT, requires_grad: bool) -> None:
    for parameter in model.backbone_parameters():
        parameter.requires_grad_(requires_grad)


def train_motion_gpt(
    model: MotionTokenGPT,
    token_datasets: dict[str, MotionTokenDataset],
    config: ExperimentConfig,
    device: torch.device,
    progress_callback: ProgressCallback | None = None,
) -> tuple[list[dict[str, Any]], dict[str, torch.Tensor], int]:
    """Train the configured head, selecting the epoch on validation only."""

    settings = config.training
    pin_memory = device.type == "cuda"
    train_loader = make_loader(
        token_datasets["train"],
        batch_size=settings.batch_size,
        shuffle=True,
        seed=config.seed,
        num_workers=settings.num_workers,
        pin_memory=pin_memory,
    )
    validation_loader = make_loader(
        token_datasets["validation"],
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
    # Freezing only makes sense when a separate classifier head exists.
    freeze_epochs = (
        0 if config.gpt.head == "generative" else settings.freeze_backbone_epochs
    )

    history: list[dict[str, Any]] = []
    best_value = _initial_best_value(settings.selection_metric)
    best_state = _detached_state(model)
    best_epoch = 0
    epochs_without_improvement = 0

    for epoch in range(1, settings.max_epochs + 1):
        backbone_frozen = epoch <= freeze_epochs
        _set_backbone_requires_grad(model, not backbone_frozen)

        model.train()
        train_losses: list[tuple[float, int]] = []
        for batch in train_loader:
            indices = batch["indices"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with _autocast(device, settings.mixed_precision):
                loss, _ = _classification_loss(
                    model, indices, labels, config, compute_score=False
                )
            scaler.scale(loss).backward()
            if settings.gradient_clip_norm is not None:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), settings.gradient_clip_norm)
            scaler.step(optimizer)
            scaler.update()
            train_losses.append((float(loss.item()), int(labels.shape[0])))

        validation_output = evaluate_with_predictions(
            model, validation_loader, config, device
        )
        selection_value = validation_output.metrics[settings.selection_metric]
        if scheduler is not None:
            scheduler.step(selection_value)

        record = {
            "stage": "classification",
            "epoch": epoch,
            "backbone_frozen": backbone_frozen,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "train_loss": _weighted_mean(train_losses),
            **{
                f"validation_{key}": value
                for key, value in validation_output.metrics.items()
            },
        }

        if _is_improvement(
            settings.selection_metric,
            selection_value,
            best_value,
            settings.early_stopping_min_delta,
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
            f"[{config.gpt.head}] epoch {epoch:3d}/{settings.max_epochs} "
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


def _prediction_rows(output: EvaluationOutput, threshold: float) -> list[str]:
    probabilities = torch.sigmoid(output.logits)
    predictions = probabilities >= threshold
    return [
        json.dumps(
            {
                "sample_id": sample_id,
                "label": int(label.item()),
                "logit": float(logit.item()),
                "probability": float(probability.item()),
                "threshold": threshold,
                "prediction": int(prediction.item()),
            }
        )
        for sample_id, label, logit, probability, prediction in zip(
            output.sample_ids,
            output.labels,
            output.logits,
            probabilities,
            predictions,
            strict=True,
        )
    ]


def _write_lines(path: Path, lines: list[str]) -> None:
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def _tokenizer_checkpoint(
    model: MotionVQVAE,
    config: ExperimentConfig,
    data: DataBundle,
    history: list[dict[str, Any]],
    metrics: dict[str, float],
) -> dict[str, Any]:
    return {
        "checkpoint_version": CHECKPOINT_VERSION,
        "checkpoint_type": "t2m_gpt_motion_vqvae",
        "model_state_dict": model.state_dict(),
        "vqvae_config": _json_ready(asdict(config.vqvae)),
        "experiment_config": _config_signature(config),
        "normalizer": None if data.normalizer is None else data.normalizer.state_dict(),
        "time_grid": data.time_grid,
        "channel_names": data.channel_names,
        "input_dim": data.num_channels,
        "num_frames": data.num_frames,
        "num_tokens": data.num_frames // config.vqvae.temporal_downsample_factor,
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
    tokenizer_history: list[dict[str, Any]],
    tokenizer_metrics: dict[str, float],
    token_statistics_by_split: dict[str, dict[str, float]],
) -> dict[str, Any]:
    return {
        "checkpoint_version": CHECKPOINT_VERSION,
        "checkpoint_type": f"t2m_gpt_{config.gpt.head}_classifier",
        "model_state_dict": model_state,
        "gpt_config": _json_ready(asdict(config.gpt)),
        "vqvae_config": _json_ready(asdict(config.vqvae)),
        "experiment_config": _config_signature(config),
        "normalizer": None if data.normalizer is None else data.normalizer.state_dict(),
        "time_grid": data.time_grid,
        "channel_names": data.channel_names,
        "num_codes": config.vqvae.num_codes,
        "num_tokens": num_tokens,
        "label_convention": {
            config.data.negative_label: 0,
            config.data.positive_label: 1,
        },
        "best_epoch": best_epoch,
        "decision_threshold": decision_threshold,
        "validation_metrics": validation_metrics,
        "test_metrics": test_metrics,
        "model_parameter_count": model_parameter_count,
        "history": history,
        "pretraining_history": pretraining_history,
        "tokenizer_history": tokenizer_history,
        "tokenizer_metrics": tokenizer_metrics,
        "token_statistics": token_statistics_by_split,
    }


def train_t2m_gpt(
    config: ExperimentConfig,
    progress_callback: ProgressCallback | None = None,
) -> TrainingResult:
    """Run both stages for one fold and evaluate the test split exactly once."""

    config.validate()
    set_reproducible_seed(config.seed, config.training.deterministic_algorithms)
    device = resolve_device(config.device)

    run_dir = Path(config.output_dir) / config.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    tokenizer_path = run_dir / "motion_vqvae.pt"
    pretrained_path = run_dir / "pretrained_backbone.pt"
    checkpoint_path = run_dir / "best_model.pt"
    history_path = run_dir / "metrics.json"
    validation_predictions_path = run_dir / "validation_predictions.jsonl"
    test_predictions_path = run_dir / "test_predictions.jsonl"

    data = prepare_data(config.data)

    tokenizer, tokenizer_history, tokenizer_metrics = train_motion_vqvae(
        data, config, device, progress_callback
    )
    torch.save(
        _tokenizer_checkpoint(tokenizer, config, data, tokenizer_history, tokenizer_metrics),
        tokenizer_path,
    )

    token_datasets = {
        split: tokenize_split(
            tokenizer,
            getattr(data, split),
            config.vqvae_training.evaluation_batch_size,
            device,
        )
        for split in SPLIT_NAMES
    }
    token_statistics_by_split = {
        split: token_statistics(dataset, config.vqvae.num_codes)
        for split, dataset in token_datasets.items()
    }
    num_tokens = token_datasets["train"].num_tokens

    set_reproducible_seed(config.seed, config.training.deterministic_algorithms)
    model = MotionTokenGPT(config.vqvae.num_codes, num_tokens, config.gpt).to(device)

    pretraining_history: list[dict[str, Any]] = []
    if config.use_pretraining:
        pretraining_history = pretrain_motion_gpt(
            model,
            token_datasets["train"],
            token_datasets["validation"],
            config,
            device,
            progress_callback,
        )
        model.synchronize_prefix_embeddings()
        torch.save(
            {
                "checkpoint_version": CHECKPOINT_VERSION,
                "checkpoint_type": "t2m_gpt_pretrained_backbone",
                "model_state_dict": model.state_dict(),
                "gpt_config": _json_ready(asdict(config.gpt)),
                "num_codes": config.vqvae.num_codes,
                "num_tokens": num_tokens,
                "pretraining_history": pretraining_history,
            },
            pretrained_path,
        )

    history, best_state, best_epoch = train_motion_gpt(
        model, token_datasets, config, device, progress_callback
    )
    model.load_state_dict(best_state)

    pin_memory = device.type == "cuda"
    validation_loader = make_loader(
        token_datasets["validation"],
        batch_size=config.training.evaluation_batch_size,
        shuffle=False,
        seed=config.seed,
        num_workers=config.training.num_workers,
        pin_memory=pin_memory,
    )
    test_loader = make_loader(
        token_datasets["test"],
        batch_size=config.training.evaluation_batch_size,
        shuffle=False,
        seed=config.seed,
        num_workers=config.training.num_workers,
        pin_memory=pin_memory,
    )

    validation_output = evaluate_with_predictions(
        model, validation_loader, config, device
    )
    decision_threshold = config.training.threshold
    if config.training.calibrate_threshold_on_validation:
        decision_threshold = optimize_binary_threshold(
            validation_output.logits,
            validation_output.labels,
            config.training.threshold_metric,
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
            tokenizer_history=tokenizer_history,
            tokenizer_metrics=tokenizer_metrics,
            token_statistics_by_split=token_statistics_by_split,
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
                    "split_sizes": {
                        split: len(token_datasets[split]) for split in SPLIT_NAMES
                    },
                    "best_epoch": best_epoch,
                    "decision_threshold": decision_threshold,
                    "model_parameter_count": model_parameter_count,
                    "tokenizer_validation_metrics": tokenizer_metrics,
                    "token_statistics": token_statistics_by_split,
                    "validation_metrics": validation_output.metrics,
                    "test_metrics": test_output.metrics,
                    "tokenizer_history": tokenizer_history,
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
        tokenizer_checkpoint_path=tokenizer_path,
        tokenizer_history=tokenizer_history,
        tokenizer_metrics=tokenizer_metrics,
        pretrained_checkpoint_path=pretrained_path if config.use_pretraining else None,
        pretraining_history=pretraining_history,
    )


# -----------------------------------------------------------------------------
# Reloading
# -----------------------------------------------------------------------------


def _load_checkpoint(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"No checkpoint at {path}")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("checkpoint_version") != CHECKPOINT_VERSION:
        raise ValueError(
            f"Unsupported checkpoint version in {path}: "
            f"{checkpoint.get('checkpoint_version')!r}"
        )
    return checkpoint


def load_trained_model(
    config: ExperimentConfig,
) -> tuple[MotionTokenGPT, MotionVQVAE, FeatureNormalizer | None, dict[str, Any]]:
    """Restore the tokenizer, the transformer, and the fitted normalizer."""

    config.validate()
    run_dir = Path(config.output_dir) / config.run_name
    classifier_checkpoint = _load_checkpoint(run_dir / "best_model.pt")
    tokenizer_checkpoint = _load_checkpoint(run_dir / "motion_vqvae.pt")

    tokenizer = MotionVQVAE(tokenizer_checkpoint["input_dim"], config.vqvae)
    tokenizer.load_state_dict(tokenizer_checkpoint["model_state_dict"])
    tokenizer.eval()

    model = MotionTokenGPT(
        classifier_checkpoint["num_codes"],
        classifier_checkpoint["num_tokens"],
        config.gpt,
    )
    model.load_state_dict(classifier_checkpoint["model_state_dict"])
    model.eval()

    normalizer_state = classifier_checkpoint.get("normalizer")
    normalizer = (
        None
        if normalizer_state is None
        else FeatureNormalizer.from_state_dict(normalizer_state)
    )
    return model, tokenizer, normalizer, classifier_checkpoint


def load_training_result(config: ExperimentConfig) -> TrainingResult:
    """Reload a finished fold, refusing checkpoints from a different configuration."""

    config.validate()
    run_dir = Path(config.output_dir) / config.run_name
    checkpoint = _load_checkpoint(run_dir / "best_model.pt")
    if checkpoint.get("experiment_config") != _config_signature(config):
        raise ValueError(
            f"Checkpoint in {run_dir} was produced by a different configuration"
        )
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
        tokenizer_checkpoint_path=run_dir / "motion_vqvae.pt",
        tokenizer_history=list(checkpoint.get("tokenizer_history", [])),
        tokenizer_metrics=dict(checkpoint.get("tokenizer_metrics", {})),
        pretrained_checkpoint_path=pretrained_path if pretrained_path.exists() else None,
        pretraining_history=list(checkpoint.get("pretraining_history", [])),
    )


__all__ = [
    "EvaluationOutput",
    "TrainingResult",
    "evaluate_vqvae",
    "evaluate_with_predictions",
    "load_trained_model",
    "load_training_result",
    "pretrain_motion_gpt",
    "resolve_device",
    "set_reproducible_seed",
    "token_statistics",
    "tokenize_split",
    "train_motion_gpt",
    "train_motion_vqvae",
    "train_t2m_gpt",
]
