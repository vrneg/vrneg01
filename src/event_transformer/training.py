"""Training and evaluation entry points for one saved dataset fold."""

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
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm

from .config import ExperimentConfig, ModelConfig
from .data import DataBundle, prepare_data
from .features import FEATURE_SCHEMA_VERSION, MODALITY_DIMS, FeatureNormalizer
from .metrics import (
    binary_classification_metrics,
    optimize_binary_threshold,
)
from .model import EventTransformer
from .pretraining import pretrain_event_transformer


ProgressCallback = Callable[[dict[str, Any]], None]
CHECKPOINT_VERSION = 2


@dataclass(slots=True)
class TrainingResult:
    """Artifacts and final metrics from one training run."""

    checkpoint_path: Path
    history_path: Path
    validation_predictions_path: Path
    predictions_path: Path
    run_name: str
    best_epoch: int
    decision_threshold: float
    validation_metrics: dict[str, float]
    test_metrics: dict[str, float]
    history: list[dict[str, Any]]
    pretrained_checkpoint_path: Path | None = None
    pretraining_history: list[dict[str, Any]] = field(default_factory=list)


@dataclass(slots=True)
class EvaluationOutput:
    """Metrics and row-level outputs collected in one evaluation pass."""

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


def _move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        "modality_features": {
            modality_id: features.to(device, non_blocking=True)
            for modality_id, features in batch["modality_features"].items()
        },
        "finger_flags": {
            modality_id: flags.to(device, non_blocking=True)
            for modality_id, flags in batch["finger_flags"].items()
        },
        "modality_token_indices": {
            modality_id: indices.to(device, non_blocking=True)
            for modality_id, indices in batch["modality_token_indices"].items()
        },
        "modality_actor_relation_ids": {
            modality_id: actor_ids.to(device, non_blocking=True)
            for modality_id, actor_ids in batch[
                "modality_actor_relation_ids"
            ].items()
        },
        "time_features": batch["time_features"].to(device, non_blocking=True),
        "event_mask": batch["event_mask"].to(device, non_blocking=True),
        "anchor_mask": batch["anchor_mask"].to(device, non_blocking=True),
        "padding_mask": batch["padding_mask"].to(device, non_blocking=True),
        "labels": batch["labels"].to(device, non_blocking=True),
    }


def _forward(model: EventTransformer, batch: dict[str, Any]) -> torch.Tensor:
    return model(
        modality_features=batch["modality_features"],
        finger_flags=batch["finger_flags"],
        modality_token_indices=batch["modality_token_indices"],
        modality_actor_relation_ids=batch["modality_actor_relation_ids"],
        time_features=batch["time_features"],
        event_mask=batch["event_mask"],
        anchor_mask=batch["anchor_mask"],
        padding_mask=batch["padding_mask"],
    )


@torch.no_grad()
def evaluate_with_predictions(
    model: EventTransformer,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    threshold: float = 0.5,
) -> EvaluationOutput:
    model.eval()
    all_logits: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []
    loss_sum = 0.0
    num_examples = 0
    sample_ids: list[str] = []

    for raw_batch in loader:
        sample_ids.extend(str(sample_id) for sample_id in raw_batch["sample_ids"])
        batch = _move_batch(raw_batch, device)
        logits = _forward(model, batch)
        loss = criterion(logits, batch["labels"])
        batch_size = batch["labels"].shape[0]
        loss_sum += float(loss.item()) * batch_size
        num_examples += batch_size
        all_logits.append(logits.detach().cpu())
        all_labels.append(batch["labels"].detach().cpu())

    if num_examples == 0:
        raise ValueError("Cannot evaluate an empty dataset")
    logits = torch.cat(all_logits)
    labels = torch.cat(all_labels)
    metrics = binary_classification_metrics(
        logits=logits,
        labels=labels,
        loss=loss_sum / num_examples,
        threshold=threshold,
    )
    return EvaluationOutput(
        metrics=metrics,
        sample_ids=sample_ids,
        labels=labels,
        logits=logits,
    )


def evaluate(
    model: EventTransformer,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    threshold: float = 0.5,
) -> dict[str, float]:
    return evaluate_with_predictions(model, loader, criterion, device, threshold).metrics


def _train_epoch(
    model: EventTransformer,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: AdamW,
    device: torch.device,
    gradient_clip_norm: float | None,
    mixed_precision: bool,
    scaler: torch.amp.GradScaler,
    backbone_frozen: bool = False,
) -> float:
    if backbone_frozen:
        # Keep dropout disabled in the frozen representation while retaining
        # ordinary training behavior in the newly initialized classifier.
        model.eval()
        model.classifier.train()
    else:
        model.train()
    use_amp = mixed_precision and device.type == "cuda"
    loss_sum = 0.0
    num_examples = 0

    for raw_batch in loader:
        batch = _move_batch(raw_batch, device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            logits = _forward(model, batch)
            loss = criterion(logits, batch["labels"])

        scaler.scale(loss).backward()
        if gradient_clip_norm is not None:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
        scaler.step(optimizer)
        scaler.update()

        batch_size = batch["labels"].shape[0]
        loss_sum += float(loss.item()) * batch_size
        num_examples += batch_size

    if num_examples == 0:
        raise ValueError("Cannot train on an empty dataset")
    return loss_sum / num_examples


def _set_backbone_trainable(model: EventTransformer, trainable: bool) -> None:
    """Freeze or unfreeze every parameter except the classification head."""

    for name, parameter in model.named_parameters():
        if not name.startswith("classifier."):
            parameter.requires_grad_(trainable)
    for parameter in model.classifier.parameters():
        parameter.requires_grad_(True)


def _build_finetuning_optimizer(
    model: EventTransformer,
    config: ExperimentConfig,
) -> AdamW:
    """Use separate learning rates for transferred and newly initialized weights."""

    backbone_parameters = []
    classifier_parameters = []
    for name, parameter in model.named_parameters():
        if name.startswith("classifier."):
            classifier_parameters.append(parameter)
        else:
            backbone_parameters.append(parameter)

    classifier_learning_rate = (
        config.training.learning_rate
        if config.training.classifier_learning_rate is None
        else config.training.classifier_learning_rate
    )
    return AdamW(
        [
            {
                "name": "backbone",
                "params": backbone_parameters,
                "lr": config.training.learning_rate,
                "weight_decay": config.training.weight_decay,
            },
            {
                "name": "classifier",
                "params": classifier_parameters,
                "lr": classifier_learning_rate,
                "weight_decay": config.training.weight_decay,
            },
        ]
    )


def _optimizer_learning_rates(optimizer: AdamW) -> dict[str, float]:
    return {
        str(group.get("name", index)): float(group["lr"])
        for index, group in enumerate(optimizer.param_groups)
    }


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


def _json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value


def _checkpoint(
    model_state: dict[str, torch.Tensor],
    config: ExperimentConfig,
    normalizer: FeatureNormalizer | None,
    best_epoch: int,
    validation_metrics: dict[str, float],
    test_metrics: dict[str, float],
    decision_threshold: float,
    pretraining_applied: bool,
    pretraining_history: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "checkpoint_version": CHECKPOINT_VERSION,
        "checkpoint_type": "fine_tuned_classifier",
        "model_state_dict": model_state,
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "experiment_config": _json_ready(asdict(config)),
        "modality_dims": dict(MODALITY_DIMS),
        "normalizer": None if normalizer is None else normalizer.state_dict(),
        "best_epoch": best_epoch,
        "validation_metrics": validation_metrics,
        "test_metrics": test_metrics,
        "decision_threshold": decision_threshold,
        "pretraining_applied": pretraining_applied,
        "pretraining_history": pretraining_history,
    }


def _pretraining_checkpoint(
    model: EventTransformer,
    config: ExperimentConfig,
    normalizer: FeatureNormalizer | None,
    history: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "checkpoint_version": CHECKPOINT_VERSION,
        "checkpoint_type": "pretrained_backbone",
        "model_state_dict": {
            key: value.detach().cpu() for key, value in model.state_dict().items()
        },
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "experiment_config": _json_ready(asdict(config)),
        "modality_dims": dict(MODALITY_DIMS),
        "normalizer": None if normalizer is None else normalizer.state_dict(),
        "pretraining_applied": True,
        "pretraining_history": history,
    }


def _prediction_rows(
    output: EvaluationOutput,
    threshold: float,
) -> list[str]:
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


def train_event_transformer(
    config: ExperimentConfig,
    *,
    use_pretraining: bool = True,
    progress_callback: ProgressCallback | None = None,
) -> TrainingResult:
    """Pretrain and fine-tune one fold, then evaluate its test split exactly once.

    Masked-modality pretraining is enabled by default and sees only the fold's
    training split. Pass ``use_pretraining=False`` to train from random initialization.
    """

    config.validate()
    set_reproducible_seed(config.seed, config.training.deterministic_algorithms)
    torch.set_float32_matmul_precision("high")
    device = resolve_device(config.device)
    data: DataBundle = prepare_data(config.data, seed=config.seed)

    model = EventTransformer(config.model).to(device)
    run_directory = config.output_dir / config.run_name
    run_directory.mkdir(parents=True, exist_ok=True)
    pretraining_history: list[dict[str, Any]] = []
    pretrained_checkpoint_path: Path | None = None
    if use_pretraining:
        pretraining_output = pretrain_event_transformer(
            model=model,
            train_loader=data.train_loader,
            config=config.pretraining,
            device=device,
            mixed_precision=config.training.mixed_precision,
            show_progress=config.training.show_progress,
            progress_callback=progress_callback,
        )
        pretraining_history = list(pretraining_output.history)
        pretrained_checkpoint_path = run_directory / "pretrained_backbone.pt"
        torch.save(
            _pretraining_checkpoint(
                model=model,
                config=config,
                normalizer=data.normalizer,
                history=pretraining_history,
            ),
            pretrained_checkpoint_path,
        )

    # Fine-tuning deliberately starts with fresh optimizer/scaler state. Only the
    # learned backbone parameters transfer from the reconstruction objective.
    positive_weight = (
        None
        if config.training.positive_class_weight is None
        else torch.tensor(config.training.positive_class_weight, device=device)
    )
    criterion = nn.BCEWithLogitsLoss(pos_weight=positive_weight)
    optimizer = _build_finetuning_optimizer(model, config)
    scheduler = (
        ReduceLROnPlateau(
            optimizer,
            mode="min" if config.training.selection_metric == "loss" else "max",
            factor=config.training.lr_scheduler_factor,
            patience=config.training.lr_scheduler_patience,
            min_lr=config.training.minimum_learning_rate,
        )
        if config.training.lr_scheduler == "reduce_on_plateau"
        else None
    )
    use_amp = config.training.mixed_precision and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    # A classifier-only warm-up is meaningful only for transferred features. Keep
    # random-initialization ablations comparable by training their whole model from
    # the first epoch.
    freeze_backbone_epochs = (
        min(
            config.training.freeze_backbone_epochs,
            config.training.max_epochs,
        )
        if use_pretraining
        else 0
    )
    _set_backbone_trainable(model, trainable=freeze_backbone_epochs == 0)

    selection_metric = config.training.selection_metric
    best_score = math.inf if selection_metric == "loss" else -math.inf
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    best_validation_metrics: dict[str, float] | None = None
    epochs_without_improvement = 0
    history: list[dict[str, Any]] = []

    progress = tqdm(
        range(1, config.training.max_epochs + 1),
        desc=config.run_name,
        disable=not config.training.show_progress,
    )
    for epoch in progress:
        backbone_frozen = epoch <= freeze_backbone_epochs
        if epoch == freeze_backbone_epochs + 1 and freeze_backbone_epochs:
            _set_backbone_trainable(model, trainable=True)
        training_loss = _train_epoch(
            model=model,
            loader=data.train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            gradient_clip_norm=config.training.gradient_clip_norm,
            mixed_precision=config.training.mixed_precision,
            scaler=scaler,
            backbone_frozen=backbone_frozen,
        )
        validation_metrics = evaluate(
            model=model,
            loader=data.validation_loader,
            criterion=criterion,
            device=device,
            threshold=config.training.threshold,
        )
        learning_rates = _optimizer_learning_rates(optimizer)
        history.append(
            {
                "epoch": epoch,
                # Preserve the original field for consumers of existing artifacts.
                "learning_rate": learning_rates["backbone"],
                "backbone_learning_rate": learning_rates["backbone"],
                "classifier_learning_rate": learning_rates["classifier"],
                "backbone_frozen": backbone_frozen,
                "training_loss": training_loss,
                "validation": validation_metrics,
            }
        )
        candidate = validation_metrics[selection_metric]
        progress.set_postfix(
            train_loss=f"{training_loss:.4f}",
            val_loss=f"{validation_metrics['loss']:.4f}",
            val_auc=f"{validation_metrics['roc_auc']:.4f}",
            val_macro_f1=f"{validation_metrics['macro_f1']:.4f}",
        )

        should_stop = False
        if _is_improvement(
            selection_metric,
            candidate,
            best_score,
            config.training.early_stopping_min_delta,
        ):
            best_score = candidate
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            best_validation_metrics = validation_metrics
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= config.training.early_stopping_patience:
                should_stop = True

        if scheduler is not None and math.isfinite(candidate):
            scheduler.step(candidate)

        if progress_callback is not None:
            progress_callback(
                {
                    "event": "epoch_complete",
                    "epoch": epoch,
                    "max_epochs": config.training.max_epochs,
                    "training_loss": training_loss,
                    "backbone_frozen": backbone_frozen,
                    "backbone_learning_rate": learning_rates["backbone"],
                    "classifier_learning_rate": learning_rates["classifier"],
                    "validation_metrics": validation_metrics,
                    "best_epoch": best_epoch,
                    "epochs_without_improvement": epochs_without_improvement,
                    "progress_epoch": (
                        config.pretraining.num_epochs if use_pretraining else 0
                    )
                    + epoch,
                }
            )
        if should_stop:
            break

    if best_state is None or best_validation_metrics is None:
        raise RuntimeError("Training completed without a finite validation metric")

    if progress_callback is not None:
        progress_callback(
            {
                "event": "training_complete",
                "epochs_completed": len(history),
                "best_epoch": best_epoch,
            }
        )

    model.load_state_dict(best_state)
    validation_output = evaluate_with_predictions(
        model,
        data.validation_loader,
        criterion,
        device,
        threshold=config.training.threshold,
    )
    decision_threshold = (
        optimize_binary_threshold(
            validation_output.logits,
            validation_output.labels,
            metric_name=config.training.threshold_metric,
        )
        if config.training.calibrate_threshold_on_validation
        else config.training.threshold
    )
    final_validation_metrics = binary_classification_metrics(
        logits=validation_output.logits,
        labels=validation_output.labels,
        loss=validation_output.metrics["loss"],
        threshold=decision_threshold,
    )
    test_output = evaluate_with_predictions(
        model,
        data.test_loader,
        criterion,
        device,
        threshold=decision_threshold,
    )
    test_metrics = test_output.metrics

    checkpoint_path = run_directory / "best_model.pt"
    history_path = run_directory / "metrics.json"
    validation_predictions_path = run_directory / "validation_predictions.jsonl"
    predictions_path = run_directory / "test_predictions.jsonl"
    torch.save(
        _checkpoint(
            model_state=best_state,
            config=config,
            normalizer=data.normalizer,
            best_epoch=best_epoch,
            validation_metrics=final_validation_metrics,
            test_metrics=test_metrics,
            decision_threshold=decision_threshold,
            pretraining_applied=use_pretraining,
            pretraining_history=pretraining_history,
        ),
        checkpoint_path,
    )
    history_path.write_text(
        json.dumps(
            _json_ready(
                {
                    "best_epoch": best_epoch,
                    "decision_threshold": decision_threshold,
                    "decision_threshold_source": (
                        "validation_calibration"
                        if config.training.calibrate_threshold_on_validation
                        else "fixed"
                    ),
                    "threshold_metric": config.training.threshold_metric,
                    "pretraining": {
                        "enabled": use_pretraining,
                        "checkpoint_path": pretrained_checkpoint_path,
                        "history": pretraining_history,
                    },
                    "validation": final_validation_metrics,
                    "test": test_metrics,
                    "history": history,
                }
            ),
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    validation_predictions_path.write_text(
        "\n".join(_prediction_rows(validation_output, decision_threshold)) + "\n",
        encoding="utf-8",
    )
    prediction_rows = _prediction_rows(test_output, decision_threshold)
    predictions_path.write_text(
        "\n".join(prediction_rows) + "\n",
        encoding="utf-8",
    )

    return TrainingResult(
        checkpoint_path=checkpoint_path,
        pretrained_checkpoint_path=pretrained_checkpoint_path,
        history_path=history_path,
        validation_predictions_path=validation_predictions_path,
        predictions_path=predictions_path,
        run_name=config.run_name,
        best_epoch=best_epoch,
        decision_threshold=decision_threshold,
        validation_metrics=final_validation_metrics,
        test_metrics=test_metrics,
        pretraining_history=pretraining_history,
        history=history,
    )


def load_trained_model(
    checkpoint_path: str | Path,
    device: str = "cpu",
) -> tuple[EventTransformer, FeatureNormalizer | None, dict[str, Any]]:
    """Restore a trained model and its training-only normalization statistics."""

    resolved_device = resolve_device(device)
    checkpoint = torch.load(
        checkpoint_path, map_location=resolved_device, weights_only=False
    )
    checkpoint_version = checkpoint.get("checkpoint_version", 1)
    if checkpoint_version != CHECKPOINT_VERSION:
        raise ValueError(
            "Checkpoint model architecture is incompatible: "
            f"found version {checkpoint_version!r}, expected {CHECKPOINT_VERSION}. "
            "Retrain the fold to create a checkpoint with masked-modality support."
        )
    checkpoint_schema_version = checkpoint.get("feature_schema_version")
    if checkpoint_schema_version != FEATURE_SCHEMA_VERSION:
        raise ValueError(
            "Checkpoint feature schema is incompatible: "
            f"found {checkpoint_schema_version!r}, expected {FEATURE_SCHEMA_VERSION}"
        )
    model_config_data = checkpoint["experiment_config"]["model"]
    model = EventTransformer(
        config=ModelConfig(**model_config_data),
        modality_dims={int(key): value for key, value in checkpoint["modality_dims"].items()},
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(resolved_device).eval()
    normalizer_state = checkpoint.get("normalizer")
    normalizer = (
        None
        if normalizer_state is None
        else FeatureNormalizer.from_state_dict(normalizer_state)
    )
    return model, normalizer, checkpoint
