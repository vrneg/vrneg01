"""Training and evaluation for the modality-aware Inception-TCN model."""

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

try:
    from event_transformer.features import FeatureNormalizer
    from event_transformer.metrics import (
        binary_classification_metrics,
        optimize_binary_threshold,
    )
except ModuleNotFoundError as error:
    if error.name != "event_transformer":
        raise
    from ..event_transformer.features import FeatureNormalizer
    from ..event_transformer.metrics import (
        binary_classification_metrics,
        optimize_binary_threshold,
    )

from .config import ExperimentConfig, ModelConfig
from .data import DataBundle, FixedGridTorchDataset, TimeSeriesSplit, prepare_data
from .model import ModalityAwareInceptionTCN


ProgressCallback = Callable[[dict[str, Any]], None]
CHECKPOINT_VERSION = 1


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


def _make_loader(
    split: TimeSeriesSplit,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
    seed: int,
    pin_memory: bool,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        FixedGridTorchDataset(split),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        generator=generator if shuffle else None,
    )


def _make_loaders(
    data: DataBundle,
    config: ExperimentConfig,
    device: torch.device,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    loader_config = config.training
    pin_memory = device.type == "cuda"
    return (
        _make_loader(
            data.train,
            loader_config.batch_size,
            loader_config.num_workers,
            True,
            config.seed,
            pin_memory,
        ),
        _make_loader(
            data.validation,
            loader_config.evaluation_batch_size,
            loader_config.num_workers,
            False,
            config.seed,
            pin_memory,
        ),
        _make_loader(
            data.test,
            loader_config.evaluation_batch_size,
            loader_config.num_workers,
            False,
            config.seed,
            pin_memory,
        ),
    )


def _move_batch(
    batch: dict[str, Any],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    values = batch["values"].to(device, non_blocking=True)
    labels = batch["label"].to(device, non_blocking=True)
    return values, labels


@torch.no_grad()
def evaluate_with_predictions(
    model: ModalityAwareInceptionTCN,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    threshold: float,
) -> EvaluationOutput:
    model.eval()
    all_logits: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []
    sample_ids: list[str] = []
    loss_sum = 0.0
    num_examples = 0

    for batch in loader:
        sample_ids.extend(str(sample_id) for sample_id in batch["sample_id"])
        values, labels = _move_batch(batch, device)
        logits = model(values)
        loss = criterion(logits, labels)
        batch_size = int(labels.shape[0])
        loss_sum += float(loss.item()) * batch_size
        num_examples += batch_size
        all_logits.append(logits.detach().cpu())
        all_labels.append(labels.detach().cpu())

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


def _train_epoch(
    model: ModalityAwareInceptionTCN,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: AdamW,
    device: torch.device,
    gradient_clip_norm: float | None,
    scaler: torch.amp.GradScaler,
    use_amp: bool,
) -> float:
    model.train()
    loss_sum = 0.0
    num_examples = 0

    for batch in loader:
        values, labels = _move_batch(batch, device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=use_amp,
        ):
            logits = model(values)
            loss = criterion(logits, labels)

        scaler.scale(loss).backward()
        if gradient_clip_norm is not None:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
        scaler.step(optimizer)
        scaler.update()

        batch_size = int(labels.shape[0])
        loss_sum += float(loss.item()) * batch_size
        num_examples += batch_size

    if num_examples == 0:
        raise ValueError("Cannot train on an empty dataset")
    return loss_sum / num_examples


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
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value


def _config_signature(config: ExperimentConfig) -> dict[str, Any]:
    """Return the result-affecting config, excluding progress rendering."""

    signature = _json_ready(asdict(config))
    signature["training"]["show_progress"] = False
    return signature


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


def _checkpoint(
    model_state: dict[str, torch.Tensor],
    config: ExperimentConfig,
    data: DataBundle,
    best_epoch: int,
    decision_threshold: float,
    validation_metrics: dict[str, float],
    test_metrics: dict[str, float],
    model_parameter_count: int,
    history: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "checkpoint_version": CHECKPOINT_VERSION,
        "checkpoint_type": "modality_aware_inception_tcn_classifier",
        "model_state_dict": model_state,
        "model_config": _json_ready(asdict(config.model)),
        "experiment_config": _config_signature(config),
        "normalizer": (
            None if data.normalizer is None else data.normalizer.state_dict()
        ),
        "time_grid": data.time_grid,
        "channel_names": data.channel_names,
        "best_epoch": best_epoch,
        "decision_threshold": decision_threshold,
        "validation_metrics": validation_metrics,
        "test_metrics": test_metrics,
        "model_parameter_count": model_parameter_count,
        "history": history,
    }


def train_inception_tcn(
    config: ExperimentConfig,
    progress_callback: ProgressCallback | None = None,
) -> TrainingResult:
    """Train one fold, select on validation only, and evaluate test once."""

    config.validate()
    set_reproducible_seed(
        config.seed,
        deterministic_algorithms=config.training.deterministic_algorithms,
    )
    torch.set_float32_matmul_precision("high")
    device = resolve_device(config.device)
    data = prepare_data(config.data)
    train_loader, validation_loader, test_loader = _make_loaders(
        data, config, device
    )
    model = ModalityAwareInceptionTCN(config.model, data.channel_names).to(device)

    positive_weight = (
        None
        if config.training.positive_class_weight is None
        else torch.tensor(config.training.positive_class_weight, device=device)
    )
    criterion = nn.BCEWithLogitsLoss(pos_weight=positive_weight)
    optimizer = AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min" if config.training.selection_metric == "loss" else "max",
        factor=config.training.lr_scheduler_factor,
        patience=config.training.lr_scheduler_patience,
        min_lr=config.training.minimum_learning_rate,
    )
    use_amp = config.training.mixed_precision and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    selection_metric = config.training.selection_metric
    best_score = math.inf if selection_metric == "loss" else -math.inf
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    epochs_without_improvement = 0
    history: list[dict[str, Any]] = []

    progress = tqdm(
        range(1, config.training.max_epochs + 1),
        desc=config.run_name,
        disable=not config.training.show_progress,
        dynamic_ncols=True,
    )
    for epoch in progress:
        training_loss = _train_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            gradient_clip_norm=config.training.gradient_clip_norm,
            scaler=scaler,
            use_amp=use_amp,
        )
        validation_output = evaluate_with_predictions(
            model,
            validation_loader,
            criterion,
            device,
            config.training.threshold,
        )
        validation_metrics = validation_output.metrics
        learning_rate = float(optimizer.param_groups[0]["lr"])
        history.append(
            {
                "epoch": epoch,
                "learning_rate": learning_rate,
                "training_loss": training_loss,
                "validation": validation_metrics,
            }
        )
        progress.set_postfix(
            train_loss=f"{training_loss:.4f}",
            val_loss=f"{validation_metrics['loss']:.4f}",
            val_auc=f"{validation_metrics['roc_auc']:.4f}",
            val_macro_f1=f"{validation_metrics['macro_f1']:.4f}",
        )

        candidate = validation_metrics[selection_metric]
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
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if math.isfinite(candidate):
            scheduler.step(candidate)
        if progress_callback is not None:
            progress_callback(
                {
                    "event": "epoch_complete",
                    "epoch": epoch,
                    "training_loss": training_loss,
                    "validation_metrics": validation_metrics,
                    "best_epoch": best_epoch,
                    "epochs_without_improvement": epochs_without_improvement,
                }
            )
        if epochs_without_improvement >= config.training.early_stopping_patience:
            break

    if best_state is None:
        raise RuntimeError("Training completed without a finite validation metric")

    model.load_state_dict(best_state)
    validation_output = evaluate_with_predictions(
        model,
        validation_loader,
        criterion,
        device,
        config.training.threshold,
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
    validation_metrics = binary_classification_metrics(
        logits=validation_output.logits,
        labels=validation_output.labels,
        loss=validation_output.metrics["loss"],
        threshold=decision_threshold,
    )
    test_output = evaluate_with_predictions(
        model,
        test_loader,
        criterion,
        device,
        decision_threshold,
    )
    test_metrics = test_output.metrics

    run_directory = config.output_dir / config.run_name
    run_directory.mkdir(parents=True, exist_ok=True)
    checkpoint_path = run_directory / "best_model.pt"
    history_path = run_directory / "metrics.json"
    validation_predictions_path = run_directory / "validation_predictions.jsonl"
    predictions_path = run_directory / "test_predictions.jsonl"
    model_parameter_count = model.parameter_count
    checkpoint = _checkpoint(
        model_state=best_state,
        config=config,
        data=data,
        best_epoch=best_epoch,
        decision_threshold=decision_threshold,
        validation_metrics=validation_metrics,
        test_metrics=test_metrics,
        model_parameter_count=model_parameter_count,
        history=history,
    )
    torch.save(checkpoint, checkpoint_path)
    history_path.write_text(
        json.dumps(
            _json_ready(
                {
                    "best_epoch": best_epoch,
                    "model_parameter_count": model_parameter_count,
                    "decision_threshold": decision_threshold,
                    "decision_threshold_source": (
                        "validation_calibration"
                        if config.training.calibrate_threshold_on_validation
                        else "fixed"
                    ),
                    "threshold_metric": config.training.threshold_metric,
                    "validation": validation_metrics,
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
    predictions_path.write_text(
        "\n".join(_prediction_rows(test_output, decision_threshold)) + "\n",
        encoding="utf-8",
    )

    if progress_callback is not None:
        progress_callback(
            {
                "event": "training_complete",
                "best_epoch": best_epoch,
                "test_metrics": test_metrics,
            }
        )
    return TrainingResult(
        checkpoint_path=checkpoint_path,
        history_path=history_path,
        validation_predictions_path=validation_predictions_path,
        predictions_path=predictions_path,
        run_name=config.run_name,
        best_epoch=best_epoch,
        decision_threshold=decision_threshold,
        validation_metrics=validation_metrics,
        test_metrics=test_metrics,
        model_parameter_count=model_parameter_count,
        history=history,
    )


def _load_checkpoint(
    checkpoint_path: str | Path,
    map_location: torch.device | str = "cpu",
) -> dict[str, Any]:
    checkpoint = torch.load(
        checkpoint_path,
        map_location=map_location,
        weights_only=False,
    )
    version = checkpoint.get("checkpoint_version")
    if version != CHECKPOINT_VERSION:
        raise ValueError(
            f"Unsupported Inception-TCN checkpoint version {version!r}; "
            f"expected {CHECKPOINT_VERSION}"
        )
    return checkpoint


def load_trained_model(
    checkpoint_path: str | Path,
    device: str = "cpu",
) -> tuple[ModalityAwareInceptionTCN, FeatureNormalizer | None, dict[str, Any]]:
    """Restore a trained model and its training-only normalization state."""

    resolved_device = resolve_device(device)
    checkpoint = _load_checkpoint(checkpoint_path, resolved_device)
    model_config_data = dict(checkpoint["model_config"])
    model_config_data["inception_kernel_sizes"] = tuple(
        model_config_data["inception_kernel_sizes"]
    )
    model_config_data["tcn_dilations"] = tuple(model_config_data["tcn_dilations"])
    model = ModalityAwareInceptionTCN(
        ModelConfig(**model_config_data),
        tuple(checkpoint["channel_names"]),
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


def load_training_result(config: ExperimentConfig) -> TrainingResult:
    """Load a complete, configuration-matched fold for CV resumption."""

    config.validate()
    run_directory = config.output_dir / config.run_name
    checkpoint_path = run_directory / "best_model.pt"
    history_path = run_directory / "metrics.json"
    validation_predictions_path = run_directory / "validation_predictions.jsonl"
    predictions_path = run_directory / "test_predictions.jsonl"
    required_paths = (
        checkpoint_path,
        history_path,
        validation_predictions_path,
        predictions_path,
    )
    missing_paths = [path for path in required_paths if not path.is_file()]
    if missing_paths:
        missing = ", ".join(str(path) for path in missing_paths)
        raise FileNotFoundError(f"Incomplete Inception-TCN run; missing: {missing}")

    checkpoint = _load_checkpoint(checkpoint_path)
    if checkpoint.get("experiment_config") != _config_signature(config):
        raise ValueError(
            f"Saved Inception-TCN run {run_directory} was created with a "
            "different experiment configuration"
        )
    return TrainingResult(
        checkpoint_path=checkpoint_path,
        history_path=history_path,
        validation_predictions_path=validation_predictions_path,
        predictions_path=predictions_path,
        run_name=config.run_name,
        best_epoch=int(checkpoint["best_epoch"]),
        decision_threshold=float(checkpoint["decision_threshold"]),
        validation_metrics=dict(checkpoint["validation_metrics"]),
        test_metrics=dict(checkpoint["test_metrics"]),
        model_parameter_count=int(checkpoint["model_parameter_count"]),
        history=list(checkpoint["history"]),
    )


__all__ = [
    "EvaluationOutput",
    "TrainingResult",
    "evaluate_with_predictions",
    "load_trained_model",
    "load_training_result",
    "resolve_device",
    "set_reproducible_seed",
    "train_inception_tcn",
]
