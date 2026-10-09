"""Three-phase MotionGPT training: tokenizer, motion-language pretraining, instruction tuning.

Phase one fits the shared VQ-VAE on the current fold's training windows.  Phase two trains
the encoder-decoder on a mixture of self-supervised motion-language tasks.  Phase three
tunes it on the supervised task.  Validation drives every selection decision, and the test
split is evaluated once, after the selected checkpoint is restored.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Sequence
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
        optimize_binary_threshold,
    )
    from t2m_gpt.training import (
        resolve_device,
        set_reproducible_seed,
        token_statistics,
        tokenize_split,
        train_motion_vqvae,
    )
    from t2m_gpt.vqvae import MotionVQVAE
except ModuleNotFoundError as error:
    if error.name not in {"event_transformer", "t2m_gpt"}:
        raise
    from ..event_transformer.features import FeatureNormalizer
    from ..event_transformer.metrics import (
        binary_classification_metrics,
        optimize_binary_threshold,
    )
    from ..t2m_gpt.training import (
        resolve_device,
        set_reproducible_seed,
        token_statistics,
        tokenize_split,
        train_motion_vqvae,
    )
    from ..t2m_gpt.vqvae import MotionVQVAE

from .config import ExperimentConfig
from .data import DataBundle, make_task_loader, prepare_data
from .model import MotionLanguageModel, count_parameters
from .tasks import (
    ClassificationTaskDataset,
    PretrainingTaskDataset,
    TaskBatchCollator,
)
from .vocabulary import MotionLanguageVocabulary

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


def _weighted_mean(values: Sequence[tuple[float, int]]) -> float:
    total_weight = sum(weight for _, weight in values)
    if total_weight == 0:
        return math.nan
    return sum(value * weight for value, weight in values) / total_weight


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


def _autocast(device: torch.device, enabled: bool):
    return torch.autocast(
        device_type=device.type, enabled=enabled and device.type == "cuda"
    )


def _move(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


# -----------------------------------------------------------------------------
# Phase two: motion-language pretraining
# -----------------------------------------------------------------------------


def pretrain_motion_language_model(
    model: MotionLanguageModel,
    train_dataset: PretrainingTaskDataset,
    validation_dataset: PretrainingTaskDataset,
    collator: TaskBatchCollator,
    config: ExperimentConfig,
    device: torch.device,
    progress_callback: ProgressCallback | None = None,
) -> list[dict[str, Any]]:
    """Train on the mixture of self-supervised motion-language tasks."""

    settings = config.pretraining
    pin_memory = device.type == "cuda"
    train_loader = make_task_loader(
        train_dataset,
        collator,
        batch_size=settings.batch_size,
        shuffle=True,
        seed=config.seed,
        pin_memory=pin_memory,
    )
    validation_loader = make_task_loader(
        validation_dataset,
        collator,
        batch_size=config.training.evaluation_batch_size,
        shuffle=False,
        seed=config.seed,
        pin_memory=pin_memory,
    )
    optimizer = torch.optim.AdamW(
        model.backbone_parameters(),
        lr=settings.learning_rate,
        weight_decay=settings.weight_decay,
    )

    history: list[dict[str, Any]] = []
    for epoch in range(1, settings.num_epochs + 1):
        # Fresh corruption every epoch, reproducible from the run seed.
        train_dataset.set_epoch(epoch)
        model.train()
        train_losses: list[tuple[float, int]] = []
        for batch in train_loader:
            batch = _move(batch, device)
            optimizer.zero_grad(set_to_none=True)
            loss = model.seq2seq_loss(
                batch["encoder_input"],
                batch["encoder_mask"],
                batch["decoder_input"],
                batch["decoder_mask"],
                batch["labels"],
            )
            loss.backward()
            if settings.gradient_clip_norm is not None:
                nn.utils.clip_grad_norm_(model.parameters(), settings.gradient_clip_norm)
            optimizer.step()
            train_losses.append((float(loss.item()), int(batch["labels"].shape[0])))

        model.eval()
        validation_losses: list[tuple[float, int]] = []
        with torch.no_grad():
            for batch in validation_loader:
                batch = _move(batch, device)
                loss = model.seq2seq_loss(
                    batch["encoder_input"],
                    batch["encoder_mask"],
                    batch["decoder_input"],
                    batch["decoder_mask"],
                    batch["labels"],
                )
                validation_losses.append(
                    (float(loss.item()), int(batch["labels"].shape[0]))
                )

        record = {
            "stage": "motion_language_pretraining",
            "epoch": epoch,
            "tasks": list(settings.tasks),
            "train_task_loss": _weighted_mean(train_losses),
            "validation_task_loss": _weighted_mean(validation_losses),
        }
        history.append(record)
        _report(
            progress_callback,
            config.training.show_progress,
            record,
            f"[pretraining] epoch {epoch:3d}/{settings.num_epochs} "
            f"train_task_loss={record['train_task_loss']:.4f} "
            f"val_task_loss={record['validation_task_loss']:.4f}",
        )

    if config.training.show_progress and progress_callback is None:
        print()
    return history


# -----------------------------------------------------------------------------
# Phase three: instruction tuning on the supervised task
# -----------------------------------------------------------------------------


def _supervised_loss(
    model: MotionLanguageModel,
    batch: dict[str, Any],
    config: ExperimentConfig,
    compute_score: bool = True,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Return the training loss and, when requested, the decision score."""

    if config.model.head == "discriminative":
        score = model.forward_classifier(batch["encoder_input"], batch["encoder_mask"])
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
            score, batch["label"], pos_weight=weight
        )
        return loss, score

    # Motion-to-text: the decoder is trained to write the label word.
    loss = model.seq2seq_loss(
        batch["encoder_input"],
        batch["encoder_mask"],
        batch["decoder_input"],
        batch["decoder_mask"],
        batch["labels"],
    )
    if not compute_score:
        return loss, None
    score = model.answer_logit(batch["encoder_input"], batch["encoder_mask"])
    return loss, score


@torch.no_grad()
def evaluate_with_predictions(
    model: MotionLanguageModel,
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
        batch = _move(batch, device)
        loss, score = _supervised_loss(model, batch, config)
        if score is None:
            raise RuntimeError("Evaluation requires a decision score")
        losses.append((float(loss.item()), int(batch["label"].shape[0])))
        all_scores.append(score.detach().float().cpu())
        all_labels.append(batch["label"].detach().float().cpu())
        sample_ids.extend(batch["sample_id"])

    scores = torch.cat(all_scores) if all_scores else torch.empty(0)
    labels = torch.cat(all_labels) if all_labels else torch.empty(0)
    metrics = binary_classification_metrics(
        scores,
        labels,
        _weighted_mean(losses),
        config.training.threshold if threshold is None else threshold,
    )
    return EvaluationOutput(
        metrics=metrics, sample_ids=sample_ids, labels=labels, logits=scores
    )


def _build_optimizer(
    model: MotionLanguageModel,
    config: ExperimentConfig,
) -> torch.optim.Optimizer:
    settings = config.training
    if config.model.head == "motion_to_text":
        # The classifier head is unused by the generative formulation.
        groups = [{"params": model.backbone_parameters(), "lr": settings.learning_rate}]
    elif settings.classifier_learning_rate is None:
        groups = [{"params": list(model.parameters()), "lr": settings.learning_rate}]
    else:
        groups = [
            {"params": model.backbone_parameters(), "lr": settings.learning_rate},
            {"params": model.head_parameters(), "lr": settings.classifier_learning_rate},
        ]
    return torch.optim.AdamW(groups, weight_decay=settings.weight_decay)


def _set_backbone_requires_grad(model: MotionLanguageModel, requires_grad: bool) -> None:
    for parameter in model.backbone_parameters():
        parameter.requires_grad_(requires_grad)


def instruction_tune(
    model: MotionLanguageModel,
    task_datasets: dict[str, ClassificationTaskDataset],
    collator: TaskBatchCollator,
    config: ExperimentConfig,
    device: torch.device,
    progress_callback: ProgressCallback | None = None,
) -> tuple[list[dict[str, Any]], dict[str, torch.Tensor], int]:
    """Tune on the supervised task, selecting the epoch on validation only."""

    settings = config.training
    pin_memory = device.type == "cuda"
    train_loader = make_task_loader(
        task_datasets["train"],
        collator,
        batch_size=settings.batch_size,
        shuffle=True,
        seed=config.seed,
        pin_memory=pin_memory,
    )
    validation_loader = make_task_loader(
        task_datasets["validation"],
        collator,
        batch_size=settings.evaluation_batch_size,
        shuffle=False,
        seed=config.seed,
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
    freeze_epochs = (
        0 if config.model.head == "motion_to_text" else settings.freeze_backbone_epochs
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
            batch = _move(batch, device)
            optimizer.zero_grad(set_to_none=True)
            with _autocast(device, settings.mixed_precision):
                loss, _ = _supervised_loss(model, batch, config, compute_score=False)
            scaler.scale(loss).backward()
            if settings.gradient_clip_norm is not None:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), settings.gradient_clip_norm)
            scaler.step(optimizer)
            scaler.update()
            train_losses.append((float(loss.item()), int(batch["label"].shape[0])))

        validation_output = evaluate_with_predictions(
            model, validation_loader, config, device
        )
        selection_value = validation_output.metrics[settings.selection_metric]
        if scheduler is not None:
            scheduler.step(selection_value)

        record = {
            "stage": "instruction_tuning",
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
            f"[{config.model.head}] epoch {epoch:3d}/{settings.max_epochs} "
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


def _checkpoint(
    model_state: dict[str, torch.Tensor],
    config: ExperimentConfig,
    data: DataBundle,
    vocabulary: MotionLanguageVocabulary,
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
        "checkpoint_type": f"motion_gpt_{config.model.head}",
        "model_state_dict": model_state,
        "model_config": _json_ready(asdict(config.model)),
        "vqvae_config": _json_ready(asdict(config.vqvae)),
        "experiment_config": _config_signature(config),
        "normalizer": None if data.normalizer is None else data.normalizer.state_dict(),
        "time_grid": data.time_grid,
        "channel_names": data.channel_names,
        "vocabulary": {
            "num_codes": vocabulary.num_codes,
            "num_sentinels": vocabulary.num_sentinels,
            "task_names": list(vocabulary.task_names),
            "answer_words": list(vocabulary.answer_words),
            "size": vocabulary.size,
        },
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


def build_vocabulary(config: ExperimentConfig) -> MotionLanguageVocabulary:
    return MotionLanguageVocabulary(
        num_codes=config.vqvae.num_codes,
        num_sentinels=config.model.num_sentinels,
    )


def train_motion_gpt(
    config: ExperimentConfig,
    progress_callback: ProgressCallback | None = None,
) -> TrainingResult:
    """Run all three phases for one fold and evaluate the test split exactly once."""

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

    # Phase one is shared with T2M-GPT: the same tokenizer on the same channels.
    tokenizer, tokenizer_history, tokenizer_metrics = train_motion_vqvae(
        data, config, device, progress_callback
    )
    torch.save(
        {
            "checkpoint_version": CHECKPOINT_VERSION,
            "checkpoint_type": "motion_gpt_motion_vqvae",
            "model_state_dict": tokenizer.state_dict(),
            "vqvae_config": _json_ready(asdict(config.vqvae)),
            "experiment_config": _config_signature(config),
            "input_dim": data.num_channels,
            "num_frames": data.num_frames,
            "channel_names": data.channel_names,
            "history": tokenizer_history,
            "validation_metrics": tokenizer_metrics,
        },
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

    vocabulary = build_vocabulary(config)
    collator = TaskBatchCollator(vocabulary)
    classification_datasets = {
        split: ClassificationTaskDataset(
            token_datasets[split].indices,
            token_datasets[split].labels,
            token_datasets[split].sample_ids,
            vocabulary,
        )
        for split in SPLIT_NAMES
    }

    set_reproducible_seed(config.seed, config.training.deterministic_algorithms)
    model = MotionLanguageModel(vocabulary, config.model).to(device)

    pretraining_history: list[dict[str, Any]] = []
    if config.use_pretraining:
        pretraining_datasets = {
            split: PretrainingTaskDataset(
                token_datasets[split].indices,
                token_datasets[split].labels,
                token_datasets[split].sample_ids,
                vocabulary,
                config.pretraining,
                seed=config.seed,
            )
            for split in ("train", "validation")
        }
        pretraining_history = pretrain_motion_language_model(
            model,
            pretraining_datasets["train"],
            pretraining_datasets["validation"],
            collator,
            config,
            device,
            progress_callback,
        )
        torch.save(
            {
                "checkpoint_version": CHECKPOINT_VERSION,
                "checkpoint_type": "motion_gpt_pretrained_backbone",
                "model_state_dict": model.state_dict(),
                "model_config": _json_ready(asdict(config.model)),
                "num_tokens": num_tokens,
                "pretraining_history": pretraining_history,
            },
            pretrained_path,
        )

    history, best_state, best_epoch = instruction_tune(
        model, classification_datasets, collator, config, device, progress_callback
    )
    model.load_state_dict(best_state)

    pin_memory = device.type == "cuda"
    validation_loader = make_task_loader(
        classification_datasets["validation"],
        collator,
        batch_size=config.training.evaluation_batch_size,
        shuffle=False,
        seed=config.seed,
        pin_memory=pin_memory,
    )
    test_loader = make_task_loader(
        classification_datasets["test"],
        collator,
        batch_size=config.training.evaluation_batch_size,
        shuffle=False,
        seed=config.seed,
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
        validation_predictions_path,
        _prediction_rows(validation_output, decision_threshold),
    )
    _write_lines(test_predictions_path, _prediction_rows(test_output, decision_threshold))

    model_parameter_count = count_parameters(model, trainable_only=False)
    torch.save(
        _checkpoint(
            model_state=best_state,
            config=config,
            data=data,
            vocabulary=vocabulary,
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
                    "vocabulary_size": vocabulary.size,
                    "split_sizes": {
                        split: len(classification_datasets[split])
                        for split in SPLIT_NAMES
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
) -> tuple[MotionLanguageModel, MotionVQVAE, FeatureNormalizer | None, dict[str, Any]]:
    """Restore the tokenizer, the sequence-to-sequence model, and the normalizer."""

    config.validate()
    run_dir = Path(config.output_dir) / config.run_name
    checkpoint = _load_checkpoint(run_dir / "best_model.pt")
    tokenizer_checkpoint = _load_checkpoint(run_dir / "motion_vqvae.pt")

    tokenizer = MotionVQVAE(tokenizer_checkpoint["input_dim"], config.vqvae)
    tokenizer.load_state_dict(tokenizer_checkpoint["model_state_dict"])
    tokenizer.eval()

    model = MotionLanguageModel(build_vocabulary(config), config.model)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    normalizer_state = checkpoint.get("normalizer")
    normalizer = (
        None
        if normalizer_state is None
        else FeatureNormalizer.from_state_dict(normalizer_state)
    )
    return model, tokenizer, normalizer, checkpoint


def load_training_result(config: ExperimentConfig) -> TrainingResult:
    """Reload a finished fold, refusing checkpoints from a different configuration."""

    config.validate()
    run_dir = Path(config.output_dir) / config.run_name
    checkpoint = _load_checkpoint(run_dir / "best_model.pt")
    if checkpoint.get("experiment_config") != _config_signature(config):
        raise ValueError(
            f"Checkpoint in {run_dir} was produced by a different configuration"
        )
    for required in (
        "validation_predictions.jsonl",
        "test_predictions.jsonl",
        "metrics.json",
    ):
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
    "build_vocabulary",
    "evaluate_with_predictions",
    "instruction_tune",
    "load_trained_model",
    "load_training_result",
    "pretrain_motion_language_model",
    "train_motion_gpt",
]
