"""Training, evaluation, and checkpointing for the frame-level negation tagger.

Two kinds of output are produced deliberately. Span metrics describe what the tagger is
for -- locating negation in time. A **window-level** score is also written, in the same
``test_predictions.jsonl`` schema every other model in this repository uses, by taking each
window's maximum per-frame positive marginal. That makes the tagger directly comparable
with the ROCKET, TCN, T2M-GPT, and MotionGPT window classifiers through
:mod:`src.comparison_stats`, instead of being an experiment whose numbers cannot be placed
next to anything.
"""

from __future__ import annotations

import json
import random
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset

try:
    from t2m_gpt.data import load_fold_dataset, prepare_data
except ModuleNotFoundError as error:  # pragma: no cover - import-path fallback
    if error.name != "t2m_gpt":
        raise
    from ..t2m_gpt.data import load_fold_dataset, prepare_data

from .config import TaggerExperimentConfig
from .labels import (
    OUTSIDE,
    TAG_NAMES,
    FrameLabelStats,
    median_anchor_duration_ms,
    split_frame_labels,
)
from .metrics import TaggerMetrics, evaluate_tagging
from .model import FrameTagger, count_parameters


def resolve_device(device: str) -> torch.device:
    if device != "auto":
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class FrameTaggingDataset(Dataset[dict[str, torch.Tensor]]):
    """One split as channel arrays with per-frame BIO targets."""

    def __init__(self, values: np.ndarray, tags: np.ndarray) -> None:
        if values.shape[0] != tags.shape[0]:
            raise ValueError("values and tags must describe the same windows")
        if values.shape[2] != tags.shape[1]:
            raise ValueError("values and tags must have the same frame count")
        self.values = values
        self.tags = tags

    def __len__(self) -> int:
        return int(self.values.shape[0])

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "values": torch.from_numpy(np.ascontiguousarray(self.values[index])).float(),
            "tags": torch.from_numpy(np.ascontiguousarray(self.tags[index])).long(),
            "index": torch.tensor(index, dtype=torch.long),
        }


@dataclass(slots=True)
class SplitPredictions:
    """Decoded tags plus the window-level score derived from them."""

    tags: np.ndarray
    frame_probability: np.ndarray
    window_probability: np.ndarray
    window_labels: np.ndarray
    sample_ids: list[str]


@dataclass(slots=True)
class TaggerTrainingResult:
    """Everything one fold produces."""

    config: TaggerExperimentConfig
    best_epoch: int
    model_parameter_count: int
    label_stats: dict[str, dict[str, float]]
    validation_metrics: dict[str, Any]
    test_metrics: dict[str, Any]
    history: list[dict[str, float]] = field(default_factory=list)
    checkpoint_path: str = ""


def _derive_labels(
    config: TaggerExperimentConfig, time_grid: np.ndarray
) -> tuple[dict[str, np.ndarray], dict[str, FrameLabelStats], float]:
    fold = load_fold_dataset(config.data.dataset)
    words = {split: list(fold[split]["word"]) for split in ("train", "validation", "test")}
    assumed = (
        config.labels.assumed_token_duration_ms
        if config.labels.assumed_token_duration_ms is not None
        else median_anchor_duration_ms(words["train"], config.labels.fallback_duration_ms)
    )
    tags: dict[str, np.ndarray] = {}
    stats: dict[str, FrameLabelStats] = {}
    for split, split_words in words.items():
        tags[split], stats[split] = split_frame_labels(split_words, time_grid, assumed)
    return tags, stats, assumed


def _evaluate(
    model: FrameTagger,
    loader: DataLoader,
    device: torch.device,
    window_labels: np.ndarray,
    sample_ids: Sequence[str],
    minimum_overlap: float,
) -> tuple[TaggerMetrics, SplitPredictions, float]:
    model.eval()
    total_loss, total_frames = 0.0, 0
    decoded: list[np.ndarray] = []
    frame_probabilities: list[np.ndarray] = []
    references: list[np.ndarray] = []
    order: list[int] = []

    with torch.no_grad():
        for batch in loader:
            values = batch["values"].to(device)
            tags = batch["tags"].to(device)
            emissions = model(values)
            loss = model.crf(emissions, tags, None, reduction="sum")
            total_loss += float(loss.item())
            total_frames += int(tags.numel())
            predicted = model.crf.decode(emissions)
            probability = model.crf.marginal_positive_probability(
                emissions, model.positive_tag_ids
            )
            decoded.append(predicted.cpu().numpy())
            frame_probabilities.append(probability.cpu().numpy())
            references.append(tags.cpu().numpy())
            order.extend(int(value) for value in batch["index"].tolist())

    predicted_tags = np.concatenate(decoded, axis=0)
    probabilities = np.concatenate(frame_probabilities, axis=0)
    reference_tags = np.concatenate(references, axis=0)

    # Restore dataset order so predictions line up with sample ids.
    restore = np.argsort(np.asarray(order))
    predicted_tags = predicted_tags[restore]
    probabilities = probabilities[restore]
    reference_tags = reference_tags[restore]

    metrics = evaluate_tagging(predicted_tags, reference_tags, minimum_overlap)
    window_probability = probabilities.max(axis=1)
    predictions = SplitPredictions(
        tags=predicted_tags,
        frame_probability=probabilities,
        window_probability=window_probability,
        window_labels=window_labels,
        sample_ids=list(sample_ids),
    )
    average_loss = total_loss / total_frames if total_frames else float("nan")
    return metrics, predictions, average_loss


def _window_metrics(predictions: SplitPredictions, threshold: float = 0.5) -> dict[str, float]:
    labels = predictions.window_labels
    scores = predictions.window_probability
    result: dict[str, float] = {}
    if len(np.unique(labels)) == 2:
        result["window_roc_auc"] = float(roc_auc_score(labels, scores))
    predicted = (scores >= threshold).astype(np.int64)
    true_positive = int(((predicted == 1) & (labels == 1)).sum())
    false_positive = int(((predicted == 1) & (labels == 0)).sum())
    false_negative = int(((predicted == 0) & (labels == 1)).sum())
    precision = (
        true_positive / (true_positive + false_positive)
        if true_positive + false_positive
        else 0.0
    )
    recall = (
        true_positive / (true_positive + false_negative)
        if true_positive + false_negative
        else 0.0
    )
    result["window_accuracy"] = float((predicted == labels).mean())
    result["window_f1"] = (
        2 * precision * recall / (precision + recall) if precision + recall else 0.0
    )
    return result


def _write_predictions(path: Path, predictions: SplitPredictions, threshold: float) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for index, sample_id in enumerate(predictions.sample_ids):
            probability = float(predictions.window_probability[index])
            handle.write(
                json.dumps(
                    {
                        "sample_id": sample_id,
                        "label": int(predictions.window_labels[index]),
                        # A marginal probability has no logit; report its log-odds so the
                        # column keeps the meaning it has for the other models.
                        "logit": float(
                            np.log(np.clip(probability, 1e-9, 1 - 1e-9))
                            - np.log(1 - np.clip(probability, 1e-9, 1 - 1e-9))
                        ),
                        "probability": probability,
                        "prediction": int(probability >= threshold),
                        "tags": [TAG_NAMES[int(tag)] for tag in predictions.tags[index]],
                    }
                )
                + "\n"
            )


def train_frame_tagger(config: TaggerExperimentConfig) -> TaggerTrainingResult:
    """Train one fold of the frame tagger and write its artifacts."""

    config.validate()
    seed_everything(config.seed)
    device = resolve_device(config.device)

    data = prepare_data(config.data)
    tags, label_stats, assumed = _derive_labels(config, data.time_grid)
    for split, split_tags in tags.items():
        expected = getattr(data, split).values.shape[0]
        if split_tags.shape[0] != expected:
            raise RuntimeError(
                f"{split} has {split_tags.shape[0]} label rows for {expected} windows; "
                "the fold rows and the encoded windows are out of alignment"
            )

    loaders = {}
    for split in ("train", "validation", "test"):
        split_data = getattr(data, split)
        dataset = FrameTaggingDataset(split_data.values, tags[split])
        loaders[split] = DataLoader(
            dataset,
            batch_size=(
                config.training.batch_size
                if split == "train"
                else config.training.evaluation_batch_size
            ),
            shuffle=split == "train",
        )

    model = FrameTagger(
        num_channels=data.train.values.shape[1],
        encoder=config.tagger.encoder,
        d_model=config.tagger.d_model,
        num_layers=config.tagger.num_layers,
        nhead=config.tagger.nhead,
        dim_feedforward=config.tagger.dim_feedforward,
        dropout=config.tagger.dropout,
        max_frames=max(256, data.time_grid.shape[0]),
        constrain_transitions=config.tagger.constrain_transitions,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min" if config.training.selection_metric == "loss" else "max",
        factor=config.training.lr_scheduler_factor,
        patience=config.training.lr_scheduler_patience,
        min_lr=config.training.minimum_learning_rate,
    )

    output_dir = Path(config.output_dir) / config.run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "best_model.pt"

    best_score = float("inf") if config.training.selection_metric == "loss" else -1.0
    best_epoch = 0
    history: list[dict[str, float]] = []
    epochs_without_improvement = 0

    for epoch in range(1, config.training.max_epochs + 1):
        model.train()
        epoch_loss, epoch_frames = 0.0, 0
        for batch in loaders["train"]:
            values = batch["values"].to(device)
            batch_tags = batch["tags"].to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = model.loss(values, batch_tags, reduction="token_mean")
            loss.backward()
            if config.training.gradient_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), config.training.gradient_clip_norm
                )
            optimizer.step()
            epoch_loss += float(loss.item()) * int(batch_tags.numel())
            epoch_frames += int(batch_tags.numel())

        validation_metrics, _, validation_loss = _evaluate(
            model,
            loaders["validation"],
            device,
            data.validation.labels,
            data.validation.sample_ids,
            config.training.minimum_span_overlap,
        )
        score = (
            validation_loss
            if config.training.selection_metric == "loss"
            else validation_metrics.primary
        )
        improved = (
            score < best_score
            if config.training.selection_metric == "loss"
            else score > best_score
        )
        if improved:
            best_score, best_epoch = score, epoch
            epochs_without_improvement = 0
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "experiment_config": json.loads(json.dumps(asdict(config), default=str)),
                    "channel_names": data.channel_names,
                    "time_grid": data.time_grid.tolist(),
                    "tag_names": TAG_NAMES,
                    "assumed_token_duration_ms": assumed,
                    "epoch": epoch,
                },
                checkpoint_path,
            )
        else:
            epochs_without_improvement += 1

        scheduler.step(score)
        record = {
            "epoch": epoch,
            "train_loss": epoch_loss / epoch_frames if epoch_frames else float("nan"),
            "validation_loss": validation_loss,
            "validation_span_f1": validation_metrics.primary,
            "validation_frame_f1": validation_metrics.frame["frame_f1"],
        }
        history.append(record)
        if config.training.show_progress:
            print(
                f"[tagger] epoch {epoch:3d}/{config.training.max_epochs} "
                f"train_loss={record['train_loss']:.4f} "
                f"val_loss={validation_loss:.4f} "
                f"val_span_f1={validation_metrics.primary:.4f} "
                f"val_frame_f1={record['validation_frame_f1']:.4f} best={best_epoch}",
                flush=True,
            )
        if epochs_without_improvement >= config.training.early_stopping_patience:
            break

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])

    results: dict[str, dict[str, Any]] = {}
    for split in ("validation", "test"):
        split_data = getattr(data, split)
        metrics, predictions, loss = _evaluate(
            model,
            loaders[split],
            device,
            split_data.labels,
            split_data.sample_ids,
            config.training.minimum_span_overlap,
        )
        payload = metrics.as_dict()
        payload["loss"] = loss
        payload.update(_window_metrics(predictions))
        results[split] = payload
        _write_predictions(output_dir / f"{split}_predictions.jsonl", predictions, 0.5)

    result = TaggerTrainingResult(
        config=config,
        best_epoch=best_epoch,
        model_parameter_count=count_parameters(model),
        label_stats={split: stats.as_dict() for split, stats in label_stats.items()},
        validation_metrics=results["validation"],
        test_metrics=results["test"],
        history=history,
        checkpoint_path=str(checkpoint_path),
    )
    (output_dir / "metrics.json").write_text(
        json.dumps(
            {
                "best_epoch": result.best_epoch,
                "model_parameter_count": result.model_parameter_count,
                "assumed_token_duration_ms": assumed,
                "label_stats": result.label_stats,
                "validation": result.validation_metrics,
                "test": result.test_metrics,
                "history": result.history,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return result


__all__ = [
    "FrameTaggingDataset",
    "SplitPredictions",
    "TaggerTrainingResult",
    "resolve_device",
    "seed_everything",
    "train_frame_tagger",
]
