"""Binary classification metrics used during validation and testing."""

from __future__ import annotations

import math

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score


THRESHOLD_OPTIMIZATION_METRICS = frozenset(
    {
        "accuracy",
        "balanced_accuracy",
        "precision",
        "recall",
        "specificity",
        "f1",
        "negative_f1",
        "macro_f1",
        "weighted_f1",
        "matthews_correlation_coefficient",
    }
)


def _threshold_metrics(
    targets: np.ndarray,
    predictions: np.ndarray,
) -> dict[str, float]:
    true_positive = int(((predictions == 1) & (targets == 1)).sum())
    true_negative = int(((predictions == 0) & (targets == 0)).sum())
    false_positive = int(((predictions == 1) & (targets == 0)).sum())
    false_negative = int(((predictions == 0) & (targets == 1)).sum())

    total = len(targets)
    positive_support = true_positive + false_negative
    negative_support = true_negative + false_positive
    positive_precision_denominator = true_positive + false_positive
    negative_precision_denominator = true_negative + false_negative
    positive_precision = (
        true_positive / positive_precision_denominator
        if positive_precision_denominator
        else 0.0
    )
    positive_recall = true_positive / positive_support if positive_support else 0.0
    negative_precision = (
        true_negative / negative_precision_denominator
        if negative_precision_denominator
        else 0.0
    )
    negative_recall = true_negative / negative_support if negative_support else 0.0
    positive_f1 = (
        2.0 * positive_precision * positive_recall
        / (positive_precision + positive_recall)
        if positive_precision + positive_recall
        else 0.0
    )
    negative_f1 = (
        2.0 * negative_precision * negative_recall
        / (negative_precision + negative_recall)
        if negative_precision + negative_recall
        else 0.0
    )
    mcc_denominator = math.sqrt(
        (true_positive + false_positive)
        * (true_positive + false_negative)
        * (true_negative + false_positive)
        * (true_negative + false_negative)
    )
    mcc = (
        (true_positive * true_negative - false_positive * false_negative)
        / mcc_denominator
        if mcc_denominator
        else 0.0
    )
    return {
        "accuracy": (true_positive + true_negative) / total if total else math.nan,
        "balanced_accuracy": (positive_recall + negative_recall) / 2.0,
        "precision": positive_precision,
        "recall": positive_recall,
        "specificity": negative_recall,
        "f1": positive_f1,
        "positive_precision": positive_precision,
        "positive_recall": positive_recall,
        "positive_f1": positive_f1,
        "negative_precision": negative_precision,
        "negative_recall": negative_recall,
        "negative_f1": negative_f1,
        "macro_f1": (positive_f1 + negative_f1) / 2.0,
        "weighted_f1": (
            positive_f1 * positive_support + negative_f1 * negative_support
        )
        / total
        if total
        else math.nan,
        "matthews_correlation_coefficient": mcc,
        "true_positive": float(true_positive),
        "true_negative": float(true_negative),
        "false_positive": float(false_positive),
        "false_negative": float(false_negative),
        "positive_support": float(positive_support),
        "negative_support": float(negative_support),
        "num_examples": float(total),
    }


def binary_classification_metrics_from_predictions(
    logits: torch.Tensor,
    labels: torch.Tensor,
    predictions: torch.Tensor | np.ndarray,
    loss: float,
) -> dict[str, float]:
    """Compute metrics for externally calibrated binary predictions."""

    probabilities = torch.sigmoid(logits.detach()).cpu().numpy()
    targets = labels.detach().cpu().numpy().astype(np.int64)
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.detach().cpu().numpy()
    predictions = np.asarray(predictions).astype(np.int64)
    if predictions.shape != targets.shape:
        raise ValueError("predictions and labels must have the same shape")

    metrics = _threshold_metrics(targets, predictions)
    if len(np.unique(targets)) == 2:
        metrics["roc_auc"] = float(roc_auc_score(targets, probabilities))
        metrics["average_precision"] = float(
            average_precision_score(targets, probabilities)
        )
    else:
        metrics["roc_auc"] = math.nan
        metrics["average_precision"] = math.nan
    return {"loss": float(loss), **metrics}


def binary_classification_metrics(
    logits: torch.Tensor,
    labels: torch.Tensor,
    loss: float,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Compute thresholded and ranking metrics from complete-split predictions."""

    probabilities = torch.sigmoid(logits.detach())
    predictions = probabilities >= threshold
    return binary_classification_metrics_from_predictions(
        logits=logits,
        labels=labels,
        predictions=predictions,
        loss=loss,
    )


def optimize_binary_threshold(
    logits: torch.Tensor,
    labels: torch.Tensor,
    metric_name: str = "macro_f1",
) -> float:
    """Choose a threshold on validation data only, with deterministic tie-breaking."""

    if metric_name not in THRESHOLD_OPTIMIZATION_METRICS:
        raise ValueError(
            f"Unsupported threshold metric {metric_name!r}; expected one of "
            f"{sorted(THRESHOLD_OPTIMIZATION_METRICS)}"
        )
    probabilities = torch.sigmoid(logits.detach()).cpu().numpy()
    targets = labels.detach().cpu().numpy().astype(np.int64)
    epsilon = np.finfo(np.float32).eps
    candidates = np.unique(
        np.concatenate(
            (
                np.asarray([epsilon, 0.5, 1.0 - epsilon]),
                probabilities,
            )
        )
    )

    best_threshold = 0.5
    best_key = (-math.inf, -math.inf, -math.inf)
    for threshold in candidates:
        metrics = _threshold_metrics(targets, probabilities >= threshold)
        key = (
            metrics[metric_name],
            metrics["balanced_accuracy"],
            -abs(float(threshold) - 0.5),
        )
        if key > best_key:
            best_key = key
            best_threshold = float(threshold)
    return best_threshold
