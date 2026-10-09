"""Binary held-out metrics used by modality attribution reports."""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    log_loss,
    roc_auc_score,
)


METRIC_NAMES: tuple[str, ...] = (
    "macro_f1",
    "balanced_accuracy",
    "average_precision",
    "roc_auc",
    "loss",
)


def sigmoid(logits: np.ndarray) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float64)
    positive = values >= 0
    output = np.empty_like(values)
    output[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exp_values = np.exp(values[~positive])
    output[~positive] = exp_values / (1.0 + exp_values)
    return output


def binary_metrics(
    labels: np.ndarray,
    logits: np.ndarray,
    threshold: float | np.ndarray,
) -> dict[str, float]:
    labels = np.asarray(labels, dtype=np.int64)
    probabilities = sigmoid(logits)
    predictions = (probabilities >= np.asarray(threshold)).astype(np.int64)
    result = {
        "macro_f1": float(f1_score(labels, predictions, average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "average_precision": float(average_precision_score(labels, probabilities)),
        "loss": float(log_loss(labels, probabilities, labels=[0, 1])),
    }
    result["roc_auc"] = (
        float(roc_auc_score(labels, probabilities))
        if np.unique(labels).size == 2
        else math.nan
    )
    return result


def metric_decrease(
    full: dict[str, float], perturbed: dict[str, float]
) -> dict[str, float]:
    """Return positive values when perturbation degrades classification."""

    return {
        name: (
            perturbed[name] - full[name]
            if name == "loss"
            else full[name] - perturbed[name]
        )
        for name in METRIC_NAMES
    }


def bootstrap_mean_interval(
    values: Sequence[float],
    *,
    seed: int,
    samples: int,
) -> tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return math.nan, math.nan
    if array.size == 1 or samples == 0:
        value = float(array.mean())
        return value, value
    generator = np.random.default_rng(seed)
    draws = generator.choice(array, size=(samples, array.size), replace=True).mean(axis=1)
    lower, upper = np.quantile(draws, [0.025, 0.975])
    return float(lower), float(upper)
