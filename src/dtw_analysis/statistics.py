"""Paper-facing pooled metrics and session-cluster uncertainty."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence

import numpy as np
from sklearn.metrics import f1_score


def cluster_bootstrap_interval(
    group_ids: Sequence[str],
    statistic: Callable[[np.ndarray], float],
    *,
    seed: int,
    samples: int,
) -> tuple[float, float]:
    """Percentile interval from resampled recording/session clusters."""

    groups = np.asarray(group_ids, dtype=object)
    unique = np.asarray(list(dict.fromkeys(groups.tolist())), dtype=object)
    if unique.size == 0:
        return math.nan, math.nan
    if samples == 0 or unique.size == 1:
        value = float(statistic(np.arange(groups.size)))
        return value, value
    indices_by_group = {group: np.flatnonzero(groups == group) for group in unique}
    generator = np.random.default_rng(seed)
    values = np.empty(samples, dtype=np.float64)
    for sample_index in range(samples):
        selected = generator.choice(unique, size=unique.size, replace=True)
        indices = np.concatenate([indices_by_group[group] for group in selected])
        values[sample_index] = statistic(indices)
    finite = values[np.isfinite(values)]
    if not finite.size:
        return math.nan, math.nan
    lower, upper = np.quantile(finite, [0.025, 0.975])
    return float(lower), float(upper)


def macro_f1_interval(
    labels: np.ndarray,
    predictions: np.ndarray,
    group_ids: Sequence[str],
    *,
    seed: int,
    samples: int,
) -> tuple[float, float]:
    labels = np.asarray(labels, dtype=np.int64)
    predictions = np.asarray(predictions, dtype=np.int64)
    return cluster_bootstrap_interval(
        group_ids,
        lambda indices: float(
            f1_score(
                labels[indices],
                predictions[indices],
                average="macro",
                labels=[0, 1],
                zero_division=0,
            )
        ),
        seed=seed,
        samples=samples,
    )


def paired_macro_f1_decrease_interval(
    labels: np.ndarray,
    baseline_predictions: np.ndarray,
    perturbed_predictions: np.ndarray,
    group_ids: Sequence[str],
    *,
    seed: int,
    samples: int,
) -> tuple[float, float]:
    labels = np.asarray(labels, dtype=np.int64)
    baseline = np.asarray(baseline_predictions, dtype=np.int64)
    perturbed = np.asarray(perturbed_predictions, dtype=np.int64)

    def statistic(indices: np.ndarray) -> float:
        baseline_score = f1_score(
            labels[indices],
            baseline[indices],
            average="macro",
            labels=[0, 1],
            zero_division=0,
        )
        perturbed_score = f1_score(
            labels[indices],
            perturbed[indices],
            average="macro",
            labels=[0, 1],
            zero_division=0,
        )
        return float(baseline_score - perturbed_score)

    return cluster_bootstrap_interval(
        group_ids, statistic, seed=seed, samples=samples
    )


def mean_interval(
    values: np.ndarray,
    group_ids: Sequence[str],
    *,
    seed: int,
    samples: int,
) -> tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    return cluster_bootstrap_interval(
        group_ids,
        lambda indices: float(array[indices].mean()),
        seed=seed,
        samples=samples,
    )


def descriptive(values: Sequence[float]) -> dict[str, float | int | None]:
    finite = np.asarray([value for value in values if np.isfinite(value)], dtype=float)
    if not finite.size:
        return {
            "num_folds": len(values),
            "mean": None,
            "standard_deviation": None,
            "minimum": None,
            "maximum": None,
        }
    return {
        "num_folds": len(values),
        "mean": float(finite.mean()),
        "standard_deviation": (
            float(finite.std(ddof=1)) if finite.size > 1 else None
        ),
        "minimum": float(finite.min()),
        "maximum": float(finite.max()),
    }
