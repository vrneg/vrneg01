"""Significance testing over pooled out-of-fold predictions from any model in this repo.

Every model here writes the same ``test_predictions.jsonl`` row schema per fold
(``sample_id``, ``label``, ``logit``, ``probability``, ``prediction``), so this module
works uniformly across the ROCKET/TCN baselines, T2M-GPT, and MotionGPT: point it at
each model's ``cross_validation_seed-<seed>`` directory and it reads the pooled
out-of-fold predictions that ``aggregate_cross_validation_results`` already produced.

A pooled AUROC or macro F1 from ten folds is a single point estimate; the fold count is
too small, and this dataset's test splits are tens of windows, so that point estimate
carries real sampling uncertainty. The tools here quantify it:

- :func:`bootstrap_metric_ci` resamples one model's pooled predictions to give a
  confidence interval on one metric.
- :func:`paired_bootstrap_comparison` resamples two models' predictions *together* (by
  matching ``sample_id``) to test whether one outperforms the other on the same windows,
  which is a tighter, correctly-paired comparison than treating the two bootstraps as
  independent.
- :func:`mcnemar_test` is the classical paired test for whether two classifiers'
  right/wrong patterns differ, using thresholded predictions rather than scores.

All resampling is at the window level. Windows from the same recording session are
correlated (shared speaker, shared session-level noise), so these intervals are somewhat
narrower than the true between-session uncertainty; treat them as a lower bound on the
uncertainty, not the final word. A fold-level bootstrap would respect that structure but
needs many more folds than ten to be stable -- not available here.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

MetricName = Literal[
    "accuracy",
    "balanced_accuracy",
    "precision",
    "recall",
    "f1",
    "macro_f1",
    "roc_auc",
    "average_precision",
]


@dataclass(slots=True)
class PooledPredictions:
    """One model's pooled out-of-fold predictions, aligned by sample id."""

    name: str
    sample_ids: np.ndarray
    labels: np.ndarray
    probabilities: np.ndarray
    predictions: np.ndarray

    def __post_init__(self) -> None:
        length = len(self.sample_ids)
        for field_name, array in (
            ("labels", self.labels),
            ("probabilities", self.probabilities),
            ("predictions", self.predictions),
        ):
            if len(array) != length:
                raise ValueError(f"{field_name} has {len(array)} rows, expected {length}")
        if len(set(self.sample_ids.tolist())) != length:
            raise ValueError(f"Duplicate sample_id values in pooled predictions for {self.name!r}")


def _read_prediction_rows(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def load_pooled_predictions(
    name: str,
    fold_directories: Sequence[Path],
    predictions_filename: str = "test_predictions.jsonl",
    max_duplicate_fraction: float = 0.02,
) -> PooledPredictions:
    """Pool one model's per-fold test predictions into one aligned set of arrays.

    ``fold_directories`` should be every fold's run directory for one model/dataset
    combination (for example every ``fold-*_seed-42`` directory under one
    ``<model>/<dataset_prefix>/<variant>/``), matching how
    :func:`event_transformer.aggregation.aggregate_cross_validation_results` pools them.

    That aggregator already tracks (rather than rejects) a small number of samples
    landing in more than one fold's test partition -- as
    ``pooled_out_of_fold.num_duplicate_sample_ids`` in its summary -- so this loader
    matches that tolerance: it keeps the first occurrence of each ``sample_id`` and
    raises only if duplicates exceed ``max_duplicate_fraction`` of the pooled rows,
    which would indicate folds are not actually disjoint rather than a handful of
    borderline windows landing in two test partitions.
    """

    rows: list[dict[str, Any]] = []
    for directory in fold_directories:
        path = directory / predictions_filename
        if not path.exists():
            raise FileNotFoundError(f"No {predictions_filename} in {directory}")
        rows.extend(_read_prediction_rows(path))
    if not rows:
        raise ValueError(f"No prediction rows found for {name!r}")

    seen: set[str] = set()
    deduplicated: list[dict[str, Any]] = []
    for row in rows:
        sample_id = str(row["sample_id"])
        if sample_id in seen:
            continue
        seen.add(sample_id)
        deduplicated.append(row)

    duplicate_fraction = (len(rows) - len(deduplicated)) / len(rows)
    if duplicate_fraction > max_duplicate_fraction:
        raise ValueError(
            f"{name!r} has {len(rows) - len(deduplicated)} duplicate sample_id values "
            f"across {len(rows)} pooled rows ({duplicate_fraction:.1%}), exceeding "
            f"max_duplicate_fraction={max_duplicate_fraction:.1%}; folds may not be disjoint"
        )

    return PooledPredictions(
        name=name,
        sample_ids=np.asarray([str(row["sample_id"]) for row in deduplicated]),
        labels=np.asarray([int(row["label"]) for row in deduplicated], dtype=np.int64),
        probabilities=np.asarray(
            [float(row["probability"]) for row in deduplicated], dtype=np.float64
        ),
        predictions=np.asarray(
            [int(row["prediction"]) for row in deduplicated], dtype=np.int64
        ),
    )


def discover_fold_directories(model_output_dir: Path) -> list[Path]:
    """Find every ``fold-*_seed-*`` directory directly under a model/variant directory."""

    directories = sorted(
        path for path in model_output_dir.glob("fold-*_seed-*") if path.is_dir()
    )
    if not directories:
        raise FileNotFoundError(f"No fold-*_seed-* directories under {model_output_dir}")
    return directories


def load_pooled_predictions_from_output_dir(
    name: str,
    model_output_dir: Path,
    predictions_filename: str = "test_predictions.jsonl",
) -> PooledPredictions:
    """Convenience wrapper: discover fold directories, then pool their predictions."""

    return load_pooled_predictions(
        name, discover_fold_directories(model_output_dir), predictions_filename
    )


def _threshold_metrics(labels: np.ndarray, predictions: np.ndarray) -> dict[str, float]:
    true_positive = int(((predictions == 1) & (labels == 1)).sum())
    true_negative = int(((predictions == 0) & (labels == 0)).sum())
    false_positive = int(((predictions == 1) & (labels == 0)).sum())
    false_negative = int(((predictions == 0) & (labels == 1)).sum())

    positive_support = true_positive + false_negative
    negative_support = true_negative + false_positive
    precision_denominator = true_positive + false_positive
    precision = true_positive / precision_denominator if precision_denominator else 0.0
    recall = true_positive / positive_support if positive_support else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    negative_precision_denominator = true_negative + false_negative
    negative_precision = (
        true_negative / negative_precision_denominator if negative_precision_denominator else 0.0
    )
    negative_recall = true_negative / negative_support if negative_support else 0.0
    negative_f1 = (
        2 * negative_precision * negative_recall / (negative_precision + negative_recall)
        if negative_precision + negative_recall
        else 0.0
    )
    return {
        "accuracy": (true_positive + true_negative) / len(labels) if len(labels) else math.nan,
        "balanced_accuracy": (recall + negative_recall) / 2.0,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "macro_f1": (f1 + negative_f1) / 2.0,
    }


def compute_metric(
    labels: np.ndarray,
    probabilities: np.ndarray,
    predictions: np.ndarray,
    metric: MetricName,
) -> float:
    """Compute one metric from a resampled (or full) set of predictions.

    Ranking metrics return ``nan`` when a resample happens to contain only one class,
    which a bootstrap over a small, roughly-balanced sample will occasionally produce;
    :func:`bootstrap_metric_ci` drops these draws rather than crashing on them.
    """

    if metric == "roc_auc":
        if len(np.unique(labels)) < 2:
            return math.nan
        return float(roc_auc_score(labels, probabilities))
    if metric == "average_precision":
        if len(np.unique(labels)) < 2:
            return math.nan
        return float(average_precision_score(labels, probabilities))
    return _threshold_metrics(labels, predictions)[metric]


@dataclass(slots=True)
class BootstrapResult:
    """A point estimate and a percentile bootstrap confidence interval."""

    metric: str
    point_estimate: float
    confidence_level: float
    lower: float
    upper: float
    num_resamples: int
    num_dropped_resamples: int


def bootstrap_metric_ci(
    predictions: PooledPredictions,
    metric: MetricName,
    num_resamples: int = 2000,
    confidence_level: float = 0.95,
    seed: int = 42,
) -> BootstrapResult:
    """Percentile bootstrap confidence interval for one metric on one model."""

    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must be in (0, 1)")
    rng = np.random.default_rng(seed)
    n = len(predictions.labels)
    point_estimate = compute_metric(
        predictions.labels, predictions.probabilities, predictions.predictions, metric
    )

    values = []
    for _ in range(num_resamples):
        indices = rng.integers(0, n, size=n)
        value = compute_metric(
            predictions.labels[indices],
            predictions.probabilities[indices],
            predictions.predictions[indices],
            metric,
        )
        if math.isfinite(value):
            values.append(value)

    dropped = num_resamples - len(values)
    if not values:
        return BootstrapResult(metric, point_estimate, confidence_level, math.nan, math.nan, num_resamples, dropped)
    lower_percentile = (1.0 - confidence_level) / 2.0 * 100.0
    upper_percentile = 100.0 - lower_percentile
    lower, upper = np.percentile(values, [lower_percentile, upper_percentile])
    return BootstrapResult(
        metric=metric,
        point_estimate=point_estimate,
        confidence_level=confidence_level,
        lower=float(lower),
        upper=float(upper),
        num_resamples=num_resamples,
        num_dropped_resamples=dropped,
    )


def _align_by_sample_id(
    first: PooledPredictions, second: PooledPredictions
) -> tuple[PooledPredictions, PooledPredictions]:
    """Restrict both prediction sets to their shared sample ids, in matching order.

    Two models trained on the same dataset variant and fold definitions should cover
    the same windows; anything less than near-total overlap signals they were not
    evaluated on comparable data (different dataset variant, different fold count) and
    a paired comparison is not meaningful.
    """

    first_order = {sample_id: index for index, sample_id in enumerate(first.sample_ids)}
    shared = [sample_id for sample_id in second.sample_ids if sample_id in first_order]
    if not shared:
        raise ValueError(
            f"{first.name!r} and {second.name!r} share no sample ids; they were not "
            "evaluated on the same windows and cannot be paired"
        )
    coverage = len(shared) / max(len(first.sample_ids), len(second.sample_ids))
    if coverage < 0.9:
        raise ValueError(
            f"{first.name!r} and {second.name!r} share only {coverage:.0%} of their "
            "windows; refusing a paired comparison across mismatched evaluation sets"
        )

    second_order = {sample_id: index for index, sample_id in enumerate(second.sample_ids)}
    first_indices = np.asarray([first_order[sample_id] for sample_id in shared])
    second_indices = np.asarray([second_order[sample_id] for sample_id in shared])

    def _subset(predictions: PooledPredictions, indices: np.ndarray) -> PooledPredictions:
        return PooledPredictions(
            name=predictions.name,
            sample_ids=predictions.sample_ids[indices],
            labels=predictions.labels[indices],
            probabilities=predictions.probabilities[indices],
            predictions=predictions.predictions[indices],
        )

    return _subset(first, first_indices), _subset(second, second_indices)


@dataclass(slots=True)
class PairedComparisonResult:
    """A paired bootstrap comparison between two models on the same windows."""

    metric: str
    first_name: str
    second_name: str
    first_point_estimate: float
    second_point_estimate: float
    difference_point_estimate: float
    confidence_level: float
    difference_lower: float
    difference_upper: float
    probability_first_better: float
    num_windows: int
    num_resamples: int


def paired_bootstrap_comparison(
    first: PooledPredictions,
    second: PooledPredictions,
    metric: MetricName,
    num_resamples: int = 2000,
    confidence_level: float = 0.95,
    seed: int = 42,
) -> PairedComparisonResult:
    """Test whether ``first`` outperforms ``second`` on their shared windows.

    Both models are resampled with the *same* drawn indices each iteration -- this is
    what makes the comparison paired rather than two independent bootstraps -- so shared
    per-window difficulty cancels out of the difference distribution.
    ``probability_first_better`` is the fraction of resamples where ``first`` scored
    higher; values near 0 or 1 indicate a consistent difference, values near 0.5 mean the
    bootstrap could not tell the two apart on these windows.
    """

    aligned_first, aligned_second = _align_by_sample_id(first, second)
    if not np.array_equal(aligned_first.labels, aligned_second.labels):
        raise ValueError(
            f"{first.name!r} and {second.name!r} disagree on the label for at least "
            "one shared sample_id"
        )

    rng = np.random.default_rng(seed)
    n = len(aligned_first.labels)
    first_point = compute_metric(
        aligned_first.labels, aligned_first.probabilities, aligned_first.predictions, metric
    )
    second_point = compute_metric(
        aligned_second.labels, aligned_second.probabilities, aligned_second.predictions, metric
    )

    differences = []
    wins = 0
    valid = 0
    for _ in range(num_resamples):
        indices = rng.integers(0, n, size=n)
        labels = aligned_first.labels[indices]
        first_value = compute_metric(
            labels, aligned_first.probabilities[indices], aligned_first.predictions[indices], metric
        )
        second_value = compute_metric(
            labels, aligned_second.probabilities[indices], aligned_second.predictions[indices], metric
        )
        if not (math.isfinite(first_value) and math.isfinite(second_value)):
            continue
        differences.append(first_value - second_value)
        wins += first_value > second_value
        valid += 1

    if not differences:
        raise RuntimeError("Every bootstrap resample produced an undefined metric value")
    lower_percentile = (1.0 - confidence_level) / 2.0 * 100.0
    upper_percentile = 100.0 - lower_percentile
    lower, upper = np.percentile(differences, [lower_percentile, upper_percentile])
    return PairedComparisonResult(
        metric=metric,
        first_name=first.name,
        second_name=second.name,
        first_point_estimate=first_point,
        second_point_estimate=second_point,
        difference_point_estimate=first_point - second_point,
        confidence_level=confidence_level,
        difference_lower=float(lower),
        difference_upper=float(upper),
        probability_first_better=wins / valid,
        num_windows=n,
        num_resamples=valid,
    )


@dataclass(slots=True)
class McNemarResult:
    """McNemar's test for whether two classifiers' error patterns differ."""

    first_name: str
    second_name: str
    only_first_correct: int
    only_second_correct: int
    both_correct: int
    both_incorrect: int
    statistic: float
    p_value: float


def mcnemar_test(first: PooledPredictions, second: PooledPredictions) -> McNemarResult:
    """Classical paired test on thresholded predictions from the shared windows.

    Uses the chi-square form with Edwards' continuity correction, valid when the
    discordant-pair count is reasonably large; for very small discordant counts the
    reported p-value is approximate.
    """

    aligned_first, aligned_second = _align_by_sample_id(first, second)
    if not np.array_equal(aligned_first.labels, aligned_second.labels):
        raise ValueError(
            f"{first.name!r} and {second.name!r} disagree on the label for at least "
            "one shared sample_id"
        )

    first_correct = aligned_first.predictions == aligned_first.labels
    second_correct = aligned_second.predictions == aligned_second.labels
    only_first = int((first_correct & ~second_correct).sum())
    only_second = int((~first_correct & second_correct).sum())
    both_correct = int((first_correct & second_correct).sum())
    both_incorrect = int((~first_correct & ~second_correct).sum())

    discordant = only_first + only_second
    if discordant == 0:
        statistic, p_value = 0.0, 1.0
    else:
        statistic = (abs(only_first - only_second) - 1) ** 2 / discordant
        p_value = math.erfc(math.sqrt(statistic / 2.0))
    return McNemarResult(
        first_name=first.name,
        second_name=second.name,
        only_first_correct=only_first,
        only_second_correct=only_second,
        both_correct=both_correct,
        both_incorrect=both_incorrect,
        statistic=statistic,
        p_value=p_value,
    )


def summarize_comparison_matrix(
    models: Sequence[PooledPredictions],
    metric: MetricName = "roc_auc",
    num_resamples: int = 2000,
    seed: int = 42,
) -> dict[str, Any]:
    """Bootstrap CIs for every model plus every pairwise paired comparison.

    Returns a JSON-ready dict: ``per_model`` gives each model's CI, ``pairwise`` gives
    every ``(i, j)`` paired bootstrap and McNemar result for models that share windows.
    """

    per_model = {
        model.name: _bootstrap_result_to_dict(
            bootstrap_metric_ci(model, metric, num_resamples, seed=seed)
        )
        for model in models
    }
    pairwise: list[dict[str, Any]] = []
    for i in range(len(models)):
        for j in range(i + 1, len(models)):
            try:
                paired = paired_bootstrap_comparison(
                    models[i], models[j], metric, num_resamples, seed=seed
                )
                mcnemar = mcnemar_test(models[i], models[j])
            except ValueError as error:
                pairwise.append(
                    {
                        "first": models[i].name,
                        "second": models[j].name,
                        "comparable": False,
                        "reason": str(error),
                    }
                )
                continue
            pairwise.append(
                {
                    "first": models[i].name,
                    "second": models[j].name,
                    "comparable": True,
                    "paired_bootstrap": _paired_result_to_dict(paired),
                    "mcnemar": _mcnemar_result_to_dict(mcnemar),
                }
            )
    return {"metric": metric, "per_model": per_model, "pairwise": pairwise}


def _bootstrap_result_to_dict(result: BootstrapResult) -> dict[str, Any]:
    return {
        "metric": result.metric,
        "point_estimate": result.point_estimate,
        "confidence_level": result.confidence_level,
        "ci_lower": result.lower,
        "ci_upper": result.upper,
        "num_resamples": result.num_resamples,
        "num_dropped_resamples": result.num_dropped_resamples,
    }


def _paired_result_to_dict(result: PairedComparisonResult) -> dict[str, Any]:
    return {
        "metric": result.metric,
        "first_name": result.first_name,
        "second_name": result.second_name,
        "first_point_estimate": result.first_point_estimate,
        "second_point_estimate": result.second_point_estimate,
        "difference_point_estimate": result.difference_point_estimate,
        "confidence_level": result.confidence_level,
        "difference_ci_lower": result.difference_lower,
        "difference_ci_upper": result.difference_upper,
        "probability_first_better": result.probability_first_better,
        "num_windows": result.num_windows,
        "num_resamples": result.num_resamples,
    }


def _mcnemar_result_to_dict(result: McNemarResult) -> dict[str, Any]:
    return {
        "first_name": result.first_name,
        "second_name": result.second_name,
        "only_first_correct": result.only_first_correct,
        "only_second_correct": result.only_second_correct,
        "both_correct": result.both_correct,
        "both_incorrect": result.both_incorrect,
        "statistic": result.statistic,
        "p_value": result.p_value,
    }


__all__ = [
    "BootstrapResult",
    "McNemarResult",
    "PairedComparisonResult",
    "PooledPredictions",
    "bootstrap_metric_ci",
    "compute_metric",
    "discover_fold_directories",
    "load_pooled_predictions",
    "load_pooled_predictions_from_output_dir",
    "mcnemar_test",
    "paired_bootstrap_comparison",
    "summarize_comparison_matrix",
]
