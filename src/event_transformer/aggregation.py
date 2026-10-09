"""Cross-validation metric and out-of-fold prediction aggregation."""

from __future__ import annotations

import csv
import json
import math
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Any

import numpy as np
import torch
from torch.nn import functional as functional

from .metrics import (
    binary_classification_metrics,
    binary_classification_metrics_from_predictions,
)
from .training import TrainingResult


SCORE_METRICS: tuple[str, ...] = (
    "loss",
    "accuracy",
    "balanced_accuracy",
    "precision",
    "recall",
    "specificity",
    "f1",
    "negative_precision",
    "negative_f1",
    "macro_f1",
    "weighted_f1",
    "matthews_correlation_coefficient",
    "roc_auc",
    "average_precision",
)
COUNT_METRICS: tuple[str, ...] = (
    "true_positive",
    "true_negative",
    "false_positive",
    "false_negative",
    "positive_support",
    "negative_support",
    "num_examples",
)


@dataclass(frozen=True, slots=True)
class CrossValidationArtifacts:
    """Files and in-memory summary produced after all requested folds finish."""

    summary_path: Path
    fold_metrics_path: Path
    metric_summary_path: Path
    summary: dict[str, Any]


def _summary_statistics(values: Sequence[float]) -> dict[str, float | int | None]:
    finite_values = np.asarray(
        [float(value) for value in values if math.isfinite(float(value))],
        dtype=np.float64,
    )
    valid_count = int(finite_values.size)
    total_count = len(values)
    if valid_count == 0:
        return {
            "num_folds": total_count,
            "num_valid_folds": 0,
            "mean": None,
            "standard_deviation": None,
            "standard_error": None,
            "median": None,
            "minimum": None,
            "maximum": None,
        }

    standard_deviation = (
        float(finite_values.std(ddof=1)) if valid_count > 1 else None
    )
    standard_error = (
        standard_deviation / math.sqrt(valid_count)
        if standard_deviation is not None
        else None
    )
    return {
        "num_folds": total_count,
        "num_valid_folds": valid_count,
        "mean": float(finite_values.mean()),
        "standard_deviation": standard_deviation,
        "standard_error": standard_error,
        "median": float(median(finite_values.tolist())),
        "minimum": float(finite_values.min()),
        "maximum": float(finite_values.max()),
    }


def _split_statistics(
    results: Sequence[TrainingResult],
    split: str,
) -> dict[str, dict[str, float | int | None]]:
    metrics_attribute = f"{split}_metrics"
    return {
        metric: _summary_statistics(
            [getattr(result, metrics_attribute)[metric] for result in results]
        )
        for metric in SCORE_METRICS
    }


def _load_prediction_rows(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as error:
            raise ValueError(f"Invalid JSON in {path} at line {line_number}") from error
    if not rows:
        raise ValueError(f"Prediction file is empty: {path}")
    return rows


def _pooled_out_of_fold_metrics(
    prediction_rows: Sequence[Mapping[str, Any]],
    threshold: float,
) -> dict[str, float]:
    logits = torch.tensor(
        [float(row["logit"]) for row in prediction_rows], dtype=torch.float32
    )
    labels = torch.tensor(
        [float(row["label"]) for row in prediction_rows], dtype=torch.float32
    )
    loss = float(functional.binary_cross_entropy_with_logits(logits, labels).item())
    if all("prediction" in row for row in prediction_rows):
        predictions = torch.tensor(
            [int(row["prediction"]) for row in prediction_rows], dtype=torch.long
        )
        return binary_classification_metrics_from_predictions(
            logits, labels, predictions, loss
        )
    return binary_classification_metrics(logits, labels, loss, threshold)


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _write_fold_metrics_csv(
    path: Path,
    fold_results: Mapping[int, TrainingResult],
) -> None:
    fieldnames = ["fold", "run_name", "best_epoch", "decision_threshold"]
    fieldnames.extend(f"validation_{metric}" for metric in (*SCORE_METRICS, *COUNT_METRICS))
    fieldnames.extend(f"test_{metric}" for metric in (*SCORE_METRICS, *COUNT_METRICS))

    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for fold, result in sorted(fold_results.items()):
            row: dict[str, Any] = {
                "fold": fold,
                "run_name": result.run_name,
                "best_epoch": result.best_epoch,
                "decision_threshold": result.decision_threshold,
            }
            row.update(
                {
                    f"validation_{metric}": result.validation_metrics[metric]
                    for metric in (*SCORE_METRICS, *COUNT_METRICS)
                }
            )
            row.update(
                {
                    f"test_{metric}": result.test_metrics[metric]
                    for metric in (*SCORE_METRICS, *COUNT_METRICS)
                }
            )
            writer.writerow(row)


def _write_metric_summary_csv(
    path: Path,
    split_statistics: Mapping[str, Mapping[str, Mapping[str, Any]]],
) -> None:
    fieldnames = [
        "split",
        "metric",
        "num_folds",
        "num_valid_folds",
        "mean",
        "standard_deviation",
        "standard_error",
        "median",
        "minimum",
        "maximum",
    ]
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for split, metrics in split_statistics.items():
            for metric, statistics in metrics.items():
                writer.writerow({"split": split, "metric": metric, **statistics})


def aggregate_cross_validation_results(
    fold_results: Mapping[int, TrainingResult],
    output_dir: str | Path,
    threshold: float = 0.5,
    artifact_prefix: str | None = None,
) -> CrossValidationArtifacts:
    """Aggregate descriptive fold statistics and pooled out-of-fold metrics.

    Mean, sample standard deviation, standard error, median, minimum, and maximum are
    reported for validation and test metrics. AUROC and average precision are also
    recomputed from all held-out predictions together, rather than inferred from fold
    averages. The fold standard deviation is descriptive: cross-validation folds have
    overlapping training data and are not independent experimental repetitions.
    """

    if not fold_results:
        raise ValueError("At least one fold result is required")
    if not 0.0 < threshold < 1.0:
        raise ValueError("threshold must be strictly between 0 and 1")
    if artifact_prefix is not None:
        artifact_prefix = artifact_prefix.strip()
        if not artifact_prefix:
            raise ValueError("artifact_prefix cannot be empty")
        if Path(artifact_prefix).name != artifact_prefix:
            raise ValueError("artifact_prefix must be a file-name-safe name")

    sorted_results = [result for _, result in sorted(fold_results.items())]
    prediction_rows: list[dict[str, Any]] = []
    prediction_rows_by_fold: dict[int, int] = {}
    for fold, result in sorted(fold_results.items()):
        rows = _load_prediction_rows(result.predictions_path)
        prediction_rows_by_fold[fold] = len(rows)
        prediction_rows.extend(rows)

    sample_id_counts = Counter(str(row["sample_id"]) for row in prediction_rows)
    duplicate_ids = {key: count for key, count in sample_id_counts.items() if count > 1}
    split_statistics = {
        "validation": _split_statistics(sorted_results, "validation"),
        "test": _split_statistics(sorted_results, "test"),
    }
    pooled_metrics = _pooled_out_of_fold_metrics(prediction_rows, threshold)

    summary: dict[str, Any] = {
        "artifact_prefix": artifact_prefix,
        "num_folds": len(fold_results),
        "fold_numbers": sorted(fold_results),
        "fallback_threshold": threshold,
        "decision_threshold": _summary_statistics(
            [result.decision_threshold for result in sorted_results]
        ),
        "best_epoch": _summary_statistics(
            [float(result.best_epoch) for result in sorted_results]
        ),
        "fold_statistics": split_statistics,
        "pooled_out_of_fold": {
            "metrics": pooled_metrics,
            "num_predictions": len(prediction_rows),
            "num_unique_sample_ids": len(sample_id_counts),
            "num_duplicate_sample_ids": len(duplicate_ids),
            "num_duplicate_occurrences": sum(count - 1 for count in duplicate_ids.values()),
            "predictions_per_fold": prediction_rows_by_fold,
        },
        "folds": {
            fold: {
                "run_name": result.run_name,
                "best_epoch": result.best_epoch,
                "decision_threshold": result.decision_threshold,
                "validation_metrics": result.validation_metrics,
                "test_metrics": result.test_metrics,
                "checkpoint_path": result.checkpoint_path,
                "validation_predictions_path": result.validation_predictions_path,
                "predictions_path": result.predictions_path,
            }
            for fold, result in sorted(fold_results.items())
        },
        "interpretation_note": (
            "Fold standard deviations describe variation across held-out folds. "
            "They are not independent-repetition uncertainty estimates because "
            "cross-validation training sets overlap."
        ),
    }

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    file_prefix = "" if artifact_prefix is None else f"{artifact_prefix}_"
    summary_path = output_dir / f"{file_prefix}cross_validation_summary.json"
    fold_metrics_path = output_dir / f"{file_prefix}fold_metrics.csv"
    metric_summary_path = output_dir / f"{file_prefix}metric_summary.csv"
    safe_summary = _json_safe(summary)
    summary_path.write_text(
        json.dumps(safe_summary, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    _write_fold_metrics_csv(fold_metrics_path, fold_results)
    _write_metric_summary_csv(metric_summary_path, split_statistics)

    return CrossValidationArtifacts(
        summary_path=summary_path,
        fold_metrics_path=fold_metrics_path,
        metric_summary_path=metric_summary_path,
        summary=safe_summary,
    )
