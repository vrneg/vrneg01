"""Cross-validation and seed-level aggregation for compact fusion-TCN runs."""

from __future__ import annotations

import csv
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Any

import numpy as np

try:
    from event_transformer.aggregation import (
        SCORE_METRICS,
        CrossValidationArtifacts,
        aggregate_cross_validation_results,
    )
except ModuleNotFoundError as error:
    if error.name != "event_transformer":
        raise
    from ..event_transformer.aggregation import (
        SCORE_METRICS,
        CrossValidationArtifacts,
        aggregate_cross_validation_results,
    )


@dataclass(frozen=True, slots=True)
class MultiSeedArtifacts:
    """Summary over independent seed-level cross-validation means."""

    summary_path: Path
    metric_summary_path: Path
    summary: dict[str, Any]


def _statistics(values: Sequence[float]) -> dict[str, float | int | None]:
    numeric_values = [float(value) for value in values if value is not None]
    finite = np.asarray(
        [value for value in numeric_values if math.isfinite(value)],
        dtype=np.float64,
    )
    if finite.size == 0:
        return {
            "num_seeds": len(values),
            "num_valid_seeds": 0,
            "mean": None,
            "standard_deviation": None,
            "standard_error": None,
            "median": None,
            "minimum": None,
            "maximum": None,
        }
    standard_deviation = float(finite.std(ddof=1)) if finite.size > 1 else None
    return {
        "num_seeds": len(values),
        "num_valid_seeds": int(finite.size),
        "mean": float(finite.mean()),
        "standard_deviation": standard_deviation,
        "standard_error": (
            standard_deviation / math.sqrt(int(finite.size))
            if standard_deviation is not None
            else None
        ),
        "median": float(median(finite.tolist())),
        "minimum": float(finite.min()),
        "maximum": float(finite.max()),
    }


def aggregate_multi_seed_results(
    seed_artifacts: Mapping[int, CrossValidationArtifacts],
    output_dir: str | Path,
    artifact_prefix: str,
) -> MultiSeedArtifacts:
    """Summarize seed-to-seed uncertainty without treating folds as repetitions."""

    if not seed_artifacts:
        raise ValueError("At least one seed artifact is required")
    artifact_prefix = artifact_prefix.strip()
    if not artifact_prefix or Path(artifact_prefix).name != artifact_prefix:
        raise ValueError("artifact_prefix must be a non-empty file-name-safe name")
    seeds = sorted(seed_artifacts)
    split_statistics: dict[str, dict[str, dict[str, Any]]] = {}
    for split in ("validation", "test"):
        split_statistics[split] = {
            metric: _statistics(
                [
                    seed_artifacts[seed].summary["fold_statistics"][split][metric][
                        "mean"
                    ]
                    for seed in seeds
                ]
            )
            for metric in SCORE_METRICS
        }
    pooled_statistics = {
        metric: _statistics(
            [
                seed_artifacts[seed].summary["pooled_out_of_fold"]["metrics"][
                    metric
                ]
                for seed in seeds
            ]
        )
        for metric in SCORE_METRICS
    }
    summary = {
        "artifact_prefix": artifact_prefix,
        "num_seeds": len(seeds),
        "seeds": seeds,
        "seed_mean_fold_statistics": split_statistics,
        "seed_pooled_out_of_fold_statistics": pooled_statistics,
        "per_seed": {
            seed: {
                "summary_path": seed_artifacts[seed].summary_path,
                "fold_statistics": seed_artifacts[seed].summary["fold_statistics"],
                "pooled_out_of_fold": seed_artifacts[seed].summary[
                    "pooled_out_of_fold"
                ],
            }
            for seed in seeds
        },
        "interpretation_note": (
            "Uncertainty is computed across complete cross-validation repetitions "
            "with different random seeds. Fold means within a seed are descriptive "
            "because their training sets overlap."
        ),
    }
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / f"{artifact_prefix}_multi_seed_summary.json"
    metric_summary_path = output_dir / f"{artifact_prefix}_multi_seed_metrics.csv"
    summary_path.write_text(
        json.dumps(summary, indent=2, default=str, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    fields = [
        "scope",
        "split",
        "metric",
        "num_seeds",
        "num_valid_seeds",
        "mean",
        "standard_deviation",
        "standard_error",
        "median",
        "minimum",
        "maximum",
    ]
    with metric_summary_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for split, metrics in split_statistics.items():
            for metric, statistics in metrics.items():
                writer.writerow(
                    {
                        "scope": "mean_across_folds",
                        "split": split,
                        "metric": metric,
                        **statistics,
                    }
                )
        for metric, statistics in pooled_statistics.items():
            writer.writerow(
                {
                    "scope": "pooled_out_of_fold",
                    "split": "test",
                    "metric": metric,
                    **statistics,
                }
            )
    return MultiSeedArtifacts(
        summary_path=summary_path,
        metric_summary_path=metric_summary_path,
        summary=summary,
    )


__all__ = [
    "SCORE_METRICS",
    "CrossValidationArtifacts",
    "MultiSeedArtifacts",
    "aggregate_cross_validation_results",
    "aggregate_multi_seed_results",
]
