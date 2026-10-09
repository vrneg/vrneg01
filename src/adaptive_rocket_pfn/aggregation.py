"""Cross-validation metrics and adaptive-representation diagnostics."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

try:
    from event_transformer.aggregation import (
        SCORE_METRICS,
        CrossValidationArtifacts,
        aggregate_cross_validation_results as _aggregate_results,
    )
except ModuleNotFoundError as error:
    if error.name != "event_transformer":
        raise
    from ..event_transformer.aggregation import (
        SCORE_METRICS,
        CrossValidationArtifacts,
        aggregate_cross_validation_results as _aggregate_results,
    )


def aggregate_cross_validation_results(
    fold_results: Mapping[int, Any],
    output_dir: str | Path,
    threshold: float = 0.5,
    artifact_prefix: str | None = None,
) -> CrossValidationArtifacts:
    """Aggregate predictive results and learned representation distributions."""

    artifacts = _aggregate_results(
        fold_results=fold_results,
        output_dir=output_dir,
        threshold=threshold,
        artifact_prefix=artifact_prefix,
    )
    ordered_results = sorted(fold_results.items())
    expert_names = sorted(
        {
            expert
            for _, result in ordered_results
            for expert in result.ensemble_weights
        }
    )
    mean_weights = {
        expert: sum(
            float(result.ensemble_weights.get(expert, 0.0))
            for _, result in ordered_results
        )
        / len(ordered_results)
        for expert in expert_names
    }

    distribution_totals: dict[str, dict[str, int]] = {}
    for _, result in ordered_results:
        for category, values in result.selected_distribution.items():
            totals = distribution_totals.setdefault(category, {})
            for name, count in values.items():
                totals[name] = totals.get(name, 0) + int(count)

    artifacts.summary["adaptive_rocket_pfn"] = {
        "mean_ensemble_weights": mean_weights,
        "selected_distribution_totals": {
            category: dict(sorted(values.items()))
            for category, values in sorted(distribution_totals.items())
        },
        "folds": {
            str(fold): {
                "num_candidate_features": result.num_candidate_features,
                "num_selected_features": result.num_selected_features,
                "selected_features_by_expert": result.selected_features_by_expert,
                "ensemble_weights": result.ensemble_weights,
                "selected_distribution": result.selected_distribution,
                "oof_ensemble_metrics": result.oof_ensemble_metrics,
            }
            for fold, result in ordered_results
        },
    }
    artifacts.summary_path.write_text(
        json.dumps(artifacts.summary, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return artifacts


__all__ = [
    "SCORE_METRICS",
    "CrossValidationArtifacts",
    "aggregate_cross_validation_results",
]
