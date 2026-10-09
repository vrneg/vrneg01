"""Cross-validation aggregation with sparse model-selection summaries."""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

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


def _count_summary(values: list[int]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "minimum": int(array.min()),
        "maximum": int(array.max()),
    }


def aggregate_cross_validation_results(
    fold_results: Mapping[int, Any],
    output_dir: str | Path,
    threshold: float = 0.5,
    artifact_prefix: str | None = None,
) -> CrossValidationArtifacts:
    """Aggregate metrics and record which sparse classifier won each fold."""

    artifacts = _aggregate_results(
        fold_results=fold_results,
        output_dir=output_dir,
        threshold=threshold,
        artifact_prefix=artifact_prefix,
    )
    ordered_results = sorted(fold_results.items())
    classifier_counts = Counter(
        result.best_classifier for _, result in ordered_results
    )
    artifacts.summary["sparse_multirocket_hydra_selection"] = {
        "best_classifier_counts": dict(sorted(classifier_counts.items())),
        "selected_feature_count": _count_summary(
            [result.num_selected_features for _, result in ordered_results]
        ),
        "nonzero_coefficient_count": _count_summary(
            [result.num_nonzero_coefficients for _, result in ordered_results]
        ),
        "folds": {
            str(fold): {
                "best_classifier": result.best_classifier,
                "best_inner_cv_score": result.best_cv_score,
                "best_hyperparameters": result.best_hyperparameters,
                "num_selected_features": result.num_selected_features,
                "num_nonzero_coefficients": result.num_nonzero_coefficients,
                "classifier_leaderboard": result.classifier_leaderboard,
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
