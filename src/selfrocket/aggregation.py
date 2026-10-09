"""Cross-validation aggregation for SelF-Rocket experiments."""

from __future__ import annotations

import json
from collections import Counter
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
    """Aggregate classification metrics plus SelF-Rocket selection behavior."""

    artifacts = _aggregate_results(
        fold_results=fold_results,
        output_dir=output_dir,
        threshold=threshold,
        artifact_prefix=artifact_prefix,
    )
    ordered_results = sorted(fold_results.items())
    selected_counts = Counter(
        result.selected_candidate for _, result in ordered_results
    )
    proposed_counts = Counter(
        result.selected_candidate_before_vote for _, result in ordered_results
    )
    selection_summary = {
        "selected_candidate_counts": dict(sorted(selected_counts.items())),
        "candidate_before_vote_counts": dict(sorted(proposed_counts.items())),
        "num_fallbacks": sum(
            bool(result.selection_used_fallback) for _, result in ordered_results
        ),
        "folds": {
            str(fold): {
                "selected_candidate": result.selected_candidate,
                "selected_candidate_before_vote": (
                    result.selected_candidate_before_vote
                ),
                "vote_support": result.selection_vote_support,
                "used_fallback": result.selection_used_fallback,
                "median_scores": result.selection_median_scores,
            }
            for fold, result in ordered_results
        },
    }
    artifacts.summary["selfrocket_selection"] = selection_summary
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
