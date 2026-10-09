"""Cross-validation aggregation for DrCIF experiments."""

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

__all__ = [
    "SCORE_METRICS",
    "CrossValidationArtifacts",
    "aggregate_cross_validation_results",
]
