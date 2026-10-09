"""Cross-validation aggregation for MotionGPT experiments.

The shared aggregator reads the attributes ``TrainingResult`` exposes, so the MotionGPT
result object is accepted without modification and folds stay comparable with the other
models in this repository.
"""

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
