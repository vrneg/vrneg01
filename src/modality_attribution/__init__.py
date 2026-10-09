"""Model-agnostic modality attribution for trained negation classifiers."""

from .analysis import (
    AnalysisConfig,
    AnalysisResult,
    analyze_cross_validation,
    attach_retraining_summary,
    load_analysis_result,
)
from .grouping import ModalityGroup, build_channel_groups, mask_fixed_grid
from .shapley import ShapleyResult, sampled_modality_shapley

__all__ = [
    "AnalysisConfig",
    "AnalysisResult",
    "ModalityGroup",
    "ShapleyResult",
    "analyze_cross_validation",
    "attach_retraining_summary",
    "build_channel_groups",
    "mask_fixed_grid",
    "load_analysis_result",
    "sampled_modality_shapley",
]
