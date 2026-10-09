"""Cross-validated DTW separability and timing-robustness analyses."""

from .analysis import DTWAnalysisResult, run_analysis
from .config import DTWAnalysisConfig

__all__ = ["DTWAnalysisConfig", "DTWAnalysisResult", "run_analysis"]
