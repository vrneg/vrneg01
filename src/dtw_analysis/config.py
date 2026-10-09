"""Configuration for standalone DTW and timing-robustness analyses."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Literal


@dataclass(frozen=True, slots=True)
class DTWAnalysisConfig:
    output_dir: Path
    mode: Literal["dtw", "robustness", "both"] = "both"
    device: str = "cpu"
    batch_size: int = 64
    n_jobs: int = 1
    fold_jobs: int = 1
    seed: int = 42
    bootstrap_samples: int = 10_000
    pca_variance: float = 0.95
    pca_max_components: int = 16
    dtw_windows: tuple[float, ...] = (0.0, 0.1, 0.2)
    neighbor_counts: tuple[int, ...] = (1, 3, 5)
    shift_steps: tuple[int, ...] = (-4, -2, -1, 1, 2, 4)
    time_scales: tuple[float, ...] = (0.8, 1.2)
    event_num_time_points: int = 32
    event_window_start_seconds: float = -0.5
    event_window_end_seconds: float = 0.5
    allow_duplicate_samples: bool = False
    attribution_summary_path: Path | None = None

    def validate(self) -> None:
        if self.mode not in {"dtw", "robustness", "both"}:
            raise ValueError("mode must be 'dtw', 'robustness', or 'both'")
        if self.batch_size < 1 or self.n_jobs == 0 or self.n_jobs < -1:
            raise ValueError(
                "batch_size must be positive and n_jobs must be -1 or positive"
            )
        if self.fold_jobs < 1:
            raise ValueError("fold_jobs must be positive")
        if self.bootstrap_samples < 0:
            raise ValueError("bootstrap_samples cannot be negative")
        if not 0.0 < self.pca_variance <= 1.0:
            raise ValueError("pca_variance must be in (0, 1]")
        if self.pca_max_components < 1:
            raise ValueError("pca_max_components must be positive")
        if not self.dtw_windows or any(
            not math.isfinite(window) or window < 0.0 or window > 1.0
            for window in self.dtw_windows
        ):
            raise ValueError("dtw_windows must contain values in [0, 1]")
        if len(set(self.dtw_windows)) != len(self.dtw_windows):
            raise ValueError("dtw_windows must not contain duplicates")
        if not self.neighbor_counts or any(
            count < 1 or count % 2 == 0 for count in self.neighbor_counts
        ):
            raise ValueError("neighbor_counts must contain positive odd integers")
        if len(set(self.neighbor_counts)) != len(self.neighbor_counts):
            raise ValueError("neighbor_counts must not contain duplicates")
        if (
            not self.shift_steps
            or 0 in self.shift_steps
            or len(set(self.shift_steps)) != len(self.shift_steps)
        ):
            raise ValueError(
                "shift_steps must be unique, non-empty, and exclude zero"
            )
        if not self.time_scales or any(
            not math.isfinite(scale) or scale <= 0.0 or scale == 1.0
            for scale in self.time_scales
        ):
            raise ValueError("time_scales must contain positive non-identity values")
        if len(set(self.time_scales)) != len(self.time_scales):
            raise ValueError("time_scales must not contain duplicates")
        if self.event_num_time_points < 9:
            raise ValueError("event_num_time_points must be at least nine")
        if self.event_window_end_seconds <= self.event_window_start_seconds:
            raise ValueError("event timing window end must exceed its start")
