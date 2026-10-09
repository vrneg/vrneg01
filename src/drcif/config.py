"""Configuration objects for Diverse Representation CIF experiments."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

try:
    from minirocket.config import DataConfig, EvaluationConfig
except ModuleNotFoundError as error:
    if error.name != "minirocket":
        raise
    from ..minirocket.config import DataConfig, EvaluationConfig


VALID_INTERVAL_RULES = frozenset({"sqrt", "sqrt-div"})
VALID_PARALLEL_BACKENDS = frozenset(
    {"loky", "multiprocessing", "threading"}
)


@dataclass(frozen=True, slots=True)
class DrCIFConfig:
    """Interval sampling, feature subsampling, and forest settings."""

    n_estimators: int = 200
    n_intervals: tuple[int | str, ...] = (4, "sqrt-div")
    min_interval_length: int | float = 3
    max_interval_length: int | float = 0.5
    att_subsample_size: int | float | None = 10
    time_limit_in_minutes: float | None = None
    contract_max_n_estimators: int = 500
    use_pycatch22: bool = False
    stabilize_near_constant_intervals: bool = True
    n_jobs: int = 4
    parallel_backend: str | None = None

    def validate(self) -> None:
        if self.n_estimators < 1:
            raise ValueError("n_estimators must be positive")
        if not self.n_intervals:
            raise ValueError("n_intervals cannot be empty")
        for value in self.n_intervals:
            if isinstance(value, bool) or not isinstance(value, (int, str)):
                raise ValueError("n_intervals must contain positive ints or rules")
            if isinstance(value, int) and value < 1:
                raise ValueError("integer n_intervals values must be positive")
            if isinstance(value, str) and value not in VALID_INTERVAL_RULES:
                raise ValueError(
                    "interval rules must be selected from "
                    f"{sorted(VALID_INTERVAL_RULES)}"
                )
        for name, value in (
            ("min_interval_length", self.min_interval_length),
            ("max_interval_length", self.max_interval_length),
        ):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{name} must be an int or float")
            if value <= 0:
                raise ValueError(f"{name} must be positive")
            if isinstance(value, float) and value > 1.0:
                raise ValueError(f"float {name} must be a proportion in (0, 1]")
        if (
            isinstance(self.min_interval_length, int)
            and isinstance(self.max_interval_length, int)
            and self.max_interval_length < self.min_interval_length
        ):
            raise ValueError(
                "integer max_interval_length cannot be below min_interval_length"
            )
        if self.att_subsample_size is not None:
            value = self.att_subsample_size
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError("att_subsample_size must be an int, float, or None")
            if isinstance(value, int) and value < 1:
                raise ValueError("integer att_subsample_size must be positive")
            if isinstance(value, float) and not 0.0 < value <= 1.0:
                raise ValueError("float att_subsample_size must be in (0, 1]")
        if self.time_limit_in_minutes is not None and self.time_limit_in_minutes <= 0:
            raise ValueError("time_limit_in_minutes must be positive or None")
        if self.contract_max_n_estimators < 1:
            raise ValueError("contract_max_n_estimators must be positive")
        if not isinstance(self.stabilize_near_constant_intervals, bool):
            raise ValueError("stabilize_near_constant_intervals must be a bool")
        if self.n_jobs == 0:
            raise ValueError("n_jobs cannot be zero")
        if (
            self.parallel_backend is not None
            and self.parallel_backend not in VALID_PARALLEL_BACKENDS
        ):
            raise ValueError(
                "parallel_backend must be None or selected from "
                f"{sorted(VALID_PARALLEL_BACKENDS)}"
            )


@dataclass(frozen=True, slots=True)
class ExperimentConfig:
    """Complete configuration for one saved DrCIF fold."""

    data: DataConfig
    output_dir: Path
    run_name: str
    model: DrCIFConfig = field(default_factory=DrCIFConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    seed: int = 42

    def validate(self) -> None:
        self.data.validate()
        self.model.validate()
        self.evaluation.validate()
        if not self.run_name.strip():
            raise ValueError("run_name cannot be empty")


__all__ = [
    "DataConfig",
    "DrCIFConfig",
    "EvaluationConfig",
    "ExperimentConfig",
]
