"""Configuration objects for HIVE-COTE 2.0 experiments."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

try:
    from minirocket.config import DataConfig, EvaluationConfig
except ModuleNotFoundError as error:
    if error.name != "minirocket":
        raise
    from ..minirocket.config import DataConfig, EvaluationConfig


VALID_PARALLEL_BACKENDS = frozenset(
    {"loky", "multiprocessing", "threading"}
)
VALID_ROCKET_TRANSFORMS = frozenset({"rocket", "minirocket", "multirocket"})


def _validate_positive_int(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive int")


@dataclass(frozen=True, slots=True)
class STCComponentConfig:
    """Paper-default Shapelet Transform Classifier settings and contract caps."""

    n_shapelet_samples: int = 10_000
    max_shapelets: int | None = None
    max_shapelet_length: int | None = None
    batch_size: int = 100
    contract_max_n_shapelet_samples: int = 10_000
    rotation_forest_n_estimators: int = 200
    rotation_forest_contract_max_n_estimators: int = 200

    def validate(self) -> None:
        _validate_positive_int("n_shapelet_samples", self.n_shapelet_samples)
        if self.max_shapelets is not None:
            _validate_positive_int("max_shapelets", self.max_shapelets)
        if self.max_shapelet_length is not None:
            _validate_positive_int("max_shapelet_length", self.max_shapelet_length)
        _validate_positive_int("batch_size", self.batch_size)
        _validate_positive_int(
            "contract_max_n_shapelet_samples",
            self.contract_max_n_shapelet_samples,
        )
        _validate_positive_int(
            "rotation_forest_n_estimators",
            self.rotation_forest_n_estimators,
        )
        _validate_positive_int(
            "rotation_forest_contract_max_n_estimators",
            self.rotation_forest_contract_max_n_estimators,
        )


@dataclass(frozen=True, slots=True)
class DrCIFComponentConfig:
    """HIVE-COTE's paper-default DrCIF settings and contract cap."""

    n_estimators: int = 500
    n_intervals: tuple[int | str, ...] = (4, "sqrt-div")
    min_interval_length: int | float = 3
    max_interval_length: int | float = 0.5
    att_subsample_size: int | float | None = 10
    contract_max_n_estimators: int = 500
    use_pycatch22: bool = False
    stabilize_near_constant_intervals: bool = True

    def validate(self) -> None:
        _validate_positive_int("DrCIF n_estimators", self.n_estimators)
        _validate_positive_int(
            "DrCIF contract_max_n_estimators",
            self.contract_max_n_estimators,
        )
        if not self.n_intervals:
            raise ValueError("DrCIF n_intervals cannot be empty")
        for value in self.n_intervals:
            if isinstance(value, bool) or not isinstance(value, (int, str)):
                raise ValueError("DrCIF n_intervals must contain ints or rules")
            if isinstance(value, int) and value < 1:
                raise ValueError("DrCIF integer interval counts must be positive")
            if isinstance(value, str) and value not in {"sqrt", "sqrt-div"}:
                raise ValueError("DrCIF interval rule must be 'sqrt' or 'sqrt-div'")
        for name, value in (
            ("DrCIF min_interval_length", self.min_interval_length),
            ("DrCIF max_interval_length", self.max_interval_length),
        ):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{name} must be an int or float")
            if value <= 0 or (isinstance(value, float) and value > 1):
                raise ValueError(f"{name} must be positive and proportional floats <= 1")
        if self.att_subsample_size is not None:
            value = self.att_subsample_size
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError("DrCIF att_subsample_size must be numeric or None")
            if isinstance(value, int) and value < 1:
                raise ValueError("DrCIF integer att_subsample_size must be positive")
            if isinstance(value, float) and not 0 < value <= 1:
                raise ValueError("DrCIF float att_subsample_size must be in (0, 1]")
        if not isinstance(self.use_pycatch22, bool):
            raise ValueError("use_pycatch22 must be a bool")
        if not isinstance(self.stabilize_near_constant_intervals, bool):
            raise ValueError("stabilize_near_constant_intervals must be a bool")


@dataclass(frozen=True, slots=True)
class ArsenalComponentConfig:
    """Paper-default Arsenal settings and contract cap."""

    n_kernels: int = 2_000
    n_estimators: int = 25
    rocket_transform: str = "rocket"
    max_dilations_per_kernel: int = 32
    n_features_per_kernel: int = 4
    contract_max_n_estimators: int = 25

    def validate(self) -> None:
        _validate_positive_int("Arsenal n_kernels", self.n_kernels)
        _validate_positive_int("Arsenal n_estimators", self.n_estimators)
        _validate_positive_int(
            "Arsenal max_dilations_per_kernel", self.max_dilations_per_kernel
        )
        _validate_positive_int(
            "Arsenal n_features_per_kernel", self.n_features_per_kernel
        )
        _validate_positive_int(
            "Arsenal contract_max_n_estimators",
            self.contract_max_n_estimators,
        )
        if self.rocket_transform not in VALID_ROCKET_TRANSFORMS:
            raise ValueError(
                "rocket_transform must be selected from "
                f"{sorted(VALID_ROCKET_TRANSFORMS)}"
            )


@dataclass(frozen=True, slots=True)
class TDEComponentConfig:
    """Paper-default Temporal Dictionary Ensemble settings and contract cap."""

    n_parameter_samples: int = 250
    max_ensemble_size: int = 50
    max_win_len_prop: float = 1.0
    min_window: int = 10
    randomly_selected_params: int = 50
    bigrams: bool | None = None
    dim_threshold: float = 0.85
    max_dims: int = 20
    contract_max_n_parameter_samples: int = 250

    def validate(self) -> None:
        _validate_positive_int("TDE n_parameter_samples", self.n_parameter_samples)
        _validate_positive_int("TDE max_ensemble_size", self.max_ensemble_size)
        _validate_positive_int("TDE min_window", self.min_window)
        _validate_positive_int(
            "TDE randomly_selected_params", self.randomly_selected_params
        )
        _validate_positive_int("TDE max_dims", self.max_dims)
        _validate_positive_int(
            "TDE contract_max_n_parameter_samples",
            self.contract_max_n_parameter_samples,
        )
        if not 0 < self.max_win_len_prop <= 1:
            raise ValueError("TDE max_win_len_prop must be in (0, 1]")
        if not 0 < self.dim_threshold <= 1:
            raise ValueError("TDE dim_threshold must be in (0, 1]")
        if self.bigrams is not None and not isinstance(self.bigrams, bool):
            raise ValueError("TDE bigrams must be a bool or None")


@dataclass(frozen=True, slots=True)
class HIVECOTEV2Config:
    """Resource policy and four component configurations for one HC2 model."""

    # aeon divides a positive total contract by six for each component. Contracts
    # are approximate because each component must finish its current work unit.
    time_limit_in_minutes: float = 360.0
    stc: STCComponentConfig = field(default_factory=STCComponentConfig)
    drcif: DrCIFComponentConfig = field(default_factory=DrCIFComponentConfig)
    arsenal: ArsenalComponentConfig = field(default_factory=ArsenalComponentConfig)
    tde: TDEComponentConfig = field(default_factory=TDEComponentConfig)
    save_component_predictions: bool = True
    verbose: int = 1
    n_jobs: int = 4
    parallel_backend: str | None = None

    def validate(self) -> None:
        if self.time_limit_in_minutes < 0:
            raise ValueError("time_limit_in_minutes cannot be negative")
        if isinstance(self.verbose, bool) or not isinstance(self.verbose, int):
            raise ValueError("verbose must be a nonnegative int")
        if self.verbose < 0:
            raise ValueError("verbose must be a nonnegative int")
        if self.n_jobs == 0:
            raise ValueError("n_jobs cannot be zero")
        if not isinstance(self.save_component_predictions, bool):
            raise ValueError("save_component_predictions must be a bool")
        if (
            self.parallel_backend is not None
            and self.parallel_backend not in VALID_PARALLEL_BACKENDS
        ):
            raise ValueError(
                "parallel_backend must be None or selected from "
                f"{sorted(VALID_PARALLEL_BACKENDS)}"
            )
        self.stc.validate()
        self.drcif.validate()
        self.arsenal.validate()
        self.tde.validate()


@dataclass(frozen=True, slots=True)
class ExperimentConfig:
    """Complete configuration for one saved HIVE-COTE 2.0 fold."""

    data: DataConfig
    output_dir: Path
    run_name: str
    model: HIVECOTEV2Config = field(default_factory=HIVECOTEV2Config)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    seed: int = 42

    def validate(self) -> None:
        self.data.validate()
        self.model.validate()
        self.evaluation.validate()
        if self.data.num_time_points < 3:
            raise ValueError("HIVE-COTE 2.0 requires at least three time points")
        if not self.run_name.strip():
            raise ValueError("run_name cannot be empty")


__all__ = [
    "ArsenalComponentConfig",
    "DataConfig",
    "DrCIFComponentConfig",
    "EvaluationConfig",
    "ExperimentConfig",
    "HIVECOTEV2Config",
    "STCComponentConfig",
    "TDEComponentConfig",
]
