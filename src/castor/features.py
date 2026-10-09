"""CASTOR feature construction and preprocessing utilities."""

from __future__ import annotations

import inspect
from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array, check_is_fitted, check_random_state

from .config import ExperimentConfig


def _patch_wildboar_sklearn_validation() -> None:
    """Bridge Wildboar 1.2's renamed scikit-learn validation argument."""

    from sklearn.utils.validation import check_array as sklearn_check_array
    from wildboar.utils import validation as wildboar_validation

    parameters = inspect.signature(sklearn_check_array).parameters
    if "force_all_finite" in parameters or "ensure_all_finite" not in parameters:
        return
    current = wildboar_validation.sklearn_check_array
    if getattr(current, "_castor_sklearn_compat", False):
        return

    def compatible_check_array(*args: Any, **kwargs: Any) -> np.ndarray:
        if "force_all_finite" in kwargs:
            kwargs["ensure_all_finite"] = kwargs.pop("force_all_finite")
        return sklearn_check_array(*args, **kwargs)

    compatible_check_array._castor_sklearn_compat = True  # type: ignore[attr-defined]
    wildboar_validation.sklearn_check_array = compatible_check_array


def load_castor_transform_class() -> type:
    """Load the official compiled CASTOR transform with a useful error message."""

    try:
        import wildboar  # noqa: F401

        _patch_wildboar_sklearn_validation()
        from wildboar.transform import CastorTransform
    except ImportError as error:
        raise ImportError(
            "CASTOR experiments require Wildboar 1.2.1. Because aeon and the "
            "released Wildboar package declare conflicting scikit-learn ranges, "
            "install it without dependencies using `venv/bin/pip install "
            "--no-deps -r requirements-castor.txt`."
        ) from error
    return CastorTransform


class FirstDifferenceTransformer(TransformerMixin, BaseEstimator):
    """Compute the discrete first difference along the time dimension."""

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray | None = None,
    ) -> FirstDifferenceTransformer:
        values = np.asarray(X)
        if values.ndim not in (2, 3):
            raise ValueError("CASTOR input must be a 2D or 3D time-series array")
        if values.shape[-1] < 2:
            raise ValueError("first differences require at least two time points")
        self.n_timesteps_in_ = int(values.shape[-1])
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(self, "n_timesteps_in_")
        values = np.asarray(X)
        if values.ndim not in (2, 3):
            raise ValueError("CASTOR input must be a 2D or 3D time-series array")
        if values.shape[-1] != self.n_timesteps_in_:
            raise ValueError("CASTOR input has a different number of time points")
        return np.diff(values, axis=-1)


class CastorSparseScaler(TransformerMixin, BaseEstimator):
    """Apply the square-root scaling used by Wildboar's CASTOR classifier."""

    def __init__(self, exp: float = 4.0):
        self.exp = exp

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> CastorSparseScaler:
        values = check_array(X, ensure_2d=True, dtype=np.float64)
        rooted = np.sqrt(values.clip(min=0.0))
        self.mean_ = rooted.mean(axis=0)
        zero_fraction = float((rooted == 0.0).mean())
        self.scale_ = rooted.std(axis=0) + zero_fraction**self.exp + 1e-8
        self.n_features_in_ = int(rooted.shape[1])
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(self, ("mean_", "scale_", "n_features_in_"))
        values = check_array(X, ensure_2d=True, dtype=np.float64)
        if values.shape[1] != self.n_features_in_:
            raise ValueError("CASTOR feature count differs from the fitted scaler")
        return (np.sqrt(values.clip(min=0.0)) - self.mean_) / self.scale_


def build_feature_transformer(config: ExperimentConfig) -> Any:
    """Build the paper's raw/difference competing-shapelet representation."""

    from sklearn.pipeline import FeatureUnion, Pipeline

    CastorTransform = load_castor_transform_class()
    model = config.model
    random_state = check_random_state(config.seed)
    groups_per_representation = (
        model.n_groups // 2 if model.use_first_difference else model.n_groups
    )

    def new_transform() -> Any:
        return CastorTransform(
            n_groups=groups_per_representation,
            n_shapelets=model.n_shapelets,
            metric=model.metric,
            normalize_prob=model.normalize_prob,
            shapelet_size=model.shapelet_size,
            lower=model.lower,
            upper=model.upper,
            soft_min=model.soft_min,
            soft_max=model.soft_max,
            soft_threshold=model.soft_threshold,
            ignore_y=model.ignore_y,
            random_state=int(random_state.randint(np.iinfo(np.int32).max)),
            n_jobs=model.n_jobs,
        )

    raw_transform = new_transform()
    if not model.use_first_difference:
        return raw_transform
    return FeatureUnion(
        [
            ("raw", raw_transform),
            (
                "first_difference",
                Pipeline(
                    [
                        ("difference", FirstDifferenceTransformer()),
                        ("castor", new_transform()),
                    ]
                ),
            ),
        ]
    )


__all__ = [
    "CastorSparseScaler",
    "FirstDifferenceTransformer",
    "build_feature_transformer",
    "load_castor_transform_class",
]
