"""Raw MultiRocket-HYDRA features and inner-fold branch standardization."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted

try:
    from multirocket.training import _HydraSparseScaler, _load_transform_classes
except ModuleNotFoundError as error:
    if error.name != "multirocket":
        raise
    from ..multirocket.training import _HydraSparseScaler, _load_transform_classes


class MultiRocketHydraRawFeatures(BaseEstimator, TransformerMixin):
    """Concatenate unscaled MultiRocket and HYDRA features.

    The expensive convolutional transforms are fitted once on an outer fold's
    training split. Scaling and supervised feature selection remain inside the
    downstream inner-CV pipeline.
    """

    def __init__(
        self,
        num_kernels: int = 6_250,
        max_dilations_per_kernel: int = 32,
        num_features_per_kernel: int = 4,
        normalise_per_instance: bool = False,
        hydra_num_kernels: int = 8,
        hydra_num_groups: int = 64,
        hydra_max_num_channels: int = 8,
        n_jobs: int = 1,
        random_state: int | None = None,
    ):
        self.num_kernels = num_kernels
        self.max_dilations_per_kernel = max_dilations_per_kernel
        self.num_features_per_kernel = num_features_per_kernel
        self.normalise_per_instance = normalise_per_instance
        self.hydra_num_kernels = hydra_num_kernels
        self.hydra_num_groups = hydra_num_groups
        self.hydra_max_num_channels = hydra_max_num_channels
        self.n_jobs = n_jobs
        self.random_state = random_state

    def _new_transformers(self) -> tuple[Any, Any]:
        MultiRocket, HydraTransformer = _load_transform_classes()
        multirocket = MultiRocket(
            n_kernels=self.num_kernels,
            max_dilations_per_kernel=self.max_dilations_per_kernel,
            n_features_per_kernel=self.num_features_per_kernel,
            normalise=self.normalise_per_instance,
            n_jobs=self.n_jobs,
            random_state=self.random_state,
        )
        hydra = HydraTransformer(
            n_kernels=self.hydra_num_kernels,
            n_groups=self.hydra_num_groups,
            max_num_channels=self.hydra_max_num_channels,
            n_jobs=self.n_jobs,
            random_state=self.random_state,
            output_type="numpy",
        )
        return multirocket, hydra

    @staticmethod
    def _combine(multirocket_values: Any, hydra_values: Any) -> np.ndarray:
        multirocket_array = np.asarray(multirocket_values)
        hydra_array = np.asarray(hydra_values)
        if multirocket_array.ndim != 2 or hydra_array.ndim != 2:
            raise RuntimeError("MultiRocket and HYDRA must return 2D feature matrices")
        if multirocket_array.shape[0] != hydra_array.shape[0]:
            raise RuntimeError("MultiRocket and HYDRA returned different sample counts")
        return np.concatenate((multirocket_array, hydra_array), axis=1).astype(
            np.float32, copy=False
        )

    def fit(self, X: np.ndarray, y: Any = None) -> MultiRocketHydraRawFeatures:
        self.fit_transform(X, y)
        return self

    def fit_transform(
        self,
        X: np.ndarray,
        y: Any = None,
        **fit_params: Any,
    ) -> np.ndarray:
        del fit_params
        self.multirocket_, self.hydra_ = self._new_transformers()
        multirocket_values = self.multirocket_.fit_transform(X, y)
        hydra_values = self.hydra_.fit_transform(X, y)
        self.n_multirocket_features_ = int(np.asarray(multirocket_values).shape[1])
        self.n_hydra_features_ = int(np.asarray(hydra_values).shape[1])
        return self._combine(multirocket_values, hydra_values)

    def transform(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(
            self,
            ("multirocket_", "hydra_", "n_multirocket_features_", "n_hydra_features_"),
        )
        return self._combine(
            self.multirocket_.transform(X),
            self.hydra_.transform(X),
        )


class MultiRocketHydraStandardizer(BaseEstimator, TransformerMixin):
    """Apply the appropriate scaler to each concatenated feature branch."""

    def __init__(self, n_multirocket_features: int):
        self.n_multirocket_features = n_multirocket_features

    def _split(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        values = np.asarray(X)
        if values.ndim != 2:
            raise ValueError("Expected a 2D concatenated feature matrix")
        if not 0 < self.n_multirocket_features < values.shape[1]:
            raise ValueError(
                "n_multirocket_features must split non-empty MultiRocket and HYDRA branches"
            )
        return (
            values[:, : self.n_multirocket_features],
            values[:, self.n_multirocket_features :],
        )

    def fit(self, X: np.ndarray, y: Any = None) -> MultiRocketHydraStandardizer:
        del y
        multirocket_values, hydra_values = self._split(X)
        self.multirocket_scaler_ = StandardScaler(with_mean=False).fit(
            multirocket_values
        )
        self.hydra_scaler_ = _HydraSparseScaler().fit(hydra_values)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(self, ("multirocket_scaler_", "hydra_scaler_"))
        multirocket_values, hydra_values = self._split(X)
        scaled_multirocket = self.multirocket_scaler_.transform(multirocket_values)
        scaled_hydra = self.hydra_scaler_.transform(hydra_values)
        return np.concatenate((scaled_multirocket, scaled_hydra), axis=1).astype(
            np.float32, copy=False
        )


__all__ = [
    "MultiRocketHydraRawFeatures",
    "MultiRocketHydraStandardizer",
]
