"""Paper-aligned MASHT feature construction and lazy TabPFN-3 inference."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

try:
    from multirocket.training import _HydraSparseScaler, _load_transform_classes
except ModuleNotFoundError as error:
    if error.name != "multirocket":
        raise
    from ..multirocket.training import _HydraSparseScaler, _load_transform_classes

from .config import MASHTConfig


class TabPFNAccessError(RuntimeError):
    """Raised before fold preprocessing when TabPFN-3 weights are unavailable."""


@dataclass(frozen=True, slots=True)
class FeaturePlan:
    """Resolved nominal budget and transform sizes for one dataset fold."""

    budget_sample_count: int
    nominal_total_budget: int
    component_budget: int
    multirocket_num_kernels: int
    hydra_num_dilations: int
    hydra_num_groups: int


def adaptive_feature_budget(num_samples: int) -> int:
    """Return the nominal feature budget defined by the MASHT paper."""

    if num_samples < 1:
        raise ValueError("num_samples must be positive")
    if num_samples < 1_000:
        return 10_000
    if num_samples < 100_000:
        return 2_000
    return 200


def resolve_feature_plan(
    model: MASHTConfig,
    num_samples: int,
    input_length: int,
) -> FeaturePlan:
    """Translate MASHT's shared budget into aeon MultiRocket/HYDRA parameters."""

    if input_length < 10:
        raise ValueError("MASHT requires at least 10 time points")
    nominal_budget = (
        model.max_features
        if model.max_features is not None
        else adaptive_feature_budget(num_samples)
    )
    component_budget = int(nominal_budget) // 2

    # Reference MASHT allocates half the MultiRocket component to the base series
    # and half to first differences, then emits four pooling features per kernel.
    multirocket_num_kernels = int((component_budget / 2) / 4)
    hydra_num_dilations = int(
        np.ceil(np.log2((input_length - 1) / (9 - 1)))
    )
    if hydra_num_dilations < 1:
        raise ValueError("The time series is too short for MASHT HYDRA dilations")
    hydra_num_groups = int(
        np.floor(component_budget / (hydra_num_dilations * model.hydra_num_kernels))
        / 2
    )
    if hydra_num_groups < 1:
        raise ValueError("The selected MASHT feature budget is too small for HYDRA")
    if multirocket_num_kernels < 84:
        raise ValueError(
            "The selected MASHT feature budget requests fewer than aeon's minimum "
            "84 MultiRocket kernels; set max_features to at least 1,344"
        )
    return FeaturePlan(
        budget_sample_count=int(num_samples),
        nominal_total_budget=int(nominal_budget),
        component_budget=component_budget,
        multirocket_num_kernels=multirocket_num_kernels,
        hydra_num_dilations=hydra_num_dilations,
        hydra_num_groups=hydra_num_groups,
    )


class MASHTFeatureTransformer(BaseEstimator, TransformerMixin):
    """Stack sparse-scaled HYDRA and unscaled MultiRocket features.

    Branch order and preprocessing follow the authors' reference implementation:
    ``[SparseScaler(HYDRA), MultiRocket]``. TabPFN performs its own tabular
    preprocessing after the two branches are concatenated.
    """

    def __init__(
        self,
        multirocket_num_kernels: int,
        hydra_num_groups: int,
        multirocket_max_dilations_per_kernel: int = 32,
        multirocket_num_features_per_kernel: int = 4,
        multirocket_normalise_per_instance: bool = False,
        hydra_num_kernels: int = 8,
        hydra_max_num_channels: int = 8,
        n_jobs: int = 1,
        random_state: int | None = None,
    ) -> None:
        self.multirocket_num_kernels = multirocket_num_kernels
        self.hydra_num_groups = hydra_num_groups
        self.multirocket_max_dilations_per_kernel = (
            multirocket_max_dilations_per_kernel
        )
        self.multirocket_num_features_per_kernel = (
            multirocket_num_features_per_kernel
        )
        self.multirocket_normalise_per_instance = (
            multirocket_normalise_per_instance
        )
        self.hydra_num_kernels = hydra_num_kernels
        self.hydra_max_num_channels = hydra_max_num_channels
        self.n_jobs = n_jobs
        self.random_state = random_state

    def _new_transformers(self) -> tuple[Any, Any]:
        MultiRocket, HydraTransformer = _load_transform_classes()
        multirocket = MultiRocket(
            n_kernels=self.multirocket_num_kernels,
            max_dilations_per_kernel=(
                self.multirocket_max_dilations_per_kernel
            ),
            n_features_per_kernel=self.multirocket_num_features_per_kernel,
            normalise=self.multirocket_normalise_per_instance,
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
    def _combine(hydra_values: Any, multirocket_values: Any) -> np.ndarray:
        hydra_array = np.asarray(hydra_values)
        multirocket_array = np.asarray(multirocket_values)
        if hydra_array.ndim != 2 or multirocket_array.ndim != 2:
            raise RuntimeError("HYDRA and MultiRocket must return 2D matrices")
        if hydra_array.shape[0] != multirocket_array.shape[0]:
            raise RuntimeError("HYDRA and MultiRocket returned different sample counts")
        return np.concatenate((hydra_array, multirocket_array), axis=1).astype(
            np.float32, copy=False
        )

    def fit(self, X: np.ndarray, y: Any = None) -> "MASHTFeatureTransformer":
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
        multirocket_values = np.asarray(self.multirocket_.fit_transform(X, y))
        hydra_values = np.asarray(self.hydra_.fit_transform(X, y))
        self.hydra_scaler_ = _HydraSparseScaler()
        hydra_values = self.hydra_scaler_.fit_transform(hydra_values)
        self.n_hydra_features_ = int(hydra_values.shape[1])
        self.n_multirocket_features_ = int(multirocket_values.shape[1])
        return self._combine(hydra_values, multirocket_values)

    def transform(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(
            self,
            (
                "multirocket_",
                "hydra_",
                "hydra_scaler_",
                "n_hydra_features_",
                "n_multirocket_features_",
            ),
        )
        hydra_values = self.hydra_scaler_.transform(
            np.asarray(self.hydra_.transform(X))
        )
        multirocket_values = np.asarray(self.multirocket_.transform(X))
        return self._combine(hydra_values, multirocket_values)


def new_feature_transformer(
    model: MASHTConfig,
    plan: FeaturePlan,
    seed: int,
) -> MASHTFeatureTransformer:
    return MASHTFeatureTransformer(
        multirocket_num_kernels=plan.multirocket_num_kernels,
        hydra_num_groups=plan.hydra_num_groups,
        multirocket_max_dilations_per_kernel=(
            model.multirocket_max_dilations_per_kernel
        ),
        multirocket_num_features_per_kernel=(
            model.multirocket_num_features_per_kernel
        ),
        multirocket_normalise_per_instance=(
            model.multirocket_normalise_per_instance
        ),
        hydra_num_kernels=model.hydra_num_kernels,
        hydra_max_num_channels=model.hydra_max_num_channels,
        n_jobs=model.transform_n_jobs,
        random_state=seed,
    )


def _load_tabpfn_classes() -> tuple[type, type]:
    try:
        from tabpfn import TabPFNClassifier
        from tabpfn.constants import ModelVersion
    except ImportError as error:
        raise ImportError(
            "MASHT requires TabPFN. Install project requirements with "
            "`pip install -r requirements.txt`."
        ) from error
    return TabPFNClassifier, ModelVersion


def new_tabpfn_classifier(model: MASHTConfig, seed: int) -> Any:
    """Create an explicitly versioned TabPFN-3 classifier with MASHT settings."""

    TabPFNClassifier, ModelVersion = _load_tabpfn_classes()
    overrides: dict[str, Any] = {
        "n_estimators": model.tabpfn_n_estimators,
        "auto_scale_n_estimators": model.tabpfn_auto_scale_n_estimators,
        "fit_mode": model.tabpfn_fit_mode,
        "device": model.tabpfn_device,
        "memory_saving_mode": model.tabpfn_memory_saving_mode,
        "inference_precision": model.tabpfn_inference_precision,
        "n_preprocessing_jobs": model.tabpfn_preprocessing_jobs,
        "tuning_config": None,
        "show_progress_bar": model.tabpfn_show_progress_bar,
        "ignore_pretraining_limits": model.tabpfn_ignore_pretraining_limits,
        "balance_probabilities": model.tabpfn_balance_probabilities,
        "random_state": seed,
    }
    if model.tabpfn_model_path is not None:
        overrides["model_path"] = str(model.tabpfn_model_path)
    return TabPFNClassifier.create_default_for_version(
        ModelVersion.V3,
        **overrides,
    )


def ensure_tabpfn_checkpoint_access(model: MASHTConfig, seed: int) -> None:
    """Fail before loading a fold when the gated TabPFN-3 weights are missing."""

    classifier = new_tabpfn_classifier(model, seed)
    configured_paths = classifier.model_path
    paths = configured_paths if isinstance(configured_paths, list) else [configured_paths]
    if paths and all(Path(path).is_file() for path in paths):
        return

    try:
        from tabpfn.browser_auth import ensure_license_accepted
        from tabpfn.errors import TabPFNError
        from tabpfn.model_loading import ModelSource
    except ImportError as error:
        raise ImportError(
            "MASHT requires TabPFN. Install project requirements with "
            "`pip install -r requirements.txt`."
        ) from error
    repository = ModelSource.get_classifier_v3().repo_id.rsplit("/", 1)[-1]
    try:
        ensure_license_accepted(hf_repo_id=repository)
    except TabPFNError as error:
        raise TabPFNAccessError(
            "TabPFN v3 weights are not cached and require one-time license "
            "acceptance. Accept the model license at https://ux.priorlabs.ai, "
            "put TABPFN_TOKEN=<your-api-key> in the project-root .env, then "
            "restart the script. Do not commit or share the token."
        ) from error


def positive_class_probabilities(classifier: Any, features: np.ndarray) -> np.ndarray:
    probabilities = np.asarray(classifier.predict_proba(features), dtype=np.float64)
    classes = np.asarray(classifier.classes_)
    positive_columns = np.flatnonzero(classes == 1)
    if positive_columns.size != 1:
        raise RuntimeError(
            "TabPFN classes must contain binary positive label 1, got "
            f"{classes.tolist()}"
        )
    expected_shape = (features.shape[0], classes.size)
    if probabilities.shape != expected_shape:
        raise RuntimeError(
            f"TabPFN returned shape {probabilities.shape}; expected {expected_shape}"
        )
    positive = probabilities[:, int(positive_columns[0])]
    if not np.all(np.isfinite(positive)):
        raise RuntimeError("TabPFN returned non-finite probabilities")
    return np.clip(positive, 0.0, 1.0)


@dataclass(slots=True)
class FittedMASHT:
    """Fitted random transforms plus training context for lazy TabPFN inference."""

    model_config: MASHTConfig
    seed: int
    train_values: np.ndarray
    train_labels: np.ndarray
    feature_transformer: MASHTFeatureTransformer

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        train_features = np.asarray(
            self.feature_transformer.transform(self.train_values),
            dtype=np.float32,
            order="C",
        )
        query_features = np.asarray(
            self.feature_transformer.transform(X),
            dtype=np.float32,
            order="C",
        )
        classifier = new_tabpfn_classifier(self.model_config, self.seed)
        classifier.fit(train_features, self.train_labels)
        positive = positive_class_probabilities(classifier, query_features)
        return np.column_stack((1.0 - positive, positive))

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= threshold).astype(np.int64)


__all__ = [
    "FeaturePlan",
    "FittedMASHT",
    "MASHTFeatureTransformer",
    "TabPFNAccessError",
    "adaptive_feature_budget",
    "ensure_tabpfn_checkpoint_access",
    "new_feature_transformer",
    "new_tabpfn_classifier",
    "positive_class_probabilities",
    "resolve_feature_plan",
]
