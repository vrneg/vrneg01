"""Large structured candidate bank for Adaptive Multi-Representation RocketPFN."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
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

from .config import AdaptiveRocketPFNConfig
from .views import temporal_views


EXPERT_NAMES: tuple[str, ...] = ("morphology", "dynamics", "prototypes")
MORPHOLOGY_VIEWS = frozenset(("raw", "smoothed", "local_norm"))
DYNAMICS_VIEWS = frozenset(("first_diff", "second_diff", "highpass"))
POOLING_ORDER: tuple[str, ...] = ("ppv", "lspv", "mpv", "mipv")


@dataclass(frozen=True, slots=True)
class FeatureFamily:
    """Metadata tying a contiguous feature block to its generating choices."""

    name: str
    expert: str
    bank_key: str
    start: int
    stop: int
    view: str
    dilation_regime: int | None
    internal_representation: str
    pooling: str

    @property
    def width(self) -> int:
        return self.stop - self.start


def _load_prototype_transform_class() -> type:
    try:
        from aeon.transformations.collection.shapelet_based import (
            RandomDilatedShapeletTransform,
        )
    except ImportError as error:
        raise ImportError(
            "The prototype expert requires aeon's random dilated shapelet transform."
        ) from error
    return RandomDilatedShapeletTransform


def _expert_for_view(view: str) -> str:
    if view in MORPHOLOGY_VIEWS:
        return "morphology"
    if view in DYNAMICS_VIEWS:
        return "dynamics"
    raise ValueError(f"No semantic expert is assigned to view {view!r}")


class AdaptiveCandidateBank(BaseEstimator, TransformerMixin):
    """Generate structured MultiRocket, HYDRA, and prototype-distance candidates.

    MultiRocket outputs are exposed as separate families for each external temporal
    view, dilation regime, internal base/difference branch, and pooling operator.
    This metadata lets the supervised selector learn a compact empirical feature
    prior while the convolution weights themselves remain random.

    The prototype family uses aeon's maintained random dilated shapelet-distance
    transform. It is SPROCKET-inspired but deliberately not labelled SPROCKET.
    """

    def __init__(self, config: AdaptiveRocketPFNConfig, random_state: int = 42):
        self.config = config
        self.random_state = random_state

    def _report(self, message: str) -> None:
        callback = getattr(self, "progress_callback_", None)
        if callback is not None:
            callback(message)

    def _views(self, X: np.ndarray) -> dict[str, np.ndarray]:
        return temporal_views(
            X,
            self.config.views,
            smoothing_window=self.config.smoothing_window,
            normalization_epsilon=self.config.local_normalization_epsilon,
        )

    def _new_multirocket(self, dilation_regime: int, seed: int) -> Any:
        MultiRocket, _ = _load_transform_classes()
        return MultiRocket(
            n_kernels=self.config.multirocket_num_kernels_per_bank,
            max_dilations_per_kernel=dilation_regime,
            n_features_per_kernel=(
                self.config.multirocket_num_features_per_kernel
            ),
            normalise=self.config.multirocket_normalise_per_instance,
            n_jobs=self.config.transform_n_jobs,
            random_state=seed,
        )

    def _new_hydra(self, seed: int) -> Any:
        _, HydraTransformer = _load_transform_classes()
        return HydraTransformer(
            n_kernels=self.config.hydra_num_kernels,
            n_groups=self.config.hydra_num_groups,
            max_num_channels=self.config.hydra_max_num_channels,
            n_jobs=self.config.transform_n_jobs,
            random_state=seed,
            output_type="numpy",
        )

    def _new_prototype_transform(self, seed: int) -> Any:
        RandomDilatedShapeletTransform = _load_prototype_transform_class()
        return RandomDilatedShapeletTransform(
            max_shapelets=self.config.prototype_max_shapelets,
            shapelet_lengths=list(self.config.prototype_shapelet_lengths),
            proba_normalization=(
                self.config.prototype_normalization_probability
            ),
            alpha_similarity=self.config.prototype_similarity,
            random_state=seed,
            n_jobs=self.config.transform_n_jobs,
        )

    @staticmethod
    def _multirocket_counts(transformer: Any, width: int) -> tuple[int, int]:
        base_count = int(np.asarray(transformer.parameter[-1]).size)
        difference_count = int(np.asarray(transformer.parameter1[-1]).size)
        expected_width = 4 * (base_count + difference_count)
        if width != expected_width:
            raise RuntimeError(
                f"Unexpected MultiRocket layout {width}; expected {expected_width}"
            )
        return base_count, difference_count

    @staticmethod
    def _multirocket_families(
        *,
        view: str,
        dilation_regime: int,
        bank_key: str,
        base_count: int,
        difference_count: int,
    ) -> list[FeatureFamily]:
        expert = _expert_for_view(view)
        families: list[FeatureFamily] = []
        offset = 0
        for internal_representation, count in (
            ("base", base_count),
            ("difference", difference_count),
        ):
            for pooling in POOLING_ORDER:
                name = (
                    f"{view}.d{dilation_regime}."
                    f"{internal_representation}.{pooling}"
                )
                families.append(
                    FeatureFamily(
                        name=name,
                        expert=expert,
                        bank_key=bank_key,
                        start=offset,
                        stop=offset + count,
                        view=view,
                        dilation_regime=dilation_regime,
                        internal_representation=internal_representation,
                        pooling=pooling,
                    )
                )
                offset += count
        return families

    @staticmethod
    def _slice_families(
        bank_outputs: dict[str, np.ndarray],
        families: tuple[FeatureFamily, ...] | list[FeatureFamily],
    ) -> dict[str, np.ndarray]:
        return {
            family.name: np.asarray(
                bank_outputs[family.bank_key][:, family.start : family.stop],
                dtype=np.float32,
            )
            for family in families
        }

    @staticmethod
    def _validate_bank_outputs(bank_outputs: dict[str, np.ndarray]) -> None:
        invalid = [
            name
            for name, values in bank_outputs.items()
            if not np.isfinite(values).all()
        ]
        if invalid:
            raise RuntimeError(
                "Candidate transforms produced non-finite values in: "
                + ", ".join(invalid)
            )

    def fit(self, X: np.ndarray, y: Any = None) -> "AdaptiveCandidateBank":
        self.fit_transform(X, y)
        return self

    def fit_transform(
        self,
        X: np.ndarray,
        y: Any = None,
        **fit_params: Any,
    ) -> dict[str, np.ndarray]:
        # Every candidate transform is deliberately unsupervised. Labels enter only
        # the nested stable-selection stage after this bank has been materialized.
        del y, fit_params
        views = self._views(X)
        self.multirocket_transformers_: dict[str, Any] = {}
        families: list[FeatureFamily] = []
        bank_outputs: dict[str, np.ndarray] = {}
        seed_offset = 0

        for view_name, values in views.items():
            for dilation_regime in self.config.dilation_regimes:
                bank_key = f"multirocket::{view_name}::d{dilation_regime}"
                transformer = self._new_multirocket(
                    dilation_regime,
                    self.random_state + seed_offset,
                )
                seed_offset += 1
                transformed = np.asarray(
                    transformer.fit_transform(values, None),
                    dtype=np.float32,
                )
                self.multirocket_transformers_[bank_key] = transformer
                bank_outputs[bank_key] = transformed
                base_count, difference_count = self._multirocket_counts(
                    transformer, transformed.shape[1]
                )
                families.extend(
                    self._multirocket_families(
                        view=view_name,
                        dilation_regime=dilation_regime,
                        bank_key=bank_key,
                        base_count=base_count,
                        difference_count=difference_count,
                    )
                )
                self._report(
                    f"Candidate bank {bank_key}: {transformed.shape[1]:,} "
                    "MultiRocket features."
                )

        self.hydra_ = self._new_hydra(self.random_state + seed_offset)
        hydra_values = np.asarray(self.hydra_.fit_transform(X, None))
        self.hydra_scaler_ = _HydraSparseScaler()
        hydra_values = np.asarray(
            self.hydra_scaler_.fit_transform(hydra_values),
            dtype=np.float32,
        )
        bank_outputs["hydra"] = hydra_values
        families.append(
            FeatureFamily(
                name="hydra.competition",
                expert="morphology",
                bank_key="hydra",
                start=0,
                stop=hydra_values.shape[1],
                view="raw",
                dilation_regime=None,
                internal_representation="competition",
                pooling="histogram",
            )
        )
        self._report(
            f"Candidate bank HYDRA: {hydra_values.shape[1]:,} features."
        )

        self.prototype_ = self._new_prototype_transform(
            self.random_state + seed_offset + 1
        )
        prototype_input = np.ascontiguousarray(X, dtype=np.float64)
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="Some invalid values .* shapelet transformation.*",
                category=UserWarning,
            )
            prototype_values = np.asarray(
                self.prototype_.fit_transform(prototype_input, None),
                dtype=np.float32,
            )
        bank_outputs["prototype"] = prototype_values
        families.append(
            FeatureFamily(
                name="prototype.rdst",
                expert="prototypes",
                bank_key="prototype",
                start=0,
                stop=prototype_values.shape[1],
                view="raw",
                dilation_regime=None,
                internal_representation="shapelet",
                pooling="distance_occurrence_location",
            )
        )
        self._report(
            f"Candidate bank prototype distances: "
            f"{prototype_values.shape[1]:,} features."
        )

        self.family_specs_ = tuple(families)
        self.n_candidate_features_ = int(sum(item.width for item in families))
        self.family_widths_ = {item.name: item.width for item in families}
        self.expert_families_ = {
            expert: tuple(item.name for item in families if item.expert == expert)
            for expert in EXPERT_NAMES
        }
        self._validate_bank_outputs(bank_outputs)
        return self._slice_families(bank_outputs, self.family_specs_)

    def transform(self, X: np.ndarray) -> dict[str, np.ndarray]:
        check_is_fitted(
            self,
            (
                "multirocket_transformers_",
                "hydra_",
                "hydra_scaler_",
                "prototype_",
                "family_specs_",
            ),
        )
        views = self._views(X)
        bank_outputs: dict[str, np.ndarray] = {}
        for bank_key, transformer in self.multirocket_transformers_.items():
            _, view_name, _ = bank_key.split("::")
            bank_outputs[bank_key] = np.asarray(
                transformer.transform(views[view_name]),
                dtype=np.float32,
            )
        bank_outputs["hydra"] = np.asarray(
            self.hydra_scaler_.transform(np.asarray(self.hydra_.transform(X))),
            dtype=np.float32,
        )
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="Some invalid values .* shapelet transformation.*",
                category=UserWarning,
            )
            bank_outputs["prototype"] = np.asarray(
                self.prototype_.transform(
                    np.ascontiguousarray(X, dtype=np.float64)
                ),
                dtype=np.float32,
            )
        self._validate_bank_outputs(bank_outputs)
        return self._slice_families(bank_outputs, self.family_specs_)


__all__ = [
    "AdaptiveCandidateBank",
    "DYNAMICS_VIEWS",
    "EXPERT_NAMES",
    "FeatureFamily",
    "MORPHOLOGY_VIEWS",
    "POOLING_ORDER",
]
