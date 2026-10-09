"""Numerical safeguards for aeon's interval-level Catch22 extraction."""

from __future__ import annotations

import numpy as np
from aeon.classification.interval_based import DrCIFClassifier
from aeon.transformations.collection import PeriodogramTransformer
from aeon.transformations.collection.feature_based import Catch22
from aeon.utils.numba.general import first_order_differences_3d
from sklearn.preprocessing import FunctionTransformer


AEON_MINIMUM_TEMPORAL_STD = 1e-7


def collapse_near_constant_traces(X: np.ndarray) -> np.ndarray:
    """Collapse only nonconstant 3D traces below aeon's variance cutoff."""

    if not isinstance(X, np.ndarray) or X.ndim != 3:
        return X

    ranges = np.nanmax(X, axis=2) - np.nanmin(X, axis=2)
    standard_deviations = np.nanstd(X, axis=2, ddof=0)
    near_constant = (
        (standard_deviations <= AEON_MINIMUM_TEMPORAL_STD) & (ranges != 0)
    )
    if not np.any(near_constant):
        return X

    stabilized = X.copy()
    case_indices, channel_indices = np.nonzero(near_constant)
    levels = np.nanmean(
        stabilized[case_indices, channel_indices, :],
        axis=1,
        dtype=np.float64,
    ).astype(stabilized.dtype, copy=False)
    stabilized[case_indices, channel_indices, :] = levels[:, None]
    return stabilized


def stabilized_first_order_differences(X: np.ndarray) -> np.ndarray:
    """Build DrCIF's difference representation and stabilize its flat traces."""

    return collapse_near_constant_traces(first_order_differences_3d(X))


def stabilized_periodogram(X: np.ndarray) -> np.ndarray:
    """Build DrCIF's periodogram representation and stabilize its flat traces."""

    transformed = PeriodogramTransformer().fit_transform(X)
    return collapse_near_constant_traces(transformed)


class NearConstantSafeCatch22(Catch22):
    """Treat sub-threshold numerical residue as an exactly constant interval.

    aeon intentionally rejects a series when its standard deviation is at most
    ``1e-7`` but its range is nonzero. DrCIF can create such series only after it
    has sampled a short interval, so whole-series preprocessing cannot reliably
    catch them. Collapsing those interval-local values to their mean preserves
    their effective constant value and lets Catch22 handle them through its normal
    constant-series path.
    """

    def _preprocess_collection(self, X, store_metadata: bool = True):
        X = collapse_near_constant_traces(X)
        return super()._preprocess_collection(X, store_metadata=store_metadata)


class NearConstantSafeDrCIFClassifier(DrCIFClassifier):
    """DrCIF variant that installs the interval safeguard after aeon's reset."""

    def _install_near_constant_safeguard(self) -> None:
        raw, _, _ = self.series_transformers
        if raw is not None:
            raise RuntimeError("Unexpected aeon DrCIF raw representation transformer")

        # Differences and periodograms can create effectively constant full traces.
        # RandomIntervals validates those representations before slicing them.
        self.series_transformers = [
            FunctionTransformer(collapse_near_constant_traces, validate=False),
            FunctionTransformer(stabilized_first_order_differences, validate=False),
            FunctionTransformer(stabilized_periodogram, validate=False),
        ]
        self.interval_features[0] = NearConstantSafeCatch22(
            outlier_norm=True,
            use_pycatch22=self.use_pycatch22,
        )

    def _fit(self, X, y):
        # BaseClassifier.fit resets all constructor state before entering _fit.
        # Install here so that reset cannot silently restore aeon's plain Catch22.
        self._install_near_constant_safeguard()
        return super()._fit(X, y)

    def _fit_predict_proba(self, X, y):
        self._install_near_constant_safeguard()
        return super()._fit_predict_proba(X, y)


__all__ = [
    "AEON_MINIMUM_TEMPORAL_STD",
    "NearConstantSafeCatch22",
    "NearConstantSafeDrCIFClassifier",
    "collapse_near_constant_traces",
    "stabilized_first_order_differences",
    "stabilized_periodogram",
]
