"""Deterministic temporal representations used by the adaptive candidate bank."""

from __future__ import annotations

import numpy as np

try:
    from minirocket.data import collapse_near_constant_traces
except ModuleNotFoundError as error:
    if error.name != "minirocket":
        raise
    from ..minirocket.data import collapse_near_constant_traces

from .config import ViewName


def moving_average(values: np.ndarray, window: int) -> np.ndarray:
    """Centered moving average with reflected boundaries and unchanged length."""

    array = np.asarray(values, dtype=np.float32)
    if array.ndim != 3:
        raise ValueError("Temporal views require [cases, channels, time] arrays")
    if window < 3 or window % 2 == 0 or window > array.shape[-1]:
        raise ValueError("window must be odd and fit inside the time dimension")
    padding = window // 2
    padded = np.pad(array, ((0, 0), (0, 0), (padding, padding)), mode="reflect")
    cumulative = np.cumsum(padded, axis=-1, dtype=np.float64)
    cumulative = np.concatenate(
        (np.zeros((*array.shape[:-1], 1), dtype=np.float64), cumulative),
        axis=-1,
    )
    smoothed = (cumulative[..., window:] - cumulative[..., :-window]) / window
    return smoothed.astype(np.float32, copy=False)


def temporal_views(
    X: np.ndarray,
    names: tuple[ViewName, ...],
    *,
    smoothing_window: int,
    normalization_epsilon: float,
) -> dict[str, np.ndarray]:
    """Generate requested raw, derivative, frequency, and normalized views."""

    values = np.asarray(X, dtype=np.float32)
    if values.ndim != 3:
        raise ValueError("X must have shape [cases, channels, time]")
    result: dict[str, np.ndarray] = {}
    smooth: np.ndarray | None = None

    for name in names:
        if name == "raw":
            view = values
        elif name == "first_diff":
            view = np.diff(values, n=1, axis=-1)
        elif name == "second_diff":
            view = np.diff(values, n=2, axis=-1)
        else:
            if smooth is None:
                smooth = moving_average(values, smoothing_window)
            if name == "smoothed":
                view = smooth
            elif name == "highpass":
                view = values - smooth
            elif name == "local_norm":
                residual = values - smooth
                local_variance = moving_average(residual * residual, smoothing_window)
                view = residual / np.sqrt(local_variance + normalization_epsilon)
            else:
                raise ValueError(f"Unknown temporal view: {name!r}")
        # Differencing and smoothing can turn a valid signal into non-zero
        # floating-point residue below aeon's 1e-7 collection cutoff. Treat that
        # residue as the effectively constant trace it represents.
        result[name] = np.ascontiguousarray(
            collapse_near_constant_traces(view),
            dtype=np.float32,
        )
    return result


__all__ = ["moving_average", "temporal_views"]
