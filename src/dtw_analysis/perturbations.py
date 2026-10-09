"""Label-independent temporal perturbations for robustness evaluation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

try:
    from event_transformer.features import MODALITY_NAMES
except ModuleNotFoundError as error:
    if error.name != "event_transformer":
        raise
    from ..event_transformer.features import MODALITY_NAMES


MINIMUM_TEMPORAL_STD = 1e-7
WARP_SCHEMA_VERSION = 4


def source_positions(
    num_time_points: int,
    kind: str,
    magnitude: float,
    center_index: float | None = None,
) -> np.ndarray:
    """Map output grid coordinates to source coordinates for one time warp."""

    coordinates = np.arange(num_time_points, dtype=np.float64)
    if kind == "shift":
        return coordinates - magnitude
    if kind == "scale":
        if magnitude <= 0.0:
            raise ValueError("scale magnitude must be positive")
        center = (
            (num_time_points - 1) / 2.0
            if center_index is None
            else float(center_index)
        )
        return center + (coordinates - center) / magnitude
    raise ValueError("kind must be 'shift' or 'scale'")


def _linear_warp(traces: np.ndarray, positions: np.ndarray) -> np.ndarray:
    grid = np.arange(traces.shape[-1], dtype=np.float64)
    flat_source = traces.reshape(-1, traces.shape[-1])
    # Modality selection uses advanced indexing and produces a non-contiguous
    # layout. ``empty_like(traces).reshape(...)`` can therefore be a copy; filling
    # that copy leaves the returned allocation uninitialized. Build the flat
    # destination directly and reshape it only after every value has been written.
    flat_output = np.empty(
        (flat_source.shape[0], traces.shape[-1]), dtype=traces.dtype
    )
    for index, trace in enumerate(flat_source):
        flat_output[index] = np.interp(positions, grid, trace, left=0.0, right=0.0)
    return flat_output.reshape(traces.shape)


def _nearest_warp(traces: np.ndarray, positions: np.ndarray) -> np.ndarray:
    rounded = np.rint(positions).astype(np.int64)
    valid = (rounded >= 0) & (rounded < traces.shape[-1])
    output = np.zeros_like(traces)
    output[..., valid] = traces[..., rounded[valid]]
    return output


def _collapse_near_constant_traces(values: np.ndarray) -> np.ndarray:
    """Remove interpolation residue rejected by Aeon collection estimators."""

    # Variance in float32 can overflow for otherwise finite normalized motion
    # outliers. The comparison is numerical housekeeping, so perform it in float64.
    stable_values = values.astype(np.float64, copy=False)
    temporal_ranges = np.ptp(stable_values, axis=2)
    temporal_stds = np.std(stable_values, axis=2, ddof=0)
    near_constant = (temporal_stds <= MINIMUM_TEMPORAL_STD) & (
        temporal_ranges != 0
    )
    if not np.any(near_constant):
        return values
    levels = np.mean(values, axis=2, keepdims=True, dtype=np.float64).astype(
        values.dtype
    )
    return np.where(near_constant[:, :, None], levels, values).astype(
        values.dtype, copy=False
    )


def warp_fixed_grid(
    values: np.ndarray,
    channel_names: tuple[str, ...],
    modalities: Sequence[str],
    *,
    kind: str,
    magnitude: float,
    center_index: float | None = None,
) -> np.ndarray:
    """Warp selected modality channels while preserving all other inputs."""

    if values.ndim != 3 or values.shape[1] != len(channel_names):
        raise ValueError("values and channel_names describe incompatible fixed grids")
    if not np.all(np.isfinite(values)):
        raise ValueError("Fixed-grid input contains non-finite values before warping")
    selected = frozenset(modalities)
    unknown = selected - set(MODALITY_NAMES)
    if unknown:
        raise ValueError(f"Unknown modalities: {sorted(unknown)}")
    positions = source_positions(
        values.shape[-1], kind, magnitude, center_index=center_index
    )
    result = np.array(values, copy=True)
    continuous_indices = [
        index
        for index, name in enumerate(channel_names)
        if name.split(".", 1)[0] in selected
        and not name.endswith(".present")
        and ".flag_" not in name
    ]
    discrete_indices = [
        index
        for index, name in enumerate(channel_names)
        if name.split(".", 1)[0] in selected
        and (name.endswith(".present") or ".flag_" in name)
    ]
    if continuous_indices:
        result[:, continuous_indices, :] = _linear_warp(
            values[:, continuous_indices, :], positions
        )
    if discrete_indices:
        result[:, discrete_indices, :] = _nearest_warp(
            values[:, discrete_indices, :], positions
        )
    if any(f"{modality}.present" in channel_names for modality in selected):
        # Warped feature values must remain zero where the corresponding modality
        # is unobserved after nearest-neighbor presence interpolation.
        for modality in selected:
            presence_name = f"{modality}.present"
            if presence_name not in channel_names:
                continue
            presence = result[:, channel_names.index(presence_name), :]
            modality_features = [
                index
                for index, name in enumerate(channel_names)
                if name.startswith(f"{modality}.") and not name.endswith(".present")
            ]
            block = result[:, modality_features, :]
            # ``block * presence`` evaluates inf*0 before applying the mask. Avoid
            # that invalid intermediate and make every absent step exactly zero.
            masked = np.zeros_like(block)
            np.multiply(
                block,
                presence[:, None, :],
                out=masked,
                where=presence[:, None, :] != 0.0,
            )
            result[:, modality_features, :] = masked
    result = _collapse_near_constant_traces(result)
    if not np.all(np.isfinite(result)):
        sample_index, channel_index, time_index = np.argwhere(
            ~np.isfinite(result)
        )[0]
        raise ValueError(
            "Timing warp produced a non-finite value at "
            f"sample={sample_index}, channel={channel_names[channel_index]!r}, "
            f"time_index={time_index}"
        )
    return result


def warp_event_rows(
    source: Sequence[Mapping[str, Any]],
    modalities: Sequence[str],
    *,
    kind: str,
    magnitude: float,
    step_milliseconds: float,
) -> list[dict[str, Any]]:
    """Warp selected raw event timestamps relative to each word anchor."""

    selected = frozenset(modalities)
    unknown = selected - set(MODALITY_NAMES)
    if unknown:
        raise ValueError(f"Unknown modalities: {sorted(unknown)}")
    if kind not in {"shift", "scale"}:
        raise ValueError("kind must be 'shift' or 'scale'")
    rows: list[dict[str, Any]] = []
    for source_row in source:
        row = dict(source_row)
        word = source_row["word"]
        anchor = float(word["timeMs"])
        context = []
        for source_event in source_row["context"]:
            event = dict(source_event)
            if str(event.get("event_type")) in selected:
                timestamp_key = "timeMs" if "timeMs" in event else "timestamp"
                timestamp = float(event[timestamp_key])
                warped = (
                    timestamp + magnitude * step_milliseconds
                    if kind == "shift"
                    else anchor + magnitude * (timestamp - anchor)
                )
                if "timeMs" in event:
                    event["timeMs"] = warped
                if "timestamp" in event:
                    event["timestamp"] = warped
            context.append(event)
        row["context"] = context
        rows.append(row)
    return rows
