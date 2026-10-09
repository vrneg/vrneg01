"""Cheap, non-deep diagnostic: which channels (if any) carry a detectable negation signal.

Before spending more compute on sequence-model architecture, this checks two things per
channel: a linear (point-biserial correlation) relationship between a single frame's
value and the label, and a nonlinear one via 1-nearest-centroid classification under
dynamic time warping (DTW) distance to each class's mean trajectory. Both operate on the
same fixed-grid representation the trained models see, so a channel that shows nothing
here is a channel unlikely to help a bigger model either -- and a channel that does show
something explains what a bank of ROCKET kernels might be keying on.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def dtw_distance(series_a: np.ndarray, series_b: np.ndarray) -> float:
    """Classic O(T^2) dynamic-time-warping distance between two 1D series.

    Frame counts here are tiny (32), so the naive dynamic program is fast enough without
    an external dependency or a banding heuristic.
    """

    length_a, length_b = len(series_a), len(series_b)
    cost = np.full((length_a + 1, length_b + 1), np.inf)
    cost[0, 0] = 0.0
    for i in range(1, length_a + 1):
        for j in range(1, length_b + 1):
            distance = (series_a[i - 1] - series_b[j - 1]) ** 2
            cost[i, j] = distance + min(cost[i - 1, j], cost[i, j - 1], cost[i - 1, j - 1])
    return float(np.sqrt(cost[length_a, length_b]))


@dataclass(slots=True)
class ChannelScanResult:
    channel_name: str
    channel_index: int
    best_frame: int
    best_frame_ms: float
    point_biserial_r: float
    dtw_nearest_centroid_accuracy: float


def point_biserial_scan(
    values: np.ndarray, labels: np.ndarray, time_grid_ms: np.ndarray
) -> list[tuple[int, int, float]]:
    """Return ``(channel, frame, correlation)`` for the best frame of every channel.

    ``values`` has shape ``[windows, channels, frames]``; correlation is the ordinary
    Pearson correlation with the 0/1 label, which is exactly the point-biserial
    coefficient when one variable is binary.
    """

    num_windows, num_channels, num_frames = values.shape
    labels_centered = labels - labels.mean()
    label_norm = np.sqrt((labels_centered**2).sum())

    results = []
    for channel in range(num_channels):
        channel_values = values[:, channel, :]
        centered = channel_values - channel_values.mean(axis=0, keepdims=True)
        numerator = centered.T @ labels_centered
        denom = np.sqrt((centered**2).sum(axis=0)) * label_norm
        with np.errstate(invalid="ignore", divide="ignore"):
            correlations = np.where(denom > 1e-12, numerator / denom, 0.0)
        best_frame = int(np.argmax(np.abs(correlations)))
        results.append((channel, best_frame, float(correlations[best_frame])))
    return results


def dtw_nearest_centroid_accuracy(
    channel_series: np.ndarray, labels: np.ndarray, seed: int = 42
) -> float:
    """Leave-nothing-out nearest-centroid accuracy under DTW distance to class means.

    Class means are computed on the full set (not held out), so this is a description of
    class separability under DTW, not an unbiased accuracy estimate -- appropriate for a
    screening diagnostic, not for reporting as a model's performance.
    """

    positive_mean = channel_series[labels == 1].mean(axis=0)
    negative_mean = channel_series[labels == 0].mean(axis=0)
    correct = 0
    for series, label in zip(channel_series, labels, strict=True):
        distance_to_positive = dtw_distance(series, positive_mean)
        distance_to_negative = dtw_distance(series, negative_mean)
        predicted = 1 if distance_to_positive < distance_to_negative else 0
        correct += predicted == label
    return correct / len(labels)


def scan_channels(
    values: np.ndarray,
    labels: np.ndarray,
    channel_names: tuple[str, ...],
    time_grid_ms: np.ndarray,
    top_k_for_dtw: int = 20,
) -> list[ChannelScanResult]:
    """Correlation-scan every channel, then run the DTW check on the top-k by |r|."""

    correlation_results = point_biserial_scan(values, labels, time_grid_ms)
    ranked = sorted(correlation_results, key=lambda row: abs(row[2]), reverse=True)

    dtw_accuracy_by_channel: dict[int, float] = {}
    for channel, _frame, _corr in ranked[:top_k_for_dtw]:
        dtw_accuracy_by_channel[channel] = dtw_nearest_centroid_accuracy(
            values[:, channel, :], labels
        )

    return [
        ChannelScanResult(
            channel_name=channel_names[channel],
            channel_index=channel,
            best_frame=frame,
            best_frame_ms=float(time_grid_ms[frame]),
            point_biserial_r=correlation,
            dtw_nearest_centroid_accuracy=dtw_accuracy_by_channel.get(channel, float("nan")),
        )
        for channel, frame, correlation in ranked
    ]


def modality_summary(
    results: list[ChannelScanResult],
) -> dict[str, tuple[float, str, float]]:
    """Best |correlation| per modality, with the channel and its DTW accuracy."""

    best: dict[str, tuple[float, str, float]] = {}
    for result in results:
        modality = result.channel_name.split(".", 1)[0]
        magnitude = abs(result.point_biserial_r)
        if modality not in best or magnitude > best[modality][0]:
            best[modality] = (
                magnitude,
                f"{result.channel_name}@{result.best_frame_ms:+.0f}ms",
                result.dtw_nearest_centroid_accuracy,
            )
    return best


__all__ = [
    "ChannelScanResult",
    "dtw_distance",
    "dtw_nearest_centroid_accuracy",
    "modality_summary",
    "point_biserial_scan",
    "scan_channels",
]
