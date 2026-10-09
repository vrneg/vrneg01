"""Train-only trajectory preprocessing and normalized Aeon DTW k-NN."""

from __future__ import annotations

import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np
from aeon.distances import dtw_alignment_path
from sklearn.decomposition import PCA
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler

from .data import Trajectory


LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class TrajectoryPreprocessor:
    variance: float
    max_components: int
    scaler: StandardScaler | None = None
    pca: PCA | None = None
    num_components_: int = 0
    explained_variance_: float = 0.0

    def fit(self, trajectories: list[Trajectory]) -> "TrajectoryPreprocessor":
        if not trajectories:
            raise ValueError("Cannot fit trajectory preprocessing without observations")
        rows = np.concatenate(
            [trajectory.values for trajectory in trajectories], axis=0
        )
        if rows.shape[0] < 2:
            raise ValueError("PCA requires at least two observed time points")
        self.scaler = StandardScaler().fit(rows)
        scaled = self.scaler.transform(rows)
        maximum = min(self.max_components, scaled.shape[0], scaled.shape[1])
        candidate = PCA(n_components=maximum, svd_solver="full").fit(scaled)
        cumulative = np.cumsum(np.nan_to_num(candidate.explained_variance_ratio_))
        selected = int(np.searchsorted(cumulative, self.variance, side="left") + 1)
        selected = min(maximum, max(1, selected))
        self.pca = PCA(n_components=selected, svd_solver="full").fit(scaled)
        self.num_components_ = selected
        self.explained_variance_ = float(
            np.nan_to_num(self.pca.explained_variance_ratio_).sum()
        )
        return self

    def transform(self, trajectories: list[Trajectory]) -> list[Trajectory]:
        if self.scaler is None or self.pca is None:
            raise RuntimeError("TrajectoryPreprocessor must be fitted before transform")
        return [
            Trajectory(
                self.pca.transform(self.scaler.transform(trajectory.values)),
                trajectory.time_indices,
            )
            for trajectory in trajectories
        ]


def normalized_dtw_alignment(
    query: Trajectory,
    reference: Trajectory,
    window: float,
) -> tuple[list[tuple[int, int]], float]:
    """Return path and RMS accumulated Aeon DTW cost per alignment step."""

    path, accumulated = dtw_alignment_path(
        np.ascontiguousarray(query.values.T),
        np.ascontiguousarray(reference.values.T),
        window=window,
    )
    if not path:
        raise RuntimeError("Aeon returned an empty DTW alignment path")
    return path, float(np.sqrt(max(0.0, accumulated) / len(path)))


def distance_matrix(
    queries: list[Trajectory],
    references: list[Trajectory],
    *,
    window: float,
    n_jobs: int,
) -> np.ndarray:
    if not queries or not references:
        raise ValueError("DTW distance matrices require queries and references")

    def row(query: Trajectory) -> np.ndarray:
        return np.asarray(
            [
                normalized_dtw_alignment(query, reference, window)[1]
                for reference in references
            ],
            dtype=np.float64,
        )

    workers = (os.cpu_count() or 1) if n_jobs < 0 else n_jobs
    if workers == 1:
        rows = [row(query) for query in queries]
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            rows = list(executor.map(row, queries))
    return np.stack(rows)


@dataclass(frozen=True, slots=True)
class KNNOutput:
    probabilities: np.ndarray
    predictions: np.ndarray
    nearest_indices: np.ndarray
    nearest_negative_distances: np.ndarray
    nearest_positive_distances: np.ndarray


def predict_inverse_distance_knn(
    distances: np.ndarray,
    reference_labels: np.ndarray,
    k: int,
) -> KNNOutput:
    labels = np.asarray(reference_labels, dtype=np.int64)
    if distances.ndim != 2 or distances.shape[1] != labels.shape[0]:
        raise ValueError("Distance matrix and reference labels are incompatible")
    if set(np.unique(labels)) != {0, 1}:
        raise ValueError("Observed DTW references must contain both classes")
    effective_k = min(k, labels.size)
    order = np.argsort(distances, axis=1, kind="stable")
    neighbors = order[:, :effective_k]
    probabilities = np.empty(distances.shape[0], dtype=np.float64)
    for row_index, indices in enumerate(neighbors):
        values = distances[row_index, indices]
        zero = values <= 1e-12
        if np.any(zero):
            probabilities[row_index] = float(labels[indices[zero]].mean())
        else:
            weights = 1.0 / values
            probabilities[row_index] = float(
                np.dot(weights, labels[indices]) / weights.sum()
            )
    negative = np.min(distances[:, labels == 0], axis=1)
    positive = np.min(distances[:, labels == 1], axis=1)
    return KNNOutput(
        probabilities=probabilities,
        predictions=(probabilities >= 0.5).astype(np.int64),
        nearest_indices=order[:, 0],
        nearest_negative_distances=negative,
        nearest_positive_distances=positive,
    )


@dataclass(frozen=True, slots=True)
class SelectedDTW:
    window: float
    k: int
    validation_macro_f1: float
    num_pca_components: int
    explained_variance: float


def select_dtw_hyperparameters(
    train: list[Trajectory],
    train_labels: np.ndarray,
    validation: list[Trajectory],
    validation_labels: np.ndarray,
    *,
    windows: tuple[float, ...],
    neighbor_counts: tuple[int, ...],
    pca_variance: float,
    pca_max_components: int,
    n_jobs: int,
) -> SelectedDTW:
    preprocessor = TrajectoryPreprocessor(
        pca_variance, pca_max_components
    ).fit(train)
    transformed_train = preprocessor.transform(train)
    transformed_validation = preprocessor.transform(validation)
    best: tuple[float, float, int] | None = None
    for window in windows:
        started = time.perf_counter()
        LOGGER.info(
            "Computing validation DTW matrix for window=%.3f (%d queries × %d "
            "references)",
            window,
            len(transformed_validation),
            len(transformed_train),
        )
        distances = distance_matrix(
            transformed_validation,
            transformed_train,
            window=window,
            n_jobs=n_jobs,
        )
        LOGGER.info(
            "Validation DTW matrix for window=%.3f completed in %.1fs",
            window,
            time.perf_counter() - started,
        )
        for k in neighbor_counts:
            output = predict_inverse_distance_knn(distances, train_labels, k)
            score = float(
                f1_score(
                    validation_labels,
                    output.predictions,
                    average="macro",
                    labels=[0, 1],
                    zero_division=0,
                )
            )
            candidate = (-score, window, k)
            if best is None or candidate < best:
                best = candidate
    assert best is not None
    return SelectedDTW(
        window=best[1],
        k=best[2],
        validation_macro_f1=-best[0],
        num_pca_components=preprocessor.num_components_,
        explained_variance=preprocessor.explained_variance_,
    )


def probabilities_to_logits(probabilities: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(probabilities, dtype=np.float64), 1e-7, 1.0 - 1e-7)
    return np.log(clipped) - np.log1p(-clipped)
