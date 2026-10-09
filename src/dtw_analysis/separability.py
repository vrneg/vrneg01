"""Nested-fold modality-wise DTW temporal separability analysis."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

try:
    from modality_attribution.metrics import binary_metrics
except ModuleNotFoundError as error:
    if error.name != "modality_attribution":
        raise
    from ..modality_attribution.metrics import binary_metrics

from .config import DTWAnalysisConfig
from .data import DTWFoldData, MODALITY_NAMES, Trajectory, observed
from .distance import (
    TrajectoryPreprocessor,
    distance_matrix,
    normalized_dtw_alignment,
    predict_inverse_distance_knn,
    probabilities_to_logits,
    select_dtw_hyperparameters,
)


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class FoldDTWResult:
    fold_rows: list[dict[str, Any]]
    prediction_rows: list[dict[str, Any]]
    presence_rows: list[dict[str, Any]]
    presence_prediction_rows: list[dict[str, Any]]
    alignment_examples: list[dict[str, Any]]


def _presence_probabilities(
    reference_presence: np.ndarray,
    reference_labels: np.ndarray,
    query_presence: np.ndarray,
) -> tuple[np.ndarray, dict[bool, float]]:
    fallback = float(reference_labels.mean())
    rates: dict[bool, float] = {}
    for state in (False, True):
        selected = reference_labels[reference_presence == state]
        rates[state] = float(selected.mean()) if selected.size else fallback
    return np.asarray([rates[bool(state)] for state in query_presence]), rates


def _alignment_examples(
    fold: DTWFoldData,
    modality: str,
    queries: list[Trajectory],
    query_indices: np.ndarray,
    references: list[Trajectory],
    reference_indices: list[tuple[str, int]],
    reference_labels: np.ndarray,
    probabilities: np.ndarray,
    predictions: np.ndarray,
    nearest_indices: np.ndarray,
    window: float,
) -> list[dict[str, Any]]:
    labels = fold.test.labels[query_indices]
    confidence = np.abs(probabilities - 0.5)
    categories = {
        "correct_negation": (predictions == labels) & (labels == 1),
        "correct_control": (predictions == labels) & (labels == 0),
        "high_confidence_error": predictions != labels,
    }
    rows: list[dict[str, Any]] = []
    for category, mask in categories.items():
        candidates = np.flatnonzero(mask)
        if not candidates.size:
            continue
        query_position = int(candidates[np.argmax(confidence[candidates])])
        reference_position = int(nearest_indices[query_position])
        query = queries[query_position]
        reference = references[reference_position]
        path, distance = normalized_dtw_alignment(query, reference, window)
        reference_split, source_index = reference_indices[reference_position]
        source_split = fold.train if reference_split == "train" else fold.validation
        rows.append(
            {
                "fold": fold.fold_name,
                "modality": modality,
                "category": category,
                "query_sample_id": fold.test.sample_ids[
                    int(query_indices[query_position])
                ],
                "query_label": int(labels[query_position]),
                "query_probability": float(probabilities[query_position]),
                "reference_sample_id": source_split.sample_ids[source_index],
                "reference_label": int(reference_labels[reference_position]),
                "normalized_distance": distance,
                "window": window,
                "query_length": int(query.values.shape[0]),
                "reference_length": int(reference.values.shape[0]),
                "path": [
                    {
                        "query_index": int(query.time_indices[query_step]),
                        "reference_index": int(reference.time_indices[reference_step]),
                        "query_time_seconds": float(
                            fold.time_grid[query.time_indices[query_step]]
                        ),
                        "reference_time_seconds": float(
                            fold.time_grid[reference.time_indices[reference_step]]
                        ),
                        "query_pca_values": query.values[query_step].tolist(),
                        "reference_pca_values": reference.values[
                            reference_step
                        ].tolist(),
                    }
                    for query_step, reference_step in path
                ],
            }
        )
    return rows


def analyze_dtw_fold(
    fold: DTWFoldData,
    config: DTWAnalysisConfig,
) -> FoldDTWResult:
    fold_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    presence_rows: list[dict[str, Any]] = []
    presence_prediction_rows: list[dict[str, Any]] = []
    alignment_examples: list[dict[str, Any]] = []

    for modality_index, modality in enumerate(MODALITY_NAMES):
        modality_started = time.perf_counter()
        LOGGER.info(
            "[%s] DTW modality %d/%d: %s",
            fold.fold_name,
            modality_index + 1,
            len(MODALITY_NAMES),
            modality,
        )
        train_trajectories, train_indices = observed(fold.train, modality)
        validation_trajectories, validation_indices = observed(
            fold.validation, modality
        )
        test_trajectories, test_indices = observed(fold.test, modality)
        train_labels = fold.train.labels[train_indices]
        validation_labels = fold.validation.labels[validation_indices]
        test_labels = fold.test.labels[test_indices]

        reference_presence = np.concatenate(
            (fold.train.presence[modality], fold.validation.presence[modality])
        )
        reference_presence_labels = np.concatenate(
            (fold.train.labels, fold.validation.labels)
        )
        presence_probabilities, rates = _presence_probabilities(
            reference_presence,
            reference_presence_labels,
            fold.test.presence[modality],
        )
        presence_metrics = binary_metrics(
            fold.test.labels, probabilities_to_logits(presence_probabilities), 0.5
        )
        presence_rows.append(
            {
                "fold": fold.fold_name,
                "modality": modality,
                "test_coverage": float(fold.test.presence[modality].mean()),
                "train_validation_observed_positive_rate": rates[True],
                "train_validation_absent_positive_rate": rates[False],
                **presence_metrics,
            }
        )
        for sample_index, probability in enumerate(presence_probabilities):
            presence_prediction_rows.append(
                {
                    "fold": fold.fold_name,
                    "sample_index": sample_index,
                    "sample_key": f"{fold.fold_name}:{sample_index}",
                    "sample_id": fold.test.sample_ids[sample_index],
                    "group_id": fold.test.group_ids[sample_index],
                    "modality": modality,
                    "label": int(fold.test.labels[sample_index]),
                    "observed": bool(fold.test.presence[modality][sample_index]),
                    "probability": float(probability),
                    "prediction": int(probability >= 0.5),
                }
            )

        unavailable_reason = None
        if set(np.unique(train_labels)) != {0, 1}:
            unavailable_reason = (
                "observed training references do not contain both classes"
            )
        elif not validation_trajectories:
            unavailable_reason = "validation split contains no usable trajectories"
        elif not test_trajectories:
            unavailable_reason = "test split contains no usable trajectories"
        if unavailable_reason is not None:
            fold_rows.append(
                {
                    "fold": fold.fold_name,
                    "modality": modality,
                    "status": "unavailable",
                    "reason": unavailable_reason,
                    "train_observed": len(train_trajectories),
                    "validation_observed": len(validation_trajectories),
                    "test_observed": len(test_trajectories),
                    "test_coverage": float(fold.test.presence[modality].mean()),
                }
            )
            LOGGER.warning(
                "[%s] Skipping %s after %.1fs: %s",
                fold.fold_name,
                modality,
                time.perf_counter() - modality_started,
                unavailable_reason,
            )
            continue

        LOGGER.info(
            "[%s] %s observations: train=%d, validation=%d, test=%d; selecting "
            "window/k",
            fold.fold_name,
            modality,
            len(train_trajectories),
            len(validation_trajectories),
            len(test_trajectories),
        )
        selected = select_dtw_hyperparameters(
            train_trajectories,
            train_labels,
            validation_trajectories,
            validation_labels,
            windows=config.dtw_windows,
            neighbor_counts=config.neighbor_counts,
            pca_variance=config.pca_variance,
            pca_max_components=config.pca_max_components,
            n_jobs=config.n_jobs,
        )
        LOGGER.info(
            "[%s] %s selected window=%.3f, k=%d (validation macro-F1=%.4f); "
            "computing outer-test distances",
            fold.fold_name,
            modality,
            selected.window,
            selected.k,
            selected.validation_macro_f1,
        )
        reference_trajectories = train_trajectories + validation_trajectories
        reference_labels = np.concatenate((train_labels, validation_labels))
        reference_indices = [
            ("train", int(index)) for index in train_indices
        ] + [("validation", int(index)) for index in validation_indices]
        preprocessor = TrajectoryPreprocessor(
            config.pca_variance, config.pca_max_components
        ).fit(reference_trajectories)
        transformed_references = preprocessor.transform(reference_trajectories)
        transformed_test = preprocessor.transform(test_trajectories)
        distances = distance_matrix(
            transformed_test,
            transformed_references,
            window=selected.window,
            n_jobs=config.n_jobs,
        )
        output = predict_inverse_distance_knn(
            distances, reference_labels, selected.k
        )
        metrics = binary_metrics(
            test_labels, probabilities_to_logits(output.probabilities), 0.5
        )
        same_class = np.where(
            test_labels == 1,
            output.nearest_positive_distances,
            output.nearest_negative_distances,
        )
        different_class = np.where(
            test_labels == 1,
            output.nearest_negative_distances,
            output.nearest_positive_distances,
        )
        fold_rows.append(
            {
                "fold": fold.fold_name,
                "modality": modality,
                "status": "complete",
                "train_observed": len(train_trajectories),
                "validation_observed": len(validation_trajectories),
                "test_observed": len(test_trajectories),
                "test_coverage": float(fold.test.presence[modality].mean()),
                "selected_window": selected.window,
                "selected_k": selected.k,
                "validation_macro_f1": selected.validation_macro_f1,
                "selection_pca_components": selected.num_pca_components,
                "selection_explained_variance": selected.explained_variance,
                "final_pca_components": preprocessor.num_components_,
                "final_explained_variance": preprocessor.explained_variance_,
                "mean_nearest_same_class_distance": float(same_class.mean()),
                "mean_nearest_different_class_distance": float(different_class.mean()),
                **metrics,
            }
        )
        for local_index, sample_index in enumerate(test_indices):
            nearest_index = int(output.nearest_indices[local_index])
            nearest_split, nearest_source_index = reference_indices[nearest_index]
            source_split = fold.train if nearest_split == "train" else fold.validation
            prediction_rows.append(
                {
                    "fold": fold.fold_name,
                    "sample_index": int(sample_index),
                    "sample_key": f"{fold.fold_name}:{int(sample_index)}",
                    "sample_id": fold.test.sample_ids[int(sample_index)],
                    "group_id": fold.test.group_ids[int(sample_index)],
                    "modality": modality,
                    "label": int(test_labels[local_index]),
                    "probability": float(output.probabilities[local_index]),
                    "prediction": int(output.predictions[local_index]),
                    "nearest_negative_distance": float(
                        output.nearest_negative_distances[local_index]
                    ),
                    "nearest_positive_distance": float(
                        output.nearest_positive_distances[local_index]
                    ),
                    "distance_margin": float(
                        output.nearest_negative_distances[local_index]
                        - output.nearest_positive_distances[local_index]
                    ),
                    "nearest_same_class_distance": float(same_class[local_index]),
                    "nearest_different_class_distance": float(
                        different_class[local_index]
                    ),
                    "nearest_reference_sample_id": source_split.sample_ids[
                        nearest_source_index
                    ],
                    "nearest_reference_label": int(reference_labels[nearest_index]),
                    "selected_window": selected.window,
                    "selected_k": selected.k,
                }
            )
        alignment_examples.extend(
            _alignment_examples(
                fold,
                modality,
                transformed_test,
                test_indices,
                transformed_references,
                reference_indices,
                reference_labels,
                output.probabilities,
                output.predictions,
                output.nearest_indices,
                selected.window,
            )
        )
        LOGGER.info(
            "[%s] Completed %s in %.1fs: test macro-F1=%.4f, coverage=%.1f%%",
            fold.fold_name,
            modality,
            time.perf_counter() - modality_started,
            metrics["macro_f1"],
            fold.test.presence[modality].mean() * 100.0,
        )

    return FoldDTWResult(
        fold_rows,
        prediction_rows,
        presence_rows,
        presence_prediction_rows,
        alignment_examples,
    )
