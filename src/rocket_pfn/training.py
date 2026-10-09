"""Train-free ROCKET feature extraction followed by in-context TabPFN inference."""

from __future__ import annotations

import json
import random
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterator

import joblib
import numpy as np
import torch
from torch.nn import functional as functional

try:
    from event_transformer.metrics import (
        binary_classification_metrics,
        optimize_binary_threshold,
    )
except ModuleNotFoundError as error:
    if error.name != "event_transformer":
        raise
    from ..event_transformer.metrics import (
        binary_classification_metrics,
        optimize_binary_threshold,
    )

from .config import ExperimentConfig
from .data import DataBundle, TimeSeriesSplit, prepare_data
from .features import (
    FittedRocketPFN,
    average_tabpfn_probabilities,
    ensure_tabpfn_checkpoint_access,
    load_reusable_multirocket_hydra_transformer,
    new_multirocket_hydra_transformer,
    new_rocket_transformer,
    ranked_feature_groups,
)


ARTIFACT_VERSION = 1
_PROBABILITY_EPSILON = 1e-7


@dataclass(slots=True)
class TrainingResult:
    """Artifacts, metrics, and representation diagnostics for one outer fold."""

    checkpoint_path: Path
    history_path: Path
    validation_predictions_path: Path
    predictions_path: Path
    run_name: str
    best_epoch: int
    decision_threshold: float
    validation_metrics: dict[str, float]
    test_metrics: dict[str, float]
    feature_representation: str
    feature_source: str
    tabpfn_version: str
    num_feature_groups: int
    feature_group_sizes: list[int]
    num_transformed_features: int
    num_selected_features: int
    history: list[dict[str, Any]] = field(default_factory=list)
    # Compatibility fields for the shared cross-validation aggregator.
    pretrained_checkpoint_path: Path | None = None
    pretraining_history: list[dict[str, Any]] = field(default_factory=list)


def _json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value


def _log(config: ExperimentConfig, message: str) -> None:
    print(f"[{config.run_name}] {message}", flush=True)


def _probabilities_to_logits(probabilities: np.ndarray) -> np.ndarray:
    clipped = np.clip(
        np.asarray(probabilities, dtype=np.float64),
        _PROBABILITY_EPSILON,
        1.0 - _PROBABILITY_EPSILON,
    )
    return np.log(clipped) - np.log1p(-clipped)


def _metrics(
    logits: np.ndarray,
    labels: np.ndarray,
    threshold: float,
) -> dict[str, float]:
    scores = torch.from_numpy(np.asarray(logits)).float()
    targets = torch.from_numpy(np.asarray(labels)).float()
    loss = float(functional.binary_cross_entropy_with_logits(scores, targets).item())
    return binary_classification_metrics(scores, targets, loss, threshold)


def _prediction_rows(
    split: TimeSeriesSplit,
    logits: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
) -> list[str]:
    return [
        json.dumps(
            {
                "sample_id": sample_id,
                "label": int(label),
                "logit": float(logit),
                "probability": float(probability),
                "threshold": threshold,
                "prediction": int(probability >= threshold),
            }
        )
        for sample_id, label, logit, probability in zip(
            split.sample_ids,
            split.labels,
            logits,
            probabilities,
            strict=True,
        )
    ]


def _rocket_feature_pairs(
    config: ExperimentConfig,
    data: DataBundle,
    query_values: np.ndarray,
    fitted_transformers: list[Any],
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    model = config.model
    for group_index in range(model.rocket_num_groups):
        started = time.monotonic()
        transformer = new_rocket_transformer(model, config.seed + group_index)
        train_features = np.asarray(
            transformer.fit_transform(data.train.values, data.train.labels)
        )
        query_features = np.asarray(transformer.transform(query_values))
        fitted_transformers.append(transformer)
        _log(
            config,
            f"ROCKET group {group_index + 1}/{model.rocket_num_groups}: "
            f"{train_features.shape[1]:,} features generated in "
            f"{time.monotonic() - started:.1f}s; running TabPFN ...",
        )
        yield train_features, query_features


def _multirocket_hydra_feature_pairs(
    train_features: np.ndarray,
    query_features: np.ndarray,
    feature_indices: tuple[np.ndarray, ...],
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    for indices in feature_indices:
        yield train_features[:, indices], query_features[:, indices]


def train_rocket_pfn(
    config: ExperimentConfig,
    *,
    prepared_data: DataBundle | None = None,
) -> TrainingResult:
    """Run one outer fold without exposing validation/test labels to the model."""

    config.validate()
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    total_start = time.monotonic()

    _log(config, "Checking TabPFN checkpoint access ...")
    ensure_tabpfn_checkpoint_access(config.model, config.seed)
    _log(
        config,
        (
            f"Loading dataset from {config.data.dataset_path} ..."
            if prepared_data is None
            else "Reusing prepared full-fold dataset with variant masks ..."
        ),
    )
    data_start = time.monotonic()
    data: DataBundle = (
        prepare_data(config.data) if prepared_data is None else prepared_data
    )
    train_classes = np.unique(data.train.labels)
    if not np.array_equal(train_classes, np.asarray([0, 1])):
        raise ValueError(
            "RocketPFN requires binary training labels encoded as 0 and 1; "
            f"received {train_classes.tolist()}"
        )
    _log(
        config,
        f"Loaded train={len(data.train.labels)}, "
        f"validation={len(data.validation.labels)}, test={len(data.test.labels)} "
        f"in {time.monotonic() - data_start:.1f}s.",
    )

    # TabPFN recomputes the training context for every predict call. Combining both
    # untouched evaluation splits makes one batched call per group substantially faster.
    query_values = np.concatenate(
        (data.validation.values, data.test.values), axis=0
    )
    fitted_transformers: list[Any] = []
    feature_indices: tuple[np.ndarray, ...] = ()
    feature_group_sizes: list[int]
    transformed_feature_count: int
    selected_feature_count: int
    feature_source: str

    if config.model.feature_representation == "rocket":
        feature_source = "fresh_independent_rocket_groups"
        expected_group_size = 2 * config.model.rocket_kernels_per_group
        feature_group_sizes = [
            expected_group_size for _ in range(config.model.rocket_num_groups)
        ]
        transformed_feature_count = sum(feature_group_sizes)
        selected_feature_count = transformed_feature_count
        feature_pairs = _rocket_feature_pairs(
            config, data, query_values, fitted_transformers
        )
        expected_groups = config.model.rocket_num_groups
    else:
        transform_start = time.monotonic()
        transformer = load_reusable_multirocket_hydra_transformer(config)
        if transformer is None:
            feature_source = "fresh_multirocket_hydra"
            _log(config, "Fitting fresh MultiRocket+HYDRA feature transforms ...")
            transformer = new_multirocket_hydra_transformer(
                config.model, config.seed
            )
            train_features = np.asarray(
                transformer.fit_transform(data.train.values, data.train.labels)
            )
        else:
            feature_source = str(config.feature_artifact_path)
            _log(
                config,
                "Reusing fold-matched transforms from "
                f"{config.feature_artifact_path} ...",
            )
            train_features = np.asarray(transformer.transform(data.train.values))
        query_features = np.asarray(transformer.transform(query_values))
        fitted_transformers.append(transformer)
        transformed_feature_count = int(train_features.shape[1])
        feature_indices = ranked_feature_groups(
            train_features,
            data.train.labels,
            config.model.reduced_feature_count,
            config.model.max_features_per_group,
        )
        feature_group_sizes = [int(indices.size) for indices in feature_indices]
        selected_feature_count = sum(feature_group_sizes)
        expected_groups = len(feature_indices)
        _log(
            config,
            f"Generated {transformed_feature_count:,} MultiRocket+HYDRA features "
            f"and selected {selected_feature_count:,} into {expected_groups} groups "
            f"in {time.monotonic() - transform_start:.1f}s.",
        )
        feature_pairs = _multirocket_hydra_feature_pairs(
            train_features, query_features, feature_indices
        )

    inference_start = time.monotonic()
    _log(
        config,
        "TabPFN group execution: "
        + (
            "batched (set TABPFN_BATCH_GROUPS=False for the original path)."
            if config.model.tabpfn_batch_groups
            else "sequential."
        ),
    )

    def report_group(group_number: int, width: int) -> None:
        _log(
            config,
            f"Completed TabPFN group {group_number}/{expected_groups} "
            f"({width:,} features).",
        )

    query_probabilities, completed_groups = average_tabpfn_probabilities(
        feature_pairs,
        data.train.labels,
        config.model,
        config.seed,
        progress_callback=report_group,
    )
    if completed_groups != expected_groups:
        raise RuntimeError(
            f"Expected {expected_groups} feature groups, completed {completed_groups}"
        )
    _log(
        config,
        f"TabPFN probability ensemble finished in "
        f"{time.monotonic() - inference_start:.1f}s.",
    )

    validation_count = len(data.validation.labels)
    validation_probabilities = query_probabilities[:validation_count]
    test_probabilities = query_probabilities[validation_count:]
    validation_logits = _probabilities_to_logits(validation_probabilities)
    test_logits = _probabilities_to_logits(test_probabilities)
    decision_threshold = (
        optimize_binary_threshold(
            torch.from_numpy(validation_logits).float(),
            torch.from_numpy(data.validation.labels).float(),
            metric_name=config.evaluation.threshold_metric,
        )
        if config.evaluation.calibrate_threshold_on_validation
        else config.evaluation.threshold
    )
    validation_metrics = _metrics(
        validation_logits, data.validation.labels, decision_threshold
    )
    test_metrics = _metrics(test_logits, data.test.labels, decision_threshold)

    fitted_model = FittedRocketPFN(
        model_config=config.model,
        seed=config.seed,
        train_values=np.asarray(data.train.values),
        train_labels=np.asarray(data.train.labels),
        feature_transformers=tuple(fitted_transformers),
        feature_indices=feature_indices,
    )
    run_directory = config.output_dir / config.run_name
    run_directory.mkdir(parents=True, exist_ok=True)
    checkpoint_path = run_directory / "model.joblib"
    history_path = run_directory / "metrics.json"
    validation_predictions_path = run_directory / "validation_predictions.jsonl"
    predictions_path = run_directory / "test_predictions.jsonl"

    artifact_type = (
        "rocket_pfn_classifier"
        if config.model.feature_representation == "rocket"
        else "multirocket_hydra_pfn_classifier"
    )
    artifact = {
        "artifact_version": ARTIFACT_VERSION,
        "artifact_type": artifact_type,
        "model": fitted_model,
        "experiment_config": _json_ready(asdict(config)),
        "normalizer": (
            None if data.normalizer is None else data.normalizer.state_dict()
        ),
        "time_grid": data.time_grid,
        "channel_names": data.channel_names,
        "feature_representation": config.model.feature_representation,
        "feature_source": feature_source,
        "tabpfn_version": config.model.tabpfn_version,
        "tabpfn_batch_groups": config.model.tabpfn_batch_groups,
        "num_feature_groups": completed_groups,
        "feature_group_sizes": feature_group_sizes,
        "num_transformed_features": transformed_feature_count,
        "num_selected_features": selected_feature_count,
        "decision_threshold": decision_threshold,
        "validation_metrics": validation_metrics,
        "test_metrics": test_metrics,
    }
    # TabPFN's large pretrained network is intentionally absent from fitted_model;
    # compress only the random transforms and small training context stored per fold.
    joblib.dump(artifact, checkpoint_path, compress=3)

    metrics_payload = {
        "feature_representation": config.model.feature_representation,
        "feature_source": feature_source,
        "tabpfn_version": config.model.tabpfn_version,
        "tabpfn_n_estimators": config.model.tabpfn_n_estimators,
        "tabpfn_batch_groups": config.model.tabpfn_batch_groups,
        "num_input_channels": len(data.channel_names),
        "num_time_points": config.data.num_time_points,
        "num_feature_groups": completed_groups,
        "feature_group_sizes": feature_group_sizes,
        "num_transformed_features": transformed_feature_count,
        "num_selected_features": selected_feature_count,
        "decision_threshold": decision_threshold,
        "decision_threshold_source": (
            "validation_calibration"
            if config.evaluation.calibrate_threshold_on_validation
            else "fixed"
        ),
        "threshold_metric": config.evaluation.threshold_metric,
        "validation": validation_metrics,
        "test": test_metrics,
        "elapsed_seconds": time.monotonic() - total_start,
    }
    history_path.write_text(
        json.dumps(_json_ready(metrics_payload), indent=2) + "\n",
        encoding="utf-8",
    )
    validation_predictions_path.write_text(
        "\n".join(
            _prediction_rows(
                data.validation,
                validation_logits,
                validation_probabilities,
                decision_threshold,
            )
        )
        + "\n",
        encoding="utf-8",
    )
    predictions_path.write_text(
        "\n".join(
            _prediction_rows(
                data.test,
                test_logits,
                test_probabilities,
                decision_threshold,
            )
        )
        + "\n",
        encoding="utf-8",
    )
    _log(
        config,
        f"Finished fold in {time.monotonic() - total_start:.1f}s; "
        f"test_balanced_accuracy={test_metrics['balanced_accuracy']:.4f}. "
        f"Artifacts: {run_directory}",
    )

    return TrainingResult(
        checkpoint_path=checkpoint_path,
        history_path=history_path,
        validation_predictions_path=validation_predictions_path,
        predictions_path=predictions_path,
        run_name=config.run_name,
        best_epoch=0,
        decision_threshold=decision_threshold,
        validation_metrics=validation_metrics,
        test_metrics=test_metrics,
        feature_representation=config.model.feature_representation,
        feature_source=feature_source,
        tabpfn_version=config.model.tabpfn_version,
        num_feature_groups=completed_groups,
        feature_group_sizes=feature_group_sizes,
        num_transformed_features=transformed_feature_count,
        num_selected_features=selected_feature_count,
        history=[],
    )


def load_trained_model(checkpoint_path: str | Path) -> dict[str, Any]:
    """Load a fitted RocketPFN artifact without loading TabPFN weights."""

    artifact = joblib.load(checkpoint_path)
    version = artifact.get("artifact_version")
    if version != ARTIFACT_VERSION:
        raise ValueError(
            f"Unsupported RocketPFN artifact version {version!r}; "
            f"expected {ARTIFACT_VERSION}"
        )
    return artifact


def load_training_result(config: ExperimentConfig) -> TrainingResult:
    """Load a complete, configuration-matched RocketPFN run from disk."""

    config.validate()
    run_directory = config.output_dir / config.run_name
    checkpoint_path = run_directory / "model.joblib"
    history_path = run_directory / "metrics.json"
    validation_predictions_path = run_directory / "validation_predictions.jsonl"
    predictions_path = run_directory / "test_predictions.jsonl"
    required_paths = (
        checkpoint_path,
        history_path,
        validation_predictions_path,
        predictions_path,
    )
    missing_paths = [path for path in required_paths if not path.is_file()]
    if missing_paths:
        missing = ", ".join(str(path) for path in missing_paths)
        raise FileNotFoundError(f"Incomplete RocketPFN run; missing: {missing}")

    artifact = load_trained_model(checkpoint_path)
    if artifact.get("experiment_config") != _json_ready(asdict(config)):
        raise ValueError(
            f"Saved RocketPFN run {run_directory} was created with a different "
            "experiment configuration"
        )
    return TrainingResult(
        checkpoint_path=checkpoint_path,
        history_path=history_path,
        validation_predictions_path=validation_predictions_path,
        predictions_path=predictions_path,
        run_name=config.run_name,
        best_epoch=0,
        decision_threshold=float(artifact["decision_threshold"]),
        validation_metrics=dict(artifact["validation_metrics"]),
        test_metrics=dict(artifact["test_metrics"]),
        feature_representation=str(artifact["feature_representation"]),
        feature_source=str(artifact["feature_source"]),
        tabpfn_version=str(artifact["tabpfn_version"]),
        num_feature_groups=int(artifact["num_feature_groups"]),
        feature_group_sizes=[int(value) for value in artifact["feature_group_sizes"]],
        num_transformed_features=int(artifact["num_transformed_features"]),
        num_selected_features=int(artifact["num_selected_features"]),
        history=[],
    )


__all__ = [
    "ARTIFACT_VERSION",
    "TrainingResult",
    "load_trained_model",
    "load_training_result",
    "train_rocket_pfn",
]
