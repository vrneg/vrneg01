"""Train and evaluate multivariate MiniRocket on one saved fold."""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import torch
from sklearn.linear_model import RidgeClassifierCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
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


ARTIFACT_VERSION = 1


@dataclass(slots=True)
class TrainingResult:
    """Artifacts and metrics for one MiniRocket fold."""

    checkpoint_path: Path
    history_path: Path
    validation_predictions_path: Path
    predictions_path: Path
    run_name: str
    best_epoch: int
    decision_threshold: float
    validation_metrics: dict[str, float]
    test_metrics: dict[str, float]
    best_alpha: float
    num_transformed_features: int
    history: list[dict[str, Any]] = field(default_factory=list)
    # These compatibility fields allow use of the shared CV aggregation code.
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


def _load_minirocket_class() -> type:
    try:
        from aeon.transformations.collection.convolution_based import MiniRocket
    except ImportError as error:
        raise ImportError(
            "MiniRocket experiments require aeon. Install project requirements "
            "with `pip install -r requirements.txt`."
        ) from error
    return MiniRocket


def _build_pipeline(config: ExperimentConfig) -> Any:
    MiniRocket = _load_minirocket_class()
    return make_pipeline(
        MiniRocket(
            n_kernels=config.model.num_kernels,
            max_dilations_per_kernel=config.model.max_dilations_per_kernel,
            n_jobs=config.model.n_jobs,
            random_state=config.seed,
        ),
        StandardScaler(with_mean=False),
        RidgeClassifierCV(
            alphas=np.asarray(config.model.alphas, dtype=np.float64),
            class_weight=config.model.class_weight,
        ),
    )


def _decision_scores(pipeline: Any, split: TimeSeriesSplit) -> np.ndarray:
    scores = np.asarray(pipeline.decision_function(split.values), dtype=np.float64)
    if scores.shape != (split.labels.shape[0],):
        raise RuntimeError(
            f"Expected one decision score per sample, received shape {scores.shape}"
        )
    return scores


def _metrics(
    scores: np.ndarray,
    labels: np.ndarray,
    threshold: float,
) -> dict[str, float]:
    logits = torch.from_numpy(scores).float()
    targets = torch.from_numpy(labels).float()
    loss = float(functional.binary_cross_entropy_with_logits(logits, targets).item())
    return binary_classification_metrics(logits, targets, loss, threshold)


def _prediction_rows(
    split: TimeSeriesSplit,
    scores: np.ndarray,
    threshold: float,
) -> list[str]:
    probabilities = torch.sigmoid(torch.from_numpy(scores).float()).numpy()
    return [
        json.dumps(
            {
                "sample_id": sample_id,
                "label": int(label),
                "logit": float(score),
                "probability": float(probability),
                "threshold": threshold,
                "prediction": int(probability >= threshold),
            }
        )
        for sample_id, label, score, probability in zip(
            split.sample_ids,
            split.labels,
            scores,
            probabilities,
            strict=True,
        )
    ]


def _ridge_estimator(pipeline: Any) -> RidgeClassifierCV:
    estimator = pipeline.steps[-1][1]
    if not isinstance(estimator, RidgeClassifierCV):
        raise TypeError("MiniRocket pipeline does not end in RidgeClassifierCV")
    return estimator


def train_minirocket(config: ExperimentConfig) -> TrainingResult:
    """Fit MiniRocket on the training split and evaluate validation/test once."""

    config.validate()
    random.seed(config.seed)
    np.random.seed(config.seed)
    data: DataBundle = prepare_data(config.data)
    pipeline = _build_pipeline(config)
    pipeline.fit(data.train.values, data.train.labels)

    validation_scores = _decision_scores(pipeline, data.validation)
    test_scores = _decision_scores(pipeline, data.test)
    decision_threshold = (
        optimize_binary_threshold(
            torch.from_numpy(validation_scores).float(),
            torch.from_numpy(data.validation.labels).float(),
            metric_name=config.evaluation.threshold_metric,
        )
        if config.evaluation.calibrate_threshold_on_validation
        else config.evaluation.threshold
    )
    validation_metrics = _metrics(
        validation_scores, data.validation.labels, decision_threshold
    )
    test_metrics = _metrics(test_scores, data.test.labels, decision_threshold)

    ridge = _ridge_estimator(pipeline)
    best_alpha = float(ridge.alpha_)
    transformed_feature_count = int(
        pipeline.steps[0][1].transform(data.train.values[:1]).shape[1]
    )
    run_directory = config.output_dir / config.run_name
    run_directory.mkdir(parents=True, exist_ok=True)
    checkpoint_path = run_directory / "model.joblib"
    history_path = run_directory / "metrics.json"
    validation_predictions_path = run_directory / "validation_predictions.jsonl"
    predictions_path = run_directory / "test_predictions.jsonl"

    artifact = {
        "artifact_version": ARTIFACT_VERSION,
        "artifact_type": "multivariate_minirocket_classifier",
        "pipeline": pipeline,
        "experiment_config": _json_ready(asdict(config)),
        "normalizer": (
            None if data.normalizer is None else data.normalizer.state_dict()
        ),
        "time_grid": data.time_grid,
        "channel_names": data.channel_names,
        "best_alpha": best_alpha,
        "num_transformed_features": transformed_feature_count,
        "decision_threshold": decision_threshold,
        "validation_metrics": validation_metrics,
        "test_metrics": test_metrics,
    }
    joblib.dump(artifact, checkpoint_path)

    metrics_payload = {
        "best_alpha": best_alpha,
        "num_input_channels": len(data.channel_names),
        "num_time_points": config.data.num_time_points,
        "num_transformed_features": transformed_feature_count,
        "decision_threshold": decision_threshold,
        "decision_threshold_source": (
            "validation_calibration"
            if config.evaluation.calibrate_threshold_on_validation
            else "fixed"
        ),
        "threshold_metric": config.evaluation.threshold_metric,
        "validation": validation_metrics,
        "test": test_metrics,
    }
    history_path.write_text(
        json.dumps(_json_ready(metrics_payload), indent=2) + "\n",
        encoding="utf-8",
    )
    validation_predictions_path.write_text(
        "\n".join(
            _prediction_rows(
                data.validation, validation_scores, decision_threshold
            )
        )
        + "\n",
        encoding="utf-8",
    )
    predictions_path.write_text(
        "\n".join(_prediction_rows(data.test, test_scores, decision_threshold))
        + "\n",
        encoding="utf-8",
    )

    return TrainingResult(
        checkpoint_path=checkpoint_path,
        history_path=history_path,
        validation_predictions_path=validation_predictions_path,
        predictions_path=predictions_path,
        run_name=config.run_name,
        # MiniRocket and ridge fitting have no epoch-selection phase.
        best_epoch=0,
        decision_threshold=decision_threshold,
        validation_metrics=validation_metrics,
        test_metrics=test_metrics,
        best_alpha=best_alpha,
        num_transformed_features=transformed_feature_count,
        history=[],
    )


def load_trained_model(checkpoint_path: str | Path) -> dict[str, Any]:
    """Load a fitted MiniRocket pipeline and its preprocessing metadata."""

    artifact = joblib.load(checkpoint_path)
    version = artifact.get("artifact_version")
    if version != ARTIFACT_VERSION:
        raise ValueError(
            f"Unsupported MiniRocket artifact version {version!r}; "
            f"expected {ARTIFACT_VERSION}"
        )
    return artifact


def load_training_result(config: ExperimentConfig) -> TrainingResult:
    """Load a complete, configuration-matched MiniRocket run from disk."""

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
        raise FileNotFoundError(f"Incomplete MiniRocket run; missing: {missing}")

    artifact = load_trained_model(checkpoint_path)
    expected_config = _json_ready(asdict(config))
    if artifact.get("experiment_config") != expected_config:
        raise ValueError(
            f"Saved MiniRocket run {run_directory} was created with a different "
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
        best_alpha=float(artifact["best_alpha"]),
        num_transformed_features=int(artifact["num_transformed_features"]),
        history=[],
    )
