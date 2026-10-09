"""Train and evaluate multivariate MrSQM on one saved fold."""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

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
from .model import MultivariateMrSQMClassifier, build_classifier


ARTIFACT_VERSION = 1


@dataclass(slots=True)
class TrainingResult:
    """Artifacts and metrics for one MrSQM fold."""

    checkpoint_path: Path
    history_path: Path
    validation_predictions_path: Path
    predictions_path: Path
    run_name: str
    best_epoch: int
    decision_threshold: float
    validation_metrics: dict[str, float]
    test_metrics: dict[str, float]
    logistic_c: float
    num_symbolic_features: int
    num_selected_channels: int
    selected_channel_names: list[str]
    history: list[dict[str, Any]] = field(default_factory=list)
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


def _decision_scores(
    classifier: MultivariateMrSQMClassifier,
    split: TimeSeriesSplit,
) -> np.ndarray:
    scores = np.asarray(classifier.decision_function(split.values), dtype=np.float64)
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


def train_mrsqm(config: ExperimentConfig) -> TrainingResult:
    """Fit MrSQM and evaluate validation/test exactly once."""

    config.validate()
    random.seed(config.seed)
    np.random.seed(config.seed)
    data: DataBundle = prepare_data(config.data)
    classifier = build_classifier(config)
    classifier.fit(data.train.values, data.train.labels)

    validation_scores = _decision_scores(classifier, data.validation)
    test_scores = _decision_scores(classifier, data.test)
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

    selected_indices = (
        classifier.channel_selector_.selected_channel_indices_.tolist()
    )
    selected_channel_names = [data.channel_names[index] for index in selected_indices]
    num_symbolic_features = int(classifier.n_symbolic_features_)

    run_directory = config.output_dir / config.run_name
    run_directory.mkdir(parents=True, exist_ok=True)
    checkpoint_path = run_directory / "model.joblib"
    history_path = run_directory / "metrics.json"
    validation_predictions_path = run_directory / "validation_predictions.jsonl"
    predictions_path = run_directory / "test_predictions.jsonl"
    artifact = {
        "artifact_version": ARTIFACT_VERSION,
        "artifact_type": "multivariate_mrsqm_classifier",
        "classifier": classifier,
        "experiment_config": _json_ready(asdict(config)),
        "normalizer": (
            None if data.normalizer is None else data.normalizer.state_dict()
        ),
        "time_grid": data.time_grid,
        "channel_names": data.channel_names,
        "selected_channel_indices": selected_indices,
        "selected_channel_names": selected_channel_names,
        "channel_scores": classifier.channel_selector_.channel_scores_,
        "logistic_c": config.model.logistic_c,
        "num_symbolic_features": num_symbolic_features,
        "decision_threshold": decision_threshold,
        "validation_metrics": validation_metrics,
        "test_metrics": test_metrics,
    }
    joblib.dump(artifact, checkpoint_path, compress=3)

    metrics_payload = {
        "strategy": config.model.strategy,
        "logistic_c": config.model.logistic_c,
        "num_input_channels": len(data.channel_names),
        "num_selected_channels": len(selected_indices),
        "selected_channel_names": selected_channel_names,
        "num_time_points": config.data.num_time_points,
        "num_symbolic_features": num_symbolic_features,
        "num_sax_representations": config.model.num_sax_representations,
        "num_sfa_representations": config.model.num_sfa_representations,
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
            _prediction_rows(data.validation, validation_scores, decision_threshold)
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
        best_epoch=0,
        decision_threshold=decision_threshold,
        validation_metrics=validation_metrics,
        test_metrics=test_metrics,
        logistic_c=config.model.logistic_c,
        num_symbolic_features=num_symbolic_features,
        num_selected_channels=len(selected_indices),
        selected_channel_names=selected_channel_names,
        history=[],
    )


def load_trained_model(checkpoint_path: str | Path) -> dict[str, Any]:
    """Load a fitted MrSQM classifier and preprocessing metadata."""

    artifact = joblib.load(checkpoint_path)
    version = artifact.get("artifact_version")
    if version != ARTIFACT_VERSION:
        raise ValueError(
            f"Unsupported MrSQM artifact version {version!r}; "
            f"expected {ARTIFACT_VERSION}"
        )
    if artifact.get("artifact_type") != "multivariate_mrsqm_classifier":
        raise ValueError("Artifact is not a MrSQM classifier")
    return artifact


def load_training_result(config: ExperimentConfig) -> TrainingResult:
    """Load a complete, configuration-matched MrSQM fold."""

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
        raise FileNotFoundError(
            "Incomplete MrSQM run; missing: "
            + ", ".join(str(path) for path in missing_paths)
        )
    artifact = load_trained_model(checkpoint_path)
    if artifact.get("experiment_config") != _json_ready(asdict(config)):
        raise ValueError(
            f"Saved MrSQM run {run_directory} was created with a different "
            "experiment configuration"
        )
    selected_channel_names = list(artifact["selected_channel_names"])
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
        logistic_c=float(artifact["logistic_c"]),
        num_symbolic_features=int(artifact["num_symbolic_features"]),
        num_selected_channels=len(selected_channel_names),
        selected_channel_names=selected_channel_names,
        history=[],
    )


__all__ = [
    "TrainingResult",
    "load_trained_model",
    "load_training_result",
    "train_mrsqm",
]
