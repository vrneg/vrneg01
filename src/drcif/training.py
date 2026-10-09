"""Train and evaluate Diverse Representation CIF on one saved fold."""

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


ARTIFACT_VERSION = 1


@dataclass(slots=True)
class TrainingResult:
    """Artifacts and metrics for one DrCIF fold."""

    checkpoint_path: Path
    history_path: Path
    validation_predictions_path: Path
    predictions_path: Path
    run_name: str
    best_epoch: int
    decision_threshold: float
    validation_metrics: dict[str, float]
    test_metrics: dict[str, float]
    n_estimators_built: int
    total_intervals_per_tree: int
    history: list[dict[str, Any]] = field(default_factory=list)
    pretrained_checkpoint_path: Path | None = None
    pretraining_history: list[dict[str, Any]] = field(default_factory=list)


@dataclass(slots=True)
class PredictionOutput:
    """Positive-class probabilities and their finite-logit representation."""

    probabilities: np.ndarray
    logits: np.ndarray


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


def _load_drcif_class() -> type:
    try:
        from aeon.classification.interval_based import DrCIFClassifier
    except ImportError as error:
        raise ImportError(
            "DrCIF experiments require aeon. Install project requirements with "
            "`pip install -r requirements.txt`."
        ) from error
    return DrCIFClassifier


def build_classifier(config: ExperimentConfig) -> Any:
    """Construct the configured aeon estimator without fitting it."""

    model = config.model
    if model.stabilize_near_constant_intervals:
        # Import only after the dependency check so missing-aeon failures retain the
        # actionable error from _load_drcif_class.
        _load_drcif_class()
        from .stabilization import NearConstantSafeDrCIFClassifier

        classifier_class = NearConstantSafeDrCIFClassifier
    else:
        classifier_class = _load_drcif_class()
    classifier = classifier_class(
        n_estimators=model.n_estimators,
        n_intervals=model.n_intervals,
        min_interval_length=model.min_interval_length,
        max_interval_length=model.max_interval_length,
        att_subsample_size=model.att_subsample_size,
        time_limit_in_minutes=model.time_limit_in_minutes,
        contract_max_n_estimators=model.contract_max_n_estimators,
        use_pycatch22=model.use_pycatch22,
        random_state=config.seed,
        n_jobs=model.n_jobs,
        parallel_backend=model.parallel_backend,
    )
    return classifier


def _positive_class_output(
    classifier: Any,
    split: TimeSeriesSplit,
) -> PredictionOutput:
    probabilities = np.asarray(
        classifier.predict_proba(split.values), dtype=np.float64
    )
    if probabilities.ndim != 2 or probabilities.shape[0] != split.labels.shape[0]:
        raise RuntimeError(
            "DrCIF predict_proba returned an unexpected shape: "
            f"{probabilities.shape}"
        )
    classes = np.asarray(classifier.classes_)
    positive_indices = np.flatnonzero(classes == 1)
    if positive_indices.size != 1:
        raise RuntimeError(
            "DrCIF must be fitted with binary labels containing positive class 1"
        )
    positive_probabilities = probabilities[:, int(positive_indices[0])]
    epsilon = np.finfo(np.float32).eps
    clipped = np.clip(positive_probabilities, epsilon, 1.0 - epsilon)
    logits = np.log(clipped) - np.log1p(-clipped)
    return PredictionOutput(
        probabilities=positive_probabilities,
        logits=logits,
    )


def _metrics(
    output: PredictionOutput,
    labels: np.ndarray,
    threshold: float,
) -> dict[str, float]:
    logits = torch.from_numpy(output.logits).float()
    targets = torch.from_numpy(labels).float()
    loss = float(functional.binary_cross_entropy_with_logits(logits, targets).item())
    return binary_classification_metrics(logits, targets, loss, threshold)


def _prediction_rows(
    split: TimeSeriesSplit,
    output: PredictionOutput,
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
            output.logits,
            output.probabilities,
            strict=True,
        )
    ]


def train_drcif(config: ExperimentConfig) -> TrainingResult:
    """Fit DrCIF on training data and evaluate validation/test exactly once."""

    config.validate()
    random.seed(config.seed)
    np.random.seed(config.seed)
    data: DataBundle = prepare_data(config.data)
    classifier = build_classifier(config)
    classifier.fit(data.train.values, data.train.labels)

    validation_output = _positive_class_output(classifier, data.validation)
    test_output = _positive_class_output(classifier, data.test)
    decision_threshold = (
        optimize_binary_threshold(
            torch.from_numpy(validation_output.logits).float(),
            torch.from_numpy(data.validation.labels).float(),
            metric_name=config.evaluation.threshold_metric,
        )
        if config.evaluation.calibrate_threshold_on_validation
        else config.evaluation.threshold
    )
    validation_metrics = _metrics(
        validation_output,
        data.validation.labels,
        decision_threshold,
    )
    test_metrics = _metrics(test_output, data.test.labels, decision_threshold)
    n_estimators_built = len(classifier.estimators_)
    total_intervals_per_tree = int(classifier.total_intervals_)

    run_directory = config.output_dir / config.run_name
    run_directory.mkdir(parents=True, exist_ok=True)
    checkpoint_path = run_directory / "model.joblib"
    history_path = run_directory / "metrics.json"
    validation_predictions_path = run_directory / "validation_predictions.jsonl"
    predictions_path = run_directory / "test_predictions.jsonl"
    artifact = {
        "artifact_version": ARTIFACT_VERSION,
        "artifact_type": "multivariate_drcif_classifier",
        "classifier": classifier,
        "experiment_config": _json_ready(asdict(config)),
        "normalizer": (
            None if data.normalizer is None else data.normalizer.state_dict()
        ),
        "time_grid": data.time_grid,
        "channel_names": data.channel_names,
        "n_estimators_built": n_estimators_built,
        "total_intervals_per_tree": total_intervals_per_tree,
        "decision_threshold": decision_threshold,
        "validation_metrics": validation_metrics,
        "test_metrics": test_metrics,
    }
    joblib.dump(artifact, checkpoint_path, compress=3)
    history_path.write_text(
        json.dumps(
            _json_ready(
                {
                    "n_estimators_requested": config.model.n_estimators,
                    "n_estimators_built": n_estimators_built,
                    "total_intervals_per_tree": total_intervals_per_tree,
                    "num_input_channels": len(data.channel_names),
                    "num_time_points": config.data.num_time_points,
                    "representations": ["raw", "first_difference", "periodogram"],
                    "stabilize_near_constant_intervals": (
                        config.model.stabilize_near_constant_intervals
                    ),
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
            ),
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    validation_predictions_path.write_text(
        "\n".join(
            _prediction_rows(
                data.validation,
                validation_output,
                decision_threshold,
            )
        )
        + "\n",
        encoding="utf-8",
    )
    predictions_path.write_text(
        "\n".join(_prediction_rows(data.test, test_output, decision_threshold))
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
        n_estimators_built=n_estimators_built,
        total_intervals_per_tree=total_intervals_per_tree,
        history=[],
    )


def load_trained_model(checkpoint_path: str | Path) -> dict[str, Any]:
    """Load a fitted DrCIF classifier and its preprocessing metadata."""

    artifact = joblib.load(checkpoint_path)
    version = artifact.get("artifact_version")
    if version != ARTIFACT_VERSION:
        raise ValueError(
            f"Unsupported DrCIF artifact version {version!r}; "
            f"expected {ARTIFACT_VERSION}"
        )
    if artifact.get("artifact_type") != "multivariate_drcif_classifier":
        raise ValueError("Artifact is not a DrCIF classifier")
    return artifact


def load_training_result(config: ExperimentConfig) -> TrainingResult:
    """Load a complete, configuration-matched DrCIF fold for resumption."""

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
            "Incomplete DrCIF run; missing: "
            + ", ".join(str(path) for path in missing_paths)
        )
    artifact = load_trained_model(checkpoint_path)
    if artifact.get("experiment_config") != _json_ready(asdict(config)):
        raise ValueError(
            f"Saved DrCIF run {run_directory} was created with a different "
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
        n_estimators_built=int(artifact["n_estimators_built"]),
        total_intervals_per_tree=int(artifact["total_intervals_per_tree"]),
        history=[],
    )


__all__ = [
    "PredictionOutput",
    "TrainingResult",
    "build_classifier",
    "load_trained_model",
    "load_training_result",
    "train_drcif",
]
