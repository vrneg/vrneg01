"""Train and evaluate HIVE-COTE 2.0 on one saved fold."""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
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
from .estimator import ResourceAwareHIVECOTEV2, build_classifier as _build_classifier


ARTIFACT_VERSION = 1
COMPONENT_NAMES = ("STC", "DrCIF", "Arsenal", "TDE")


@dataclass(slots=True)
class TrainingResult:
    """Artifacts and metrics for one HIVE-COTE 2.0 fold."""

    checkpoint_path: Path
    history_path: Path
    validation_predictions_path: Path
    predictions_path: Path
    status_path: Path
    run_name: str
    best_epoch: int
    decision_threshold: float
    validation_metrics: dict[str, float]
    test_metrics: dict[str, float]
    component_weights: dict[str, float]
    component_train_accuracies: dict[str, float]
    component_fit_seconds: dict[str, float]
    fit_seconds: float
    validation_component_predictions_path: Path | None = None
    component_predictions_path: Path | None = None
    history: list[dict[str, Any]] = field(default_factory=list)
    pretrained_checkpoint_path: Path | None = None
    pretraining_history: list[dict[str, Any]] = field(default_factory=list)


@dataclass(slots=True)
class PredictionOutput:
    """Positive-class probabilities and their finite-logit representation."""

    probabilities: np.ndarray
    logits: np.ndarray


@dataclass(slots=True)
class EnsemblePredictionOutput:
    """Final HC2 prediction plus optional predictions from each component."""

    ensemble: PredictionOutput
    components: dict[str, PredictionOutput]


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


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(
        json.dumps(_json_ready(payload), indent=2) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(path)


def _write_status(path: Path, state: str, **details: Any) -> None:
    _atomic_write_json(
        path,
        {
            "state": state,
            "updated_at_utc": datetime.now(timezone.utc).isoformat(),
            **details,
        },
    )


def build_classifier(config: ExperimentConfig) -> ResourceAwareHIVECOTEV2:
    """Construct the configured official aeon HC2 estimator without fitting it."""

    return _build_classifier(config.model, config.seed)


def _to_prediction_output(probabilities: np.ndarray) -> PredictionOutput:
    epsilon = np.finfo(np.float32).eps
    clipped = np.clip(probabilities, epsilon, 1.0 - epsilon)
    return PredictionOutput(
        probabilities=probabilities,
        logits=np.log(clipped) - np.log1p(-clipped),
    )


def _positive_class_outputs(
    classifier: ResourceAwareHIVECOTEV2,
    split: TimeSeriesSplit,
    include_components: bool,
) -> EnsemblePredictionOutput:
    if include_components:
        ensemble_probabilities, component_probabilities = (
            classifier.predict_proba_with_components(split.values)
        )
    else:
        ensemble_probabilities = classifier.predict_proba(split.values)
        component_probabilities = {}

    classes = np.asarray(classifier.classes_)
    positive_indices = np.flatnonzero(classes == 1)
    if positive_indices.size != 1:
        raise RuntimeError("HIVE-COTE 2.0 requires binary positive class 1")
    positive_index = int(positive_indices[0])
    ensemble_probabilities = np.asarray(ensemble_probabilities, dtype=np.float64)
    expected_shape = (split.labels.shape[0], classes.shape[0])
    if ensemble_probabilities.shape != expected_shape:
        raise RuntimeError(
            "HIVE-COTE 2.0 predict_proba returned an unexpected shape: "
            f"{ensemble_probabilities.shape}"
        )

    components: dict[str, PredictionOutput] = {}
    estimator_by_name = dict(
        zip(classifier.component_names_, classifier.fitted_estimators_, strict=True)
    )
    for name, probabilities in component_probabilities.items():
        probabilities = np.asarray(probabilities, dtype=np.float64)
        if probabilities.shape != expected_shape:
            raise RuntimeError(
                f"HC2 component {name} returned shape {probabilities.shape}; "
                f"expected {expected_shape}"
            )
        component_classes = np.asarray(estimator_by_name[name].classes_)
        component_positive = np.flatnonzero(component_classes == 1)
        if component_positive.size != 1:
            raise RuntimeError(f"HC2 component {name} has no unique positive class 1")
        components[name] = _to_prediction_output(
            probabilities[:, int(component_positive[0])]
        )

    return EnsemblePredictionOutput(
        ensemble=_to_prediction_output(ensemble_probabilities[:, positive_index]),
        components=components,
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


def _component_prediction_rows(
    split: TimeSeriesSplit,
    components: dict[str, PredictionOutput],
) -> list[str]:
    names = [name for name in COMPONENT_NAMES if name in components]
    return [
        json.dumps(
            {
                "sample_id": sample_id,
                "label": int(split.labels[index]),
                "component_probabilities": {
                    name: float(components[name].probabilities[index])
                    for name in names
                },
            }
        )
        for index, sample_id in enumerate(split.sample_ids)
    ]


def train_hivecote_v2(config: ExperimentConfig) -> TrainingResult:
    """Fit HC2 on training data and evaluate validation/test exactly once."""

    config.validate()
    random.seed(config.seed)
    np.random.seed(config.seed)
    run_directory = config.output_dir / config.run_name
    run_directory.mkdir(parents=True, exist_ok=True)
    status_path = run_directory / "status.json"
    _write_status(status_path, "loading_data", run_name=config.run_name)

    try:
        data: DataBundle = prepare_data(config.data)
        classifier = build_classifier(config)
        _write_status(
            status_path,
            "training",
            run_name=config.run_name,
            train_shape=list(data.train.values.shape),
            approximate_total_contract_minutes=config.model.time_limit_in_minutes,
            component_order=list(COMPONENT_NAMES),
        )
        fit_start = perf_counter()
        classifier.fit(data.train.values, data.train.labels)
        fit_seconds = perf_counter() - fit_start

        _write_status(status_path, "predicting_validation", fit_seconds=fit_seconds)
        validation_output = _positive_class_outputs(
            classifier,
            data.validation,
            config.model.save_component_predictions,
        )
        _write_status(status_path, "predicting_test", fit_seconds=fit_seconds)
        test_output = _positive_class_outputs(
            classifier,
            data.test,
            config.model.save_component_predictions,
        )
    except BaseException as error:
        _write_status(
            status_path,
            "failed",
            error_type=type(error).__name__,
            error_message=str(error),
        )
        raise

    decision_threshold = (
        optimize_binary_threshold(
            torch.from_numpy(validation_output.ensemble.logits).float(),
            torch.from_numpy(data.validation.labels).float(),
            metric_name=config.evaluation.threshold_metric,
        )
        if config.evaluation.calibrate_threshold_on_validation
        else config.evaluation.threshold
    )
    validation_metrics = _metrics(
        validation_output.ensemble,
        data.validation.labels,
        decision_threshold,
    )
    test_metrics = _metrics(
        test_output.ensemble,
        data.test.labels,
        decision_threshold,
    )
    component_weights = {
        str(name): float(weight)
        for name, weight in classifier.get_component_weights().items()
    }
    component_train_accuracies = {
        str(name): float(value)
        for name, value in classifier.component_train_accuracies_.items()
    }
    component_fit_seconds = {
        str(name): float(value)
        for name, value in classifier.component_fit_seconds_.items()
    }

    checkpoint_path = run_directory / "model.joblib"
    history_path = run_directory / "metrics.json"
    validation_predictions_path = run_directory / "validation_predictions.jsonl"
    predictions_path = run_directory / "test_predictions.jsonl"
    validation_component_predictions_path = (
        run_directory / "validation_component_predictions.jsonl"
        if config.model.save_component_predictions
        else None
    )
    component_predictions_path = (
        run_directory / "test_component_predictions.jsonl"
        if config.model.save_component_predictions
        else None
    )
    artifact = {
        "artifact_version": ARTIFACT_VERSION,
        "artifact_type": "multivariate_hivecote_v2_classifier",
        "classifier": classifier,
        "experiment_config": _json_ready(asdict(config)),
        "normalizer": (
            None if data.normalizer is None else data.normalizer.state_dict()
        ),
        "time_grid": data.time_grid,
        "channel_names": data.channel_names,
        "component_weights": component_weights,
        "component_train_accuracies": component_train_accuracies,
        "component_fit_seconds": component_fit_seconds,
        "fit_seconds": fit_seconds,
        "decision_threshold": decision_threshold,
        "validation_metrics": validation_metrics,
        "test_metrics": test_metrics,
    }
    temporary_checkpoint_path = checkpoint_path.with_suffix(".joblib.tmp")
    _write_status(status_path, "saving_artifact", fit_seconds=fit_seconds)
    joblib.dump(artifact, temporary_checkpoint_path, compress=3)
    temporary_checkpoint_path.replace(checkpoint_path)

    metrics_payload = {
        "implementation": "aeon.classification.hybrid.HIVECOTEV2",
        "component_order": list(COMPONENT_NAMES),
        "component_weights": component_weights,
        "component_train_accuracies": component_train_accuracies,
        "component_fit_seconds": component_fit_seconds,
        "fit_seconds": fit_seconds,
        "approximate_total_contract_minutes": config.model.time_limit_in_minutes,
        "num_input_channels": len(data.channel_names),
        "num_time_points": config.data.num_time_points,
        "compute_backend": "cpu",
        "n_jobs": config.model.n_jobs,
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
    _atomic_write_json(history_path, metrics_payload)
    validation_predictions_path.write_text(
        "\n".join(
            _prediction_rows(
                data.validation,
                validation_output.ensemble,
                decision_threshold,
            )
        )
        + "\n",
        encoding="utf-8",
    )
    predictions_path.write_text(
        "\n".join(
            _prediction_rows(data.test, test_output.ensemble, decision_threshold)
        )
        + "\n",
        encoding="utf-8",
    )
    if validation_component_predictions_path is not None:
        validation_component_predictions_path.write_text(
            "\n".join(
                _component_prediction_rows(
                    data.validation,
                    validation_output.components,
                )
            )
            + "\n",
            encoding="utf-8",
        )
    if component_predictions_path is not None:
        component_predictions_path.write_text(
            "\n".join(
                _component_prediction_rows(data.test, test_output.components)
            )
            + "\n",
            encoding="utf-8",
        )
    _write_status(
        status_path,
        "completed",
        fit_seconds=fit_seconds,
        checkpoint_path=checkpoint_path,
    )
    return TrainingResult(
        checkpoint_path=checkpoint_path,
        history_path=history_path,
        validation_predictions_path=validation_predictions_path,
        predictions_path=predictions_path,
        status_path=status_path,
        run_name=config.run_name,
        best_epoch=0,
        decision_threshold=decision_threshold,
        validation_metrics=validation_metrics,
        test_metrics=test_metrics,
        component_weights=component_weights,
        component_train_accuracies=component_train_accuracies,
        component_fit_seconds=component_fit_seconds,
        fit_seconds=fit_seconds,
        validation_component_predictions_path=(
            validation_component_predictions_path
        ),
        component_predictions_path=component_predictions_path,
        history=[],
    )


def load_trained_model(checkpoint_path: str | Path) -> dict[str, Any]:
    """Load a fitted HC2 classifier and its preprocessing metadata."""

    artifact = joblib.load(checkpoint_path)
    version = artifact.get("artifact_version")
    if version != ARTIFACT_VERSION:
        raise ValueError(
            f"Unsupported HIVE-COTE 2.0 artifact version {version!r}; "
            f"expected {ARTIFACT_VERSION}"
        )
    if artifact.get("artifact_type") != "multivariate_hivecote_v2_classifier":
        raise ValueError("Artifact is not a HIVE-COTE 2.0 classifier")
    return artifact


def load_training_result(config: ExperimentConfig) -> TrainingResult:
    """Load a complete, configuration-matched HC2 fold for resumption."""

    config.validate()
    run_directory = config.output_dir / config.run_name
    checkpoint_path = run_directory / "model.joblib"
    history_path = run_directory / "metrics.json"
    validation_predictions_path = run_directory / "validation_predictions.jsonl"
    predictions_path = run_directory / "test_predictions.jsonl"
    status_path = run_directory / "status.json"
    validation_component_predictions_path = (
        run_directory / "validation_component_predictions.jsonl"
        if config.model.save_component_predictions
        else None
    )
    component_predictions_path = (
        run_directory / "test_component_predictions.jsonl"
        if config.model.save_component_predictions
        else None
    )
    required_paths = [
        checkpoint_path,
        history_path,
        validation_predictions_path,
        predictions_path,
        status_path,
    ]
    if validation_component_predictions_path is not None:
        required_paths.append(validation_component_predictions_path)
    if component_predictions_path is not None:
        required_paths.append(component_predictions_path)
    missing_paths = [path for path in required_paths if not path.is_file()]
    if missing_paths:
        raise FileNotFoundError(
            "Incomplete HIVE-COTE 2.0 run; missing: "
            + ", ".join(str(path) for path in missing_paths)
        )
    artifact = load_trained_model(checkpoint_path)
    if artifact.get("experiment_config") != _json_ready(asdict(config)):
        raise ValueError(
            f"Saved HIVE-COTE 2.0 run {run_directory} was created with a "
            "different experiment configuration"
        )
    status = json.loads(status_path.read_text(encoding="utf-8"))
    if status.get("state") != "completed":
        raise FileNotFoundError(
            f"HIVE-COTE 2.0 run is not complete: state={status.get('state')!r}"
        )
    return TrainingResult(
        checkpoint_path=checkpoint_path,
        history_path=history_path,
        validation_predictions_path=validation_predictions_path,
        predictions_path=predictions_path,
        status_path=status_path,
        run_name=config.run_name,
        best_epoch=0,
        decision_threshold=float(artifact["decision_threshold"]),
        validation_metrics=dict(artifact["validation_metrics"]),
        test_metrics=dict(artifact["test_metrics"]),
        component_weights=dict(artifact["component_weights"]),
        component_train_accuracies=dict(
            artifact["component_train_accuracies"]
        ),
        component_fit_seconds=dict(artifact["component_fit_seconds"]),
        fit_seconds=float(artifact["fit_seconds"]),
        validation_component_predictions_path=(
            validation_component_predictions_path
        ),
        component_predictions_path=component_predictions_path,
        history=[],
    )


__all__ = [
    "EnsemblePredictionOutput",
    "PredictionOutput",
    "TrainingResult",
    "build_classifier",
    "load_trained_model",
    "load_training_result",
    "train_hivecote_v2",
]
