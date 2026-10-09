"""Train MASHT features and run in-context classification with TabPFN-3."""

from __future__ import annotations

import json
import random
import time
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
from .features import (
    FeaturePlan,
    FittedMASHT,
    ensure_tabpfn_checkpoint_access,
    new_feature_transformer,
    new_tabpfn_classifier,
    positive_class_probabilities,
    resolve_feature_plan,
)


ARTIFACT_VERSION = 1
_PROBABILITY_EPSILON = 1e-7


@dataclass(slots=True)
class TrainingResult:
    """Artifacts, metrics, and resolved MASHT feature diagnostics for one fold."""

    checkpoint_path: Path
    history_path: Path
    validation_predictions_path: Path
    predictions_path: Path
    run_name: str
    best_epoch: int
    decision_threshold: float
    validation_metrics: dict[str, float]
    test_metrics: dict[str, float]
    budget_sample_count: int
    nominal_feature_budget: int
    num_transformed_features: int
    num_hydra_features: int
    num_multirocket_features: int
    multirocket_num_kernels: int
    hydra_num_groups: int
    tabpfn_version: str = "3"
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


def _budget_sample_count(config: ExperimentConfig, data: DataBundle) -> int:
    if config.model.feature_budget_scope == "train":
        return len(data.train.labels)
    return (
        len(data.train.labels)
        + len(data.validation.labels)
        + len(data.test.labels)
    )


def _plan_payload(plan: FeaturePlan) -> dict[str, int]:
    return {name: int(value) for name, value in asdict(plan).items()}


def train_masht(config: ExperimentConfig) -> TrainingResult:
    """Fit one MASHT outer fold without exposing held-out labels to the model."""

    config.validate()
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    total_start = time.monotonic()

    _log(config, "Checking local TabPFN-3 checkpoint access ...")
    ensure_tabpfn_checkpoint_access(config.model, config.seed)
    _log(config, f"Loading dataset from {config.data.dataset_path} ...")
    data_start = time.monotonic()
    data: DataBundle = prepare_data(config.data)
    train_classes = np.unique(data.train.labels)
    if not np.array_equal(train_classes, np.asarray([0, 1])):
        raise ValueError(
            "MASHT requires binary training labels encoded as 0 and 1; "
            f"received {train_classes.tolist()}"
        )
    _log(
        config,
        f"Loaded train={len(data.train.labels)}, "
        f"validation={len(data.validation.labels)}, test={len(data.test.labels)} "
        f"in {time.monotonic() - data_start:.1f}s.",
    )

    budget_sample_count = _budget_sample_count(config, data)
    plan = resolve_feature_plan(
        config.model,
        budget_sample_count,
        data.train.values.shape[-1],
    )
    _log(
        config,
        f"MASHT nominal budget={plan.nominal_total_budget:,} from "
        f"{plan.budget_sample_count:,} samples: MultiRocket kernels="
        f"{plan.multirocket_num_kernels:,}, HYDRA groups={plan.hydra_num_groups:,} "
        f"across {plan.hydra_num_dilations} dilations.",
    )

    feature_start = time.monotonic()
    transformer = new_feature_transformer(config.model, plan, config.seed)
    _log(config, "Fitting HYDRA and MultiRocket transforms on training data ...")
    train_features = np.asarray(
        transformer.fit_transform(data.train.values, data.train.labels),
        dtype=np.float32,
        order="C",
    )
    # TabPFN recomputes its training context for every prediction. Combining both
    # untouched evaluation splits avoids a second full TabPFN-3 inference pass.
    query_values = np.concatenate(
        (data.validation.values, data.test.values), axis=0
    )
    query_features = np.asarray(
        transformer.transform(query_values),
        dtype=np.float32,
        order="C",
    )
    num_hydra_features = int(transformer.n_hydra_features_)
    num_multirocket_features = int(transformer.n_multirocket_features_)
    num_transformed_features = int(train_features.shape[1])
    if num_hydra_features + num_multirocket_features != num_transformed_features:
        raise RuntimeError("MASHT branch feature counts do not match output width")
    _log(
        config,
        f"Generated {num_transformed_features:,} effective features "
        f"(HYDRA={num_hydra_features:,}, MultiRocket="
        f"{num_multirocket_features:,}) in "
        f"{time.monotonic() - feature_start:.1f}s.",
    )

    classifier = new_tabpfn_classifier(config.model, config.seed)
    inference_start = time.monotonic()
    _log(
        config,
        f"Fitting one TabPFN-3 context with {config.model.tabpfn_n_estimators} "
        "estimators ...",
    )
    classifier.fit(train_features, data.train.labels)
    _log(
        config,
        f"TabPFN-3 fit completed in {time.monotonic() - inference_start:.1f}s; "
        f"predicting {len(query_features):,} held-out samples ...",
    )
    query_probabilities = positive_class_probabilities(classifier, query_features)
    _log(
        config,
        f"TabPFN-3 fit and prediction finished in "
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

    fitted_model = FittedMASHT(
        model_config=config.model,
        seed=config.seed,
        train_values=np.asarray(data.train.values),
        train_labels=np.asarray(data.train.labels),
        feature_transformer=transformer,
    )
    run_directory = config.output_dir / config.run_name
    run_directory.mkdir(parents=True, exist_ok=True)
    checkpoint_path = run_directory / "model.joblib"
    history_path = run_directory / "metrics.json"
    validation_predictions_path = run_directory / "validation_predictions.jsonl"
    predictions_path = run_directory / "test_predictions.jsonl"

    artifact = {
        "artifact_version": ARTIFACT_VERSION,
        "artifact_type": "masht_tabpfn_v3_classifier",
        # The 203 MiB pretrained TabPFN checkpoint is intentionally not duplicated
        # in each fold artifact. FittedMASHT recreates it lazily from model_config.
        "model": fitted_model,
        "experiment_config": _json_ready(asdict(config)),
        "normalizer": (
            None if data.normalizer is None else data.normalizer.state_dict()
        ),
        "time_grid": data.time_grid,
        "channel_names": data.channel_names,
        "feature_plan": _plan_payload(plan),
        "num_transformed_features": num_transformed_features,
        "num_hydra_features": num_hydra_features,
        "num_multirocket_features": num_multirocket_features,
        "tabpfn_version": "3",
        "decision_threshold": decision_threshold,
        "validation_metrics": validation_metrics,
        "test_metrics": test_metrics,
    }
    joblib.dump(artifact, checkpoint_path, compress=3)

    metrics_payload = {
        "method": "MASHT",
        "tabpfn_version": "3",
        "tabpfn_n_estimators": config.model.tabpfn_n_estimators,
        "feature_budget_scope": config.model.feature_budget_scope,
        "feature_plan": _plan_payload(plan),
        "num_input_channels": len(data.channel_names),
        "num_time_points": config.data.num_time_points,
        "num_transformed_features": num_transformed_features,
        "num_hydra_features": num_hydra_features,
        "num_multirocket_features": num_multirocket_features,
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
        budget_sample_count=plan.budget_sample_count,
        nominal_feature_budget=plan.nominal_total_budget,
        num_transformed_features=num_transformed_features,
        num_hydra_features=num_hydra_features,
        num_multirocket_features=num_multirocket_features,
        multirocket_num_kernels=plan.multirocket_num_kernels,
        hydra_num_groups=plan.hydra_num_groups,
    )


def load_trained_model(checkpoint_path: str | Path) -> dict[str, Any]:
    """Load a fitted MASHT artifact without loading TabPFN-3 weights."""

    artifact = joblib.load(checkpoint_path)
    version = artifact.get("artifact_version")
    if version != ARTIFACT_VERSION:
        raise ValueError(
            f"Unsupported MASHT artifact version {version!r}; "
            f"expected {ARTIFACT_VERSION}"
        )
    if artifact.get("artifact_type") != "masht_tabpfn_v3_classifier":
        raise ValueError("The checkpoint is not a MASHT TabPFN-3 artifact")
    return artifact


def load_training_result(config: ExperimentConfig) -> TrainingResult:
    """Load a complete, configuration-matched MASHT run from disk."""

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
        raise FileNotFoundError(f"Incomplete MASHT run; missing: {missing}")

    artifact = load_trained_model(checkpoint_path)
    if artifact.get("experiment_config") != _json_ready(asdict(config)):
        raise ValueError(
            f"Saved MASHT run {run_directory} was created with a different "
            "experiment configuration"
        )
    plan = artifact["feature_plan"]
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
        budget_sample_count=int(plan["budget_sample_count"]),
        nominal_feature_budget=int(plan["nominal_total_budget"]),
        num_transformed_features=int(artifact["num_transformed_features"]),
        num_hydra_features=int(artifact["num_hydra_features"]),
        num_multirocket_features=int(artifact["num_multirocket_features"]),
        multirocket_num_kernels=int(plan["multirocket_num_kernels"]),
        hydra_num_groups=int(plan["hydra_num_groups"]),
    )


__all__ = [
    "ARTIFACT_VERSION",
    "TrainingResult",
    "load_trained_model",
    "load_training_result",
    "train_masht",
]
