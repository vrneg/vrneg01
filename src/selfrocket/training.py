"""Train multivariate SelF-Rocket on one saved dataset fold."""

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
from .features import SelFRocketFeatures


ARTIFACT_VERSION = 1


@dataclass(slots=True)
class TrainingResult:
    """Artifacts, selection diagnostics, and metrics for one SelF-Rocket fold."""

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
    selected_candidate: str
    selected_candidate_before_vote: str
    selection_vote_support: float
    selection_used_fallback: bool
    selection_median_scores: dict[str, float]
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


def _build_pipeline(config: ExperimentConfig) -> Any:
    model = config.model
    transform = SelFRocketFeatures(
        num_kernels=model.num_kernels,
        max_dilations_per_kernel=model.max_dilations_per_kernel,
        normalise_per_instance=model.normalise_per_instance,
        only_mix=model.only_mix,
        selection_num_folds=model.selection_num_folds,
        selection_num_runs=model.selection_num_runs,
        selection_num_features=model.selection_num_features,
        selection_max_samples=model.selection_max_samples,
        selection_alphas=model.selection_alphas,
        vote_top=model.vote_top,
        vote_threshold=model.vote_threshold,
        length_threshold=model.length_threshold,
        class_weight=model.class_weight,
        n_jobs=model.n_jobs,
        random_state=config.seed,
    )
    ridge = RidgeClassifierCV(
        alphas=np.asarray(model.alphas, dtype=np.float64),
        class_weight=model.class_weight,
    )
    return make_pipeline(transform, ridge)


def _feature_transformer(pipeline: Any) -> SelFRocketFeatures:
    transformer = pipeline.steps[0][1]
    if not isinstance(transformer, SelFRocketFeatures):
        raise TypeError("SelF-Rocket pipeline has an unexpected feature transformer")
    return transformer


def _ridge_estimator(pipeline: Any) -> RidgeClassifierCV:
    estimator = pipeline.steps[-1][1]
    if not isinstance(estimator, RidgeClassifierCV):
        raise TypeError("SelF-Rocket pipeline does not end in RidgeClassifierCV")
    return estimator


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


def train_selfrocket(config: ExperimentConfig) -> TrainingResult:
    """Select a SelF-Rocket feature set, fit ridge, and evaluate one fold."""

    config.validate()
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
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

    transformer = _feature_transformer(pipeline)
    ridge = _ridge_estimator(pipeline)
    best_alpha = float(ridge.alpha_)
    transformed_feature_count = int(transformer.n_transformed_features_)
    selected_candidate = str(transformer.selected_candidate_)
    selected_before_vote = str(transformer.selected_candidate_before_vote_)
    vote_support = float(transformer.selection_vote_support_)
    used_fallback = bool(transformer.selection_used_fallback_)
    median_scores = dict(transformer.selection_median_scores_)

    run_directory = config.output_dir / config.run_name
    run_directory.mkdir(parents=True, exist_ok=True)
    checkpoint_path = run_directory / "model.joblib"
    history_path = run_directory / "metrics.json"
    validation_predictions_path = run_directory / "validation_predictions.jsonl"
    predictions_path = run_directory / "test_predictions.jsonl"

    artifact = {
        "artifact_version": ARTIFACT_VERSION,
        "artifact_type": "multivariate_selfrocket_classifier",
        "pipeline": pipeline,
        "experiment_config": _json_ready(asdict(config)),
        "normalizer": (
            None if data.normalizer is None else data.normalizer.state_dict()
        ),
        "time_grid": data.time_grid,
        "channel_names": data.channel_names,
        "best_alpha": best_alpha,
        "num_transformed_features": transformed_feature_count,
        "selected_candidate": selected_candidate,
        "selected_candidate_before_vote": selected_before_vote,
        "selection_vote_support": vote_support,
        "selection_used_fallback": used_fallback,
        "selection_median_scores": median_scores,
        "selection_performances": transformer.selection_performances_,
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
        "selected_candidate": selected_candidate,
        "selected_candidate_before_vote": selected_before_vote,
        "selection_vote_support": vote_support,
        "selection_used_fallback": used_fallback,
        "selection_median_scores": median_scores,
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
        best_alpha=best_alpha,
        num_transformed_features=transformed_feature_count,
        selected_candidate=selected_candidate,
        selected_candidate_before_vote=selected_before_vote,
        selection_vote_support=vote_support,
        selection_used_fallback=used_fallback,
        selection_median_scores=median_scores,
        history=[],
    )


def load_trained_model(checkpoint_path: str | Path) -> dict[str, Any]:
    """Load a fitted SelF-Rocket pipeline and preprocessing metadata."""

    artifact = joblib.load(checkpoint_path)
    version = artifact.get("artifact_version")
    if version != ARTIFACT_VERSION:
        raise ValueError(
            f"Unsupported SelF-Rocket artifact version {version!r}; "
            f"expected {ARTIFACT_VERSION}"
        )
    return artifact


def load_training_result(config: ExperimentConfig) -> TrainingResult:
    """Load a complete, configuration-matched SelF-Rocket run from disk."""

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
        raise FileNotFoundError(f"Incomplete SelF-Rocket run; missing: {missing}")

    artifact = load_trained_model(checkpoint_path)
    if artifact.get("experiment_config") != _json_ready(asdict(config)):
        raise ValueError(
            f"Saved SelF-Rocket run {run_directory} was created with a different "
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
        selected_candidate=str(artifact["selected_candidate"]),
        selected_candidate_before_vote=str(
            artifact["selected_candidate_before_vote"]
        ),
        selection_vote_support=float(artifact["selection_vote_support"]),
        selection_used_fallback=bool(artifact["selection_used_fallback"]),
        selection_median_scores=dict(artifact["selection_median_scores"]),
        history=[],
    )


__all__ = [
    "TrainingResult",
    "load_trained_model",
    "load_training_result",
    "train_selfrocket",
]
