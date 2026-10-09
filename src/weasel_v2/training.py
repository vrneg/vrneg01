"""Train and evaluate multivariate WEASEL 2.0 on one saved fold."""

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
from sklearn.pipeline import Pipeline
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
from .features import MultivariateWEASELTransformerV2, SupervisedChannelSelector


ARTIFACT_VERSION = 1


@dataclass(slots=True)
class TrainingResult:
    """Artifacts and metrics for one WEASEL 2.0 fold."""

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


def build_pipeline(config: ExperimentConfig) -> Pipeline:
    """Construct the channel-screened WEASEL 2.0 plus ridge pipeline."""

    model = config.model
    return Pipeline(
        [
            (
                "channel_selection",
                SupervisedChannelSelector(
                    max_channels=model.max_channels,
                    epsilon=model.channel_score_epsilon,
                ),
            ),
            (
                "weasel_v2",
                MultivariateWEASELTransformerV2(
                    min_window=model.min_window,
                    norm_options=model.norm_options,
                    word_lengths=model.word_lengths,
                    use_first_differences=model.use_first_differences,
                    feature_selection=model.feature_selection,
                    max_feature_count=model.max_feature_count,
                    ensemble_size=model.ensemble_size,
                    random_state=config.seed,
                    n_jobs=model.n_jobs,
                ),
            ),
            (
                "ridge",
                RidgeClassifierCV(
                    alphas=np.asarray(model.alphas, dtype=np.float64),
                    class_weight=model.class_weight,
                ),
            ),
        ]
    )


def _decision_scores(pipeline: Pipeline, split: TimeSeriesSplit) -> np.ndarray:
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


def train_weasel_v2(config: ExperimentConfig) -> TrainingResult:
    """Fit WEASEL 2.0 and evaluate validation/test exactly once."""

    config.validate()
    random.seed(config.seed)
    np.random.seed(config.seed)
    data: DataBundle = prepare_data(config.data)
    pipeline = build_pipeline(config)
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

    selector = pipeline.named_steps["channel_selection"]
    transformer = pipeline.named_steps["weasel_v2"]
    ridge = pipeline.named_steps["ridge"]
    if not isinstance(selector, SupervisedChannelSelector):
        raise TypeError("WEASEL 2.0 pipeline has an unexpected channel selector")
    if not isinstance(transformer, MultivariateWEASELTransformerV2):
        raise TypeError("WEASEL 2.0 pipeline has an unexpected dictionary transform")
    if not isinstance(ridge, RidgeClassifierCV):
        raise TypeError("WEASEL 2.0 pipeline does not end in RidgeClassifierCV")
    selected_indices = selector.selected_channel_indices_.tolist()
    selected_channel_names = [data.channel_names[index] for index in selected_indices]
    num_transformed_features = int(transformer.total_features_count_)
    best_alpha = float(ridge.alpha_)

    run_directory = config.output_dir / config.run_name
    run_directory.mkdir(parents=True, exist_ok=True)
    checkpoint_path = run_directory / "model.joblib"
    history_path = run_directory / "metrics.json"
    validation_predictions_path = run_directory / "validation_predictions.jsonl"
    predictions_path = run_directory / "test_predictions.jsonl"
    artifact = {
        "artifact_version": ARTIFACT_VERSION,
        "artifact_type": "multivariate_weasel_v2_classifier",
        "pipeline": pipeline,
        "experiment_config": _json_ready(asdict(config)),
        "normalizer": (
            None if data.normalizer is None else data.normalizer.state_dict()
        ),
        "time_grid": data.time_grid,
        "channel_names": data.channel_names,
        "selected_channel_indices": selected_indices,
        "selected_channel_names": selected_channel_names,
        "channel_scores": selector.channel_scores_,
        "best_alpha": best_alpha,
        "num_transformed_features": num_transformed_features,
        "decision_threshold": decision_threshold,
        "validation_metrics": validation_metrics,
        "test_metrics": test_metrics,
    }
    joblib.dump(artifact, checkpoint_path, compress=3)

    metrics_payload = {
        "best_alpha": best_alpha,
        "num_input_channels": len(data.channel_names),
        "num_selected_channels": len(selected_indices),
        "selected_channel_names": selected_channel_names,
        "num_time_points": config.data.num_time_points,
        "ensemble_size": transformer.ensemble_size_,
        "num_transformed_features": num_transformed_features,
        "channel_configuration_counts": transformer.channel_configuration_counts_,
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
        num_transformed_features=num_transformed_features,
        num_selected_channels=len(selected_indices),
        selected_channel_names=selected_channel_names,
        history=[],
    )


def load_trained_model(checkpoint_path: str | Path) -> dict[str, Any]:
    """Load a fitted WEASEL 2.0 pipeline and preprocessing metadata."""

    artifact = joblib.load(checkpoint_path)
    version = artifact.get("artifact_version")
    if version != ARTIFACT_VERSION:
        raise ValueError(
            f"Unsupported WEASEL 2.0 artifact version {version!r}; "
            f"expected {ARTIFACT_VERSION}"
        )
    if artifact.get("artifact_type") != "multivariate_weasel_v2_classifier":
        raise ValueError("Artifact is not a WEASEL 2.0 classifier")
    return artifact


def load_training_result(config: ExperimentConfig) -> TrainingResult:
    """Load a complete, configuration-matched WEASEL 2.0 fold."""

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
            "Incomplete WEASEL 2.0 run; missing: "
            + ", ".join(str(path) for path in missing_paths)
        )
    artifact = load_trained_model(checkpoint_path)
    if artifact.get("experiment_config") != _json_ready(asdict(config)):
        raise ValueError(
            f"Saved WEASEL 2.0 run {run_directory} was created with a different "
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
        best_alpha=float(artifact["best_alpha"]),
        num_transformed_features=int(artifact["num_transformed_features"]),
        num_selected_channels=len(selected_channel_names),
        selected_channel_names=selected_channel_names,
        history=[],
    )


__all__ = [
    "TrainingResult",
    "build_pipeline",
    "load_trained_model",
    "load_training_result",
    "train_weasel_v2",
]
