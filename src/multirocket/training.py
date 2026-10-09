"""Train MultiRocket with an optional Hydra feature branch on one saved fold."""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import torch
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import RidgeClassifierCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted
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
    """Artifacts and metrics for one MultiRocket fold."""

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
    hydra_enabled: bool
    num_multirocket_features: int
    num_hydra_features: int
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


def _load_transform_classes() -> tuple[type, type]:
    try:
        from aeon.transformations.collection.convolution_based import (
            HydraTransformer,
            MultiRocket,
        )
    except ImportError as error:
        raise ImportError(
            "MultiRocket experiments require aeon and Hydra requires torch. "
            "Install project requirements with `pip install -r requirements.txt`."
        ) from error
    return MultiRocket, HydraTransformer


class _HydraSparseScaler(BaseEstimator, TransformerMixin):
    """NumPy equivalent of Hydra's sparse square-root standardization."""

    def __init__(self, mask: bool = True, exponent: int = 4):
        self.mask = mask
        self.exponent = exponent

    @staticmethod
    def _root_counts(values: np.ndarray) -> np.ndarray:
        return np.sqrt(np.clip(np.asarray(values), 0.0, None))

    def fit(self, X: np.ndarray, y: Any = None) -> _HydraSparseScaler:
        del y
        values = self._root_counts(X)
        self.dtype_ = values.dtype
        self.epsilon_ = (
            np.mean(values == 0, axis=0) ** self.exponent + 1e-8
        ).astype(self.dtype_)
        self.mean_ = np.mean(values, axis=0, dtype=np.float64).astype(self.dtype_)
        # torch.std, used by aeon's Hydra scaler, applies Bessel's correction.
        self.scale_ = np.std(
            values, axis=0, ddof=1, dtype=np.float64
        ).astype(self.dtype_) + self.epsilon_
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(self, ("mean_", "scale_"))
        values = self._root_counts(X)
        centered = values - self.mean_
        if self.mask:
            centered = centered * (values != 0)
        return (centered / self.scale_).astype(self.dtype_, copy=False)


class MultiRocketHydraFeatures(BaseEstimator, TransformerMixin):
    """Concatenate consistently scaled MultiRocket and Hydra features."""

    def __init__(
        self,
        num_kernels: int = 6_250,
        max_dilations_per_kernel: int = 32,
        num_features_per_kernel: int = 4,
        normalise_per_instance: bool = False,
        hydra_num_kernels: int = 8,
        hydra_num_groups: int = 64,
        hydra_max_num_channels: int = 8,
        n_jobs: int = 1,
        random_state: int | None = None,
    ):
        self.num_kernels = num_kernels
        self.max_dilations_per_kernel = max_dilations_per_kernel
        self.num_features_per_kernel = num_features_per_kernel
        self.normalise_per_instance = normalise_per_instance
        self.hydra_num_kernels = hydra_num_kernels
        self.hydra_num_groups = hydra_num_groups
        self.hydra_max_num_channels = hydra_max_num_channels
        self.n_jobs = n_jobs
        self.random_state = random_state

    def _new_transformers(self) -> tuple[Any, Any]:
        MultiRocket, HydraTransformer = _load_transform_classes()
        multirocket = MultiRocket(
            n_kernels=self.num_kernels,
            max_dilations_per_kernel=self.max_dilations_per_kernel,
            n_features_per_kernel=self.num_features_per_kernel,
            normalise=self.normalise_per_instance,
            n_jobs=self.n_jobs,
            random_state=self.random_state,
        )
        hydra = HydraTransformer(
            n_kernels=self.hydra_num_kernels,
            n_groups=self.hydra_num_groups,
            max_num_channels=self.hydra_max_num_channels,
            n_jobs=self.n_jobs,
            random_state=self.random_state,
            output_type="numpy",
        )
        return multirocket, hydra

    def fit(self, X: np.ndarray, y: Any = None) -> MultiRocketHydraFeatures:
        self.fit_transform(X, y)
        return self

    def fit_transform(self, X: np.ndarray, y: Any = None, **fit_params: Any) -> np.ndarray:
        del fit_params
        self.multirocket_, self.hydra_ = self._new_transformers()
        multirocket_values = np.asarray(self.multirocket_.fit_transform(X, y))
        hydra_values = np.asarray(self.hydra_.fit_transform(X, y))

        self.multirocket_scaler_ = StandardScaler(with_mean=False)
        self.hydra_scaler_ = _HydraSparseScaler()
        multirocket_values = self.multirocket_scaler_.fit_transform(
            multirocket_values
        )
        hydra_values = self.hydra_scaler_.fit_transform(hydra_values)
        self.n_multirocket_features_ = int(multirocket_values.shape[1])
        self.n_hydra_features_ = int(hydra_values.shape[1])
        return np.concatenate((multirocket_values, hydra_values), axis=1)

    def transform(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(
            self,
            (
                "multirocket_",
                "hydra_",
                "multirocket_scaler_",
                "hydra_scaler_",
            ),
        )
        multirocket_values = self.multirocket_scaler_.transform(
            np.asarray(self.multirocket_.transform(X))
        )
        hydra_values = self.hydra_scaler_.transform(
            np.asarray(self.hydra_.transform(X))
        )
        return np.concatenate((multirocket_values, hydra_values), axis=1)


def _build_pipeline(config: ExperimentConfig) -> Any:
    MultiRocket, _ = _load_transform_classes()
    ridge = RidgeClassifierCV(
        alphas=np.asarray(config.model.alphas, dtype=np.float64),
        class_weight=config.model.class_weight,
    )
    if config.model.use_hydra:
        transform = MultiRocketHydraFeatures(
            num_kernels=config.model.num_kernels,
            max_dilations_per_kernel=config.model.max_dilations_per_kernel,
            num_features_per_kernel=config.model.num_features_per_kernel,
            normalise_per_instance=config.model.normalise_per_instance,
            hydra_num_kernels=config.model.hydra_num_kernels,
            hydra_num_groups=config.model.hydra_num_groups,
            hydra_max_num_channels=config.model.hydra_max_num_channels,
            n_jobs=config.model.n_jobs,
            random_state=config.seed,
        )
        return make_pipeline(transform, ridge)

    return make_pipeline(
        MultiRocket(
            n_kernels=config.model.num_kernels,
            max_dilations_per_kernel=config.model.max_dilations_per_kernel,
            n_features_per_kernel=config.model.num_features_per_kernel,
            normalise=config.model.normalise_per_instance,
            n_jobs=config.model.n_jobs,
            random_state=config.seed,
        ),
        StandardScaler(with_mean=False),
        ridge,
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
        raise TypeError("MultiRocket pipeline does not end in RidgeClassifierCV")
    return estimator


def _feature_counts(pipeline: Any, data: DataBundle) -> tuple[int, int, int]:
    transformer = pipeline.steps[0][1]
    transformed = np.asarray(transformer.transform(data.train.values[:1]))
    total = int(transformed.shape[1])
    multirocket = int(getattr(transformer, "n_multirocket_features_", total))
    hydra = int(getattr(transformer, "n_hydra_features_", 0))
    if multirocket + hydra != total:
        raise RuntimeError("MultiRocket/Hydra feature counts do not match output")
    return total, multirocket, hydra


def train_multirocket(config: ExperimentConfig) -> TrainingResult:
    """Fit MultiRocket, optionally add Hydra, and evaluate validation/test."""

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

    ridge = _ridge_estimator(pipeline)
    best_alpha = float(ridge.alpha_)
    total_features, multirocket_features, hydra_features = _feature_counts(
        pipeline, data
    )
    run_directory = config.output_dir / config.run_name
    run_directory.mkdir(parents=True, exist_ok=True)
    checkpoint_path = run_directory / "model.joblib"
    history_path = run_directory / "metrics.json"
    validation_predictions_path = run_directory / "validation_predictions.jsonl"
    predictions_path = run_directory / "test_predictions.jsonl"

    artifact_type = (
        "multivariate_multirocket_hydra_classifier"
        if config.model.use_hydra
        else "multivariate_multirocket_classifier"
    )
    artifact = {
        "artifact_version": ARTIFACT_VERSION,
        "artifact_type": artifact_type,
        "pipeline": pipeline,
        "experiment_config": _json_ready(asdict(config)),
        "normalizer": (
            None if data.normalizer is None else data.normalizer.state_dict()
        ),
        "time_grid": data.time_grid,
        "channel_names": data.channel_names,
        "hydra_enabled": config.model.use_hydra,
        "best_alpha": best_alpha,
        "num_transformed_features": total_features,
        "num_multirocket_features": multirocket_features,
        "num_hydra_features": hydra_features,
        "decision_threshold": decision_threshold,
        "validation_metrics": validation_metrics,
        "test_metrics": test_metrics,
    }
    joblib.dump(artifact, checkpoint_path)

    metrics_payload = {
        "best_alpha": best_alpha,
        "hydra_enabled": config.model.use_hydra,
        "num_input_channels": len(data.channel_names),
        "num_time_points": config.data.num_time_points,
        "num_transformed_features": total_features,
        "num_multirocket_features": multirocket_features,
        "num_hydra_features": hydra_features,
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
        num_transformed_features=total_features,
        hydra_enabled=config.model.use_hydra,
        num_multirocket_features=multirocket_features,
        num_hydra_features=hydra_features,
        history=[],
    )


def load_trained_model(checkpoint_path: str | Path) -> dict[str, Any]:
    """Load a fitted MultiRocket pipeline and preprocessing metadata."""

    artifact = joblib.load(checkpoint_path)
    version = artifact.get("artifact_version")
    if version != ARTIFACT_VERSION:
        raise ValueError(
            f"Unsupported MultiRocket artifact version {version!r}; "
            f"expected {ARTIFACT_VERSION}"
        )
    return artifact


def load_training_result(config: ExperimentConfig) -> TrainingResult:
    """Load a complete, configuration-matched MultiRocket run from disk."""

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
        raise FileNotFoundError(f"Incomplete MultiRocket run; missing: {missing}")

    artifact = load_trained_model(checkpoint_path)
    if artifact.get("experiment_config") != _json_ready(asdict(config)):
        raise ValueError(
            f"Saved MultiRocket run {run_directory} was created with a different "
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
        hydra_enabled=bool(artifact["hydra_enabled"]),
        num_multirocket_features=int(artifact["num_multirocket_features"]),
        num_hydra_features=int(artifact["num_hydra_features"]),
        history=[],
    )
