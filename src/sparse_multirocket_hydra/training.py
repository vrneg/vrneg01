"""Train sparse MultiRocket-HYDRA with leakage-safe inner model selection."""

from __future__ import annotations

import json
import random
import tempfile
import time
import warnings
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import torch
from joblib import Memory
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import (
    LogisticRegression,
    RidgeClassifier,
    SGDClassifier,
)
from sklearn.model_selection import GridSearchCV, ParameterGrid, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
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

from .config import ClassifierName, ExperimentConfig
from .data import DataBundle, TimeSeriesSplit, prepare_data
from .features import MultiRocketHydraRawFeatures, MultiRocketHydraStandardizer


ARTIFACT_VERSION = 1
_SCORING_NAMES = {
    "accuracy": "accuracy",
    "balanced_accuracy": "balanced_accuracy",
    "macro_f1": "f1_macro",
    "roc_auc": "roc_auc",
    "average_precision": "average_precision",
}


def _stable_f_classif(
    features: np.ndarray,
    labels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute ANOVA scores without flooding runs for constant ROCKET features."""

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        scores, p_values = f_classif(features, labels)
    # Constant features have undefined scores and should always sort last.
    scores = np.nan_to_num(scores, nan=-np.inf)
    return scores, p_values


@dataclass(slots=True)
class FittedSparseMultiRocketHydra:
    """Fitted convolutional feature map plus selected linear classifier."""

    feature_transformer: MultiRocketHydraRawFeatures
    model_search: GridSearchCV

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        raw_features = self.feature_transformer.transform(X)
        scores = self.model_search.decision_function(raw_features)
        return np.asarray(scores, dtype=np.float64)

    def predict(self, X: np.ndarray) -> np.ndarray:
        raw_features = self.feature_transformer.transform(X)
        return np.asarray(self.model_search.predict(raw_features), dtype=np.int64)


@dataclass(slots=True)
class TrainingResult:
    """Artifacts, metrics, and sparsity diagnostics for one outer fold."""

    checkpoint_path: Path
    history_path: Path
    validation_predictions_path: Path
    predictions_path: Path
    run_name: str
    best_epoch: int
    decision_threshold: float
    validation_metrics: dict[str, float]
    test_metrics: dict[str, float]
    best_classifier: str
    best_cv_score: float
    best_hyperparameters: dict[str, Any]
    classifier_leaderboard: list[dict[str, Any]]
    num_transformed_features: int
    num_multirocket_features: int
    num_hydra_features: int
    num_selected_features: int
    num_selected_multirocket_features: int
    num_selected_hydra_features: int
    num_nonzero_coefficients: int
    num_nonzero_multirocket_coefficients: int
    num_nonzero_hydra_coefficients: int
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


def _build_feature_transformer(config: ExperimentConfig) -> MultiRocketHydraRawFeatures:
    model = config.model
    return MultiRocketHydraRawFeatures(
        num_kernels=model.num_kernels,
        max_dilations_per_kernel=model.max_dilations_per_kernel,
        num_features_per_kernel=model.num_features_per_kernel,
        normalise_per_instance=model.normalise_per_instance,
        hydra_num_kernels=model.hydra_num_kernels,
        hydra_num_groups=model.hydra_num_groups,
        hydra_max_num_channels=model.hydra_max_num_channels,
        n_jobs=model.n_jobs,
        random_state=config.seed,
    )


def _load_reusable_feature_transformer(
    config: ExperimentConfig,
) -> MultiRocketHydraRawFeatures | None:
    """Load compatible fitted random transforms from the baseline artifact."""

    checkpoint_path = config.feature_artifact_path
    if checkpoint_path is None:
        return None
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            "Configured MultiRocket+HYDRA feature artifact does not exist: "
            f"{checkpoint_path}. Run the baseline first or set "
            "feature_artifact_path=None."
        )
    artifact = joblib.load(checkpoint_path)
    if artifact.get("artifact_type") != "multivariate_multirocket_hydra_classifier":
        raise ValueError(
            f"Feature artifact {checkpoint_path} is not a MultiRocket+HYDRA model"
        )
    saved_config = artifact.get("experiment_config", {})
    if saved_config.get("data") != _json_ready(asdict(config.data)):
        raise ValueError(
            f"Feature artifact {checkpoint_path} uses a different data configuration"
        )
    if saved_config.get("seed") != config.seed:
        raise ValueError(
            f"Feature artifact {checkpoint_path} uses a different random seed"
        )
    saved_model = saved_config.get("model", {})
    expected_settings = {
        "num_kernels": config.model.num_kernels,
        "max_dilations_per_kernel": config.model.max_dilations_per_kernel,
        "num_features_per_kernel": config.model.num_features_per_kernel,
        "normalise_per_instance": config.model.normalise_per_instance,
        "hydra_num_kernels": config.model.hydra_num_kernels,
        "hydra_num_groups": config.model.hydra_num_groups,
        "hydra_max_num_channels": config.model.hydra_max_num_channels,
    }
    mismatches = {
        key: (saved_model.get(key), value)
        for key, value in expected_settings.items()
        if saved_model.get(key) != value
    }
    if not saved_model.get("use_hydra") or mismatches:
        raise ValueError(
            f"Feature artifact {checkpoint_path} has incompatible transform settings: "
            f"{mismatches}"
        )

    source_transformer = artifact["pipeline"].steps[0][1]
    required_attributes = (
        "multirocket_",
        "hydra_",
        "n_multirocket_features_",
        "n_hydra_features_",
    )
    missing = [
        name for name in required_attributes if not hasattr(source_transformer, name)
    ]
    if missing:
        raise ValueError(
            f"Feature artifact {checkpoint_path} lacks fitted attributes: {missing}"
        )

    transformer = _build_feature_transformer(config)
    for name in required_attributes:
        setattr(transformer, name, getattr(source_transformer, name))
    return transformer


def _new_classifier(name: ClassifierName, config: ExperimentConfig) -> Any:
    model = config.model
    if name == "ridge":
        return RidgeClassifier(
            class_weight=model.class_weight,
            solver="lsqr",
            tol=model.tolerance,
        )
    if name == "logistic_l2":
        return LogisticRegression(
            solver="liblinear",
            l1_ratio=0.0,
            class_weight=model.class_weight,
            max_iter=model.max_iter,
            tol=model.tolerance,
            random_state=config.seed,
        )
    if name == "elastic_net":
        return SGDClassifier(
            loss="log_loss",
            penalty="elasticnet",
            alpha=1e-2,
            l1_ratio=0.5,
            class_weight=model.class_weight,
            max_iter=model.max_iter,
            tol=model.tolerance,
            random_state=config.seed,
        )
    if name == "linear_svm":
        return LinearSVC(
            dual="auto",
            class_weight=model.class_weight,
            max_iter=model.max_iter,
            tol=model.tolerance,
            random_state=config.seed,
        )
    raise ValueError(f"Unsupported classifier {name!r}")


def _effective_feature_counts(
    requested_counts: tuple[int, ...],
    num_available_features: int,
) -> tuple[int, ...]:
    if num_available_features < 1:
        raise ValueError("num_available_features must be positive")
    return tuple(
        sorted({min(int(count), num_available_features) for count in requested_counts})
    )


def _parameter_grid(
    config: ExperimentConfig,
    effective_feature_counts: tuple[int, ...],
) -> list[dict[str, Any]]:
    model = config.model
    grids: list[dict[str, Any]] = []
    for name in model.classifier_candidates:
        common: dict[str, Any] = {
            "feature_selection__k": effective_feature_counts,
            "classifier": [_new_classifier(name, config)],
        }
        if name == "ridge":
            common["classifier__alpha"] = model.ridge_alphas
        elif name == "logistic_l2":
            common["classifier__C"] = model.logistic_l2_cs
        elif name == "elastic_net":
            common["classifier__alpha"] = model.elastic_net_alphas
            common["classifier__l1_ratio"] = model.elastic_net_l1_ratios
        elif name == "linear_svm":
            common["classifier__C"] = model.linear_svm_cs
        grids.append(common)
    return grids


def _build_model_search(
    config: ExperimentConfig,
    num_multirocket_features: int,
    num_available_features: int,
    memory: Memory | None = None,
) -> GridSearchCV:
    """Build the inner-CV selector/classifier comparison.

    Scaling and SelectKBest live inside this pipeline, so both are refitted on each
    inner training split. The validation and test splits supplied to the outer fold
    are never passed to GridSearchCV.
    """

    effective_counts = _effective_feature_counts(
        config.model.feature_counts, num_available_features
    )
    first_classifier = _new_classifier(config.model.classifier_candidates[0], config)
    pipeline = Pipeline(
        steps=[
            (
                "standardize",
                MultiRocketHydraStandardizer(num_multirocket_features),
            ),
            (
                "feature_selection",
                SelectKBest(score_func=_stable_f_classif, k=effective_counts[0]),
            ),
            ("classifier", first_classifier),
        ],
        memory=memory,
    )
    inner_cv = StratifiedKFold(
        n_splits=config.model.inner_cv_folds,
        shuffle=True,
        random_state=config.seed,
    )
    return GridSearchCV(
        estimator=pipeline,
        param_grid=_parameter_grid(config, effective_counts),
        scoring=_SCORING_NAMES[config.model.inner_scoring],
        cv=inner_cv,
        refit=True,
        n_jobs=config.model.search_n_jobs,
        pre_dispatch=config.model.search_pre_dispatch,
        verbose=config.model.search_verbose,
        error_score="raise",
        return_train_score=False,
    )


def _classifier_name(estimator: Any) -> str:
    if isinstance(estimator, RidgeClassifier):
        return "ridge"
    if isinstance(estimator, LogisticRegression):
        return "logistic_l2"
    if isinstance(estimator, SGDClassifier):
        if estimator.loss == "log_loss" and estimator.penalty == "elasticnet":
            return "elastic_net"
    if isinstance(estimator, LinearSVC):
        return "linear_svm"
    raise TypeError(f"Unsupported fitted classifier type: {type(estimator).__name__}")


def _clean_hyperparameters(parameters: dict[str, Any]) -> dict[str, Any]:
    classifier = parameters.get("classifier")
    cleaned: dict[str, Any] = {
        "classifier": _classifier_name(classifier) if classifier is not None else None
    }
    for key, value in parameters.items():
        if key == "classifier":
            continue
        if key == "feature_selection__k":
            cleaned["num_selected_features"] = int(value)
        elif key.startswith("classifier__"):
            cleaned[key.removeprefix("classifier__")] = _json_ready(value)
        else:
            cleaned[key] = _json_ready(value)
    return cleaned


def _classifier_leaderboard(search: GridSearchCV) -> list[dict[str, Any]]:
    results = search.cv_results_
    best_by_classifier: dict[str, dict[str, Any]] = {}
    for index, parameters in enumerate(results["params"]):
        classifier = _classifier_name(parameters["classifier"])
        score = float(results["mean_test_score"][index])
        row = {
            "classifier": classifier,
            "mean_inner_cv_score": score,
            "standard_deviation": float(results["std_test_score"][index]),
            "global_rank": int(results["rank_test_score"][index]),
            "hyperparameters": _clean_hyperparameters(parameters),
        }
        previous = best_by_classifier.get(classifier)
        if previous is None or score > previous["mean_inner_cv_score"]:
            best_by_classifier[classifier] = row
    return sorted(best_by_classifier.values(), key=lambda row: row["global_rank"])


def _validate_inner_cv_labels(labels: np.ndarray, inner_cv_folds: int) -> None:
    classes, counts = np.unique(labels, return_counts=True)
    if classes.size != 2:
        raise ValueError(
            f"Sparse MultiRocket-HYDRA requires two training classes, got {classes.tolist()}"
        )
    if int(counts.min()) < inner_cv_folds:
        raise ValueError(
            "Each training class needs at least inner_cv_folds samples; "
            f"smallest class has {int(counts.min())}, requested {inner_cv_folds} folds"
        )


def _selection_diagnostics(
    search: GridSearchCV,
    num_multirocket_features: int,
) -> dict[str, int]:
    best_pipeline = search.best_estimator_
    selector = best_pipeline.named_steps["feature_selection"]
    selected_indices = np.flatnonzero(selector.get_support())
    classifier = best_pipeline.named_steps["classifier"]
    coefficients = np.asarray(classifier.coef_)
    coefficients = coefficients.reshape(-1, selected_indices.size)
    nonzero_selected_mask = np.any(coefficients != 0.0, axis=0)
    nonzero_indices = selected_indices[nonzero_selected_mask]
    return {
        "num_selected_features": int(selected_indices.size),
        "num_selected_multirocket_features": int(
            np.count_nonzero(selected_indices < num_multirocket_features)
        ),
        "num_selected_hydra_features": int(
            np.count_nonzero(selected_indices >= num_multirocket_features)
        ),
        "num_nonzero_coefficients": int(nonzero_indices.size),
        "num_nonzero_multirocket_coefficients": int(
            np.count_nonzero(nonzero_indices < num_multirocket_features)
        ),
        "num_nonzero_hydra_coefficients": int(
            np.count_nonzero(nonzero_indices >= num_multirocket_features)
        ),
    }


def _decision_scores(
    model: FittedSparseMultiRocketHydra,
    split: TimeSeriesSplit,
) -> np.ndarray:
    scores = np.asarray(model.decision_function(split.values), dtype=np.float64)
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


def train_sparse_multirocket_hydra(config: ExperimentConfig) -> TrainingResult:
    """Fit transforms, tune sparse linear models on train, and evaluate once."""

    config.validate()
    total_start = time.monotonic()
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    _log(config, f"Loading and encoding dataset from {config.data.dataset_path} ...")
    stage_start = time.monotonic()
    data: DataBundle = prepare_data(config.data)
    _log(
        config,
        "Dataset ready in "
        f"{time.monotonic() - stage_start:.1f}s "
        f"(train={len(data.train.labels)}, validation={len(data.validation.labels)}, "
        f"test={len(data.test.labels)}, channels={len(data.channel_names)}).",
    )
    _validate_inner_cv_labels(data.train.labels, config.model.inner_cv_folds)

    stage_start = time.monotonic()
    feature_transformer = _load_reusable_feature_transformer(config)
    if feature_transformer is None:
        _log(
            config,
            "Fitting fresh MultiRocket+HYDRA transforms. This is the expensive "
            "path; configure feature_artifact_path to reuse the baseline.",
        )
        feature_transformer = _build_feature_transformer(config)
        raw_train_features = feature_transformer.fit_transform(
            data.train.values, data.train.labels
        )
        feature_source = "fresh_fit"
    else:
        _log(
            config,
            f"Reusing fitted transforms from {config.feature_artifact_path} ...",
        )
        raw_train_features = feature_transformer.transform(data.train.values)
        feature_source = str(config.feature_artifact_path)
    num_multirocket_features = feature_transformer.n_multirocket_features_
    num_hydra_features = feature_transformer.n_hydra_features_
    num_transformed_features = int(raw_train_features.shape[1])
    if num_multirocket_features + num_hydra_features != num_transformed_features:
        raise RuntimeError("MultiRocket and HYDRA feature counts do not match output")
    _log(
        config,
        f"Generated {num_transformed_features:,} training features in "
        f"{time.monotonic() - stage_start:.1f}s "
        f"({num_multirocket_features:,} MultiRocket + "
        f"{num_hydra_features:,} HYDRA).",
    )

    # Grid candidates share the same standardized matrices and top-k subsets.
    # A temporary pipeline cache avoids recomputing those O(n*p) steps for every
    # regularization value and is removed immediately after the refitted winner is
    # available.
    with tempfile.TemporaryDirectory(prefix="sparse-mrh-inner-cv-") as cache_dir:
        model_search = _build_model_search(
            config,
            num_multirocket_features=num_multirocket_features,
            num_available_features=num_transformed_features,
            memory=Memory(location=cache_dir, verbose=0),
        )
        # ParameterGrid computes the true Cartesian-product size per family.
        candidate_count = len(ParameterGrid(model_search.param_grid))
        fit_count = candidate_count * config.model.inner_cv_folds
        _log(
            config,
            f"Starting inner search: {candidate_count} candidates × "
            f"{config.model.inner_cv_folds} folds = {fit_count} fits, "
            f"workers={config.model.search_n_jobs}. Per-fit progress follows.",
        )
        stage_start = time.monotonic()
        model_search.fit(raw_train_features, data.train.labels)
    _log(
        config,
        f"Inner search finished in {time.monotonic() - stage_start:.1f}s; "
        f"best {config.model.inner_scoring}={model_search.best_score_:.4f}.",
    )
    fitted_model = FittedSparseMultiRocketHydra(feature_transformer, model_search)

    validation_scores = _decision_scores(fitted_model, data.validation)
    test_scores = _decision_scores(fitted_model, data.test)
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

    best_classifier = _classifier_name(
        model_search.best_estimator_.named_steps["classifier"]
    )
    best_hyperparameters = _clean_hyperparameters(model_search.best_params_)
    classifier_leaderboard = _classifier_leaderboard(model_search)
    diagnostics = _selection_diagnostics(model_search, num_multirocket_features)

    run_directory = config.output_dir / config.run_name
    run_directory.mkdir(parents=True, exist_ok=True)
    checkpoint_path = run_directory / "model.joblib"
    history_path = run_directory / "metrics.json"
    validation_predictions_path = run_directory / "validation_predictions.jsonl"
    predictions_path = run_directory / "test_predictions.jsonl"

    artifact = {
        "artifact_version": ARTIFACT_VERSION,
        "artifact_type": "sparse_multirocket_hydra_classifier",
        "model": fitted_model,
        "experiment_config": _json_ready(asdict(config)),
        "normalizer": (
            None if data.normalizer is None else data.normalizer.state_dict()
        ),
        "time_grid": data.time_grid,
        "channel_names": data.channel_names,
        "best_classifier": best_classifier,
        "best_cv_score": float(model_search.best_score_),
        "best_hyperparameters": best_hyperparameters,
        "classifier_leaderboard": classifier_leaderboard,
        "feature_source": feature_source,
        "num_transformed_features": num_transformed_features,
        "num_multirocket_features": num_multirocket_features,
        "num_hydra_features": num_hydra_features,
        **diagnostics,
        "decision_threshold": decision_threshold,
        "validation_metrics": validation_metrics,
        "test_metrics": test_metrics,
    }
    joblib.dump(artifact, checkpoint_path)

    metrics_payload = {
        "inner_cv_folds": config.model.inner_cv_folds,
        "inner_scoring": config.model.inner_scoring,
        "best_classifier": best_classifier,
        "best_cv_score": float(model_search.best_score_),
        "best_hyperparameters": best_hyperparameters,
        "classifier_leaderboard": classifier_leaderboard,
        "feature_source": feature_source,
        "num_input_channels": len(data.channel_names),
        "num_time_points": config.data.num_time_points,
        "num_transformed_features": num_transformed_features,
        "num_multirocket_features": num_multirocket_features,
        "num_hydra_features": num_hydra_features,
        **diagnostics,
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
        json.dumps(_json_ready(metrics_payload), indent=2, allow_nan=False) + "\n",
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
    _log(
        config,
        f"Finished fold in {time.monotonic() - total_start:.1f}s; "
        f"classifier={best_classifier}, selected={diagnostics['num_selected_features']:,}, "
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
        best_classifier=best_classifier,
        best_cv_score=float(model_search.best_score_),
        best_hyperparameters=best_hyperparameters,
        classifier_leaderboard=classifier_leaderboard,
        num_transformed_features=num_transformed_features,
        num_multirocket_features=num_multirocket_features,
        num_hydra_features=num_hydra_features,
        **diagnostics,
        history=[],
    )


def load_trained_model(checkpoint_path: str | Path) -> dict[str, Any]:
    """Load a fitted sparse MultiRocket-HYDRA artifact."""

    artifact = joblib.load(checkpoint_path)
    version = artifact.get("artifact_version")
    if version != ARTIFACT_VERSION:
        raise ValueError(
            f"Unsupported sparse MultiRocket-HYDRA artifact version {version!r}; "
            f"expected {ARTIFACT_VERSION}"
        )
    return artifact


def load_training_result(config: ExperimentConfig) -> TrainingResult:
    """Load a complete, configuration-matched sparse model run from disk."""

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
        raise FileNotFoundError(
            f"Incomplete sparse MultiRocket-HYDRA run; missing: {missing}"
        )

    artifact = load_trained_model(checkpoint_path)
    if artifact.get("experiment_config") != _json_ready(asdict(config)):
        raise ValueError(
            f"Saved sparse MultiRocket-HYDRA run {run_directory} was created "
            "with a different experiment configuration"
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
        best_classifier=str(artifact["best_classifier"]),
        best_cv_score=float(artifact["best_cv_score"]),
        best_hyperparameters=dict(artifact["best_hyperparameters"]),
        classifier_leaderboard=list(artifact["classifier_leaderboard"]),
        num_transformed_features=int(artifact["num_transformed_features"]),
        num_multirocket_features=int(artifact["num_multirocket_features"]),
        num_hydra_features=int(artifact["num_hydra_features"]),
        num_selected_features=int(artifact["num_selected_features"]),
        num_selected_multirocket_features=int(
            artifact["num_selected_multirocket_features"]
        ),
        num_selected_hydra_features=int(artifact["num_selected_hydra_features"]),
        num_nonzero_coefficients=int(artifact["num_nonzero_coefficients"]),
        num_nonzero_multirocket_coefficients=int(
            artifact["num_nonzero_multirocket_coefficients"]
        ),
        num_nonzero_hydra_coefficients=int(
            artifact["num_nonzero_hydra_coefficients"]
        ),
        history=[],
    )


__all__ = [
    "ARTIFACT_VERSION",
    "FittedSparseMultiRocketHydra",
    "TrainingResult",
    "load_trained_model",
    "load_training_result",
    "train_sparse_multirocket_hydra",
]
