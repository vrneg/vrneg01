"""Semantic TabPFN-3 experts and leakage-safe OOF ensemble weighting."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
from scipy.optimize import minimize
from sklearn.model_selection import StratifiedKFold

from .config import AdaptiveRocketPFNConfig
from .features import AdaptiveCandidateBank, EXPERT_NAMES
from .selection import (
    ExpertSelection,
    assemble_expert_matrix,
    select_all_experts,
)


class TabPFNAccessError(RuntimeError):
    """Raised before feature generation when TabPFN-3 weights are unavailable."""


def _load_tabpfn_classes() -> tuple[type, type]:
    try:
        from tabpfn import TabPFNClassifier
        from tabpfn.constants import ModelVersion
    except ImportError as error:
        raise ImportError(
            "Adaptive RocketPFN requires TabPFN. Install project requirements "
            "with `pip install -r requirements.txt`."
        ) from error
    return TabPFNClassifier, ModelVersion


def new_tabpfn_classifier(
    model: AdaptiveRocketPFNConfig,
    seed: int,
    *,
    n_estimators: int | None = None,
) -> Any:
    """Create an explicitly versioned TabPFN-3 semantic expert."""

    TabPFNClassifier, ModelVersion = _load_tabpfn_classes()
    overrides: dict[str, Any] = {
        "n_estimators": (
            model.tabpfn_n_estimators if n_estimators is None else n_estimators
        ),
        "auto_scale_n_estimators": model.tabpfn_auto_scale_n_estimators,
        "fit_mode": model.tabpfn_fit_mode,
        "device": model.tabpfn_device,
        "memory_saving_mode": model.tabpfn_memory_saving_mode,
        "inference_precision": model.tabpfn_inference_precision,
        "n_preprocessing_jobs": model.tabpfn_preprocessing_jobs,
        "tuning_config": None,
        "show_progress_bar": model.tabpfn_show_progress_bar,
        "ignore_pretraining_limits": model.tabpfn_ignore_pretraining_limits,
        "balance_probabilities": model.tabpfn_balance_probabilities,
        "random_state": seed,
    }
    if model.tabpfn_model_path is not None:
        overrides["model_path"] = str(model.tabpfn_model_path)
    return TabPFNClassifier.create_default_for_version(ModelVersion.V3, **overrides)


def ensure_tabpfn_checkpoint_access(
    model: AdaptiveRocketPFNConfig,
    seed: int,
) -> None:
    classifier = new_tabpfn_classifier(model, seed)
    configured_paths = classifier.model_path
    paths = configured_paths if isinstance(configured_paths, list) else [configured_paths]
    if paths and all(Path(path).is_file() for path in paths):
        return

    try:
        from tabpfn.browser_auth import ensure_license_accepted
        from tabpfn.errors import TabPFNError
        from tabpfn.model_loading import ModelSource
    except ImportError as error:
        raise ImportError("Adaptive RocketPFN requires TabPFN.") from error
    repository = ModelSource.get_classifier_v3().repo_id.rsplit("/", 1)[-1]
    try:
        ensure_license_accepted(hf_repo_id=repository)
    except TabPFNError as error:
        raise TabPFNAccessError(
            "TabPFN v3 weights are unavailable. Accept the license at "
            "https://ux.priorlabs.ai, put TABPFN_TOKEN=<your-api-key> in the "
            "project-root .env, and restart. Do not commit or share the token."
        ) from error


def positive_class_probabilities(classifier: Any, features: np.ndarray) -> np.ndarray:
    probabilities = np.asarray(classifier.predict_proba(features), dtype=np.float64)
    classes = np.asarray(classifier.classes_)
    positive_columns = np.flatnonzero(classes == 1)
    if positive_columns.size != 1:
        raise RuntimeError(
            "TabPFN classes must contain positive label 1, got "
            f"{classes.tolist()}"
        )
    expected_shape = (features.shape[0], classes.size)
    if probabilities.shape != expected_shape:
        raise RuntimeError(
            f"TabPFN returned shape {probabilities.shape}; expected {expected_shape}"
        )
    positive = probabilities[:, int(positive_columns[0])]
    if not np.all(np.isfinite(positive)):
        raise RuntimeError("TabPFN returned non-finite probabilities")
    return np.clip(positive, 0.0, 1.0)


def learn_simplex_weights(
    oof_probabilities: np.ndarray,
    labels: np.ndarray,
    l2_regularization: float,
) -> np.ndarray:
    """Minimize regularized OOF log loss under nonnegative sum-to-one weights."""

    probabilities = np.asarray(oof_probabilities, dtype=np.float64)
    targets = np.asarray(labels, dtype=np.float64)
    if probabilities.ndim != 2 or probabilities.shape[0] != targets.size:
        raise ValueError("OOF probability matrix and labels do not align")
    if not np.all(np.isfinite(probabilities)):
        raise ValueError("OOF probabilities must be finite")
    num_experts = probabilities.shape[1]
    uniform = np.full(num_experts, 1.0 / num_experts, dtype=np.float64)

    def objective(weights: np.ndarray) -> float:
        combined = np.clip(probabilities @ weights, 1e-7, 1.0 - 1e-7)
        log_loss = -np.mean(
            targets * np.log(combined) + (1.0 - targets) * np.log1p(-combined)
        )
        penalty = l2_regularization * np.sum((weights - uniform) ** 2)
        return float(log_loss + penalty)

    result = minimize(
        objective,
        uniform,
        method="SLSQP",
        bounds=[(0.0, 1.0)] * num_experts,
        constraints={"type": "eq", "fun": lambda weights: weights.sum() - 1.0},
        options={"maxiter": 500, "ftol": 1e-10},
    )
    if not result.success or not np.all(np.isfinite(result.x)):
        raise RuntimeError(f"OOF ensemble weight optimization failed: {result.message}")
    weights = np.clip(np.asarray(result.x, dtype=np.float64), 0.0, 1.0)
    return weights / weights.sum()


def oof_semantic_predictions(
    train_values: np.ndarray,
    labels: np.ndarray,
    model: AdaptiveRocketPFNConfig,
    seed: int,
    log: Callable[[str], None] | None = None,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Produce OOF predictions with fold-fitted banks and feature selection."""

    classes, counts = np.unique(labels, return_counts=True)
    if classes.size != 2:
        raise ValueError("Semantic TabPFN ensemble requires binary labels")
    n_splits = min(model.oof_folds, int(counts.min()))
    if n_splits < 2:
        raise ValueError("Not enough examples per class for OOF weighting")
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof = np.full((len(labels), len(EXPERT_NAMES)), np.nan, dtype=np.float64)
    diagnostics: list[dict[str, Any]] = []

    for fold_index, (train_indices, validation_indices) in enumerate(
        splitter.split(np.zeros(len(labels)), labels), start=1
    ):
        if log is not None:
            log(
                f"OOF fold {fold_index}/{n_splits}: fitting a candidate bank on "
                f"{len(train_indices):,} inner-training examples ..."
            )
        fold_bank = AdaptiveCandidateBank(
            model,
            seed + 100_000 * fold_index,
        )
        if log is not None:
            fold_bank.progress_callback_ = lambda message: log(
                f"OOF fold {fold_index}/{n_splits}: {message}"
            )
        fold_train_candidates = fold_bank.fit_transform(
            train_values[train_indices], None
        )
        fold_bank.progress_callback_ = None
        fold_validation_candidates = fold_bank.transform(
            train_values[validation_indices]
        )
        selections = select_all_experts(
            fold_train_candidates,
            labels[train_indices],
            np.arange(len(train_indices), dtype=np.int64),
            fold_bank.expert_families_,
            model,
            seed + 100 * fold_index,
        )
        fold_diagnostic: dict[str, Any] = {
            "fold": fold_index,
            "num_candidate_features": fold_bank.n_candidate_features_,
            "num_families": len(fold_bank.family_specs_),
            "experts": {},
        }
        for expert_index, expert in enumerate(EXPERT_NAMES):
            selection = selections[expert]
            train_features = assemble_expert_matrix(
                fold_train_candidates, selection
            )
            validation_features = assemble_expert_matrix(
                fold_validation_candidates, selection
            )
            if log is not None:
                log(
                    f"OOF fold {fold_index}/{n_splits}, {expert}: "
                    f"TabPFN-3 on {train_features.shape[1]:,} selected features ..."
                )
            classifier = new_tabpfn_classifier(
                model,
                seed + 1_000 * fold_index + expert_index,
                n_estimators=model.oof_tabpfn_n_estimators,
            )
            classifier.fit(train_features, labels[train_indices])
            oof[validation_indices, expert_index] = positive_class_probabilities(
                classifier, validation_features
            )
            fold_diagnostic["experts"][expert] = {
                "num_features": selection.num_features,
                "family_budgets": dict(selection.family_budgets),
                "selected_counts": {
                    name: int(indices.size)
                    for name, indices in selection.family_indices.items()
                },
            }
        diagnostics.append(fold_diagnostic)
    if not np.all(np.isfinite(oof)):
        raise RuntimeError("OOF prediction matrix is incomplete")
    return oof, diagnostics


def predict_semantic_experts(
    train_candidates: dict[str, np.ndarray],
    query_candidates: dict[str, np.ndarray],
    labels: np.ndarray,
    selections: dict[str, ExpertSelection],
    model: AdaptiveRocketPFNConfig,
    seed: int,
    log: Callable[[str], None] | None = None,
) -> np.ndarray:
    probabilities = np.empty(
        (next(iter(query_candidates.values())).shape[0], len(EXPERT_NAMES)),
        dtype=np.float64,
    )
    for expert_index, expert in enumerate(EXPERT_NAMES):
        train_features = assemble_expert_matrix(
            train_candidates, selections[expert]
        )
        query_features = assemble_expert_matrix(
            query_candidates, selections[expert]
        )
        if log is not None:
            log(
                f"Final {expert} expert: fitting TabPFN-3 on "
                f"{train_features.shape[1]:,} selected features ..."
            )
        classifier = new_tabpfn_classifier(
            model,
            seed + expert_index,
            n_estimators=model.tabpfn_n_estimators,
        )
        classifier.fit(train_features, labels)
        probabilities[:, expert_index] = positive_class_probabilities(
            classifier, query_features
        )
    return probabilities


@dataclass(slots=True)
class FittedAdaptiveRocketPFN:
    """Fitted candidate bank, stable selections, and OOF ensemble weights."""

    model_config: AdaptiveRocketPFNConfig
    seed: int
    train_values: np.ndarray
    train_labels: np.ndarray
    candidate_bank: AdaptiveCandidateBank
    selections: dict[str, ExpertSelection]
    ensemble_weights: np.ndarray

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        train_candidates = self.candidate_bank.transform(self.train_values)
        query_candidates = self.candidate_bank.transform(X)
        expert_probabilities = predict_semantic_experts(
            train_candidates,
            query_candidates,
            self.train_labels,
            self.selections,
            self.model_config,
            self.seed,
        )
        positive = expert_probabilities @ self.ensemble_weights
        return np.column_stack((1.0 - positive, positive))

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= threshold).astype(np.int64)


__all__ = [
    "FittedAdaptiveRocketPFN",
    "TabPFNAccessError",
    "ensure_tabpfn_checkpoint_access",
    "learn_simplex_weights",
    "new_tabpfn_classifier",
    "oof_semantic_predictions",
    "positive_class_probabilities",
    "predict_semantic_experts",
]
