"""Channel-bounded wrapper around the official multivariate MrSQM transformer."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.utils.validation import check_is_fitted

try:
    from weasel_v2.features import SupervisedChannelSelector
except ModuleNotFoundError as error:
    if error.name != "weasel_v2":
        raise
    from ..weasel_v2.features import SupervisedChannelSelector

from .config import ExperimentConfig


def load_mrsqm_transformer_class() -> type:
    """Load the official optional MrSQM transformer dependency."""

    try:
        from mrsqm import MrSQMTransformer
    except ImportError as error:
        raise ImportError(
            "MrSQM experiments require the optional `mrsqm==0.0.7` package. "
            "Install the project requirements with `pip install -r requirements.txt`."
        ) from error
    return MrSQMTransformer


class MultivariateMrSQMClassifier(ClassifierMixin, BaseEstimator):
    """Fit official MrSQM RS features followed by compatible logistic regression."""

    def __init__(
        self,
        strategy: str = "RS",
        features_per_representation: int = 500,
        selection_per_representation: int = 2_000,
        num_sax_representations: int = 0,
        num_sfa_representations: int = 5,
        sfa_normalize: bool = True,
        use_first_difference: bool = True,
        max_channels: int = 8,
        channel_score_epsilon: float = 1e-8,
        logistic_c: float = 1.0,
        logistic_solver: str = "newton-cg",
        logistic_max_iterations: int = 1_000,
        class_weight: str | None = "balanced",
        random_state: int | None = None,
    ):
        self.strategy = strategy
        self.features_per_representation = features_per_representation
        self.selection_per_representation = selection_per_representation
        self.num_sax_representations = num_sax_representations
        self.num_sfa_representations = num_sfa_representations
        self.sfa_normalize = sfa_normalize
        self.use_first_difference = use_first_difference
        self.max_channels = max_channels
        self.channel_score_epsilon = channel_score_epsilon
        self.logistic_c = logistic_c
        self.logistic_solver = logistic_solver
        self.logistic_max_iterations = logistic_max_iterations
        self.class_weight = class_weight
        self.random_state = random_state

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> MultivariateMrSQMClassifier:
        MrSQMTransformer = load_mrsqm_transformer_class()
        labels = np.asarray(y)
        self.channel_selector_ = SupervisedChannelSelector(
            max_channels=self.max_channels,
            epsilon=self.channel_score_epsilon,
        )
        selected_values = self.channel_selector_.fit_transform(X, labels)
        self.transformer_ = MrSQMTransformer(
            strat=self.strategy,
            features_per_rep=self.features_per_representation,
            selection_per_rep=self.selection_per_representation,
            nsax=self.num_sax_representations,
            nsfa=self.num_sfa_representations,
            sfa_norm=self.sfa_normalize,
            first_diff=self.use_first_difference,
            random_state=self.random_state,
        )
        symbolic_features = csr_matrix(
            self.transformer_.fit_transform(selected_values, labels),
            dtype=np.float64,
        )
        if symbolic_features.shape[1] == 0:
            raise RuntimeError("MrSQM selected no symbolic features")
        self.classifier_ = LogisticRegression(
            C=self.logistic_c,
            solver=self.logistic_solver,
            class_weight=self.class_weight,
            max_iter=self.logistic_max_iterations,
            random_state=self.random_state,
        )
        self.classifier_.fit(symbolic_features, labels)
        self.classes_ = self.classifier_.classes_
        self.n_symbolic_features_ = int(symbolic_features.shape[1])
        self.n_selected_channels_ = int(selected_values.shape[1])
        return self

    def _transform(self, X: np.ndarray) -> csr_matrix:
        check_is_fitted(
            self,
            ("channel_selector_", "transformer_", "classifier_", "classes_"),
        )
        selected_values = self.channel_selector_.transform(X)
        transformed = self.transformer_.transform(selected_values)
        return csr_matrix(transformed, dtype=np.float64)

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(self.classifier_.decision_function(self._transform(X)))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(self.classifier_.predict_proba(self._transform(X)))

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(self.classifier_.predict(self._transform(X)))


def build_classifier(config: ExperimentConfig) -> MultivariateMrSQMClassifier:
    """Build the configured MrSQM estimator without fitting it."""

    model = config.model
    return MultivariateMrSQMClassifier(
        strategy=model.strategy,
        features_per_representation=model.features_per_representation,
        selection_per_representation=model.selection_per_representation,
        num_sax_representations=model.num_sax_representations,
        num_sfa_representations=model.num_sfa_representations,
        sfa_normalize=model.sfa_normalize,
        use_first_difference=model.use_first_difference,
        max_channels=model.max_channels,
        channel_score_epsilon=model.channel_score_epsilon,
        logistic_c=model.logistic_c,
        logistic_solver=model.logistic_solver,
        logistic_max_iterations=model.logistic_max_iterations,
        class_weight=model.class_weight,
        random_state=config.seed,
    )


__all__ = [
    "MultivariateMrSQMClassifier",
    "build_classifier",
    "load_mrsqm_transformer_class",
]
