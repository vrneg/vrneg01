"""Multivariate channel screening and WEASEL 2.0 dictionary features."""

from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np
from joblib import Parallel, delayed
from scipy.sparse import csr_matrix, hstack, issparse
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_random_state
from sklearn.utils.validation import check_is_fitted


def _as_3d_finite_array(X: np.ndarray) -> np.ndarray:
    values = np.asarray(X, dtype=np.float64)
    if values.ndim != 3:
        raise ValueError(
            "Symbolic multivariate input must have shape "
            "(samples, channels, time_points)"
        )
    if not np.isfinite(values).all():
        raise ValueError("Symbolic multivariate input must contain only finite values")
    return values


class SupervisedChannelSelector(TransformerMixin, BaseEstimator):
    """Select channels by training-only standardized class-trajectory separation."""

    def __init__(self, max_channels: int = 32, epsilon: float = 1e-8):
        self.max_channels = max_channels
        self.epsilon = epsilon

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> SupervisedChannelSelector:
        values = _as_3d_finite_array(X)
        labels = np.asarray(y)
        classes = np.unique(labels)
        if classes.size != 2:
            raise ValueError("channel screening currently requires binary labels")
        class_values = [values[labels == label] for label in classes]
        if any(group.shape[0] == 0 for group in class_values):
            raise ValueError("both classes need at least one training sample")

        mean_difference = class_values[1].mean(axis=0) - class_values[0].mean(axis=0)
        between_class = np.mean(np.square(mean_difference), axis=-1)
        within_class = np.mean(
            np.stack(
                [group.var(axis=0).mean(axis=-1) for group in class_values],
                axis=0,
            ),
            axis=0,
        )
        self.channel_scores_ = between_class / (within_class + self.epsilon)
        selected_count = min(int(self.max_channels), values.shape[1])
        ranking = np.argsort(-self.channel_scores_, kind="stable")
        self.selected_channel_indices_ = ranking[:selected_count].astype(np.int64)
        self.n_channels_in_ = int(values.shape[1])
        self.n_timesteps_in_ = int(values.shape[2])
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(
            self,
            ("selected_channel_indices_", "n_channels_in_", "n_timesteps_in_"),
        )
        values = _as_3d_finite_array(X)
        if values.shape[1:] != (self.n_channels_in_, self.n_timesteps_in_):
            raise ValueError("input channel/time shape differs from fitted data")
        return values[:, self.selected_channel_indices_, :]


def _fit_configuration(
    configuration_index: int,
    channel_index: int,
    X: np.ndarray,
    y: np.ndarray,
    window_sizes: np.ndarray,
    norm_options: tuple[bool, ...],
    word_lengths: tuple[int, ...],
    use_first_differences: tuple[bool, ...],
    feature_selection: str,
    features_per_configuration: int,
    n_timepoints: int,
    random_state: int | None,
) -> list[tuple[int, Any, Any]]:
    from aeon.transformations.collection.dictionary_based import SFAFast

    seed = None if random_state is None else random_state + configuration_index
    rng = check_random_state(seed)
    window_size = int(rng.choice(window_sizes))
    maximum_dilation = (n_timepoints - 1) / (window_size - 1)
    dilation = max(
        1,
        int(2 ** rng.uniform(0.0, np.log2(maximum_dilation))),
    )
    word_length = min(window_size - 2, int(rng.choice(word_lengths)))
    norm = bool(rng.choice(norm_options))
    binning_strategy = str(rng.choice(("equi-depth", "equi-width")))

    fitted: list[tuple[int, Any, Any]] = []
    channel_values = X[:, channel_index, :]
    for first_difference in use_first_differences:
        transformer = SFAFast(
            variance=True,
            word_length=word_length,
            alphabet_size=2,
            window_size=window_size,
            norm=norm,
            anova=False,
            binning_method=binning_strategy,
            remove_repeat_words=False,
            bigrams=False,
            dilation=dilation,
            lower_bounding=True,
            first_difference=first_difference,
            feature_selection=feature_selection,
            max_feature_count=features_per_configuration,
            random_state=configuration_index,
            return_sparse=True,
            n_jobs=1,
        )
        words = transformer.fit_transform(channel_values, y)
        fitted.append((channel_index, transformer, words))
    return fitted


class MultivariateWEASELTransformerV2(TransformerMixin, BaseEstimator):
    """Share WEASEL 2.0's randomized configuration budget across channels."""

    def __init__(
        self,
        min_window: int = 4,
        norm_options: tuple[bool, ...] = (False,),
        word_lengths: tuple[int, ...] = (7, 8),
        use_first_differences: tuple[bool, ...] = (True, False),
        feature_selection: str = "chi2_top_k",
        max_feature_count: int = 30_000,
        ensemble_size: int | None = None,
        random_state: int | None = None,
        n_jobs: int = 1,
    ):
        self.min_window = min_window
        self.norm_options = norm_options
        self.word_lengths = word_lengths
        self.use_first_differences = use_first_differences
        self.feature_selection = feature_selection
        self.max_feature_count = max_feature_count
        self.ensemble_size = ensemble_size
        self.random_state = random_state
        self.n_jobs = n_jobs

    @staticmethod
    def _automatic_size(n_cases: int, n_timepoints: int) -> tuple[int, int]:
        if n_cases < 250:
            return 50, 24
        if n_timepoints < 100:
            return 100, 44
        return 150, 84

    def _channel_schedule(self, n_channels: int, size: int) -> np.ndarray:
        rng = check_random_state(self.random_state)
        cycles = []
        remaining = size
        while remaining > 0:
            permutation = rng.permutation(n_channels)
            cycles.append(permutation[:remaining])
            remaining -= min(remaining, n_channels)
        return np.concatenate(cycles).astype(np.int64)

    def fit_transform(
        self,
        X: np.ndarray,
        y: np.ndarray | None = None,
        **fit_params: Any,
    ) -> csr_matrix:
        values = _as_3d_finite_array(X)
        if y is None:
            raise ValueError("WEASEL 2.0 feature fitting requires class labels")
        labels = np.asarray(y)
        automatic_size, automatic_max_window = self._automatic_size(
            values.shape[0], values.shape[2]
        )
        self.ensemble_size_ = (
            automatic_size if self.ensemble_size is None else int(self.ensemble_size)
        )
        self.max_window_ = min(values.shape[2], automatic_max_window)
        if self.min_window > self.max_window_:
            raise ValueError("min_window exceeds the available time-series length")
        window_sizes = np.arange(self.min_window, self.max_window_ + 1)
        channel_schedule = self._channel_schedule(
            values.shape[1], self.ensemble_size_
        )
        features_per_configuration = max(
            1, self.max_feature_count // self.ensemble_size_
        )

        fitted_groups = Parallel(n_jobs=self.n_jobs, prefer="threads")(
            delayed(_fit_configuration)(
                index,
                int(channel_schedule[index]),
                values,
                labels.copy(),
                window_sizes,
                self.norm_options,
                self.word_lengths,
                self.use_first_differences,
                self.feature_selection,
                features_per_configuration,
                values.shape[2],
                self.random_state,
            )
            for index in range(self.ensemble_size_)
        )
        flattened = [item for group in fitted_groups for item in group]
        self.channel_transformers_ = [
            (channel_index, transformer)
            for channel_index, transformer, _ in flattened
        ]
        matrices = [matrix for _, _, matrix in flattened]
        transformed = hstack(matrices, format="csr", dtype=np.float32)
        self.n_channels_in_ = int(values.shape[1])
        self.n_timesteps_in_ = int(values.shape[2])
        self.total_features_count_ = int(transformed.shape[1])
        self.channel_configuration_counts_ = dict(
            sorted(Counter(int(item) for item in channel_schedule).items())
        )
        return transformed

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray | None = None,
        **fit_params: Any,
    ) -> MultivariateWEASELTransformerV2:
        self.fit_transform(X, y, **fit_params)
        return self

    def transform(self, X: np.ndarray) -> csr_matrix:
        check_is_fitted(
            self,
            ("channel_transformers_", "n_channels_in_", "n_timesteps_in_"),
        )
        values = _as_3d_finite_array(X)
        if values.shape[1:] != (self.n_channels_in_, self.n_timesteps_in_):
            raise ValueError("input channel/time shape differs from fitted data")
        matrices = Parallel(n_jobs=self.n_jobs, prefer="threads")(
            delayed(transformer.transform)(values[:, channel_index, :])
            for channel_index, transformer in self.channel_transformers_
        )
        sparse_matrices = [
            matrix if issparse(matrix) else csr_matrix(matrix) for matrix in matrices
        ]
        return hstack(sparse_matrices, format="csr", dtype=np.float32)


__all__ = [
    "MultivariateWEASELTransformerV2",
    "SupervisedChannelSelector",
]
