"""Multivariate SelF-Rocket feature generation and wrapper selection."""

from __future__ import annotations

import multiprocessing
from typing import Any

import numpy as np
from numba import get_num_threads, njit, prange, set_num_threads
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import RidgeClassifierCV
from sklearn.metrics import accuracy_score
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted

from .config import SELECTION_RIDGE_ALPHAS


POOLING_OPERATORS: tuple[str, ...] = ("PPV", "ZC", "MPV", "MIPV", "LSPV")
CANDIDATE_NAMES: tuple[str, ...] = (
    "PPV",
    "ZC",
    "MPV",
    "MIPV",
    "LSPV",
    "PPV_DIFF",
    "ZC_DIFF",
    "MPV_DIFF",
    "MIPV_DIFF",
    "LSPV_DIFF",
    "PPV_MIX",
    "ZC_MIX",
    "MPV_MIX",
    "MIPV_MIX",
    "LSPV_MIX",
)


def candidate_names(only_mix: bool = False) -> tuple[str, ...]:
    """Return the paper's five MIX candidates or all fifteen candidates."""

    return CANDIDATE_NAMES[10:] if only_mix else CANDIDATE_NAMES


def highest_median_choice(
    performances: np.ndarray,
    names: tuple[str, ...],
) -> tuple[str, dict[str, float]]:
    """Choose the candidate with the highest median validation accuracy."""

    values = np.asarray(performances, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != len(names):
        raise ValueError("performances must have one column per candidate name")
    if values.shape[0] < 1 or not np.all(np.isfinite(values)):
        raise ValueError("performances must be a non-empty finite matrix")
    medians = np.median(values, axis=0)
    winner_index = int(np.argmax(medians))
    return names[winner_index], {
        name: float(score) for name, score in zip(names, medians, strict=True)
    }


def vote_validated_choice(
    performances: np.ndarray,
    names: tuple[str, ...],
    proposed: str,
    *,
    vote_top: int,
    vote_threshold: float,
    series_length: int,
    length_threshold: int,
) -> tuple[str, float, bool]:
    """Validate a median-vote winner and apply the paper's MIX fallback."""

    values = np.asarray(performances, dtype=np.float64)
    if proposed not in names:
        raise ValueError(f"Unknown proposed candidate: {proposed!r}")
    if values.ndim != 2 or values.shape[1] != len(names):
        raise ValueError("performances must have one column per candidate name")
    if not 1 <= vote_top <= len(names):
        raise ValueError("vote_top is outside the candidate range")

    proposed_index = names.index(proposed)
    top_thresholds = np.sort(values, axis=1)[:, -vote_top]
    support = float(np.mean(values[:, proposed_index] >= top_thresholds))
    if support >= vote_threshold:
        return proposed, support, False

    fallback = "ZC_MIX" if series_length >= length_threshold else "PPV_MIX"
    if fallback not in names:
        raise RuntimeError(f"Required fallback {fallback} is not a candidate")
    return fallback, support, True


def _load_multirocket_class() -> type:
    try:
        from aeon.transformations.collection.convolution_based import MultiRocket
    except ImportError as error:
        raise ImportError(
            "SelF-Rocket experiments require aeon. Install project requirements "
            "with `pip install -r requirements.txt`."
        ) from error
    return MultiRocket


@njit(inline="always")
def _zero_crossing_rate(values: np.ndarray, bias: float) -> float:
    crossings = 0
    for index in range(values.shape[0] - 1):
        left = values[index]
        right = values[index + 1]
        if (left > bias and right < bias) or (left < bias and right > bias):
            crossings += 1
    return crossings / values.shape[0]


# Do not persist these dispatchers to ``__pycache__``.  The project imports this
# module both as ``selfrocket.features`` (editable training scripts) and as
# ``src.selfrocket.features`` (tests/package imports).  Numba's on-disk cache is
# keyed by the source file, so one import style can otherwise load a serialized
# environment containing the other, unavailable module name.
@njit(fastmath=True, parallel=True)
def _zero_crossings_univariate(
    X: np.ndarray,
    parameters: tuple[np.ndarray, np.ndarray, np.ndarray],
    indices: np.ndarray,
) -> np.ndarray:
    n_cases, n_timepoints = X.shape
    dilations, n_features_per_dilation, biases = parameters
    n_kernels = len(indices)
    n_features = n_kernels * np.sum(n_features_per_dilation)
    features = np.zeros((n_cases, n_features), dtype=np.float32)

    for case_index in prange(n_cases):
        series = X[case_index]
        alpha = -series
        gamma = 3 * series
        feature_start = 0

        for dilation_index in range(len(dilations)):
            padding_selector = dilation_index % 2
            dilation = dilations[dilation_index]
            padding = (8 * dilation) // 2
            features_this_dilation = n_features_per_dilation[dilation_index]

            convolution_alpha = np.zeros(n_timepoints, dtype=np.float32)
            convolution_alpha[:] = alpha
            convolution_gamma = np.zeros((9, n_timepoints), dtype=np.float32)
            convolution_gamma[4] = gamma

            start = dilation
            end = n_timepoints - padding
            for gamma_index in range(4):
                convolution_alpha[-end:] += alpha[:end]
                convolution_gamma[gamma_index, -end:] = gamma[:end]
                end += dilation
            for gamma_index in range(5, 9):
                convolution_alpha[:-start] += alpha[start:]
                convolution_gamma[gamma_index, :-start] = gamma[start:]
                start += dilation

            for kernel_index in range(n_kernels):
                feature_end = feature_start + features_this_dilation
                index_0, index_1, index_2 = indices[kernel_index]
                convolution = (
                    convolution_alpha
                    + convolution_gamma[index_0]
                    + convolution_gamma[index_1]
                    + convolution_gamma[index_2]
                )
                if (padding_selector + kernel_index) % 2 == 1:
                    convolution = convolution[padding:-padding]
                for feature_index in range(feature_start, feature_end):
                    features[case_index, feature_index] = _zero_crossing_rate(
                        convolution, biases[feature_index]
                    )
                feature_start = feature_end

    return features


@njit(fastmath=True, parallel=True)
def _zero_crossings_multivariate(
    X: np.ndarray,
    parameters: tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
    ],
    indices: np.ndarray,
) -> np.ndarray:
    n_cases, n_channels, n_timepoints = X.shape
    (
        n_channels_per_combination,
        channel_indices,
        dilations,
        n_features_per_dilation,
        biases,
    ) = parameters
    n_kernels = len(indices)
    n_features = n_kernels * np.sum(n_features_per_dilation)
    features = np.zeros((n_cases, n_features), dtype=np.float32)

    for case_index in prange(n_cases):
        series = X[case_index]
        alpha = -series
        gamma = 3 * series
        feature_start = 0
        combination_index = 0
        channel_start = 0

        for dilation_index in range(len(dilations)):
            padding_selector = dilation_index % 2
            dilation = dilations[dilation_index]
            padding = (8 * dilation) // 2
            features_this_dilation = n_features_per_dilation[dilation_index]

            convolution_alpha = np.zeros(
                (n_channels, n_timepoints), dtype=np.float32
            )
            convolution_alpha[:] = alpha
            convolution_gamma = np.zeros(
                (9, n_channels, n_timepoints), dtype=np.float32
            )
            convolution_gamma[4] = gamma

            start = dilation
            end = n_timepoints - padding
            for gamma_index in range(4):
                convolution_alpha[:, -end:] += alpha[:, :end]
                convolution_gamma[gamma_index, :, -end:] = gamma[:, :end]
                end += dilation
            for gamma_index in range(5, 9):
                convolution_alpha[:, :-start] += alpha[:, start:]
                convolution_gamma[gamma_index, :, :-start] = gamma[:, start:]
                start += dilation

            for kernel_index in range(n_kernels):
                feature_end = feature_start + features_this_dilation
                channels_this_combination = n_channels_per_combination[
                    combination_index
                ]
                channel_end = channel_start + channels_this_combination
                channels = channel_indices[channel_start:channel_end]
                index_0, index_1, index_2 = indices[kernel_index]
                convolution = np.sum(
                    convolution_alpha[channels]
                    + convolution_gamma[index_0][channels]
                    + convolution_gamma[index_1][channels]
                    + convolution_gamma[index_2][channels],
                    axis=0,
                )
                if (padding_selector + kernel_index) % 2 == 1:
                    convolution = convolution[padding:-padding]
                for feature_index in range(feature_start, feature_end):
                    features[case_index, feature_index] = _zero_crossing_rate(
                        convolution, biases[feature_index]
                    )

                feature_start = feature_end
                combination_index += 1
                channel_start = channel_end

    return features


class SelFRocketFeatures(BaseEstimator, TransformerMixin):
    """Generate fifteen IR-PO candidates and select one with median voting."""

    def __init__(
        self,
        num_kernels: int = 10_000,
        max_dilations_per_kernel: int = 32,
        normalise_per_instance: bool = False,
        only_mix: bool = False,
        selection_num_folds: int = 2,
        selection_num_runs: int = 10,
        selection_num_features: int = 2_500,
        selection_max_samples: int = 500,
        selection_alphas: tuple[float, ...] = SELECTION_RIDGE_ALPHAS,
        vote_top: int = 5,
        vote_threshold: float = 0.9,
        length_threshold: int = 512,
        class_weight: str | None = None,
        n_jobs: int = 1,
        random_state: int | None = None,
    ):
        self.num_kernels = num_kernels
        self.max_dilations_per_kernel = max_dilations_per_kernel
        self.normalise_per_instance = normalise_per_instance
        self.only_mix = only_mix
        self.selection_num_folds = selection_num_folds
        self.selection_num_runs = selection_num_runs
        self.selection_num_features = selection_num_features
        self.selection_max_samples = selection_max_samples
        self.selection_alphas = selection_alphas
        self.vote_top = vote_top
        self.vote_threshold = vote_threshold
        self.length_threshold = length_threshold
        self.class_weight = class_weight
        self.n_jobs = n_jobs
        self.random_state = random_state

    def _normalise(self, X: np.ndarray) -> np.ndarray:
        values = np.asarray(X, dtype=np.float32)
        if not self.normalise_per_instance:
            return values
        return (values - values.mean(axis=-1, keepdims=True)) / (
            values.std(axis=-1, keepdims=True) + 1e-8
        )

    def _zero_crossings(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        values = self._normalise(X)
        first_difference = np.diff(values, axis=-1)
        indices = self.generator_._indices
        previous_threads = get_num_threads()
        worker_count = self.generator_._n_jobs
        if worker_count < 1 or worker_count > multiprocessing.cpu_count():
            worker_count = multiprocessing.cpu_count()
        try:
            set_num_threads(worker_count)
            if values.shape[1] == 1:
                base = _zero_crossings_univariate(
                    values[:, 0, :], self.generator_.parameter, indices
                )
                difference = _zero_crossings_univariate(
                    first_difference[:, 0, :], self.generator_.parameter1, indices
                )
            else:
                base = _zero_crossings_multivariate(
                    values, self.generator_.parameter, indices
                )
                difference = _zero_crossings_multivariate(
                    first_difference, self.generator_.parameter1, indices
                )
        finally:
            set_num_threads(previous_threads)
        return base, difference

    def _raw_feature_matrix(self, X: np.ndarray) -> np.ndarray:
        multirocket = np.asarray(self.generator_.transform(X), dtype=np.float32)
        base_zc, difference_zc = self._zero_crossings(X)
        base_count = base_zc.shape[1]
        difference_count = difference_zc.shape[1]
        expected = 4 * (base_count + difference_count)
        if multirocket.shape[1] != expected:
            raise RuntimeError(
                "Unexpected MultiRocket feature layout: "
                f"expected {expected}, received {multirocket.shape[1]}"
            )

        difference_offset = 4 * base_count
        base_ppv = multirocket[:, 0:base_count]
        base_lspv = multirocket[:, base_count : 2 * base_count]
        base_mpv = multirocket[:, 2 * base_count : 3 * base_count]
        base_mipv = multirocket[:, 3 * base_count : 4 * base_count]
        difference_ppv = multirocket[
            :, difference_offset : difference_offset + difference_count
        ]
        difference_lspv = multirocket[
            :,
            difference_offset + difference_count : difference_offset
            + 2 * difference_count,
        ]
        difference_mpv = multirocket[
            :,
            difference_offset + 2 * difference_count : difference_offset
            + 3 * difference_count,
        ]
        difference_mipv = multirocket[
            :,
            difference_offset + 3 * difference_count : difference_offset
            + 4 * difference_count,
        ]

        self.base_features_per_operator_ = base_count
        self.difference_features_per_operator_ = difference_count
        return np.concatenate(
            (
                base_ppv,
                base_zc,
                base_mpv,
                base_mipv,
                base_lspv,
                difference_ppv,
                difference_zc,
                difference_mpv,
                difference_mipv,
                difference_lspv,
            ),
            axis=1,
        )

    @staticmethod
    def _candidate_parts(name: str) -> tuple[int, str]:
        representation = "BASE"
        operator = name
        if name.endswith("_DIFF"):
            representation = "DIFF"
            operator = name.removesuffix("_DIFF")
        elif name.endswith("_MIX"):
            representation = "MIX"
            operator = name.removesuffix("_MIX")
        if operator not in POOLING_OPERATORS:
            raise ValueError(f"Unknown SelF-Rocket candidate: {name!r}")
        return POOLING_OPERATORS.index(operator), representation

    def _candidate_width(self, name: str) -> int:
        _, representation = self._candidate_parts(name)
        if representation == "BASE":
            return self.base_features_per_operator_
        if representation == "DIFF":
            return self.difference_features_per_operator_
        return (
            self.base_features_per_operator_
            + self.difference_features_per_operator_
        )

    def _candidate_matrix(self, values: np.ndarray, name: str) -> np.ndarray:
        operator_index, representation = self._candidate_parts(name)
        base_count = self.base_features_per_operator_
        difference_count = self.difference_features_per_operator_
        base_start = operator_index * base_count
        difference_start = 5 * base_count + operator_index * difference_count
        base = values[:, base_start : base_start + base_count]
        difference = values[
            :, difference_start : difference_start + difference_count
        ]
        if representation == "BASE":
            return base
        if representation == "DIFF":
            return difference
        return np.concatenate((base, difference), axis=1)

    def _candidate_columns(
        self,
        values: np.ndarray,
        name: str,
        rows: np.ndarray,
        columns: np.ndarray,
    ) -> np.ndarray:
        operator_index, representation = self._candidate_parts(name)
        base_count = self.base_features_per_operator_
        difference_count = self.difference_features_per_operator_
        base_start = operator_index * base_count
        difference_start = 5 * base_count + operator_index * difference_count

        if representation == "BASE":
            return values[np.ix_(rows, base_start + columns)]
        if representation == "DIFF":
            return values[np.ix_(rows, difference_start + columns)]

        selected = np.empty((len(rows), len(columns)), dtype=values.dtype)
        base_mask = columns < base_count
        if np.any(base_mask):
            selected[:, base_mask] = values[
                np.ix_(rows, base_start + columns[base_mask])
            ]
        if np.any(~base_mask):
            difference_columns = columns[~base_mask] - base_count
            selected[:, ~base_mask] = values[
                np.ix_(rows, difference_start + difference_columns)
            ]
        return selected

    def _selection_splits(
        self,
        values: np.ndarray,
        labels: np.ndarray,
    ) -> Any:
        if len(labels) <= self.selection_max_samples:
            splitter = RepeatedStratifiedKFold(
                n_splits=self.selection_num_folds,
                n_repeats=self.selection_num_runs,
                random_state=self.random_state,
            )
        else:
            half = self.selection_max_samples // 2
            splitter = StratifiedShuffleSplit(
                n_splits=self.selection_num_folds * self.selection_num_runs,
                train_size=half,
                test_size=half,
                random_state=self.random_state,
            )
        return splitter.split(values, labels)

    def _select_candidate(
        self,
        values: np.ndarray,
        labels: np.ndarray,
        series_length: int,
    ) -> None:
        names = candidate_names(self.only_mix)
        rng = np.random.default_rng(self.random_state)
        split_indices = list(self._selection_splits(values, labels))
        performances = np.empty((len(split_indices), len(names)), dtype=np.float32)

        for split_index, (train_indices, validation_indices) in enumerate(
            split_indices
        ):
            for candidate_index, name in enumerate(names):
                width = self._candidate_width(name)
                feature_count = min(self.selection_num_features, width)
                feature_indices = np.sort(
                    rng.choice(width, size=feature_count, replace=False)
                )
                train_features = self._candidate_columns(
                    values, name, train_indices, feature_indices
                )
                validation_features = self._candidate_columns(
                    values, name, validation_indices, feature_indices
                )
                classifier = RidgeClassifierCV(
                    alphas=np.asarray(self.selection_alphas, dtype=np.float64),
                    class_weight=self.class_weight,
                )
                classifier.fit(train_features, labels[train_indices])
                predictions = classifier.predict(validation_features)
                performances[split_index, candidate_index] = accuracy_score(
                    labels[validation_indices], predictions
                )

        proposed, medians = highest_median_choice(performances, names)
        selected, support, used_fallback = vote_validated_choice(
            performances,
            names,
            proposed,
            vote_top=self.vote_top,
            vote_threshold=self.vote_threshold,
            series_length=series_length,
            length_threshold=self.length_threshold,
        )
        self.candidate_names_ = names
        self.selection_performances_ = performances
        self.selection_median_scores_ = medians
        self.selected_candidate_before_vote_ = proposed
        self.selected_candidate_ = selected
        self.selection_vote_support_ = support
        self.selection_used_fallback_ = used_fallback
        self.n_transformed_features_ = self._candidate_width(selected)

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> SelFRocketFeatures:
        self.fit_transform(X, y)
        return self

    def fit_transform(
        self,
        X: np.ndarray,
        y: np.ndarray | None = None,
        **fit_params: Any,
    ) -> np.ndarray:
        del fit_params
        if y is None:
            raise ValueError("SelF-Rocket feature selection requires class labels")
        values = np.asarray(X, dtype=np.float32)
        labels = np.asarray(y)
        if values.ndim != 3:
            raise ValueError("X must have shape (samples, channels, time points)")
        if labels.ndim != 1 or len(labels) != len(values):
            raise ValueError("y must be one-dimensional with one label per sample")
        _, class_counts = np.unique(labels, return_counts=True)
        if len(class_counts) != 2:
            raise ValueError("SelF-Rocket currently requires exactly two classes")
        if int(class_counts.min()) < self.selection_num_folds:
            raise ValueError(
                "Each class needs at least selection_num_folds training samples"
            )

        MultiRocket = _load_multirocket_class()
        self.generator_ = MultiRocket(
            n_kernels=self.num_kernels,
            max_dilations_per_kernel=self.max_dilations_per_kernel,
            n_features_per_kernel=4,
            normalise=self.normalise_per_instance,
            n_jobs=self.n_jobs,
            random_state=self.random_state,
        )
        self.generator_.fit(values, labels)
        raw_features = self._raw_feature_matrix(values)
        self.scaler_ = StandardScaler()
        scaled_features = self.scaler_.fit_transform(raw_features)
        self._select_candidate(scaled_features, labels, values.shape[-1])
        return self._candidate_matrix(scaled_features, self.selected_candidate_)

    def transform(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(
            self,
            (
                "generator_",
                "scaler_",
                "selected_candidate_",
                "n_transformed_features_",
            ),
        )
        raw_features = self._raw_feature_matrix(np.asarray(X, dtype=np.float32))
        scaled_features = self.scaler_.transform(raw_features)
        selected = self._candidate_matrix(
            scaled_features, self.selected_candidate_
        )
        if selected.shape[1] != self.n_transformed_features_:
            raise RuntimeError("Selected SelF-Rocket feature count changed")
        return selected


__all__ = [
    "CANDIDATE_NAMES",
    "POOLING_OPERATORS",
    "SelFRocketFeatures",
    "candidate_names",
    "highest_median_choice",
    "vote_validated_choice",
]
