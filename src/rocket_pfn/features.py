"""Feature grouping and lazy TabPFN inference for RocketPFN."""

from __future__ import annotations

import logging
import math
import warnings
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.feature_selection import f_classif

try:
    from sparse_multirocket_hydra.features import MultiRocketHydraRawFeatures
except ModuleNotFoundError as error:
    if error.name != "sparse_multirocket_hydra":
        raise
    from ..sparse_multirocket_hydra.features import MultiRocketHydraRawFeatures

from .config import ExperimentConfig, RocketPFNConfig


LOGGER = logging.getLogger(__name__)


class TabPFNAccessError(RuntimeError):
    """Raised before data loading when pretrained weights cannot be accessed."""


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


def _load_rocket_class() -> type:
    try:
        from aeon.transformations.collection.convolution_based import Rocket
    except ImportError as error:
        raise ImportError(
            "RocketPFN requires aeon. Install project requirements with "
            "`pip install -r requirements.txt`."
        ) from error
    return Rocket


def _load_tabpfn_classes() -> tuple[type, type]:
    try:
        from tabpfn import TabPFNClassifier
        from tabpfn.constants import ModelVersion
    except ImportError as error:
        raise ImportError(
            "RocketPFN requires TabPFN. Install project requirements with "
            "`pip install -r requirements.txt`. The pretrained checkpoint is "
            "downloaded on first use."
        ) from error
    return TabPFNClassifier, ModelVersion


def new_tabpfn_classifier(model: RocketPFNConfig, seed: int) -> Any:
    """Construct the explicitly versioned TabPFN classifier used by all groups."""

    TabPFNClassifier, ModelVersion = _load_tabpfn_classes()
    versions = {
        "2.5": ModelVersion.V2_5,
        "3": ModelVersion.V3,
    }
    overrides: dict[str, Any] = {
        "n_estimators": model.tabpfn_n_estimators,
        "device": model.tabpfn_device,
        "fit_mode": model.tabpfn_fit_mode,
        "memory_saving_mode": model.tabpfn_memory_saving_mode,
        "inference_precision": model.tabpfn_inference_precision,
        "balance_probabilities": model.tabpfn_balance_probabilities,
        "n_preprocessing_jobs": model.tabpfn_preprocessing_jobs,
        "random_state": seed,
        "show_progress_bar": model.tabpfn_show_progress_bar,
    }
    if model.tabpfn_model_path is not None:
        overrides["model_path"] = str(model.tabpfn_model_path)
    return TabPFNClassifier.create_default_for_version(
        versions[model.tabpfn_version],
        **overrides,
    )


def ensure_tabpfn_checkpoint_access(model: RocketPFNConfig, seed: int) -> None:
    """Verify gated-checkpoint access before performing fold preprocessing."""

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
        raise ImportError(
            "RocketPFN requires TabPFN. Install project requirements with "
            "`pip install -r requirements.txt`."
        ) from error

    sources = {
        "2.5": ModelSource.get_classifier_v2_5,
        "3": ModelSource.get_classifier_v3,
    }
    repository = sources[model.tabpfn_version]().repo_id.rsplit("/", 1)[-1]
    try:
        ensure_license_accepted(hf_repo_id=repository)
    except TabPFNError as error:
        raise TabPFNAccessError(
            f"TabPFN v{model.tabpfn_version} weights are not cached and require "
            "one-time license acceptance.\n\n"
            "1. Open https://ux.priorlabs.ai and accept the license.\n"
            "2. Copy the API key from https://ux.priorlabs.ai/account.\n"
            "3. Set TABPFN_TOKEN in the process environment, or place\n"
            "   TABPFN_TOKEN=<your-api-key> in the project-root .env file.\n"
            "4. Restart this script.\n\n"
            "Do not commit or share the token; .env is already git-ignored."
        ) from error


def new_rocket_transformer(model: RocketPFNConfig, seed: int) -> Any:
    Rocket = _load_rocket_class()
    return Rocket(
        n_kernels=model.rocket_kernels_per_group,
        normalise=model.rocket_normalise_per_instance,
        n_jobs=model.transform_n_jobs,
        random_state=seed,
    )


def new_multirocket_hydra_transformer(
    model: RocketPFNConfig,
    seed: int,
) -> MultiRocketHydraRawFeatures:
    return MultiRocketHydraRawFeatures(
        num_kernels=model.multirocket_num_kernels,
        max_dilations_per_kernel=model.multirocket_max_dilations_per_kernel,
        num_features_per_kernel=model.multirocket_num_features_per_kernel,
        normalise_per_instance=model.multirocket_normalise_per_instance,
        hydra_num_kernels=model.hydra_num_kernels,
        hydra_num_groups=model.hydra_num_groups,
        hydra_max_num_channels=model.hydra_max_num_channels,
        n_jobs=model.transform_n_jobs,
        random_state=seed,
    )


def load_reusable_multirocket_hydra_transformer(
    config: ExperimentConfig,
) -> MultiRocketHydraRawFeatures | None:
    """Extract compatible fitted random transforms from a baseline artifact."""

    checkpoint_path = config.feature_artifact_path
    if checkpoint_path is None:
        return None
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            "Configured MultiRocket+HYDRA feature artifact does not exist: "
            f"{checkpoint_path}"
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
        raise ValueError(f"Feature artifact {checkpoint_path} uses a different seed")
    saved_model = saved_config.get("model", {})
    expected = {
        "num_kernels": config.model.multirocket_num_kernels,
        "max_dilations_per_kernel": (
            config.model.multirocket_max_dilations_per_kernel
        ),
        "num_features_per_kernel": (
            config.model.multirocket_num_features_per_kernel
        ),
        "normalise_per_instance": (
            config.model.multirocket_normalise_per_instance
        ),
        "hydra_num_kernels": config.model.hydra_num_kernels,
        "hydra_num_groups": config.model.hydra_num_groups,
        "hydra_max_num_channels": config.model.hydra_max_num_channels,
    }
    mismatches = {
        name: (saved_model.get(name), value)
        for name, value in expected.items()
        if saved_model.get(name) != value
    }
    if not saved_model.get("use_hydra") or mismatches:
        raise ValueError(
            f"Feature artifact {checkpoint_path} is incompatible: {mismatches}"
        )

    source = artifact["pipeline"].steps[0][1]
    fitted_attributes = (
        "multirocket_",
        "hydra_",
        "n_multirocket_features_",
        "n_hydra_features_",
    )
    missing = [name for name in fitted_attributes if not hasattr(source, name)]
    if missing:
        raise ValueError(
            f"Feature artifact {checkpoint_path} lacks fitted attributes: {missing}"
        )
    transformer = new_multirocket_hydra_transformer(config.model, config.seed)
    for name in fitted_attributes:
        setattr(transformer, name, getattr(source, name))
    return transformer


def ranked_feature_groups(
    features: np.ndarray,
    labels: np.ndarray,
    requested_feature_count: int,
    max_features_per_group: int,
) -> tuple[np.ndarray, ...]:
    """Select top ANOVA features on training data and distribute rank across groups."""

    values = np.asarray(features)
    if values.ndim != 2 or values.shape[1] < 1:
        raise ValueError("features must be a non-empty 2D matrix")
    count = min(requested_feature_count, values.shape[1])
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        scores, _ = f_classif(values, labels)
    scores = np.nan_to_num(scores, nan=-np.inf)
    ranked = np.argsort(-scores, kind="stable")[:count]
    num_groups = math.ceil(count / max_features_per_group)
    # Round-robin assignment prevents one group from receiving only weak tail features.
    return tuple(
        np.asarray(ranked[group_index::num_groups], dtype=np.int64)
        for group_index in range(num_groups)
    )


def positive_class_probabilities(classifier: Any, features: np.ndarray) -> np.ndarray:
    probabilities = np.asarray(classifier.predict_proba(features), dtype=np.float64)
    classes = np.asarray(classifier.classes_)
    positive_columns = np.flatnonzero(classes == 1)
    if positive_columns.size != 1:
        raise RuntimeError(
            "TabPFN classes must contain binary positive label 1, got "
            f"{classes.tolist()}"
        )
    expected_shape = (features.shape[0], classes.size)
    if probabilities.shape != expected_shape:
        raise RuntimeError(
            "TabPFN returned probability shape "
            f"{probabilities.shape}; expected {expected_shape}"
        )
    positive = probabilities[:, int(positive_columns[0])]
    if not np.all(np.isfinite(positive)):
        raise RuntimeError("TabPFN returned non-finite probabilities")
    return np.clip(positive, 0.0, 1.0)


def _has_nonconstant_training_feature(features: np.ndarray) -> bool:
    """Match TabPFN's check for at least one usable non-constant column."""

    if features.shape[0] == 0:
        raise ValueError("Training feature groups must contain at least one sample")
    nonconstant = (features[0:1, :] == features).mean(axis=0) < 1.0
    not_all_nan = ~np.all(np.isnan(features), axis=0)
    return bool(np.any(nonconstant & not_all_nan))


def average_tabpfn_probabilities(
    feature_pairs: Iterable[tuple[np.ndarray, np.ndarray]],
    labels: np.ndarray,
    model: RocketPFNConfig,
    seed: int,
    progress_callback: Any | None = None,
) -> tuple[np.ndarray, int]:
    """Score every feature group and average its positive-class probabilities."""

    # Artifact-v1 checkpoints predate this appended slots-dataclass field. Preserve
    # their original sequential behavior when they are loaded for fresh predictions.
    if getattr(model, "tabpfn_batch_groups", False):
        return _average_tabpfn_probabilities_batched(
            feature_pairs,
            labels,
            model,
            seed,
            progress_callback,
        )

    return _average_tabpfn_probabilities_sequential(
        feature_pairs,
        labels,
        model,
        seed,
        progress_callback,
    )


def _validated_feature_pair(
    train_features: np.ndarray,
    query_features: np.ndarray,
    labels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    train_values = np.asarray(train_features, dtype=np.float32, order="C")
    query_values = np.asarray(query_features, dtype=np.float32, order="C")
    if train_values.ndim != 2 or query_values.ndim != 2:
        raise ValueError("Each TabPFN feature group must be a 2D matrix")
    if train_values.shape[1] != query_values.shape[1]:
        raise ValueError("Training and query feature group widths must match")
    if train_values.shape[0] != labels.shape[0]:
        raise ValueError("Training feature rows and labels must have equal length")
    return train_values, query_values


def _prior_probabilities(
    labels: np.ndarray,
    query_count: int,
    group_number: int,
) -> np.ndarray:
    positive_prior = float(np.asarray(labels, dtype=np.float64).mean())
    warnings.warn(
        f"TabPFN feature group {group_number} has no non-constant training "
        f"features; using the positive-class training prior "
        f"({positive_prior:.6g}) for this group.",
        RuntimeWarning,
        stacklevel=3,
    )
    return np.full(query_count, positive_prior, dtype=np.float64)


def _average_tabpfn_probabilities_sequential(
    feature_pairs: Iterable[tuple[np.ndarray, np.ndarray]],
    labels: np.ndarray,
    model: RocketPFNConfig,
    seed: int,
    progress_callback: Any | None,
) -> tuple[np.ndarray, int]:
    """Run the original one-fit-and-predict-per-group inference path."""

    label_values = np.asarray(labels)
    classifier = new_tabpfn_classifier(model, seed)
    probability_sum: np.ndarray | None = None
    group_count = 0
    for group_count, (train_features, query_features) in enumerate(
        feature_pairs, start=1
    ):
        train_values, query_values = _validated_feature_pair(
            train_features, query_features, label_values
        )
        if _has_nonconstant_training_feature(train_values):
            classifier.fit(train_values, label_values)
            group_probabilities = positive_class_probabilities(
                classifier, query_values
            )
        else:
            group_probabilities = _prior_probabilities(
                label_values, query_values.shape[0], group_count
            )
        if probability_sum is None:
            probability_sum = np.zeros_like(group_probabilities, dtype=np.float64)
        elif probability_sum.shape != group_probabilities.shape:
            raise RuntimeError("TabPFN groups returned different sample counts")
        probability_sum += group_probabilities
        if progress_callback is not None:
            progress_callback(group_count, train_values.shape[1])
    if probability_sum is None or group_count == 0:
        raise ValueError("At least one feature group is required")
    return probability_sum / group_count, group_count


def _average_tabpfn_probabilities_batched(
    feature_pairs: Iterable[tuple[np.ndarray, np.ndarray]],
    labels: np.ndarray,
    model: RocketPFNConfig,
    seed: int,
    progress_callback: Any | None,
) -> tuple[np.ndarray, int]:
    """Fuse same-shaped informative groups with TabPFN's dataset-batch API."""

    label_values = np.asarray(labels)
    group_probabilities: list[np.ndarray | None] = []
    group_widths: list[int] = []
    informative_train: list[np.ndarray] = []
    informative_query: list[np.ndarray] = []
    informative_indices: list[int] = []
    query_count: int | None = None

    for group_number, (train_features, query_features) in enumerate(
        feature_pairs, start=1
    ):
        train_values, query_values = _validated_feature_pair(
            train_features, query_features, label_values
        )
        if query_count is None:
            query_count = query_values.shape[0]
        elif query_values.shape[0] != query_count:
            raise ValueError("All TabPFN query groups must have equal length")

        group_widths.append(train_values.shape[1])
        if _has_nonconstant_training_feature(train_values):
            informative_indices.append(group_number - 1)
            informative_train.append(train_values)
            informative_query.append(query_values)
            group_probabilities.append(None)
        else:
            group_probabilities.append(
                _prior_probabilities(
                    label_values,
                    query_values.shape[0],
                    group_number,
                )
            )

    group_count = len(group_probabilities)
    if group_count == 0 or query_count is None:
        raise ValueError("At least one feature group is required")

    if informative_train:
        classifier = new_tabpfn_classifier(model, seed)
        # TabPFN's dataset-batch API rejects ragged matrices. Keep the speed-up for
        # same-shaped groups and score singleton shape buckets through the ordinary
        # path, which also makes custom MultiRocket group sizes work.
        shape_buckets: dict[
            tuple[tuple[int, ...], tuple[int, ...]], list[int]
        ] = {}
        for local_index, (train_values, query_values) in enumerate(
            zip(informative_train, informative_query, strict=True)
        ):
            key = (train_values.shape, query_values.shape)
            shape_buckets.setdefault(key, []).append(local_index)

        if any(len(local_indices) > 1 for local_indices in shape_buckets.values()):
            warnings.warn(
                "TabPFN group batching can produce different probabilities than "
                "independent group inference because TabPFN v8.1 handles internal "
                "constant columns differently for multi-dataset batches. Use "
                "tabpfn_batch_groups=False for the original prediction path.",
                RuntimeWarning,
                stacklevel=2,
            )

        classes = np.unique(label_values)
        positive_columns = np.flatnonzero(classes == 1)
        if positive_columns.size != 1:
            raise RuntimeError(
                "TabPFN classes must contain binary positive label 1, got "
                f"{classes.tolist()}"
            )
        positive_column = int(positive_columns[0])

        for local_indices in shape_buckets.values():
            if len(local_indices) == 1:
                local_index = local_indices[0]
                classifier.fit(informative_train[local_index], label_values)
                values = positive_class_probabilities(
                    classifier, informative_query[local_index]
                )
                group_probabilities[informative_indices[local_index]] = values
                continue

            batched_train = [informative_train[index] for index in local_indices]
            batched_query = [informative_query[index] for index in local_indices]
            probabilities = np.asarray(
                classifier.predict_proba_batched(
                    batched_train,
                    [label_values] * len(local_indices),
                    batched_query,
                ),
                dtype=np.float64,
            )
            expected_shape = (len(local_indices), query_count, classes.size)
            if probabilities.shape != expected_shape:
                raise RuntimeError(
                    "Batched TabPFN returned probability shape "
                    f"{probabilities.shape}; expected {expected_shape}"
                )
            positive = probabilities[:, :, positive_column]
            if not np.all(np.isfinite(positive)):
                raise RuntimeError(
                    "Batched TabPFN returned non-finite probabilities"
                )
            for local_index, values in zip(
                local_indices,
                np.clip(positive, 0.0, 1.0),
                strict=True,
            ):
                group_probabilities[informative_indices[local_index]] = values

    if progress_callback is not None:
        for group_number, width in enumerate(group_widths, start=1):
            progress_callback(group_number, width)

    completed_probabilities = [
        values for values in group_probabilities if values is not None
    ]
    if len(completed_probabilities) != group_count:
        raise RuntimeError("Batched TabPFN did not return every feature group")
    return np.mean(np.stack(completed_probabilities), axis=0), group_count


@dataclass(slots=True)
class FittedRocketPFN:
    """Fitted random feature maps plus training context for lazy TabPFN inference.

    Pretrained TabPFN weights are deliberately not serialized into every fold. A
    classifier is recreated from its versioned checkpoint when predictions are made.
    """

    model_config: RocketPFNConfig
    seed: int
    train_values: np.ndarray
    train_labels: np.ndarray
    feature_transformers: tuple[Any, ...]
    feature_indices: tuple[np.ndarray, ...] = ()

    def _feature_pairs(self, X: np.ndarray) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        query_values = np.asarray(X)
        if self.model_config.feature_representation == "rocket":
            for transformer in self.feature_transformers:
                yield (
                    np.asarray(transformer.transform(self.train_values)),
                    np.asarray(transformer.transform(query_values)),
                )
            return

        if len(self.feature_transformers) != 1 or not self.feature_indices:
            raise RuntimeError("Invalid fitted MultiRocket+HYDRA feature state")
        transformer = self.feature_transformers[0]
        train_features = np.asarray(transformer.transform(self.train_values))
        query_features = np.asarray(transformer.transform(query_values))
        for indices in self.feature_indices:
            yield train_features[:, indices], query_features[:, indices]

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        positive, _ = average_tabpfn_probabilities(
            self._feature_pairs(X),
            self.train_labels,
            self.model_config,
            self.seed,
        )
        return np.column_stack((1.0 - positive, positive))

    def predict_proba_many(self, inputs: Iterable[np.ndarray]) -> list[np.ndarray]:
        """Predict several query sets while reusing fixed training work.

        Robustness analyses score many synthetic variants of the same held-out split.
        The ordinary scikit-learn API treats every variant as an unrelated prediction:
        it transforms the unchanged training split again and refits every TabPFN
        feature-group context.  Here, query rows are concatenated, each training and
        query representation is transformed once, and each TabPFN group is fitted once.
        Concatenating query rows is the same inference pattern used during training to
        score validation and test rows together; rows are split back into their
        original query sets before returning.
        """

        query_sets = [np.asarray(values) for values in inputs]
        if not query_sets:
            return []
        reference_shape = query_sets[0].shape[1:]
        if any(values.ndim != 3 for values in query_sets):
            raise ValueError("RocketPFN query sets must be three-dimensional")
        if any(values.shape[1:] != reference_shape for values in query_sets):
            raise ValueError("RocketPFN query sets must share channel and time shapes")

        # The experimental group-batched path has different inference semantics in
        # TabPFN 8.1. Preserve that path exactly instead of combining it with this
        # repeated-query optimization.
        if getattr(self.model_config, "tabpfn_batch_groups", False):
            return [self.predict_proba(values) for values in query_sets]

        query_counts = [int(values.shape[0]) for values in query_sets]
        combined_query = np.concatenate(query_sets, axis=0)
        combined_count = int(combined_query.shape[0])
        label_values = np.asarray(self.train_labels)
        probability_sum = np.zeros(combined_count, dtype=np.float64)
        classifier = new_tabpfn_classifier(self.model_config, self.seed)
        group_count = 0
        expected_groups = (
            len(self.feature_transformers)
            if self.model_config.feature_representation == "rocket"
            else len(self.feature_indices)
        )
        LOGGER.info(
            "Repeated RocketPFN inference: %d query sets, %d combined rows, "
            "%d feature groups",
            len(query_sets),
            combined_count,
            expected_groups,
        )

        def accumulate(
            train_features: np.ndarray,
            query_features: np.ndarray,
        ) -> None:
            nonlocal group_count
            group_count += 1
            train_values, query_values = _validated_feature_pair(
                train_features, query_features, label_values
            )
            if query_values.shape[0] != combined_count:
                raise RuntimeError("RocketPFN transformed query count changed")
            if _has_nonconstant_training_feature(train_values):
                classifier.fit(train_values, label_values)
                probabilities = positive_class_probabilities(
                    classifier, query_values
                )
            else:
                probabilities = _prior_probabilities(
                    label_values, combined_count, group_count
                )
            probability_sum[:] += probabilities

        if self.model_config.feature_representation == "rocket":
            for group_index, transformer in enumerate(
                self.feature_transformers, start=1
            ):
                LOGGER.info(
                    "Repeated RocketPFN feature group %d/%d",
                    group_index,
                    expected_groups,
                )
                # The training transform is invariant across every perturbation and is
                # therefore intentionally computed only once per ROCKET group.
                accumulate(
                    np.asarray(transformer.transform(self.train_values)),
                    np.asarray(transformer.transform(combined_query)),
                )
        else:
            if len(self.feature_transformers) != 1 or not self.feature_indices:
                raise RuntimeError("Invalid fitted MultiRocket+HYDRA feature state")
            transformer = self.feature_transformers[0]
            train_features = np.asarray(transformer.transform(self.train_values))
            query_features = np.asarray(transformer.transform(combined_query))
            for group_index, indices in enumerate(self.feature_indices, start=1):
                LOGGER.info(
                    "Repeated RocketPFN feature group %d/%d",
                    group_index,
                    expected_groups,
                )
                accumulate(train_features[:, indices], query_features[:, indices])

        if group_count == 0:
            raise ValueError("At least one feature group is required")
        combined_positive = probability_sum / group_count
        boundaries = np.cumsum(query_counts[:-1], dtype=np.int64)
        split_probabilities = np.split(combined_positive, boundaries)
        return [
            np.column_stack((1.0 - positive, positive))
            for positive in split_probabilities
        ]

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= threshold).astype(np.int64)


__all__ = [
    "FittedRocketPFN",
    "TabPFNAccessError",
    "average_tabpfn_probabilities",
    "ensure_tabpfn_checkpoint_access",
    "load_reusable_multirocket_hydra_transformer",
    "new_multirocket_hydra_transformer",
    "new_rocket_transformer",
    "new_tabpfn_classifier",
    "positive_class_probabilities",
    "ranked_feature_groups",
]
