"""Resource-aware wrapper around aeon's authorship implementation of HC2."""

from __future__ import annotations

from time import perf_counter
from typing import Any

from sklearn.base import clone
from sklearn.metrics import accuracy_score

from aeon.classification.convolution_based import Arsenal
from aeon.classification.dictionary_based import TemporalDictionaryEnsemble
from aeon.classification.hybrid import HIVECOTEV2
from aeon.classification.interval_based import DrCIFClassifier
from aeon.classification.shapelet_based import ShapeletTransformClassifier
from aeon.classification.sklearn import RotationForestClassifier
from aeon.utils.validation import check_n_jobs

try:
    from drcif.stabilization import NearConstantSafeDrCIFClassifier
except ModuleNotFoundError as error:
    if error.name != "drcif":
        raise
    from ..drcif.stabilization import NearConstantSafeDrCIFClassifier

from .config import HIVECOTEV2Config


def component_parameter_dicts(
    config: HIVECOTEV2Config,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Translate serializable experiment settings into aeon component parameters."""

    stc = config.stc
    stc_params = {
        "n_shapelet_samples": stc.n_shapelet_samples,
        "max_shapelets": stc.max_shapelets,
        "max_shapelet_length": stc.max_shapelet_length,
        "batch_size": stc.batch_size,
        "contract_max_n_shapelet_samples": stc.contract_max_n_shapelet_samples,
        "estimator": RotationForestClassifier(
            n_estimators=stc.rotation_forest_n_estimators,
            contract_max_n_estimators=(
                stc.rotation_forest_contract_max_n_estimators
            ),
        ),
    }

    drcif = config.drcif
    drcif_params = {
        "n_estimators": drcif.n_estimators,
        "n_intervals": drcif.n_intervals,
        "min_interval_length": drcif.min_interval_length,
        "max_interval_length": drcif.max_interval_length,
        "att_subsample_size": drcif.att_subsample_size,
        "contract_max_n_estimators": drcif.contract_max_n_estimators,
        "use_pycatch22": drcif.use_pycatch22,
        # Consumed by ResourceAwareHIVECOTEV2 before constructing the component.
        "stabilize_near_constant_intervals": (
            drcif.stabilize_near_constant_intervals
        ),
    }

    arsenal = config.arsenal
    arsenal_params = {
        "n_kernels": arsenal.n_kernels,
        "n_estimators": arsenal.n_estimators,
        "rocket_transform": arsenal.rocket_transform,
        "max_dilations_per_kernel": arsenal.max_dilations_per_kernel,
        "n_features_per_kernel": arsenal.n_features_per_kernel,
        "contract_max_n_estimators": arsenal.contract_max_n_estimators,
    }

    tde = config.tde
    tde_params = {
        "n_parameter_samples": tde.n_parameter_samples,
        "max_ensemble_size": tde.max_ensemble_size,
        "max_win_len_prop": tde.max_win_len_prop,
        "min_window": tde.min_window,
        "randomly_selected_params": tde.randomly_selected_params,
        "bigrams": tde.bigrams,
        "dim_threshold": tde.dim_threshold,
        "max_dims": tde.max_dims,
        "contract_max_n_parameter_samples": (
            tde.contract_max_n_parameter_samples
        ),
    }
    return stc_params, drcif_params, arsenal_params, tde_params


class ResourceAwareHIVECOTEV2(HIVECOTEV2):
    """Official aeon HC2 with progress output and variance-safe DrCIF.

    Ensemble composition, train-estimate generation, fourth-power CAWPE weighting,
    and prediction remain identical to aeon's implementation. This override makes
    component construction explicit so the project's DrCIF numerical safeguard can
    be used, and reports each long-running component boundary.
    """

    def _fit(self, X, y):
        self._stc_params = dict(
            self.stc_params
            if self.stc_params is not None
            else {"n_shapelet_samples": self._DEFAULT_N_SHAPELETS}
        )
        self._drcif_params = dict(
            self.drcif_params
            if self.drcif_params is not None
            else {"n_estimators": self._DEFAULT_N_TREES}
        )
        self._arsenal_params = dict(
            self.arsenal_params
            if self.arsenal_params is not None
            else {
                "n_kernels": self._DEFAULT_N_KERNELS,
                "n_estimators": self._DEFAULT_N_ESTIMATORS,
            }
        )
        self._tde_params = dict(
            self.tde_params
            if self.tde_params is not None
            else {
                "n_parameter_samples": self._DEFAULT_N_PARA_SAMPLES,
                "max_ensemble_size": self._DEFAULT_MAX_ENSEMBLE_SIZE,
                "randomly_selected_params": self._DEFAULT_RAND_PARAMS,
            }
        )

        if self.time_limit_in_minutes > 0:
            component_limit = self.time_limit_in_minutes / 6
            for parameters in (
                self._stc_params,
                self._drcif_params,
                self._arsenal_params,
                self._tde_params,
            ):
                parameters["time_limit_in_minutes"] = component_limit

        stabilize_drcif = self._drcif_params.pop(
            "stabilize_near_constant_intervals",
            True,
        )
        if self.parallel_backend is not None:
            self._drcif_params["parallel_backend"] = self.parallel_backend
        drcif_class = (
            NearConstantSafeDrCIFClassifier
            if stabilize_drcif
            else DrCIFClassifier
        )
        self._estimators = [
            ("STC", ShapeletTransformClassifier(**self._stc_params)),
            ("DrCIF", drcif_class(**self._drcif_params)),
            ("Arsenal", Arsenal(**self._arsenal_params)),
            ("TDE", TemporalDictionaryEnsemble(**self._tde_params)),
        ]
        return self._fit_components_with_progress(X, y)

    def _fit_components_with_progress(self, X, y):
        """Run aeon's CAWPE fit while exposing otherwise silent component progress."""

        self._n_jobs = check_n_jobs(self.n_jobs)
        self.fitted_estimators_ = []
        self.weights_ = []
        self.component_names_ = []
        self.component_fit_seconds_ = {}
        self.component_train_accuracies_ = {}

        for position, (name, estimator) in enumerate(self._estimators, start=1):
            if self.verbose:
                print(
                    f"[HIVE-COTE 2.0] Starting component {position}/4: {name}",
                    flush=True,
                )
            start = perf_counter()
            fitted = clone(estimator)
            if hasattr(fitted, "random_state") and self.random_state is not None:
                fitted.random_state = self.random_state
            if hasattr(fitted, "n_jobs"):
                fitted.n_jobs = self._n_jobs
            if hasattr(fitted, "verbose"):
                fitted.verbose = self.verbose

            train_predictions = fitted.fit_predict(X, y)
            train_accuracy = float(accuracy_score(y, train_predictions))
            weight = train_accuracy**self.alpha
            elapsed = perf_counter() - start

            self.fitted_estimators_.append(fitted)
            self.weights_.append(weight)
            self.component_names_.append(name)
            self.component_fit_seconds_[name] = elapsed
            self.component_train_accuracies_[name] = train_accuracy
            if self.verbose:
                print(
                    f"[HIVE-COTE 2.0] Finished {name} in {elapsed / 60:.2f} min "
                    f"(train estimate accuracy={train_accuracy:.4f}, "
                    f"CAWPE weight={weight:.6f})",
                    flush=True,
                )

        return self


def build_classifier(config: HIVECOTEV2Config, seed: int) -> ResourceAwareHIVECOTEV2:
    """Build the configured aeon HIVE-COTE 2.0 estimator."""

    stc_params, drcif_params, arsenal_params, tde_params = (
        component_parameter_dicts(config)
    )
    return ResourceAwareHIVECOTEV2(
        stc_params=stc_params,
        drcif_params=drcif_params,
        arsenal_params=arsenal_params,
        tde_params=tde_params,
        time_limit_in_minutes=config.time_limit_in_minutes,
        save_component_probas=False,
        verbose=config.verbose,
        random_state=seed,
        n_jobs=config.n_jobs,
        parallel_backend=config.parallel_backend,
    )


__all__ = [
    "ResourceAwareHIVECOTEV2",
    "build_classifier",
    "component_parameter_dicts",
]
