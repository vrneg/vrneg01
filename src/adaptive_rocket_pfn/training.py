"""Train Adaptive Multi-Representation RocketPFN on one outer fold."""

from __future__ import annotations

import json
import random
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import torch
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
from .ensemble import (
    FittedAdaptiveRocketPFN,
    ensure_tabpfn_checkpoint_access,
    learn_simplex_weights,
    oof_semantic_predictions,
    predict_semantic_experts,
)
from .features import AdaptiveCandidateBank, EXPERT_NAMES
from .selection import ExpertSelection, select_all_experts


ARTIFACT_VERSION = 1
_PROBABILITY_EPSILON = 1e-7


@dataclass(slots=True)
class TrainingResult:
    """Artifacts, metrics, selected distributions, and OOF ensemble diagnostics."""

    checkpoint_path: Path
    history_path: Path
    validation_predictions_path: Path
    predictions_path: Path
    run_name: str
    best_epoch: int
    decision_threshold: float
    validation_metrics: dict[str, float]
    test_metrics: dict[str, float]
    num_candidate_features: int
    num_selected_features: int
    selected_features_by_expert: dict[str, int]
    ensemble_weights: dict[str, float]
    oof_ensemble_metrics: dict[str, float]
    selected_distribution: dict[str, dict[str, int]]
    history: list[dict[str, Any]] = field(default_factory=list)
    pretrained_checkpoint_path: Path | None = None
    pretraining_history: list[dict[str, Any]] = field(default_factory=list)


def _json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value


def _log(config: ExperimentConfig, message: str) -> None:
    print(f"[{config.run_name}] {message}", flush=True)


def _probabilities_to_logits(probabilities: np.ndarray) -> np.ndarray:
    clipped = np.clip(
        np.asarray(probabilities, dtype=np.float64),
        _PROBABILITY_EPSILON,
        1.0 - _PROBABILITY_EPSILON,
    )
    return np.log(clipped) - np.log1p(-clipped)


def _metrics(
    probabilities: np.ndarray,
    labels: np.ndarray,
    threshold: float,
) -> dict[str, float]:
    logits = torch.from_numpy(_probabilities_to_logits(probabilities)).float()
    targets = torch.from_numpy(np.asarray(labels)).float()
    loss = float(functional.binary_cross_entropy_with_logits(logits, targets).item())
    return binary_classification_metrics(logits, targets, loss, threshold)


def _prediction_rows(
    split: TimeSeriesSplit,
    probabilities: np.ndarray,
    threshold: float,
) -> list[str]:
    logits = _probabilities_to_logits(probabilities)
    return [
        json.dumps(
            {
                "sample_id": sample_id,
                "label": int(label),
                "logit": float(logit),
                "probability": float(probability),
                "threshold": threshold,
                "prediction": int(probability >= threshold),
            }
        )
        for sample_id, label, logit, probability in zip(
            split.sample_ids,
            split.labels,
            logits,
            probabilities,
            strict=True,
        )
    ]


def _selection_payload(
    selections: dict[str, ExpertSelection],
) -> dict[str, dict[str, Any]]:
    return {
        expert: {
            "num_features": selection.num_features,
            "family_order": list(selection.family_order),
            "family_budgets": dict(selection.family_budgets),
            "selected_counts": {
                name: int(indices.size)
                for name, indices in selection.family_indices.items()
            },
            "family_utilities": dict(selection.family_utilities),
        }
        for expert, selection in selections.items()
    }


def _selected_distribution(
    bank: AdaptiveCandidateBank,
    selections: dict[str, ExpertSelection],
) -> dict[str, dict[str, int]]:
    specs = {item.name: item for item in bank.family_specs_}
    distributions: dict[str, dict[str, int]] = {
        "expert": {},
        "view": {},
        "dilation_regime": {},
        "internal_representation": {},
        "pooling": {},
    }
    for expert, selection in selections.items():
        for family_name, indices in selection.family_indices.items():
            count = int(indices.size)
            spec = specs[family_name]
            attributes = {
                "expert": expert,
                "view": spec.view,
                "dilation_regime": (
                    "none"
                    if spec.dilation_regime is None
                    else str(spec.dilation_regime)
                ),
                "internal_representation": spec.internal_representation,
                "pooling": spec.pooling,
            }
            for category, value in attributes.items():
                distributions[category][value] = (
                    distributions[category].get(value, 0) + count
                )
    return distributions


def train_adaptive_rocket_pfn(config: ExperimentConfig) -> TrainingResult:
    """Fit one leakage-safe outer fold of the adaptive semantic ensemble."""

    config.validate()
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    total_start = time.monotonic()

    _log(config, "Checking local TabPFN-3 checkpoint access ...")
    ensure_tabpfn_checkpoint_access(config.model, config.seed)
    _log(config, f"Loading dataset from {config.data.dataset_path} ...")
    data_start = time.monotonic()
    data: DataBundle = prepare_data(config.data)
    train_classes = np.unique(data.train.labels)
    if not np.array_equal(train_classes, np.asarray([0, 1])):
        raise ValueError(
            "Adaptive RocketPFN requires binary labels encoded as 0 and 1; "
            f"received {train_classes.tolist()}"
        )
    _log(
        config,
        f"Loaded train={len(data.train.labels)}, "
        f"validation={len(data.validation.labels)}, test={len(data.test.labels)} "
        f"in {time.monotonic() - data_start:.1f}s.",
    )

    candidate_start = time.monotonic()
    bank = AdaptiveCandidateBank(config.model, config.seed)
    bank.progress_callback_ = lambda message: _log(config, message)
    _log(config, "Generating the structured multi-representation candidate bank ...")
    train_candidates = bank.fit_transform(data.train.values, None)
    bank.progress_callback_ = None
    _log(
        config,
        f"Generated {bank.n_candidate_features_:,} candidates in "
        f"{time.monotonic() - candidate_start:.1f}s across "
        f"{len(bank.family_specs_):,} structured families.",
    )

    oof_start = time.monotonic()
    _log(config, "Starting nested OOF feature selection and semantic TabPFN fits ...")
    oof_probabilities, oof_diagnostics = oof_semantic_predictions(
        data.train.values,
        data.train.labels,
        config.model,
        config.seed,
        log=lambda message: _log(config, message),
    )
    weights = learn_simplex_weights(
        oof_probabilities,
        data.train.labels,
        config.model.ensemble_weight_l2,
    )
    weight_map = {
        expert: float(weights[index])
        for index, expert in enumerate(EXPERT_NAMES)
    }
    oof_ensemble_probabilities = oof_probabilities @ weights
    oof_ensemble_metrics = _metrics(
        oof_ensemble_probabilities,
        data.train.labels,
        config.evaluation.threshold,
    )
    oof_expert_metrics = {
        expert: _metrics(
            oof_probabilities[:, index],
            data.train.labels,
            config.evaluation.threshold,
        )
        for index, expert in enumerate(EXPERT_NAMES)
    }
    _log(
        config,
        f"OOF adaptation finished in {time.monotonic() - oof_start:.1f}s; "
        f"learned weights={weight_map}.",
    )

    selection_start = time.monotonic()
    _log(config, "Selecting final stable features using the complete training split ...")
    selections = select_all_experts(
        train_candidates,
        data.train.labels,
        np.arange(len(data.train.labels), dtype=np.int64),
        bank.expert_families_,
        config.model,
        config.seed + 10_000,
    )
    selected_by_expert = {
        expert: selections[expert].num_features for expert in EXPERT_NAMES
    }
    num_selected_features = int(sum(selected_by_expert.values()))
    _log(
        config,
        f"Selected {num_selected_features:,} final features "
        f"({selected_by_expert}) in {time.monotonic() - selection_start:.1f}s.",
    )

    query_values = np.concatenate(
        (data.validation.values, data.test.values), axis=0
    )
    _log(config, "Transforming validation and test data with the fitted candidate bank ...")
    query_candidates = bank.transform(query_values)
    final_start = time.monotonic()
    final_seed = config.seed + 20_000
    expert_query_probabilities = predict_semantic_experts(
        train_candidates,
        query_candidates,
        data.train.labels,
        selections,
        config.model,
        final_seed,
        log=lambda message: _log(config, message),
    )
    query_probabilities = expert_query_probabilities @ weights
    _log(
        config,
        f"Final semantic TabPFN ensemble finished in "
        f"{time.monotonic() - final_start:.1f}s.",
    )

    validation_count = len(data.validation.labels)
    validation_probabilities = query_probabilities[:validation_count]
    test_probabilities = query_probabilities[validation_count:]
    validation_logits = _probabilities_to_logits(validation_probabilities)
    decision_threshold = (
        optimize_binary_threshold(
            torch.from_numpy(validation_logits).float(),
            torch.from_numpy(data.validation.labels).float(),
            metric_name=config.evaluation.threshold_metric,
        )
        if config.evaluation.calibrate_threshold_on_validation
        else config.evaluation.threshold
    )
    validation_metrics = _metrics(
        validation_probabilities, data.validation.labels, decision_threshold
    )
    test_metrics = _metrics(
        test_probabilities, data.test.labels, decision_threshold
    )
    selected_distribution = _selected_distribution(bank, selections)

    fitted_model = FittedAdaptiveRocketPFN(
        model_config=config.model,
        seed=final_seed,
        train_values=np.asarray(data.train.values),
        train_labels=np.asarray(data.train.labels),
        candidate_bank=bank,
        selections=selections,
        ensemble_weights=weights,
    )
    run_directory = config.output_dir / config.run_name
    run_directory.mkdir(parents=True, exist_ok=True)
    checkpoint_path = run_directory / "model.joblib"
    history_path = run_directory / "metrics.json"
    validation_predictions_path = run_directory / "validation_predictions.jsonl"
    predictions_path = run_directory / "test_predictions.jsonl"

    artifact = {
        "artifact_version": ARTIFACT_VERSION,
        "artifact_type": "adaptive_multi_representation_rocket_pfn_v3",
        "model": fitted_model,
        "experiment_config": _json_ready(asdict(config)),
        "normalizer": (
            None if data.normalizer is None else data.normalizer.state_dict()
        ),
        "time_grid": data.time_grid,
        "channel_names": data.channel_names,
        "num_candidate_features": bank.n_candidate_features_,
        "num_selected_features": num_selected_features,
        "selected_features_by_expert": selected_by_expert,
        "selection": _selection_payload(selections),
        "selected_distribution": selected_distribution,
        "ensemble_weights": weight_map,
        "oof_diagnostics": oof_diagnostics,
        "oof_expert_metrics": oof_expert_metrics,
        "oof_ensemble_metrics": oof_ensemble_metrics,
        "decision_threshold": decision_threshold,
        "validation_metrics": validation_metrics,
        "test_metrics": test_metrics,
    }
    joblib.dump(artifact, checkpoint_path, compress=3)

    metrics_payload = {
        "method": "Adaptive Multi-Representation RocketPFN",
        "tabpfn_version": "3",
        "num_candidate_features": bank.n_candidate_features_,
        "num_candidate_families": len(bank.family_specs_),
        "num_selected_features": num_selected_features,
        "selected_features_by_expert": selected_by_expert,
        "selection": _selection_payload(selections),
        "selected_distribution": selected_distribution,
        "ensemble_weights": weight_map,
        "oof_expert_metrics": oof_expert_metrics,
        "oof_ensemble_metrics": oof_ensemble_metrics,
        "decision_threshold": decision_threshold,
        "decision_threshold_source": (
            "validation_calibration"
            if config.evaluation.calibrate_threshold_on_validation
            else "fixed"
        ),
        "threshold_metric": config.evaluation.threshold_metric,
        "validation": validation_metrics,
        "test": test_metrics,
        "elapsed_seconds": time.monotonic() - total_start,
    }
    history_path.write_text(
        json.dumps(_json_ready(metrics_payload), indent=2) + "\n",
        encoding="utf-8",
    )
    validation_predictions_path.write_text(
        "\n".join(
            _prediction_rows(
                data.validation, validation_probabilities, decision_threshold
            )
        )
        + "\n",
        encoding="utf-8",
    )
    predictions_path.write_text(
        "\n".join(
            _prediction_rows(data.test, test_probabilities, decision_threshold)
        )
        + "\n",
        encoding="utf-8",
    )
    _log(
        config,
        f"Finished fold in {time.monotonic() - total_start:.1f}s; "
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
        num_candidate_features=bank.n_candidate_features_,
        num_selected_features=num_selected_features,
        selected_features_by_expert=selected_by_expert,
        ensemble_weights=weight_map,
        oof_ensemble_metrics=oof_ensemble_metrics,
        selected_distribution=selected_distribution,
    )


def load_trained_model(checkpoint_path: str | Path) -> dict[str, Any]:
    """Load an adaptive RocketPFN artifact without loading TabPFN weights."""

    artifact = joblib.load(checkpoint_path)
    version = artifact.get("artifact_version")
    if version != ARTIFACT_VERSION:
        raise ValueError(
            f"Unsupported adaptive RocketPFN artifact version {version!r}; "
            f"expected {ARTIFACT_VERSION}"
        )
    if artifact.get("artifact_type") != "adaptive_multi_representation_rocket_pfn_v3":
        raise ValueError("The checkpoint is not an adaptive RocketPFN artifact")
    return artifact


def load_training_result(config: ExperimentConfig) -> TrainingResult:
    """Load a complete, configuration-matched adaptive RocketPFN run."""

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
        raise FileNotFoundError(f"Incomplete adaptive RocketPFN run; missing: {missing}")

    artifact = load_trained_model(checkpoint_path)
    if artifact.get("experiment_config") != _json_ready(asdict(config)):
        raise ValueError(
            f"Saved adaptive RocketPFN run {run_directory} uses a different "
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
        num_candidate_features=int(artifact["num_candidate_features"]),
        num_selected_features=int(artifact["num_selected_features"]),
        selected_features_by_expert={
            str(name): int(value)
            for name, value in artifact["selected_features_by_expert"].items()
        },
        ensemble_weights={
            str(name): float(value)
            for name, value in artifact["ensemble_weights"].items()
        },
        oof_ensemble_metrics=dict(artifact["oof_ensemble_metrics"]),
        selected_distribution={
            str(category): {str(name): int(value) for name, value in values.items()}
            for category, values in artifact["selected_distribution"].items()
        },
    )


__all__ = [
    "ARTIFACT_VERSION",
    "TrainingResult",
    "load_trained_model",
    "load_training_result",
    "train_adaptive_rocket_pfn",
]
