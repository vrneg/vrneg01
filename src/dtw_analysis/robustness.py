"""Checkpoint sensitivity to controlled modality timing perturbations."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

try:
    from event_transformer.features import MODALITY_NAMES
    from modality_attribution.adapters import CheckpointAdapter
    from modality_attribution.metrics import binary_metrics, metric_decrease, sigmoid
except ModuleNotFoundError as error:
    if error.name not in {"event_transformer", "modality_attribution"}:
        raise
    from ..event_transformer.features import MODALITY_NAMES
    from ..modality_attribution.adapters import CheckpointAdapter
    from ..modality_attribution.metrics import binary_metrics, metric_decrease, sigmoid

from .config import DTWAnalysisConfig
from .perturbations import WARP_SCHEMA_VERSION


LOGGER = logging.getLogger(__name__)
ROBUSTNESS_INFERENCE_SCHEMA_VERSION = 2


@dataclass(frozen=True, slots=True)
class FoldRobustnessResult:
    fold_rows: list[dict[str, Any]]
    prediction_rows: list[dict[str, Any]]


def _cache_name(target: str, kind: str, magnitude: float) -> str:
    encoded = str(magnitude).replace("-", "minus-").replace(".", "p")
    return f"{target}_{kind}_{encoded}.npz"


def _cache_signature(
    adapter: CheckpointAdapter,
    event_step_milliseconds: float,
) -> dict[str, Any]:
    checkpoint = adapter.checkpoint_path.resolve()
    checkpoint_stat = checkpoint.stat()
    return {
        "checkpoint_path": str(checkpoint),
        "checkpoint_size": checkpoint_stat.st_size,
        "checkpoint_mtime_ns": checkpoint_stat.st_mtime_ns,
        "checkpoint_type": adapter.checkpoint_type,
        "event_step_milliseconds": event_step_milliseconds,
        "warp_schema_version": WARP_SCHEMA_VERSION,
        "robustness_inference_schema_version": (
            ROBUSTNESS_INFERENCE_SCHEMA_VERSION
        ),
    }


def _cached_logits(
    adapter: CheckpointAdapter,
    target: str,
    kind: str,
    magnitude: float,
    event_step_milliseconds: float,
    cache_dir: Path,
) -> np.ndarray | None:
    cache_path = cache_dir / _cache_name(target, kind, magnitude)
    signature = _cache_signature(adapter, event_step_milliseconds)
    if not cache_path.is_file():
        return None
    with np.load(cache_path) as cached:
        cached_signature = {
            key: cached[key].item() for key in signature if key in cached
        }
        logits = np.asarray(cached["logits"], dtype=np.float64)
    if cached_signature != signature:
        LOGGER.info("Ignoring stale timing predictions: %s", cache_path.name)
        return None
    if logits.shape != adapter.labels.shape:
        raise ValueError(f"Cached robustness logits have wrong shape: {cache_path}")
    if not np.all(np.isfinite(logits)):
        LOGGER.warning(
            "Ignoring cached timing predictions with non-finite logits: %s",
            cache_path.name,
        )
        return None
    LOGGER.info("Using cached timing predictions: %s", cache_path.name)
    return logits


def _write_cached_logits(
    adapter: CheckpointAdapter,
    logits: np.ndarray,
    target: str,
    kind: str,
    magnitude: float,
    event_step_milliseconds: float,
    cache_dir: Path,
) -> None:
    cache_path = cache_dir / _cache_name(target, kind, magnitude)
    signature = _cache_signature(adapter, event_step_milliseconds)
    cache_dir.mkdir(parents=True, exist_ok=True)
    temporary = cache_path.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, logits=logits, **signature)
    temporary.replace(cache_path)


def _validated_logits(
    adapter: CheckpointAdapter,
    logits: np.ndarray,
    *,
    description: str,
) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float64)
    if values.shape != adapter.labels.shape:
        raise RuntimeError(
            f"{description} changed the checkpoint sample count: "
            f"expected {adapter.labels.shape}, got {values.shape}"
        )
    if not np.all(np.isfinite(values)):
        invalid = np.flatnonzero(~np.isfinite(values))[:10].tolist()
        raise ValueError(
            f"Checkpoint inference returned non-finite logits for {description}; "
            f"sample indices={invalid}"
        )
    return values


def _condition_logits(
    adapter: CheckpointAdapter,
    modalities: tuple[str, ...],
    target: str,
    kind: str,
    magnitude: float,
    event_step_milliseconds: float,
    cache_dir: Path,
) -> np.ndarray:
    cached = _cached_logits(
        adapter,
        target,
        kind,
        magnitude,
        event_step_milliseconds,
        cache_dir,
    )
    if cached is not None:
        return cached
    LOGGER.info(
        "Running warped checkpoint inference: target=%s, kind=%s, magnitude=%s",
        target,
        kind,
        magnitude,
    )
    logits = _validated_logits(
        adapter,
        adapter.predict_timing_perturbation(
            modalities,
            kind=kind,
            magnitude=magnitude,
            event_step_milliseconds=event_step_milliseconds,
        ),
        description=(
            f"timing condition target={target}, kind={kind}, magnitude={magnitude}"
        ),
    )
    _write_cached_logits(
        adapter,
        logits,
        target,
        kind,
        magnitude,
        event_step_milliseconds,
        cache_dir,
    )
    return logits


def analyze_robustness_fold(
    adapter: CheckpointAdapter,
    config: DTWAnalysisConfig,
    cache_dir: Path,
) -> FoldRobustnessResult:
    event_step_milliseconds = (
        (config.event_window_end_seconds - config.event_window_start_seconds)
        / (config.event_num_time_points - 1)
        * 1_000.0
    )
    timing_step_milliseconds = adapter.timing_step_milliseconds(
        event_step_milliseconds
    )
    targets = [(name, (name,)) for name in MODALITY_NAMES]
    targets.append(("All", MODALITY_NAMES))
    conditions = [("shift", float(value)) for value in config.shift_steps]
    conditions.extend(("scale", float(value)) for value in config.time_scales)
    condition_specs = [
        (target, modalities, kind, magnitude)
        for target, modalities in targets
        for kind, magnitude in conditions
    ]

    baseline_logits = _cached_logits(
        adapter,
        "Baseline",
        "identity",
        0.0,
        event_step_milliseconds,
        cache_dir,
    )
    condition_logits: dict[tuple[str, str, float], np.ndarray] = {}
    missing_specs: list[tuple[str, tuple[str, ...], str, float]] = []
    for target, modalities, kind, magnitude in condition_specs:
        cached = _cached_logits(
            adapter,
            target,
            kind,
            magnitude,
            event_step_milliseconds,
            cache_dir,
        )
        if cached is None:
            missing_specs.append((target, modalities, kind, magnitude))
        else:
            condition_logits[(target, kind, magnitude)] = cached

    if baseline_logits is None or missing_specs:
        LOGGER.info(
            "[%s] Running repeated checkpoint inference: baseline=%s, "
            "missing_conditions=%d, cached_conditions=%d",
            adapter.fold_name,
            baseline_logits is None,
            len(missing_specs),
            len(condition_specs) - len(missing_specs),
        )
        inferred_baseline, inferred_conditions = adapter.predict_robustness_logits(
            [
                (modalities, kind, magnitude)
                for _, modalities, kind, magnitude in missing_specs
            ],
            event_step_milliseconds=event_step_milliseconds,
            include_baseline=baseline_logits is None,
        )
        if len(inferred_conditions) != len(missing_specs):
            raise RuntimeError(
                "Repeated checkpoint inference returned the wrong number of "
                "timing conditions"
            )
        if baseline_logits is None:
            if inferred_baseline is None:
                raise RuntimeError("Repeated checkpoint inference omitted the baseline")
            baseline_logits = _validated_logits(
                adapter, inferred_baseline, description="the baseline condition"
            )
            _write_cached_logits(
                adapter,
                baseline_logits,
                "Baseline",
                "identity",
                0.0,
                event_step_milliseconds,
                cache_dir,
            )
        for spec, raw_logits in zip(
            missing_specs, inferred_conditions, strict=True
        ):
            target, _, kind, magnitude = spec
            logits = _validated_logits(
                adapter,
                raw_logits,
                description=(
                    f"timing condition target={target}, kind={kind}, "
                    f"magnitude={magnitude}"
                ),
            )
            condition_logits[(target, kind, magnitude)] = logits
            _write_cached_logits(
                adapter,
                logits,
                target,
                kind,
                magnitude,
                event_step_milliseconds,
                cache_dir,
            )

    if baseline_logits is None:
        raise RuntimeError("Robustness inference produced no baseline logits")
    baseline_probabilities = sigmoid(baseline_logits)
    baseline_predictions = (baseline_probabilities >= adapter.threshold).astype(
        np.int64
    )
    baseline_metrics = binary_metrics(
        adapter.labels, baseline_logits, adapter.threshold
    )
    fold_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    total_conditions = len(targets) * len(conditions)
    condition_index = 0
    for target, modalities in targets:
        for kind, magnitude in conditions:
            condition_index += 1
            condition_started = time.perf_counter()
            LOGGER.info(
                "[%s] Timing condition %d/%d: target=%s, %s=%s",
                adapter.fold_name,
                condition_index,
                total_conditions,
                target,
                kind,
                magnitude,
            )
            logits = condition_logits[(target, kind, magnitude)]
            probabilities = sigmoid(logits)
            predictions = (probabilities >= adapter.threshold).astype(np.int64)
            metrics = binary_metrics(adapter.labels, logits, adapter.threshold)
            decreases = metric_decrease(baseline_metrics, metrics)
            correlation = (
                float(np.corrcoef(baseline_logits, logits)[0, 1])
                if np.std(baseline_logits) > 0.0 and np.std(logits) > 0.0
                else float("nan")
            )
            fold_rows.append(
                {
                    "fold": adapter.fold_name,
                    "checkpoint_type": adapter.checkpoint_type,
                    "target": target,
                    "kind": kind,
                    "magnitude": magnitude,
                    "shift_milliseconds": (
                        magnitude * timing_step_milliseconds
                        if kind == "shift"
                        else None
                    ),
                    "num_test_samples": int(adapter.labels.size),
                    "macro_f1_decrease": decreases["macro_f1"],
                    "balanced_accuracy_decrease": decreases["balanced_accuracy"],
                    "average_precision_decrease": decreases["average_precision"],
                    "roc_auc_decrease": decreases["roc_auc"],
                    "loss_increase": decreases["loss"],
                    "prediction_flip_rate": float(
                        np.mean(predictions != baseline_predictions)
                    ),
                    "mean_absolute_probability_change": float(
                        np.mean(np.abs(probabilities - baseline_probabilities))
                    ),
                    "logit_correlation": correlation,
                    **{
                        f"baseline_{key}": value
                        for key, value in baseline_metrics.items()
                    },
                    **{f"perturbed_{key}": value for key, value in metrics.items()},
                }
            )
            for sample_index, sample_id in enumerate(adapter.sample_ids):
                prediction_rows.append(
                    {
                        "fold": adapter.fold_name,
                        "sample_index": sample_index,
                        "sample_key": f"{adapter.fold_name}:{sample_index}",
                        "sample_id": sample_id,
                        "group_id": adapter.group_ids[sample_index],
                        "label": int(adapter.labels[sample_index]),
                        "target": target,
                        "kind": kind,
                        "magnitude": magnitude,
                        "shift_milliseconds": (
                            magnitude * timing_step_milliseconds
                            if kind == "shift"
                            else None
                        ),
                        "threshold": adapter.threshold,
                        "baseline_logit": float(baseline_logits[sample_index]),
                        "perturbed_logit": float(logits[sample_index]),
                        "baseline_probability": float(
                            baseline_probabilities[sample_index]
                        ),
                        "perturbed_probability": float(probabilities[sample_index]),
                        "baseline_prediction": int(
                            baseline_predictions[sample_index]
                        ),
                        "perturbed_prediction": int(predictions[sample_index]),
                        "prediction_flipped": bool(
                            predictions[sample_index]
                            != baseline_predictions[sample_index]
                        ),
                    }
                )
            LOGGER.info(
                "[%s] Finished condition %d/%d in %.1fs: |Δp|=%.4f, flip=%.1f%%, "
                "macro-F1 decrease=%.4f",
                adapter.fold_name,
                condition_index,
                total_conditions,
                time.perf_counter() - condition_started,
                np.mean(np.abs(probabilities - baseline_probabilities)),
                np.mean(predictions != baseline_predictions) * 100.0,
                decreases["macro_f1"],
            )
    return FoldRobustnessResult(fold_rows, prediction_rows)
