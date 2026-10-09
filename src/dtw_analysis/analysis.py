"""Orchestration, aggregation, and reporting for standalone DTW analyses."""

from __future__ import annotations

import csv
import json
import logging
import math
import platform
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from importlib.metadata import version
from multiprocessing import get_context
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import rankdata, spearmanr

try:
    from event_transformer.features import MODALITY_NAMES
    from modality_attribution.adapters import load_checkpoint_adapter
    from modality_attribution.metrics import binary_metrics, metric_decrease
except ModuleNotFoundError as error:
    if error.name not in {"event_transformer", "modality_attribution"}:
        raise
    from ..event_transformer.features import MODALITY_NAMES
    from ..modality_attribution.adapters import load_checkpoint_adapter
    from ..modality_attribution.metrics import binary_metrics, metric_decrease

from .config import DTWAnalysisConfig
from .data import duplicate_sample_rows, load_dtw_fold, validate_dtw_folds
from .distance import probabilities_to_logits
from .robustness import FoldRobustnessResult, analyze_robustness_fold
from .separability import analyze_dtw_fold
from .statistics import (
    descriptive,
    macro_f1_interval,
    mean_interval,
    paired_macro_f1_decrease_interval,
)


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class DTWAnalysisResult:
    summary_path: Path
    report_path: Path
    summary: dict[str, Any]


def _run_robustness_fold_worker(
    checkpoint_path: Path,
    config: DTWAnalysisConfig,
    cache_dir: Path,
    log_level: int,
    configure_logging: bool,
) -> tuple[str, FoldRobustnessResult, float]:
    """Load and evaluate one checkpoint inside the calling process."""

    if configure_logging:
        logging.basicConfig(
            level=log_level,
            format="%(asctime)s | %(processName)s | %(levelname)s | %(message)s",
            datefmt="%H:%M:%S",
            force=True,
        )
    started = time.perf_counter()
    LOGGER.info("Loading checkpoint adapter from %s", checkpoint_path)
    adapter = load_checkpoint_adapter(
        checkpoint_path,
        device=config.device,
        batch_size=config.batch_size,
    )
    LOGGER.info(
        "Running timing perturbations for %s (%d held-out rows)",
        adapter.fold_name,
        len(adapter.labels),
    )
    result = analyze_robustness_fold(adapter, config, cache_dir)
    elapsed = time.perf_counter() - started
    LOGGER.info("Completed robustness fold %s in %.1fs", adapter.fold_name, elapsed)
    return adapter.fold_name, result, elapsed


def _run_robustness_folds(
    paths: list[Path],
    config: DTWAnalysisConfig,
    output_dir: Path,
) -> list[FoldRobustnessResult]:
    """Evaluate fold checkpoints sequentially or in CUDA-safe spawned workers."""

    worker_count = min(config.fold_jobs, len(paths))
    log_level = LOGGER.getEffectiveLevel()
    if worker_count == 1:
        results: list[FoldRobustnessResult] = []
        for fold_index, path in enumerate(paths, 1):
            LOGGER.info(
                "Robustness fold %d/%d: %s", fold_index, len(paths), path.parent.name
            )
            _, result, _ = _run_robustness_fold_worker(
                path,
                config,
                output_dir / "robustness_cache" / path.parent.name,
                log_level,
                False,
            )
            results.append(result)
        return results

    LOGGER.info(
        "Running %d robustness folds with %d spawned worker processes",
        len(paths),
        worker_count,
    )
    ordered: list[FoldRobustnessResult | None] = [None] * len(paths)
    context = get_context("spawn")
    with ProcessPoolExecutor(
        max_workers=worker_count,
        mp_context=context,
    ) as executor:
        futures = {
            executor.submit(
                _run_robustness_fold_worker,
                path,
                config,
                output_dir / "robustness_cache" / path.parent.name,
                log_level,
                True,
            ): index
            for index, path in enumerate(paths)
        }
        for future in as_completed(futures):
            index = futures[future]
            try:
                fold_name, result, elapsed = future.result()
            except Exception as error:
                raise RuntimeError(
                    f"Robustness analysis failed for {paths[index].parent.name}"
                ) from error
            ordered[index] = result
            LOGGER.info(
                "Collected robustness fold %s (%d/%d) after %.1fs",
                fold_name,
                sum(item is not None for item in ordered),
                len(paths),
                elapsed,
            )
    if any(result is None for result in ordered):
        raise RuntimeError("Not every robustness fold worker returned a result")
    return [result for result in ordered if result is not None]


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            safe = _json_safe(row)
            writer.writerow(
                {
                    key: (
                        json.dumps(value, sort_keys=True)
                        if isinstance(value, (dict, list))
                        else value
                    )
                    for key, value in safe.items()
                }
            )


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.write_text(
        "\n".join(json.dumps(_json_safe(row)) for row in rows) + "\n",
        encoding="utf-8",
    )


def _aggregate_dtw(
    fold_rows: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    presence_predictions: list[dict[str, Any]],
    *,
    config: DTWAnalysisConfig,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    modality_rows: list[dict[str, Any]] = []
    presence_rows: list[dict[str, Any]] = []
    total_samples = len({row["sample_key"] for row in presence_predictions})
    for modality_index, modality in enumerate(MODALITY_NAMES):
        LOGGER.info(
            "Aggregating DTW modality %d/%d: %s",
            modality_index + 1,
            len(MODALITY_NAMES),
            modality,
        )
        rows = [row for row in predictions if row["modality"] == modality]
        if rows:
            labels = np.asarray([row["label"] for row in rows], dtype=np.int64)
            probabilities = np.asarray([row["probability"] for row in rows])
            predictions_array = np.asarray(
                [row["prediction"] for row in rows], dtype=np.int64
            )
            groups = [str(row["group_id"]) for row in rows]
            metrics = binary_metrics(
                labels, probabilities_to_logits(probabilities), 0.5
            )
            lower, upper = macro_f1_interval(
                labels,
                predictions_array,
                groups,
                seed=config.seed + modality_index,
                samples=config.bootstrap_samples,
            )
            modality_folds = [
                row
                for row in fold_rows
                if row["modality"] == modality and row.get("status") == "complete"
            ]
            unavailable_folds = [
                row
                for row in fold_rows
                if row["modality"] == modality
                and row.get("status") == "unavailable"
            ]
            settings = Counter(
                (float(row["selected_window"]), int(row["selected_k"]))
                for row in modality_folds
            )
            modality_rows.append(
                {
                    "modality": modality,
                    "num_conditional_test_samples": len(rows),
                    "conditional_coverage": len(rows) / total_samples,
                    "num_complete_folds": len(modality_folds),
                    "num_unavailable_folds": len(unavailable_folds),
                    "unavailable_reasons": sorted(
                        {str(row["reason"]) for row in unavailable_folds}
                    ),
                    "macro_f1_ci_lower": lower,
                    "macro_f1_ci_upper": upper,
                    "mean_distance_margin": float(
                        np.mean([row["distance_margin"] for row in rows])
                    ),
                    "mean_nearest_same_class_distance": float(
                        np.mean([row["nearest_same_class_distance"] for row in rows])
                    ),
                    "mean_nearest_different_class_distance": float(
                        np.mean(
                            [
                                row["nearest_different_class_distance"]
                                for row in rows
                            ]
                        )
                    ),
                    "fold_macro_f1": descriptive(
                        [float(row["macro_f1"]) for row in modality_folds]
                    ),
                    "selected_settings": [
                        {"window": window, "k": k, "num_folds": count}
                        for (window, k), count in sorted(settings.items())
                    ],
                    **metrics,
                }
            )
        state_rows = [
            row for row in presence_predictions if row["modality"] == modality
        ]
        labels = np.asarray([row["label"] for row in state_rows], dtype=np.int64)
        probabilities = np.asarray([row["probability"] for row in state_rows])
        predictions_array = np.asarray(
            [row["prediction"] for row in state_rows], dtype=np.int64
        )
        groups = [str(row["group_id"]) for row in state_rows]
        metrics = binary_metrics(labels, probabilities_to_logits(probabilities), 0.5)
        lower, upper = macro_f1_interval(
            labels,
            predictions_array,
            groups,
            seed=config.seed + 100 + modality_index,
            samples=config.bootstrap_samples,
        )
        presence_rows.append(
            {
                "modality": modality,
                "num_test_samples": len(state_rows),
                "observed_coverage": float(
                    np.mean([row["observed"] for row in state_rows])
                ),
                "macro_f1_ci_lower": lower,
                "macro_f1_ci_upper": upper,
                **metrics,
            }
        )
    modality_rows.sort(key=lambda row: row["macro_f1"], reverse=True)
    for rank, row in enumerate(modality_rows, 1):
        row["rank"] = rank
    modality_rows = [{"rank": row.pop("rank"), **row} for row in modality_rows]
    return modality_rows, presence_rows


def _aggregate_robustness(
    fold_rows: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    *,
    config: DTWAnalysisConfig,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    conditions = sorted(
        {
            (str(row["target"]), str(row["kind"]), float(row["magnitude"]))
            for row in predictions
        }
    )
    summary_rows: list[dict[str, Any]] = []
    for condition_index, (target, kind, magnitude) in enumerate(conditions):
        LOGGER.info(
            "Bootstrapping robustness condition %d/%d: target=%s, %s=%s",
            condition_index + 1,
            len(conditions),
            target,
            kind,
            magnitude,
        )
        rows = [
            row
            for row in predictions
            if row["target"] == target
            and row["kind"] == kind
            and float(row["magnitude"]) == magnitude
        ]
        labels = np.asarray([row["label"] for row in rows], dtype=np.int64)
        baseline_logits = np.asarray([row["baseline_logit"] for row in rows])
        perturbed_logits = np.asarray([row["perturbed_logit"] for row in rows])
        thresholds = np.asarray([row["threshold"] for row in rows])
        baseline_probabilities = np.asarray(
            [row["baseline_probability"] for row in rows]
        )
        perturbed_probabilities = np.asarray(
            [row["perturbed_probability"] for row in rows]
        )
        baseline_predictions = np.asarray(
            [row["baseline_prediction"] for row in rows], dtype=np.int64
        )
        perturbed_predictions = np.asarray(
            [row["perturbed_prediction"] for row in rows], dtype=np.int64
        )
        groups = [str(row["group_id"]) for row in rows]
        baseline_metrics = binary_metrics(labels, baseline_logits, thresholds)
        perturbed_metrics = binary_metrics(labels, perturbed_logits, thresholds)
        decreases = metric_decrease(baseline_metrics, perturbed_metrics)
        macro_lower, macro_upper = paired_macro_f1_decrease_interval(
            labels,
            baseline_predictions,
            perturbed_predictions,
            groups,
            seed=config.seed + condition_index,
            samples=config.bootstrap_samples,
        )
        flips = (baseline_predictions != perturbed_predictions).astype(float)
        probability_changes = np.abs(
            perturbed_probabilities - baseline_probabilities
        )
        flip_lower, flip_upper = mean_interval(
            flips,
            groups,
            seed=config.seed + 1_000 + condition_index,
            samples=config.bootstrap_samples,
        )
        probability_lower, probability_upper = mean_interval(
            probability_changes,
            groups,
            seed=config.seed + 2_000 + condition_index,
            samples=config.bootstrap_samples,
        )
        correlation = (
            float(np.corrcoef(baseline_logits, perturbed_logits)[0, 1])
            if np.std(baseline_logits) > 0.0 and np.std(perturbed_logits) > 0.0
            else float("nan")
        )
        matching_folds = [
            row
            for row in fold_rows
            if row["target"] == target
            and row["kind"] == kind
            and float(row["magnitude"]) == magnitude
        ]
        shift_durations = {
            float(row["shift_milliseconds"])
            for row in matching_folds
            if row.get("shift_milliseconds") is not None
        }
        if len(shift_durations) > 1:
            raise ValueError(
                "Fold checkpoints map shift steps to different physical durations"
            )
        summary_rows.append(
            {
                "target": target,
                "kind": kind,
                "magnitude": magnitude,
                "shift_milliseconds": (
                    next(iter(shift_durations)) if shift_durations else None
                ),
                "num_test_samples": len(rows),
                "macro_f1_decrease": decreases["macro_f1"],
                "macro_f1_decrease_ci_lower": macro_lower,
                "macro_f1_decrease_ci_upper": macro_upper,
                "balanced_accuracy_decrease": decreases["balanced_accuracy"],
                "average_precision_decrease": decreases["average_precision"],
                "roc_auc_decrease": decreases["roc_auc"],
                "loss_increase": decreases["loss"],
                "prediction_flip_rate": float(flips.mean()),
                "prediction_flip_rate_ci_lower": flip_lower,
                "prediction_flip_rate_ci_upper": flip_upper,
                "mean_absolute_probability_change": float(probability_changes.mean()),
                "probability_change_ci_lower": probability_lower,
                "probability_change_ci_upper": probability_upper,
                "logit_correlation": correlation,
                **{
                    f"baseline_{key}": value
                    for key, value in baseline_metrics.items()
                },
                **{
                    f"perturbed_{key}": value
                    for key, value in perturbed_metrics.items()
                },
                "fold_macro_f1_decrease": descriptive(
                    [float(row["macro_f1_decrease"]) for row in matching_folds]
                ),
            }
        )
    target_rows: list[dict[str, Any]] = []
    for target in (*MODALITY_NAMES, "All"):
        rows = [row for row in summary_rows if row["target"] == target]
        target_rows.append(
            {
                "target": target,
                "mean_absolute_probability_change": float(
                    np.mean([row["mean_absolute_probability_change"] for row in rows])
                ),
                "mean_prediction_flip_rate": float(
                    np.mean([row["prediction_flip_rate"] for row in rows])
                ),
                "worst_macro_f1_decrease": float(
                    max(row["macro_f1_decrease"] for row in rows)
                ),
            }
        )
    modality_target_rows = [row for row in target_rows if row["target"] != "All"]
    modality_target_rows.sort(
        key=lambda row: row["mean_absolute_probability_change"], reverse=True
    )
    for rank, row in enumerate(modality_target_rows, 1):
        row["rank"] = rank
    all_row = next(row for row in target_rows if row["target"] == "All")
    target_rows = [
        {"rank": row.pop("rank"), **row} for row in modality_target_rows
    ] + [{"rank": None, **all_row}]
    return summary_rows, target_rows


def _ranks(values: dict[str, float]) -> dict[str, float]:
    names = list(MODALITY_NAMES)
    ranks = rankdata([-values[name] for name in names], method="average")
    return {name: float(rank) for name, rank in zip(names, ranks, strict=True)}


def _compare_attribution(
    path: Path,
    dtw_rows: list[dict[str, Any]],
    robustness_targets: list[dict[str, Any]],
    *,
    checkpoint_type: str,
    checkpoint_paths: list[Path],
) -> dict[str, Any]:
    attribution = json.loads(path.read_text(encoding="utf-8"))
    if attribution.get("checkpoint_type") != checkpoint_type:
        raise ValueError(
            "Attribution comparison must use the same checkpoint classifier type"
        )
    attribution_paths = {
        str(Path(row["path"]).resolve()) for row in attribution.get("checkpoints", [])
    }
    expected_paths = {str(checkpoint.resolve()) for checkpoint in checkpoint_paths}
    if attribution_paths != expected_paths:
        raise ValueError(
            "Attribution comparison must use exactly the same fold checkpoints"
        )
    attribution_rows = {
        row["modality"]: row for row in attribution["modality_importance"]
    }
    dtw = {row["modality"]: float(row["macro_f1"]) for row in dtw_rows}
    ablation = {
        name: float(attribution_rows[name]["pooled_macro_f1_decrease"])
        for name in MODALITY_NAMES
    }
    shapley = {
        name: float(attribution_rows[name]["mean_absolute_shapley"])
        for name in MODALITY_NAMES
    }
    robustness = {
        row["target"]: float(row["mean_absolute_probability_change"])
        for row in robustness_targets
        if row["target"] != "All"
    }
    measures: dict[str, dict[str, float]] = {
        "dtw_conditional_macro_f1": dtw,
        "attribution_ablation": ablation,
        "attribution_absolute_shapley": shapley,
    }
    if len(robustness) == len(MODALITY_NAMES):
        measures["timing_sensitivity"] = robustness
    correlations = []
    names = list(MODALITY_NAMES)
    measure_names = list(measures)
    for left_index, left in enumerate(measure_names):
        for right in measure_names[left_index + 1 :]:
            result = spearmanr(
                [measures[left][name] for name in names],
                [measures[right][name] for name in names],
            )
            correlations.append(
                {
                    "left": left,
                    "right": right,
                    "rho": float(result.statistic),
                    "p_value": float(result.pvalue),
                    "num_modalities": len(names),
                }
            )
    rank_maps = {name: _ranks(values) for name, values in measures.items()}
    table = []
    for modality in names:
        row: dict[str, Any] = {"modality": modality}
        for measure, values in measures.items():
            row[measure] = values[modality]
            row[f"{measure}_rank"] = rank_maps[measure][modality]
        ranks = [rank_maps[measure][modality] for measure in measures]
        row["rank_range"] = float(max(ranks) - min(ranks))
        row["interpretation"] = "discrepant" if row["rank_range"] >= 4 else "concordant"
        table.append(row)
    return {
        "attribution_summary_path": str(path.resolve()),
        "spearman_rank_correlations": correlations,
        "modality_comparison": table,
    }


def _report(summary: dict[str, Any]) -> str:
    lines = [
        "# DTW temporal separability and timing robustness",
        "",
        f"Analysis mode: `{summary['mode']}` across {summary['num_folds']} outer folds "
        f"and {summary['num_outer_test_samples']} held-out rows.",
        "",
    ]
    lines.extend(
        [
            "The DTW experiment uses checkpoint data configurations only: scaling and "
            "PCA are learned on outer training trajectories, DTW window and k are "
            "selected on validation, and train+validation references are rebuilt "
            "before "
            "one untouched outer-test evaluation. The robustness experiment keeps all "
            "checkpoint weights fixed and runs inference on label-independent timing "
            "warps.",
            "",
        ]
    )
    if "dtw" in summary:
        modality_summary = summary["dtw"]["modality_summary"]
        if not modality_summary:
            lines.extend(
                [
                    "## DTW temporal separability",
                    "",
                    "No modality had sufficient observed train, validation, and test "
                    "trajectories for conditional DTW evaluation. See the fold and "
                    "presence tables for diagnostics.",
                    "",
                ]
            )
        else:
            top = modality_summary[0]
            lines.extend(
                [
                    "## DTW temporal separability",
                    "",
                    f"**{top['modality']}** ranks first by conditional pooled macro F1 "
                    f"({top['macro_f1']:.4f}, 95% session-cluster CI "
                    f"[{top['macro_f1_ci_lower']:.4f}, "
                    f"{top['macro_f1_ci_upper']:.4f}]) "
                    f"at {top['conditional_coverage']:.1%} coverage.",
                    "",
                    "| Rank | Modality | Conditional macro F1 | 95% CI | Coverage |",
                    "|---:|---|---:|---:|---:|",
                ]
            )
            for row in modality_summary:
                lines.append(
                    f"| {row['rank']} | {row['modality']} | "
                    f"{row['macro_f1']:.4f} | "
                    f"[{row['macro_f1_ci_lower']:.4f}, "
                    f"{row['macro_f1_ci_upper']:.4f}] | "
                    f"{row['conditional_coverage']:.1%} |"
                )
            lines.extend(
                [
                    "",
                    "DTW results are conditional on the modality being observed. "
                    "Presence-only classification is reported separately and is not "
                    "part of this ranking.",
                    "",
                ]
            )
    if "robustness" in summary:
        targets = [
            row
            for row in summary["robustness"]["target_summary"]
            if row["target"] != "All"
        ]
        top = targets[0]
        lines.extend(
            [
                "## Checkpoint timing robustness",
                "",
                f"**{top['target']}** is most timing-sensitive by average absolute "
                f"probability change ({top['mean_absolute_probability_change']:.4f}).",
                "",
                "| Rank | Target | Mean absolute probability change | Mean flip rate | "
                "Worst macro-F1 decrease |",
                "|---:|---|---:|---:|---:|",
            ]
        )
        for row in targets:
            lines.append(
                f"| {row['rank']} | {row['target']} | "
                f"{row['mean_absolute_probability_change']:.4f} | "
                f"{row['mean_prediction_flip_rate']:.4f} | "
                f"{row['worst_macro_f1_decrease']:.4f} |"
            )
        lines.extend(
            [
                "",
                "Timing sensitivity is a fitted-model robustness property, not a "
                "modality importance or causal effect.",
                "",
            ]
        )
    if "attribution_comparison" in summary:
        lines.extend(
            [
                "## Comparison with modality attribution",
                "",
                "Scores remain on their original scales. Spearman correlations compare "
                "rankings only; no composite score is created.",
                "",
            ]
        )
        for row in summary["attribution_comparison"]["spearman_rank_correlations"]:
            lines.append(
                f"- {row['left']} vs. {row['right']}: ρ={row['rho']:.3f}, "
                f"p={row['p_value']:.3f}."
            )
        lines.append("")
    lines.extend(["## Interpretation limits", ""])
    lines.extend(f"- {warning}" for warning in summary["warnings"])
    lines.append("")
    return "\n".join(lines)


def run_analysis(
    checkpoint_paths: list[str | Path],
    config: DTWAnalysisConfig,
) -> DTWAnalysisResult:
    config.validate()
    analysis_started = time.perf_counter()
    output_dir = config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = [Path(path) for path in checkpoint_paths]
    LOGGER.info(
        "Preparing fold data for %d checkpoints (this reconstructs train, validation, "
        "and test trajectories)",
        len(paths),
    )
    folds = []
    for index, path in enumerate(paths, 1):
        started = time.perf_counter()
        LOGGER.info("Loading fold %d/%d from %s", index, len(paths), path)
        fold = load_dtw_fold(path, config)
        folds.append(fold)
        LOGGER.info(
            "Loaded %s in %.1fs: train=%d, validation=%d, test=%d",
            fold.fold_name,
            time.perf_counter() - started,
            len(fold.train.labels),
            len(fold.validation.labels),
            len(fold.test.labels),
        )
    duplicate_rows = duplicate_sample_rows(folds)
    duplicate_path = output_dir / "duplicate_samples.csv"
    if duplicate_rows:
        _write_csv(duplicate_path, duplicate_rows)
        LOGGER.warning(
            "Detected %d duplicate row occurrences; audit written to %s",
            len(duplicate_rows),
            duplicate_path,
        )
    else:
        duplicate_path.write_text("", encoding="utf-8")
    duplicate_count = validate_dtw_folds(
        folds, allow_duplicate_samples=config.allow_duplicate_samples
    )
    LOGGER.info(
        "Fold validation passed: %d folds, %d outer-test rows, %d duplicate-ID issues",
        len(folds),
        sum(len(fold.test.labels) for fold in folds),
        duplicate_count,
    )
    if len({fold.checkpoint_type for fold in folds}) != 1:
        raise ValueError("All fold checkpoints must use the same classifier type")
    summary: dict[str, Any] = {
        "analysis_version": 1,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "software_versions": {
            "python": platform.python_version(),
            "aeon": version("aeon"),
            "numpy": version("numpy"),
            "scikit_learn": version("scikit-learn"),
        },
        "mode": config.mode,
        "checkpoint_type": folds[0].checkpoint_type,
        "num_folds": len(folds),
        "num_outer_test_samples": sum(len(fold.test.labels) for fold in folds),
        "configuration": asdict(config),
        "checkpoints": [
            {
                "path": str(path.resolve()),
                "size_bytes": path.stat().st_size,
                "modified_time_ns": path.stat().st_mtime_ns,
            }
            for path in paths
        ],
        "warnings": [
            "DTW scores measure predictive temporal separability, not causal "
            "influence.",
            "Conditional DTW scores must always be interpreted together with coverage.",
            "Robustness warps are synthetic stress tests and not augmented natural "
            "observations.",
            "Fold dispersion is descriptive because cross-validation training sets "
            "overlap.",
        ],
    }
    if duplicate_count:
        summary["warnings"].append(
            f"The source folds contain {duplicate_count} duplicate sample-ID issues; "
            "rows were retained and audited in duplicate_samples.csv."
        )

    dtw_rows: list[dict[str, Any]] = []
    dtw_predictions: list[dict[str, Any]] = []
    presence_fold_rows: list[dict[str, Any]] = []
    presence_predictions: list[dict[str, Any]] = []
    alignment_examples: list[dict[str, Any]] = []
    if config.mode in {"dtw", "both"}:
        LOGGER.info("Starting modality-wise DTW separability analysis")
        for fold_index, fold in enumerate(folds, 1):
            started = time.perf_counter()
            LOGGER.info(
                "DTW fold %d/%d: %s", fold_index, len(folds), fold.fold_name
            )
            result = analyze_dtw_fold(fold, config)
            dtw_rows.extend(result.fold_rows)
            dtw_predictions.extend(result.prediction_rows)
            presence_fold_rows.extend(result.presence_rows)
            presence_predictions.extend(result.presence_prediction_rows)
            alignment_examples.extend(result.alignment_examples)
            LOGGER.info(
                "Completed DTW fold %s in %.1fs",
                fold.fold_name,
                time.perf_counter() - started,
            )
        started = time.perf_counter()
        LOGGER.info(
            "Aggregating DTW predictions with %d session-cluster bootstrap samples",
            config.bootstrap_samples,
        )
        modality_summary, presence_summary = _aggregate_dtw(
            dtw_rows,
            dtw_predictions,
            presence_predictions,
            config=config,
        )
        LOGGER.info("DTW aggregation completed in %.1fs", time.perf_counter() - started)
        summary["dtw"] = {
            "primary_ranking_metric": "conditional_pooled_macro_f1",
            "modality_summary": modality_summary,
            "presence_summary": presence_summary,
        }
        _write_csv(output_dir / "dtw_modality_summary.csv", modality_summary)
        _write_csv(output_dir / "dtw_fold_metrics.csv", dtw_rows)
        _write_jsonl(output_dir / "dtw_predictions.jsonl", dtw_predictions)
        _write_csv(output_dir / "presence_summary.csv", presence_summary)
        _write_csv(output_dir / "presence_fold_metrics.csv", presence_fold_rows)
        _write_jsonl(output_dir / "alignment_examples.jsonl", alignment_examples)

    robustness_fold_rows: list[dict[str, Any]] = []
    robustness_predictions: list[dict[str, Any]] = []
    robustness_targets: list[dict[str, Any]] = []
    if config.mode in {"robustness", "both"}:
        LOGGER.info("Starting frozen-checkpoint timing robustness analysis")
        for result in _run_robustness_folds(paths, config, output_dir):
            robustness_fold_rows.extend(result.fold_rows)
            robustness_predictions.extend(result.prediction_rows)
        started = time.perf_counter()
        LOGGER.info(
            "Aggregating robustness predictions with %d session-cluster bootstrap "
            "samples",
            config.bootstrap_samples,
        )
        robustness_summary, robustness_targets = _aggregate_robustness(
            robustness_fold_rows,
            robustness_predictions,
            config=config,
        )
        LOGGER.info(
            "Robustness aggregation completed in %.1fs",
            time.perf_counter() - started,
        )
        summary["robustness"] = {
            "primary_sensitivity_metric": "mean_absolute_probability_change",
            "condition_summary": robustness_summary,
            "target_summary": robustness_targets,
        }
        _write_csv(output_dir / "robustness_summary.csv", robustness_summary)
        _write_csv(output_dir / "robustness_target_summary.csv", robustness_targets)
        _write_csv(output_dir / "robustness_fold_metrics.csv", robustness_fold_rows)
        _write_jsonl(
            output_dir / "robustness_predictions.jsonl", robustness_predictions
        )

    if config.attribution_summary_path is not None:
        if not dtw_rows:
            raise ValueError("Attribution comparison requires DTW separability mode")
        summary["attribution_comparison"] = _compare_attribution(
            config.attribution_summary_path,
            summary["dtw"]["modality_summary"],
            robustness_targets,
            checkpoint_type=folds[0].checkpoint_type,
            checkpoint_paths=paths,
        )
        _write_csv(
            output_dir / "attribution_comparison.csv",
            summary["attribution_comparison"]["modality_comparison"],
        )

    summary_path = output_dir / "summary.json"
    report_path = output_dir / "report.md"
    summary_path.write_text(
        json.dumps(_json_safe(summary), indent=2) + "\n", encoding="utf-8"
    )
    report_path.write_text(_report(summary), encoding="utf-8")
    LOGGER.info(
        "Analysis completed in %.1fs; summary=%s; report=%s",
        time.perf_counter() - analysis_started,
        summary_path,
        report_path,
    )
    return DTWAnalysisResult(summary_path, report_path, summary)
