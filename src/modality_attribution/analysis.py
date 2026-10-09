"""Cross-validated post-hoc modality attribution and report generation."""

from __future__ import annotations

import csv
import copy
import json
import logging
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .adapters import CheckpointAdapter, load_checkpoint_adapter
from .metrics import (
    METRIC_NAMES,
    binary_metrics,
    bootstrap_mean_interval,
    metric_decrease,
)
from .shapley import sampled_modality_shapley


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class AnalysisConfig:
    output_dir: Path
    device: str = "cpu"
    batch_size: int = 64
    run_shapley: bool = True
    shapley_orderings: int = 32
    seed: int = 42
    bootstrap_samples: int = 2_000
    allow_duplicate_samples: bool = False

    def validate(self) -> None:
        if self.batch_size < 1:
            raise ValueError("batch_size must be positive")
        if (
            self.run_shapley
            and (self.shapley_orderings < 2 or self.shapley_orderings % 2)
        ):
            raise ValueError("shapley_orderings must be an even number of at least 2")
        if self.bootstrap_samples < 0:
            raise ValueError("bootstrap_samples cannot be negative")


@dataclass(frozen=True, slots=True)
class AnalysisResult:
    summary_path: Path
    report_path: Path
    fold_importance_path: Path
    modality_importance_path: Path
    sample_attributions_path: Path
    summary: dict[str, Any]


def _anchored_path_identity(path: str | Path, anchor: str) -> tuple[str, ...] | None:
    """Return the stable part of a path beginning at its last named anchor."""

    parts = Path(path).parts
    indices = [index for index, part in enumerate(parts) if part == anchor]
    if not indices:
        return None
    return tuple(parts[indices[-1] :])


def load_analysis_result(
    output_dir: str | Path,
    checkpoint_paths: list[str | Path],
) -> AnalysisResult:
    """Restore a completed post-hoc result before resumable retraining."""

    root = Path(output_dir)
    paths = {
        "summary": root / "summary.json",
        "report": root / "report.md",
        "fold_importance": root / "fold_importance.csv",
        "modality_importance": root / "modality_importance.csv",
        "sample_attributions": root / "sample_attributions.jsonl",
    }
    missing = [path for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "--retraining-only requires completed post-hoc outputs: "
            + ", ".join(str(path) for path in missing)
        )
    summary = json.loads(paths["summary"].read_text(encoding="utf-8"))
    saved_checkpoints = [
        str(Path(entry["path"]).resolve())
        for entry in summary.get("checkpoints", [])
    ]
    requested_checkpoints = [str(Path(path).resolve()) for path in checkpoint_paths]
    exact_match = saved_checkpoints == requested_checkpoints
    saved_identities = [
        _anchored_path_identity(path, "outputs") for path in saved_checkpoints
    ]
    requested_identities = [
        _anchored_path_identity(path, "outputs") for path in requested_checkpoints
    ]
    relocated_match = (
        bool(saved_identities)
        and all(identity is not None for identity in saved_identities)
        and saved_identities == requested_identities
    )
    if not exact_match and not relocated_match:
        raise ValueError(
            "Existing post-hoc analysis uses different or differently ordered "
            "checkpoints"
        )
    if relocated_match and not exact_match:
        LOGGER.info(
            "Reusing post-hoc analysis after project relocation; ordered paths "
            "beneath outputs/ are unchanged."
        )
        summary = copy.deepcopy(summary)
        for entry, requested_path in zip(
            summary["checkpoints"], requested_checkpoints, strict=True
        ):
            entry["path"] = requested_path
        configuration = summary.get("configuration")
        if isinstance(configuration, dict):
            configuration["output_dir"] = str(root.resolve())
    return AnalysisResult(
        summary_path=paths["summary"],
        report_path=paths["report"],
        fold_importance_path=paths["fold_importance"],
        modality_importance_path=paths["modality_importance"],
        sample_attributions_path=paths["sample_attributions"],
        summary=summary,
    )


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
        raise ValueError(f"Cannot write an empty CSV: {path}")
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(_json_safe(rows))


def _pooled_metrics(
    adapters: list[CheckpointAdapter],
    logits_by_fold: list[np.ndarray],
) -> dict[str, float]:
    labels = np.concatenate([adapter.labels for adapter in adapters])
    logits = np.concatenate(logits_by_fold)
    thresholds = np.concatenate(
        [np.full(adapter.labels.shape, adapter.threshold) for adapter in adapters]
    )
    return binary_metrics(labels, logits, thresholds)


def _validate_adapters(
    adapters: list[CheckpointAdapter], *, allow_duplicate_samples: bool
) -> int:
    if len(adapters) < 2:
        raise ValueError("Cross-validation attribution requires at least two folds")
    checkpoint_types = {adapter.checkpoint_type for adapter in adapters}
    if len(checkpoint_types) != 1:
        raise ValueError(
            "All fold checkpoints must have the same classifier type; "
            f"found {sorted(checkpoint_types)}"
        )
    fold_names = [adapter.fold_name for adapter in adapters]
    if len(fold_names) != len(set(fold_names)):
        raise ValueError(f"Duplicate fold directory names: {fold_names}")
    sample_ids = [sample_id for adapter in adapters for sample_id in adapter.sample_ids]
    duplicate_count = len(sample_ids) - len(set(sample_ids))
    if duplicate_count and not allow_duplicate_samples:
        raise ValueError(
            "Cross-validation test folds contain "
            f"{duplicate_count} duplicate sample IDs. Remove duplicate rows or rerun "
            "with allow_duplicate_samples=True after verifying they are intentional."
        )
    modality_sizes = [adapter.modality_sizes for adapter in adapters]
    if any(sizes != modality_sizes[0] for sizes in modality_sizes[1:]):
        raise ValueError("Fold checkpoints use incompatible modality schemas")
    signatures = []
    for adapter in adapters:
        signature = copy.deepcopy(adapter.config_payload)
        signature.pop("output_dir", None)
        signature.pop("run_name", None)
        signature.pop("feature_artifact_path", None)
        signature.get("data", {}).pop("dataset_path", None)
        signature.get("data", {}).setdefault("included_modalities", None)
        signature.get("data", {}).setdefault("masked_channels", None)
        signatures.append(signature)
    if any(signature != signatures[0] for signature in signatures[1:]):
        raise ValueError(
            "Fold checkpoints use incompatible experiment configurations after "
            "excluding fold-specific paths"
        )
    return duplicate_count


def duplicate_sample_rows(
    adapters: list[CheckpointAdapter],
) -> list[dict[str, Any]]:
    """Return an occurrence-level audit of repeated outer-test sample IDs."""

    outer_counts: dict[str, int] = {}
    outer_folds: dict[str, set[str]] = {}
    for adapter in adapters:
        for sample_id in adapter.sample_ids:
            outer_counts[sample_id] = outer_counts.get(sample_id, 0) + 1
            outer_folds.setdefault(sample_id, set()).add(adapter.fold_name)

    rows: list[dict[str, Any]] = []
    for adapter in adapters:
        local_counts: dict[str, int] = {}
        for sample_id in adapter.sample_ids:
            local_counts[sample_id] = local_counts.get(sample_id, 0) + 1
        for sample_index, sample_id in enumerate(adapter.sample_ids):
            if outer_counts[sample_id] < 2:
                continue
            reasons = []
            if local_counts[sample_id] > 1:
                reasons.append("repeated_within_fold_test")
            if len(outer_folds[sample_id]) > 1:
                reasons.append("repeated_across_outer_test_folds")
            rows.append(
                {
                    "sample_id": sample_id,
                    "within_fold_occurrences": local_counts[sample_id],
                    "outer_test_occurrences": outer_counts[sample_id],
                    "reason": ";".join(reasons),
                    "fold": adapter.fold_name,
                    "split": "test",
                    "sample_index": sample_index,
                    "sample_key": f"{adapter.fold_name}:{sample_index}",
                    "group_id": adapter.group_ids[sample_index],
                    "label": int(adapter.labels[sample_index]),
                }
            )
    return rows


def _report(summary: dict[str, Any]) -> str:
    rows = summary["modality_importance"]
    top_ablation = rows[0]
    shapley_enabled = bool(summary.get("shapley", {}).get("enabled", True))
    main_result = (
        f"By pooled macro-F1 ablation, **{top_ablation['modality']}** ranks first "
        f"(decrease {top_ablation['pooled_macro_f1_decrease']:.4f})."
    )
    if shapley_enabled:
        shapley_rows = sorted(
            rows, key=lambda row: row["mean_absolute_shapley"], reverse=True
        )
        top_shapley = shapley_rows[0]
        main_result += (
            f" By mean absolute per-sample Shapley magnitude, "
            f"**{top_shapley['modality']}** ranks first "
            f"({top_shapley['mean_absolute_shapley']:.4f} logit units)."
        )
    lines = [
        "# Cross-validated modality attribution",
        "",
        f"Classifier: `{summary['checkpoint_type']}` across {summary['num_folds']} folds "
        f"and {summary['num_test_samples']} held-out samples.",
        "",
        "## Main result",
        "",
        main_result,
        "",
        "## Modality ranking",
        "",
    ]
    if shapley_enabled:
        lines.extend(
            [
                "| Rank | Modality | Pooled macro-F1 decrease | Fold-mean decrease (95% CI) | Mean absolute Shapley | Channels |",
                "|---:|---|---:|---:|---:|---:|",
            ]
        )
    else:
        lines.extend(
            [
                "| Rank | Modality | Pooled macro-F1 decrease | Fold-mean decrease (95% CI) | Channels |",
                "|---:|---|---:|---:|---:|",
            ]
        )
    for row in rows:
        prefix = (
            f"| {row['rank']} | {row['modality']} | "
            f"{row['pooled_macro_f1_decrease']:.4f} | "
            f"{row['fold_mean_macro_f1_decrease']:.4f} "
            f"[{row['fold_ci_lower']:.4f}, {row['fold_ci_upper']:.4f}] | "
        )
        if shapley_enabled:
            lines.append(
                prefix
                + f"{row['mean_absolute_shapley']:.4f} | {row['num_channels']} |"
            )
        else:
            lines.append(prefix + f"{row['num_channels']} |")
    lines.extend(["", "## Interpretation", ""])
    if shapley_enabled:
        lines.extend(
            [
                "A positive ablation decrease means that removing the modality hurt held-out "
                "classification; negative values mean removal improved it. Signed sample-level "
                "Shapley values support negation when positive and oppose it when negative.",
                "",
            ]
        )
    else:
        lines.extend(
            [
                "A positive ablation decrease means that removing the modality hurt held-out "
                "classification; negative values mean removal improved it. Shapley analysis "
                "was skipped for this run.",
                "",
            ]
        )
    lines.extend(
        [
            "These are predictive, not causal, attributions. Correlated modalities can "
            "substitute for each other, so a small leave-one-out effect does not prove that "
            "a sensor carries no negation information.",
            "",
        ]
    )
    retraining = summary.get("retraining")
    if retraining:
        evidence = []
        modality_only = retraining.get("modality_only_ranking") or []
        leave_one_out = retraining.get("leave_one_out_ranking") or []
        if modality_only:
            top_only = modality_only[0]
            evidence.append(
                f"The strongest modality by itself is **{top_only['modality']}** "
                f"(pooled macro F1 {top_only['pooled_macro_f1']:.4f})."
            )
        if leave_one_out:
            top_unique = leave_one_out[0]
            evidence.append(
                f"The largest unique-value loss comes from omitting "
                f"**{top_unique['modality']}** "
                f"({top_unique['pooled_macro_f1_change_from_full']:.4f})."
            )
        if evidence:
            comparison_targets = ["fitted-model ablation"]
            if shapley_enabled:
                comparison_targets.append("Shapley reliance")
            lines.extend(
                [
                    "## Retraining evidence",
                    "",
                    " ".join(evidence),
                    "",
                    "Retraining evidence is intentionally reported separately from "
                    + " and ".join(comparison_targets)
                    + ". Disagreement between them is expected when modalities are "
                    "redundant or interact.",
                    "",
                ]
            )
    lines.extend(["## Warnings", ""])
    lines.extend(f"- {warning}" for warning in summary["warnings"])
    lines.append("")
    return "\n".join(lines)


def attach_retraining_summary(
    result: AnalysisResult, retraining_summary: dict[str, Any]
) -> AnalysisResult:
    """Add completed retraining evidence to the main JSON and Markdown reports."""

    summary = dict(result.summary)
    summary["retraining"] = retraining_summary
    result.summary_path.write_text(
        json.dumps(_json_safe(summary), indent=2) + "\n", encoding="utf-8"
    )
    result.report_path.write_text(_report(summary), encoding="utf-8")
    return AnalysisResult(
        summary_path=result.summary_path,
        report_path=result.report_path,
        fold_importance_path=result.fold_importance_path,
        modality_importance_path=result.modality_importance_path,
        sample_attributions_path=result.sample_attributions_path,
        summary=summary,
    )


def analyze_cross_validation(
    checkpoint_paths: list[str | Path],
    config: AnalysisConfig,
) -> AnalysisResult:
    """Analyze fitted fold checkpoints without using train/validation outcomes."""

    config.validate()
    adapters = [
        load_checkpoint_adapter(
            path, device=config.device, batch_size=config.batch_size
        )
        for path in checkpoint_paths
    ]
    output_dir = config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    duplicate_rows = duplicate_sample_rows(adapters)
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
    duplicate_count = _validate_adapters(
        adapters, allow_duplicate_samples=config.allow_duplicate_samples
    )
    LOGGER.info(
        "Fold validation passed: %d folds, %d outer-test rows, %d duplicate rows beyond first occurrences",
        len(adapters),
        sum(adapter.labels.size for adapter in adapters),
        duplicate_count,
    )

    modality_names = adapters[0].modality_names
    full_logits = [adapter.predict_logits(modality_names) for adapter in adapters]
    pooled_full = _pooled_metrics(adapters, full_logits)
    fold_rows: list[dict[str, Any]] = []
    ablated_logits: dict[str, list[np.ndarray]] = {name: [] for name in modality_names}
    fold_decreases: dict[str, list[dict[str, float]]] = {
        name: [] for name in modality_names
    }
    shapley_by_fold = []
    sample_rows: list[dict[str, Any]] = []

    for fold_index, (adapter, fold_full_logits) in enumerate(
        zip(adapters, full_logits, strict=True)
    ):
        fold_full_metrics = binary_metrics(
            adapter.labels, fold_full_logits, adapter.threshold
        )
        for modality in modality_names:
            coalition = tuple(name for name in modality_names if name != modality)
            scores = adapter.predict_logits(coalition)
            ablated_logits[modality].append(scores)
            ablated_metrics = binary_metrics(adapter.labels, scores, adapter.threshold)
            decreases = metric_decrease(fold_full_metrics, ablated_metrics)
            fold_decreases[modality].append(decreases)
            row: dict[str, Any] = {
                "fold": adapter.fold_name,
                "modality": modality,
                "num_test_samples": int(adapter.labels.size),
            }
            for metric in METRIC_NAMES:
                row[f"full_{metric}"] = fold_full_metrics[metric]
                row[f"ablated_{metric}"] = ablated_metrics[metric]
                row[f"{metric}_decrease"] = decreases[metric]
            fold_rows.append(row)

        if config.run_shapley:
            shapley = sampled_modality_shapley(
                modality_names,
                adapter.predict_logits,
                num_orderings=config.shapley_orderings,
                seed=config.seed + fold_index,
            )
            shapley_by_fold.append(shapley)
            if not np.allclose(shapley.reconstruction_error, 0.0, atol=1e-10):
                raise RuntimeError(
                    "Sampled Shapley values failed efficiency reconstruction"
                )
            for sample_index, sample_id in enumerate(adapter.sample_ids):
                values = shapley.values[sample_index]
                absolute_total = float(np.abs(values).sum())
                sample_rows.append(
                    {
                        "fold": adapter.fold_name,
                        "sample_index": sample_index,
                        "sample_key": f"{adapter.fold_name}:{sample_index}",
                        "sample_id": sample_id,
                        "label": int(adapter.labels[sample_index]),
                        "full_logit": float(shapley.full_logits[sample_index]),
                        "empty_logit": float(shapley.empty_logits[sample_index]),
                        "prediction_threshold": adapter.threshold,
                        "prediction": int(
                            1.0 / (1.0 + np.exp(-shapley.full_logits[sample_index]))
                            >= adapter.threshold
                        ),
                        "shapley": {
                            modality: float(values[index])
                            for index, modality in enumerate(modality_names)
                        },
                        "standard_error": {
                            modality: float(
                                shapley.standard_errors[sample_index, index]
                            )
                            for index, modality in enumerate(modality_names)
                        },
                        "absolute_share_percent": {
                            modality: (
                                float(abs(values[index]) / absolute_total * 100.0)
                                if absolute_total
                                else 0.0
                            )
                            for index, modality in enumerate(modality_names)
                        },
                    }
                )

    all_shapley = (
        np.concatenate([result.values for result in shapley_by_fold], axis=0)
        if config.run_shapley
        else None
    )
    modality_rows: list[dict[str, Any]] = []
    for modality_index, modality in enumerate(modality_names):
        pooled_ablated = _pooled_metrics(adapters, ablated_logits[modality])
        pooled_decrease = metric_decrease(pooled_full, pooled_ablated)
        macro_deltas = [
            values["macro_f1"] for values in fold_decreases[modality]
        ]
        ci_lower, ci_upper = bootstrap_mean_interval(
            macro_deltas,
            seed=config.seed + modality_index,
            samples=config.bootstrap_samples,
        )
        mean_signed_shapley = (
            float(all_shapley[:, modality_index].mean())
            if all_shapley is not None
            else None
        )
        mean_absolute_shapley = (
            float(np.abs(all_shapley[:, modality_index]).mean())
            if all_shapley is not None
            else None
        )
        num_channels = adapters[0].modality_sizes[modality]
        row = {
            "modality": modality,
            "num_channels": num_channels,
            "pooled_macro_f1_decrease": pooled_decrease["macro_f1"],
            "fold_mean_macro_f1_decrease": float(np.mean(macro_deltas)),
            "fold_ci_lower": ci_lower,
            "fold_ci_upper": ci_upper,
            "mean_signed_shapley": mean_signed_shapley,
            "mean_absolute_shapley": mean_absolute_shapley,
            "absolute_shapley_per_channel": (
                mean_absolute_shapley / num_channels
                if mean_absolute_shapley is not None
                else None
            ),
        }
        for metric in METRIC_NAMES:
            row[f"pooled_full_{metric}"] = pooled_full[metric]
            row[f"pooled_ablated_{metric}"] = pooled_ablated[metric]
            row[f"pooled_{metric}_decrease"] = pooled_decrease[metric]
        modality_rows.append(row)
    modality_rows.sort(key=lambda row: row["pooled_macro_f1_decrease"], reverse=True)
    for rank, row in enumerate(modality_rows, 1):
        row["rank"] = rank
    modality_rows = [
        {"rank": row.pop("rank"), **row}
        for row in modality_rows
    ]

    summary = {
        "analysis_version": 1,
        "checkpoint_type": adapters[0].checkpoint_type,
        "num_folds": len(adapters),
        "num_test_samples": sum(adapter.labels.size for adapter in adapters),
        "duplicate_sample_ids": duplicate_count,
        "duplicate_audit": "duplicate_samples.csv" if duplicate_rows else None,
        "primary_ranking_metric": "pooled_macro_f1_decrease",
        "configuration": asdict(config),
        "checkpoints": [
            {
                "fold": adapter.fold_name,
                "path": str(adapter.checkpoint_path.resolve()),
                "threshold": adapter.threshold,
                "num_test_samples": int(adapter.labels.size),
            }
            for adapter in adapters
        ],
        "pooled_full_metrics": pooled_full,
        "modality_sizes": adapters[0].modality_sizes,
        "modality_importance": modality_rows,
        "shapley": (
            {
                "enabled": True,
                "scale": "model_logit",
                "num_orderings_per_fold": config.shapley_orderings,
                "antithetic_orderings": True,
                "coalitions_evaluated_by_fold": [
                    result.num_coalitions_evaluated for result in shapley_by_fold
                ],
            }
            if config.run_shapley
            else {"enabled": False}
        ),
        "warnings": [
            "Attributions measure predictive reliance, not causal influence.",
            "Correlated modalities may substitute for one another.",
        ]
        + (
            [
                "Per-channel normalization is diagnostic only and is not the primary ranking.",
            ]
            if config.run_shapley
            else []
        )
        + (
            [
                f"Input test folds contained {duplicate_count} duplicate sample IDs; "
                "rows were preserved, audited in duplicate_samples.csv, and "
                "disambiguated with sample_key."
            ]
            if duplicate_count
            else []
        ),
    }

    summary_path = output_dir / "summary.json"
    report_path = output_dir / "report.md"
    fold_path = output_dir / "fold_importance.csv"
    modality_path = output_dir / "modality_importance.csv"
    sample_path = output_dir / "sample_attributions.jsonl"
    summary_path.write_text(
        json.dumps(_json_safe(summary), indent=2) + "\n", encoding="utf-8"
    )
    report_path.write_text(_report(summary), encoding="utf-8")
    _write_csv(fold_path, fold_rows)
    _write_csv(modality_path, modality_rows)
    sample_content = (
        "\n".join(json.dumps(_json_safe(row)) for row in sample_rows) + "\n"
        if sample_rows
        else ""
    )
    sample_path.write_text(sample_content, encoding="utf-8")
    return AnalysisResult(
        summary_path=summary_path,
        report_path=report_path,
        fold_importance_path=fold_path,
        modality_importance_path=modality_path,
        sample_attributions_path=sample_path,
        summary=summary,
    )
