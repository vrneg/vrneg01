"""Collect fold-average test and validation metrics for symmetric event windows.

The script searches ``outputs/<model>/<dataset-family>/...`` for shared
``*_cross_validation_summary.json`` artifacts for the cue target. It groups the
test- and validation-fold means for macro F1 and AUROC by selected symmetric event
window and model while preserving separate experiment variants and random seeds.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

try:
    from .main_utils.windowing import (
        SIGNED_MILLISECONDS_PATTERN,
        context_window_bounds_ms,
        decode_signed_milliseconds,
        encode_signed_milliseconds,
    )
except ImportError:  # Direct execution adds ``src`` to ``sys.path``.
    from main_utils.windowing import (
        SIGNED_MILLISECONDS_PATTERN,
        context_window_bounds_ms,
        decode_signed_milliseconds,
        encode_signed_milliseconds,
    )


TARGET = "cue"
EVENT_SOURCES = ("speaker", "listener", "both")
DEFAULT_SYMMETRIC_WINDOWS: tuple[int, ...] = (100, 500, 1000, 2500)
_TARGET_PATTERN = re.compile(
    r"^(?:target|tgt)-(?P<target>cueandscope|cue|scope)(?:_|$)"
)
_SOURCE_PATTERN = re.compile(
    r"(?:^|_)(?:src|source)-(?P<source>speaker|listener|both)(?:_|$)"
)
_ANY_SOURCE_PATTERN = re.compile(
    r"(?:^|_)(?:src|source)-(?P<source>[^_]+)(?:_|$)"
)
_SYMMETRIC_WINDOW_PATTERN = re.compile(r"(?:^|_)window-(?P<window>\d+)(?:_|$)")
_ASYMMETRIC_WINDOW_PATTERNS = (
    re.compile(
        rf"(?:^|_)windowL-(?P<left>{SIGNED_MILLISECONDS_PATTERN})_"
        rf"windowR-(?P<right>{SIGNED_MILLISECONDS_PATTERN})(?:_|$)"
    ),
    re.compile(
        rf"(?:^|_)wL-(?P<left>{SIGNED_MILLISECONDS_PATTERN})_"
        rf"wR-(?P<right>{SIGNED_MILLISECONDS_PATTERN})(?:_|$)"
    ),
)
_CROSS_VALIDATION_DIRECTORY_PATTERN = re.compile(
    r"cross_validation_seed-(?P<seed>\d+)"
)


class SummaryFormatError(ValueError):
    """Raised when a discovered file is not a compatible CV summary."""


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _dataset_target(dataset_family: str) -> str | None:
    match = _TARGET_PATTERN.match(dataset_family)
    return match.group("target") if match else None


def _dataset_source(dataset_family: str) -> tuple[str, bool]:
    """Return the event source and whether it was inferred from a legacy name."""

    match = _SOURCE_PATTERN.search(dataset_family)
    if match:
        return match.group("source"), False

    unknown_source = _ANY_SOURCE_PATTERN.search(dataset_family)
    if unknown_source:
        raise SummaryFormatError(
            f"unsupported dataset source {unknown_source.group('source')!r}"
        )

    # Dataset families created before the source field was added always contain
    # speaker events.
    return "speaker", True


def _window_bounds_ms(dataset_family: str) -> tuple[int, int]:
    for pattern in _ASYMMETRIC_WINDOW_PATTERNS:
        match = pattern.search(dataset_family)
        if match:
            left_ms = decode_signed_milliseconds(match.group("left"))
            right_ms = decode_signed_milliseconds(match.group("right"))
            try:
                context_window_bounds_ms(left_ms, right_ms)
            except (TypeError, ValueError) as error:
                raise SummaryFormatError(str(error)) from error
            return left_ms, right_ms

    match = _SYMMETRIC_WINDOW_PATTERN.search(dataset_family)
    if match:
        window_ms = int(match.group("window"))
        return window_ms, window_ms

    raise SummaryFormatError(
        "dataset family has no supported window definition "
        "(window-N, windowL-L_windowR-R, or wL-L_wR-R)"
    )


def _window_label(left_ms: int, right_ms: int) -> str:
    if left_ms == right_ms:
        return f"window-{left_ms}"
    left_label = encode_signed_milliseconds(left_ms)
    right_label = encode_signed_milliseconds(right_ms)
    return f"windowL-{left_label}_windowR-{right_label}"


def _metric_mean(
    summary: Mapping[str, Any], metric: str, split: str = "test"
) -> tuple[float | None, int | None]:
    if split not in {"test", "validation"}:
        raise ValueError("split must be 'test' or 'validation'")
    metric_path = f"fold_statistics.{split}.{metric}"
    try:
        statistics = summary["fold_statistics"][split][metric]
    except (KeyError, TypeError) as error:
        raise SummaryFormatError(f"missing {metric_path}") from error

    if not isinstance(statistics, Mapping) or "mean" not in statistics:
        raise SummaryFormatError(f"{metric_path} has no mean")

    mean = statistics["mean"]
    if mean is not None:
        if isinstance(mean, bool) or not isinstance(mean, (int, float)):
            raise SummaryFormatError(
                f"{metric_path}.mean is not numeric or null"
            )
        mean = float(mean)
        if not math.isfinite(mean):
            raise SummaryFormatError(f"{metric_path}.mean is not finite")

    num_valid_folds = statistics.get("num_valid_folds")
    if num_valid_folds is not None:
        if isinstance(num_valid_folds, bool) or not isinstance(num_valid_folds, int):
            raise SummaryFormatError(
                f"{metric_path}.num_valid_folds is not an integer"
            )

    return mean, num_valid_folds


def _run_location(
    summary_path: Path, outputs_dir: Path
) -> tuple[str, str, str | None, int]:
    try:
        relative_path = summary_path.relative_to(outputs_dir)
    except ValueError as error:
        raise SummaryFormatError("summary is outside the outputs directory") from error

    parts = relative_path.parts
    if len(parts) < 4:
        raise SummaryFormatError(
            "expected outputs/<model>/<dataset-family>/.../<summary-file>"
        )

    model, dataset_family = parts[:2]
    cv_index: int | None = None
    seed: int | None = None
    for index, part in enumerate(parts[2:-1], start=2):
        match = _CROSS_VALIDATION_DIRECTORY_PATTERN.fullmatch(part)
        if match:
            cv_index = index
            seed = int(match.group("seed"))
            break

    if cv_index is None or seed is None:
        raise SummaryFormatError("summary is not below cross_validation_seed-N")

    variant_parts = parts[2:cv_index]
    variant = "/".join(variant_parts) if variant_parts else None
    return model, dataset_family, variant, seed


def _optional_fold_count(summary: Mapping[str, Any]) -> int | None:
    num_folds = summary.get("num_folds")
    if num_folds is None:
        return None
    if isinstance(num_folds, bool) or not isinstance(num_folds, int):
        raise SummaryFormatError("num_folds is not an integer")
    return num_folds


def _optional_finite_number(value: Any, path: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SummaryFormatError(f"{path} is not numeric or null")
    number = float(value)
    if not math.isfinite(number):
        raise SummaryFormatError(f"{path} is not finite")
    return number


def _fold_results(
    summary: Mapping[str, Any], splits: tuple[str, ...] = ("test",)
) -> list[dict[str, Any]]:
    """Return sorted fold-level metrics from a shared CV summary.

    The test-only default preserves the helper's API for the other result
    collectors. The symmetric comparison requests both available result splits.
    """

    if not splits or any(split not in {"test", "validation"} for split in splits):
        raise ValueError("splits must contain only 'test' and/or 'validation'")
    if len(set(splits)) != len(splits):
        raise ValueError("splits must not contain duplicates")

    folds = summary.get("folds")
    if not isinstance(folds, Mapping) or not folds:
        raise SummaryFormatError("folds is not a non-empty object")

    results: list[dict[str, Any]] = []
    seen_folds: set[int] = set()
    for raw_fold, raw_result in folds.items():
        try:
            fold = int(raw_fold)
        except (TypeError, ValueError) as error:
            raise SummaryFormatError(f"invalid fold identifier {raw_fold!r}") from error
        if fold < 0 or fold in seen_folds:
            raise SummaryFormatError(
                f"invalid or duplicate fold identifier {raw_fold!r}"
            )
        seen_folds.add(fold)
        if not isinstance(raw_result, Mapping):
            raise SummaryFormatError(f"folds.{raw_fold} is not an object")

        split_results: dict[str, dict[str, float | None]] = {}
        for split in splits:
            metrics_name = f"{split}_metrics"
            metrics = raw_result.get(metrics_name)
            metric_path = f"folds.{raw_fold}.{metrics_name}"
            if not isinstance(metrics, Mapping):
                raise SummaryFormatError(f"{metric_path} is not an object")
            if "macro_f1" not in metrics or "roc_auc" not in metrics:
                raise SummaryFormatError(
                    f"{metric_path} must contain macro_f1 and roc_auc"
                )
            collected_metrics: dict[str, float | None] = {}
            for metric, value in metrics.items():
                if not isinstance(metric, str) or not metric:
                    raise SummaryFormatError(
                        f"{metric_path} has an invalid metric name"
                    )
                collected_metrics[metric] = _optional_finite_number(
                    value,
                    f"{metric_path}.{metric}",
                )
            split_results[metrics_name] = collected_metrics

        run_name = raw_result.get("run_name")
        if run_name is not None and not isinstance(run_name, str):
            raise SummaryFormatError(f"folds.{raw_fold}.run_name is not a string")
        results.append(
            {
                "fold": fold,
                "run_name": run_name,
                "decision_threshold": _optional_finite_number(
                    raw_result.get("decision_threshold"),
                    f"folds.{raw_fold}.decision_threshold",
                ),
                **split_results,
            }
        )

    results.sort(key=lambda result: result["fold"])
    num_folds = _optional_fold_count(summary)
    if num_folds is not None and num_folds != len(results):
        raise SummaryFormatError(
            f"num_folds is {num_folds}, but folds contains {len(results)} results"
        )
    return results


def collect_model_results(
    outputs_dir: str | Path,
    event_source: str = "speaker",
    symmetric_windows: tuple[int, ...] = DEFAULT_SYMMETRIC_WINDOWS,
    excluded_models: tuple[str, ...] = (),
) -> dict[str, Any]:
    """Return results for the selected symmetric cue-target windows and source.

    Dataset names without an explicit source are legacy speaker datasets. Invalid
    summaries belonging to a selected symmetric window and source are reported in
    ``issues`` rather than preventing other completed model runs from being
    collected.
    """

    if event_source not in EVENT_SOURCES:
        choices = ", ".join(EVENT_SOURCES)
        raise ValueError(f"event_source must be one of: {choices}")
    if not isinstance(symmetric_windows, tuple):
        raise ValueError("symmetric_windows must be a tuple of positive integers")
    if not symmetric_windows:
        raise ValueError("symmetric_windows must not be empty")
    if any(
        isinstance(window_ms, bool)
        or not isinstance(window_ms, int)
        or window_ms < 1
        for window_ms in symmetric_windows
    ):
        raise ValueError("symmetric_windows must be a tuple of positive integers")
    if not isinstance(excluded_models, tuple) or any(
        not isinstance(model, str) or not model
        for model in excluded_models
    ):
        raise ValueError("excluded_models must be a tuple of non-empty strings")
    selected_symmetric_windows = frozenset(symmetric_windows)
    excluded_model_names = frozenset(excluded_models)

    outputs_dir = Path(outputs_dir).resolve()
    if not outputs_dir.is_dir():
        raise FileNotFoundError(f"Outputs directory does not exist: {outputs_dir}")

    buckets: dict[
        tuple[int, int], dict[str, list[dict[str, Any]]]
    ] = defaultdict(lambda: defaultdict(list))
    issues: list[dict[str, str]] = []
    summaries_scanned = 0
    summaries_for_source = 0

    for summary_path in sorted(
        outputs_dir.rglob("*_cross_validation_summary.json")
    ):
        summaries_scanned += 1
        try:
            relative_path = summary_path.relative_to(outputs_dir)
            if len(relative_path.parts) < 2:
                continue
            if relative_path.parts[0] in excluded_model_names:
                continue
            dataset_family = relative_path.parts[1]
            if _dataset_target(dataset_family) != TARGET:
                continue
            dataset_source, source_inferred = _dataset_source(dataset_family)
            if dataset_source != event_source:
                continue

            left_ms, right_ms = _window_bounds_ms(dataset_family)
            if (
                left_ms != right_ms
                or left_ms not in selected_symmetric_windows
            ):
                continue
            summaries_for_source += 1
            model, dataset_family, variant, seed = _run_location(
                summary_path, outputs_dir
            )
            with summary_path.open("r", encoding="utf-8") as file:
                summary = json.load(file)
            if not isinstance(summary, Mapping):
                raise SummaryFormatError("top-level JSON value is not an object")

            macro_f1, macro_f1_valid_folds = _metric_mean(
                summary, "macro_f1", "test"
            )
            auroc, auroc_valid_folds = _metric_mean(summary, "roc_auc", "test")
            validation_macro_f1, validation_macro_f1_valid_folds = _metric_mean(
                summary, "macro_f1", "validation"
            )
            validation_auroc, validation_auroc_valid_folds = _metric_mean(
                summary, "roc_auc", "validation"
            )
            fold_results = _fold_results(summary, ("test", "validation"))
            run = {
                "dataset_family": dataset_family,
                "event_source": dataset_source,
                "event_source_inferred": source_inferred,
                "variant": variant,
                "seed": seed,
                "num_folds": _optional_fold_count(summary),
                "valid_folds": {
                    "macro_f1": macro_f1_valid_folds,
                    "auroc": auroc_valid_folds,
                },
                "validation_valid_folds": {
                    "macro_f1": validation_macro_f1_valid_folds,
                    "auroc": validation_auroc_valid_folds,
                },
                "macro_f1": macro_f1,
                "auroc": auroc,
                "validation_macro_f1": validation_macro_f1,
                "validation_auroc": validation_auroc,
                "fold_results": fold_results,
                "summary_file": relative_path.as_posix(),
            }
            buckets[(left_ms, right_ms)][model].append(run)
        except (OSError, json.JSONDecodeError, SummaryFormatError) as error:
            issues.append(
                {
                    "summary_file": summary_path.relative_to(outputs_dir).as_posix(),
                    "reason": str(error),
                }
            )

    windows: dict[str, dict[str, Any]] = {}
    all_models: set[str] = set()
    num_results = 0
    for (left_ms, right_ms), models in sorted(buckets.items()):
        sorted_models: dict[str, list[dict[str, Any]]] = {}
        for model, runs in sorted(models.items()):
            sorted_runs = sorted(
                runs,
                key=lambda run: (
                    run["dataset_family"],
                    run["variant"] or "",
                    run["seed"],
                    run["summary_file"],
                ),
            )
            sorted_models[model] = sorted_runs
            all_models.add(model)
            num_results += len(sorted_runs)

        windows[_window_label(left_ms, right_ms)] = {
            "left_ms": left_ms,
            "right_ms": right_ms,
            "window_start_ms": -left_ms,
            "window_end_ms": right_ms,
            "models": sorted_models,
        }

    return {
        "target": TARGET,
        "event_source": event_source,
        "metric_scope": "mean_across_test_folds",
        "validation_metric_scope": "mean_across_validation_folds",
        "source_metric_paths": {
            "macro_f1": "fold_statistics.test.macro_f1.mean",
            "auroc": "fold_statistics.test.roc_auc.mean",
        },
        "validation_source_metric_paths": {
            "macro_f1": "fold_statistics.validation.macro_f1.mean",
            "auroc": "fold_statistics.validation.roc_auc.mean",
        },
        "source_fold_results_path": "folds.<fold>.test_metrics",
        "validation_source_fold_results_path": (
            "folds.<fold>.validation_metrics"
        ),
        "outputs_dir": str(outputs_dir),
        "excluded_models": sorted(excluded_model_names),
        "summaries_scanned": summaries_scanned,
        "summaries_for_source": summaries_for_source,
        "num_results": num_results,
        "num_models": len(all_models),
        "windows": windows,
        "issues": sorted(issues, key=lambda issue: issue["summary_file"]),
    }


def write_results(result: Mapping[str, Any], output_path: str | Path) -> Path:
    """Write ``result`` as standards-compliant, human-readable JSON."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return output_path


def _ranking_rows(
    result: Mapping[str, Any],
    metric: str,
    group: str,
    split: str = "test",
) -> list[tuple[str, float | None, int]]:
    """Rank models or windows after averaging repeated runs within each pairing."""

    if split not in {"test", "validation"}:
        raise ValueError("split must be 'test' or 'validation'")
    run_metric = metric if split == "test" else f"validation_{metric}"

    windows = result["windows"]
    grouped_scores: dict[str, list[float]] = {}
    if group == "window":
        grouped_scores.update((window_label, []) for window_label in windows)
    elif group != "model":
        raise ValueError("group must be 'model' or 'window'")

    for window_label, window in windows.items():
        for model, runs in window["models"].items():
            if group == "model":
                grouped_scores.setdefault(model, [])
            run_scores = [
                run[run_metric]
                for run in runs
                if run[run_metric] is not None
            ]
            if not run_scores:
                continue
            model_window_average = sum(run_scores) / len(run_scores)
            group_name = model if group == "model" else window_label
            grouped_scores[group_name].append(model_window_average)

    rows = [
        (
            name,
            sum(scores) / len(scores) if scores else None,
            len(scores),
        )
        for name, scores in grouped_scores.items()
    ]
    return sorted(
        rows,
        key=lambda row: (
            row[1] is None,
            -(row[1] if row[1] is not None else 0.0),
            row[0],
        ),
    )


def _print_ranking(
    title: str,
    rows: Sequence[tuple[str, float | None, int]],
    averaged_entity: str,
) -> None:
    print(f"\n{title}", file=sys.stderr)
    if not rows:
        print("  No matching results.", file=sys.stderr)
        return

    for rank, (name, score, count) in enumerate(rows, start=1):
        if score is None:
            detail = f"N/A (no valid {averaged_entity} scores)"
        else:
            suffix = "" if count == 1 else "s"
            detail = f"{score:.4f} ({count} {averaged_entity}{suffix})"
        print(f"  {rank:>2}. {name}: {detail}", file=sys.stderr)


def _print_split_rankings(result: Mapping[str, Any], split: str) -> None:
    """Print all four model/window rankings for one result split."""

    print("\nModel rankings (each symmetric window weighted equally)", file=sys.stderr)
    _print_ranking(
        "By average AUROC",
        _ranking_rows(result, "auroc", "model", split),
        "window",
    )
    _print_ranking(
        "By average macro F1",
        _ranking_rows(result, "macro_f1", "model", split),
        "window",
    )

    print("\nWindow rankings (each model weighted equally)", file=sys.stderr)
    _print_ranking(
        "By average macro F1",
        _ranking_rows(result, "macro_f1", "window", split),
        "model",
    )
    _print_ranking(
        "By average AUROC",
        _ranking_rows(result, "auroc", "window", split),
        "model",
    )


def print_comparison_rankings(result: Mapping[str, Any]) -> None:
    """Print test and validation rankings while keeping stdout parseable."""

    print("\n=== Test results ===", file=sys.stderr)
    _print_split_rankings(result, "test")
    print("\n=== Validation results ===", file=sys.stderr)
    _print_split_rankings(result, "validation")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Collect cue-target macro F1 and AUROC test- and validation-fold means "
            "from all model cross-validation summaries for selected symmetric "
            "windows and one event source."
        )
    )
    parser.add_argument(
        "event_source",
        nargs="?",
        choices=EVENT_SOURCES,
        default="speaker",
        help="Event source to collect (default: speaker).",
    )
    parser.add_argument(
        "--symmetric-windows",
        nargs="+",
        type=int,
        default=DEFAULT_SYMMETRIC_WINDOWS,
        metavar="MS",
        help=(
            "Symmetric window sizes in milliseconds to collect "
            "(default: 100 500 1000 2500)."
        ),
    )
    parser.add_argument(
        "--outputs-dir",
        type=Path,
        default=_project_root() / "outputs",
        help="Model output root (default: <repository>/outputs).",
    )
    parser.add_argument(
        "--exclude-models",
        nargs="+",
        # default=("t2m_gpt_v2", "ghtt", "motiongpt3"),
        default=(),
        metavar="MODEL",
        help=(
            "Exact model directory names to exclude, for example: "
            "--exclude-models t2m_gpt_v2 motion_gpt."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        help=(
            "Destination JSON file (default: "
            "<outputs-dir>/fold_average_test_results_cue_<event-source>.json; "
            "the compatibility filename now contains test and validation results)."
        ),
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help=(
            "Fail without writing when a summary for a selected window and source "
            "is invalid."
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        result = collect_model_results(
            args.outputs_dir,
            args.event_source,
            tuple(args.symmetric_windows),
            tuple(args.exclude_models),
        )
    except (FileNotFoundError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    if args.strict and result["issues"]:
        print(
            f"error: found {len(result['issues'])} invalid matching summaries",
            file=sys.stderr,
        )
        return 1

    output_path = args.output or (
        args.outputs_dir
        / f"fold_average_test_results_{TARGET}_{args.event_source}.json"
    )
    try:
        written_path = write_results(result, output_path)
    except OSError as error:
        print(f"error: could not write {output_path}: {error}", file=sys.stderr)
        return 2

    print(json.dumps(result, indent=2, allow_nan=False))
    print_comparison_rankings(result)
    print(
        f"Wrote {result['num_results']} results from {result['num_models']} models "
        f"to {written_path}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
