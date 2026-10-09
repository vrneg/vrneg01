"""Collect one model's test metrics for an explicit list of event windows.

Unlike the symmetric and asymmetric collectors, this script includes exactly the
requested left/right window pairs, in the provided order, and permits symmetric,
anchor-crossing, fully pre-anchor, and fully post-anchor intervals together.

CLI values use the dataset-name convention in which ``mN`` means ``-N``::

    --windows 2500,m2000 2000,m1500 500,500 m500,1000

For example, ``2500,m2000`` has actual anchor-relative bounds
``[-2500, -2000]`` milliseconds.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

try:
    from .collect_model_results_asymmetric_source_temp_comparison import (
        DEFAULT_MODEL,
        _print_ranking,
        _ranking_rows,
        _validate_event_sources,
        _validate_model_name,
    )
    from .collect_model_results_symmetric_model_comparison import (
        EVENT_SOURCES,
        TARGET,
        SummaryFormatError,
        _dataset_source,
        _dataset_target,
        _fold_results,
        _metric_mean,
        _optional_fold_count,
        _project_root,
        _run_location,
        _window_bounds_ms,
        _window_label,
        write_results,
    )
    from .main_utils.windowing import (
        context_window_bounds_ms,
        decode_signed_milliseconds,
    )
except ImportError:  # Direct execution adds ``src`` to ``sys.path``.
    from collect_model_results_asymmetric_source_temp_comparison import (
        DEFAULT_MODEL,
        _print_ranking,
        _ranking_rows,
        _validate_event_sources,
        _validate_model_name,
    )
    from collect_model_results_symmetric_model_comparison import (
        EVENT_SOURCES,
        TARGET,
        SummaryFormatError,
        _dataset_source,
        _dataset_target,
        _fold_results,
        _metric_mean,
        _optional_fold_count,
        _project_root,
        _run_location,
        _window_bounds_ms,
        _window_label,
        write_results,
    )
    from main_utils.windowing import (
        context_window_bounds_ms,
        decode_signed_milliseconds,
    )


Window = tuple[int, int]


def _validate_window(left_ms: int, right_ms: int) -> Window:
    try:
        context_window_bounds_ms(left_ms, right_ms)
    except (TypeError, ValueError) as error:
        raise ValueError(str(error)) from error
    return left_ms, right_ms


def _parse_window(value: str) -> Window:
    """Parse one CLI ``LEFT,RIGHT`` pair using ``mN`` for negative values."""

    token = value.strip()
    if token.startswith("(") and token.endswith(")"):
        token = token[1:-1].strip()
    parts = [part.strip() for part in token.split(",")]
    if len(parts) != 2 or not all(parts):
        raise argparse.ArgumentTypeError(
            f"invalid window {value!r}; expected LEFT,RIGHT, for example "
            "2500,m2000"
        )
    try:
        left_ms = decode_signed_milliseconds(parts[0])
        right_ms = decode_signed_milliseconds(parts[1])
        return _validate_window(left_ms, right_ms)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            f"invalid window {value!r}: {error}"
        ) from error


def _normalize_windows(windows: Sequence[Sequence[int]]) -> tuple[Window, ...]:
    if isinstance(windows, (str, bytes)) or not isinstance(windows, Sequence):
        raise ValueError("windows must be a non-empty sequence of integer pairs")
    if not windows:
        raise ValueError("windows must not be empty")

    normalized: list[Window] = []
    seen: set[Window] = set()
    for index, window in enumerate(windows):
        if (
            isinstance(window, (str, bytes))
            or not isinstance(window, Sequence)
            or len(window) != 2
        ):
            raise ValueError(f"windows[{index}] must be a pair of integers")
        left_ms, right_ms = window
        if (
            isinstance(left_ms, bool)
            or not isinstance(left_ms, int)
            or isinstance(right_ms, bool)
            or not isinstance(right_ms, int)
        ):
            raise ValueError(f"windows[{index}] must be a pair of integers")
        pair = _validate_window(left_ms, right_ms)
        if pair in seen:
            raise ValueError(
                f"windows must not contain duplicate pair {_window_label(*pair)!r}"
            )
        seen.add(pair)
        normalized.append(pair)
    return tuple(normalized)


def collect_model_results(
    outputs_dir: str | Path,
    model: str,
    windows: Sequence[Sequence[int]],
    event_sources: tuple[str, ...] = EVENT_SOURCES,
) -> dict[str, Any]:
    """Return one model's cue-target results for exactly ``windows``.

    Each requested pair is represented even when no matching summary exists.
    Dataset families without an explicit source are treated as legacy speaker
    datasets. Invalid matching summaries are retained in ``issues``.
    """

    _validate_model_name(model)
    requested_windows = _normalize_windows(windows)
    _validate_event_sources(event_sources)
    selected_windows = frozenset(requested_windows)
    selected_sources = frozenset(event_sources)

    outputs_dir = Path(outputs_dir).resolve()
    if not outputs_dir.is_dir():
        raise FileNotFoundError(f"Outputs directory does not exist: {outputs_dir}")
    model_dir = outputs_dir / model
    if not model_dir.is_dir():
        raise FileNotFoundError(f"Model output directory does not exist: {model_dir}")

    buckets: dict[
        Window, dict[str, list[dict[str, Any]]]
    ] = defaultdict(lambda: defaultdict(list))
    issues: list[dict[str, str]] = []
    summaries_scanned = 0
    summaries_matching = 0

    for summary_path in sorted(model_dir.rglob("*_cross_validation_summary.json")):
        summaries_scanned += 1
        try:
            relative_path = summary_path.relative_to(outputs_dir)
            if len(relative_path.parts) < 2:
                continue
            dataset_family = relative_path.parts[1]
            if _dataset_target(dataset_family) != TARGET:
                continue

            event_source, source_inferred = _dataset_source(dataset_family)
            if event_source not in selected_sources:
                continue
            window = _window_bounds_ms(dataset_family)
            if window not in selected_windows:
                continue
            summaries_matching += 1

            discovered_model, dataset_family, variant, seed = _run_location(
                summary_path, outputs_dir
            )
            if discovered_model != model:
                raise SummaryFormatError(
                    f"expected model directory {model!r}, found {discovered_model!r}"
                )
            with summary_path.open("r", encoding="utf-8") as file:
                summary = json.load(file)
            if not isinstance(summary, Mapping):
                raise SummaryFormatError("top-level JSON value is not an object")

            macro_f1, macro_f1_valid_folds = _metric_mean(summary, "macro_f1")
            auroc, auroc_valid_folds = _metric_mean(summary, "roc_auc")
            buckets[window][event_source].append(
                {
                    "dataset_family": dataset_family,
                    "event_source": event_source,
                    "event_source_inferred": source_inferred,
                    "variant": variant,
                    "seed": seed,
                    "num_folds": _optional_fold_count(summary),
                    "valid_folds": {
                        "auroc": auroc_valid_folds,
                        "macro_f1": macro_f1_valid_folds,
                    },
                    "auroc": auroc,
                    "macro_f1": macro_f1,
                    "fold_results": _fold_results(summary),
                    "summary_file": relative_path.as_posix(),
                }
            )
        except (OSError, json.JSONDecodeError, SummaryFormatError) as error:
            issues.append(
                {
                    "summary_file": summary_path.relative_to(outputs_dir).as_posix(),
                    "reason": str(error),
                }
            )

    result_windows: dict[str, dict[str, Any]] = {}
    missing_windows: list[str] = []
    sources_with_results: set[str] = set()
    num_results = 0
    num_windows_with_results = 0
    for left_ms, right_ms in requested_windows:
        window = (left_ms, right_ms)
        sources = buckets.get(window, {})
        sorted_sources: dict[str, list[dict[str, Any]]] = {}
        window_has_results = False
        for event_source in event_sources:
            runs = sorted(
                sources.get(event_source, []),
                key=lambda run: (
                    run["dataset_family"],
                    run["variant"] or "",
                    run["seed"],
                    run["summary_file"],
                ),
            )
            sorted_sources[event_source] = runs
            if runs:
                window_has_results = True
                sources_with_results.add(event_source)
                num_results += len(runs)

        label = _window_label(left_ms, right_ms)
        if window_has_results:
            num_windows_with_results += 1
        else:
            missing_windows.append(label)
        result_windows[label] = {
            "left_ms": left_ms,
            "right_ms": right_ms,
            "window_start_ms": -left_ms,
            "window_end_ms": right_ms,
            "sources": sorted_sources,
        }

    return {
        "target": TARGET,
        "model": model,
        "event_sources": list(event_sources),
        "window_scope": "explicit_selection",
        "requested_windows": [
            _window_label(left_ms, right_ms)
            for left_ms, right_ms in requested_windows
        ],
        "primary_metric": "auroc",
        "metric_scope": "mean_across_test_folds",
        "source_metric_paths": {
            "auroc": "fold_statistics.test.roc_auc.mean",
            "macro_f1": "fold_statistics.test.macro_f1.mean",
        },
        "source_fold_results_path": "folds.<fold>.test_metrics",
        "outputs_dir": str(outputs_dir),
        "summaries_scanned": summaries_scanned,
        "summaries_matching": summaries_matching,
        "num_results": num_results,
        "num_windows": len(result_windows),
        "num_windows_with_results": num_windows_with_results,
        "num_sources_with_results": len(sources_with_results),
        "missing_windows": missing_windows,
        "windows": result_windows,
        "issues": sorted(issues, key=lambda issue: issue["summary_file"]),
    }


def print_comparison_rankings(result: Mapping[str, Any]) -> None:
    """Print AUROC-first rankings while keeping JSON stdout parseable."""

    print(
        "\nSource rankings (each available selected window weighted equally)",
        file=sys.stderr,
    )
    _print_ranking(
        "By average AUROC",
        _ranking_rows(result, "auroc", "source"),
        "window",
    )
    _print_ranking(
        "By average macro F1",
        _ranking_rows(result, "macro_f1", "source"),
        "window",
    )

    print(
        "\nWindow rankings (each available event source weighted equally)",
        file=sys.stderr,
    )
    _print_ranking(
        "By average AUROC",
        _ranking_rows(result, "auroc", "window"),
        "event source",
    )
    _print_ranking(
        "By average macro F1",
        _ranking_rows(result, "macro_f1", "window"),
        "event source",
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Collect cue-target AUROC and macro-F1 test-fold means for one model "
            "and an explicit ordered list of event windows."
        )
    )
    parser.add_argument(
        "model",
        nargs="?",
        default="rocket_pfn",
        help=f"Exact model output directory name (default: {DEFAULT_MODEL}).",
    )
    parser.add_argument(
        "--windows",
        nargs="+",
        type=_parse_window,
        default=[(2500, -2000), (2000, -1500), (1500, -1000),(1000, -500), (500, 0), (0, 500),(-500, 1000), (-1000, 1500), (-1500, 2000), (-2000, 2500)],
        required=False,
        metavar="LEFT,RIGHT",
        help=(
            "Exact ordered window pairs to collect. Use mN for a negative value, "
            "for example: --windows 2500,m2000 2000,m1500 500,500."
        ),
    )
    parser.add_argument(
        "--event-sources",
        nargs="+",
        choices=EVENT_SOURCES,
        default=EVENT_SOURCES,
        metavar="SOURCE",
        help="Event sources to collect (default: speaker listener both).",
    )
    parser.add_argument(
        "--outputs-dir",
        type=Path,
        default=_project_root() / "outputs",
        help="Model output root (default: <repository>/outputs).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help=(
            "Destination JSON file (default: <outputs-dir>/"
            "fold_average_test_results_cue_<model>_selected_windows.json)."
        ),
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail without writing when a matching summary is invalid.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        result = collect_model_results(
            args.outputs_dir,
            args.model,
            args.windows,
            tuple(args.event_sources),
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
        / f"fold_average_test_results_{TARGET}_{args.model}_selected_windows.json"
    )
    try:
        written_path = write_results(result, output_path)
    except OSError as error:
        print(f"error: could not write {output_path}: {error}", file=sys.stderr)
        return 2

    print(json.dumps(result, indent=2, allow_nan=False))
    print_comparison_rankings(result)
    print(
        f"Wrote {result['num_results']} results across "
        f"{result['num_windows_with_results']}/{result['num_windows']} requested "
        f"windows and {result['num_sources_with_results']} sources to {written_path}",
        file=sys.stderr,
    )
    if result["missing_windows"]:
        print(
            "No matching results for: " + ", ".join(result["missing_windows"]),
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
