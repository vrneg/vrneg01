"""Collect one model's test metrics across asymmetric windows and event sources.

The script searches one ``outputs/<model>`` directory for cue-target
``*_cross_validation_summary.json`` artifacts. It retains every asymmetric event
window, groups results by window and event source, and preserves separate experiment
variants, random seeds, and complete fold-level test metrics. ROCKET-PFN is selected
by default.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

if __package__:
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
else:
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


DEFAULT_MODEL = "rocket_pfn"


def _validate_model_name(model: str) -> None:
    if (
        not isinstance(model, str)
        or not model
        or model in {".", ".."}
        or Path(model).name != model
    ):
        raise ValueError("model must be one non-empty output directory name")


def _validate_event_sources(event_sources: tuple[str, ...]) -> None:
    if not isinstance(event_sources, tuple) or not event_sources:
        raise ValueError("event_sources must be a non-empty tuple")
    unsupported = [source for source in event_sources if source not in EVENT_SOURCES]
    if unsupported:
        choices = ", ".join(EVENT_SOURCES)
        raise ValueError(
            f"event_sources contains unsupported values {unsupported!r}; "
            f"expected only: {choices}"
        )
    if len(set(event_sources)) != len(event_sources):
        raise ValueError("event_sources must not contain duplicates")


def collect_model_results(
    outputs_dir: str | Path,
    model: str = DEFAULT_MODEL,
    event_sources: tuple[str, ...] = EVENT_SOURCES,
) -> dict[str, Any]:
    """Return one model's cue-target results for every asymmetric event window.

    Windows with equal left and right context are excluded. Dataset families without
    an explicit source are treated as legacy speaker datasets, consistent with the
    symmetric model-comparison collector. Invalid matching summaries are recorded in
    ``issues`` so other completed runs remain available.
    """

    _validate_model_name(model)
    _validate_event_sources(event_sources)
    selected_sources = frozenset(event_sources)

    outputs_dir = Path(outputs_dir).resolve()
    if not outputs_dir.is_dir():
        raise FileNotFoundError(f"Outputs directory does not exist: {outputs_dir}")
    model_dir = outputs_dir / model
    if not model_dir.is_dir():
        raise FileNotFoundError(f"Model output directory does not exist: {model_dir}")

    buckets: dict[
        tuple[int, int], dict[str, list[dict[str, Any]]]
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
            left_ms, right_ms = _window_bounds_ms(dataset_family)
            if left_ms == right_ms:
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
            fold_results = _fold_results(summary)
            buckets[(left_ms, right_ms)][event_source].append(
                {
                    "dataset_family": dataset_family,
                    "event_source": event_source,
                    "event_source_inferred": source_inferred,
                    "variant": variant,
                    "seed": seed,
                    "num_folds": _optional_fold_count(summary),
                    "valid_folds": {
                        "macro_f1": macro_f1_valid_folds,
                        "auroc": auroc_valid_folds,
                    },
                    "macro_f1": macro_f1,
                    "auroc": auroc,
                    "fold_results": fold_results,
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

    windows: dict[str, dict[str, Any]] = {}
    sources_with_results: set[str] = set()
    num_results = 0
    for (left_ms, right_ms), sources in sorted(buckets.items()):
        sorted_sources: dict[str, list[dict[str, Any]]] = {}
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
                sources_with_results.add(event_source)
                num_results += len(runs)

        windows[_window_label(left_ms, right_ms)] = {
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
        "window_scope": "asymmetric_only",
        "metric_scope": "mean_across_test_folds",
        "source_metric_paths": {
            "macro_f1": "fold_statistics.test.macro_f1.mean",
            "auroc": "fold_statistics.test.roc_auc.mean",
        },
        "source_fold_results_path": "folds.<fold>.test_metrics",
        "outputs_dir": str(outputs_dir),
        "summaries_scanned": summaries_scanned,
        "summaries_matching": summaries_matching,
        "num_results": num_results,
        "num_windows": len(windows),
        "num_sources_with_results": len(sources_with_results),
        "windows": windows,
        "issues": sorted(issues, key=lambda issue: issue["summary_file"]),
    }


def _ranking_rows(
    result: Mapping[str, Any],
    metric: str,
    group: str,
) -> list[tuple[str, float | None, int]]:
    """Rank sources or windows with equal weight per source-window pairing."""

    windows = result["windows"]
    grouped_scores: dict[str, list[float]] = {}
    if group == "source":
        grouped_scores.update((source, []) for source in result["event_sources"])
    elif group == "window":
        grouped_scores.update((window_label, []) for window_label in windows)
    else:
        raise ValueError("group must be 'source' or 'window'")

    for window_label, window in windows.items():
        for event_source, runs in window["sources"].items():
            run_scores = [run[metric] for run in runs if run[metric] is not None]
            if not run_scores:
                continue
            source_window_average = sum(run_scores) / len(run_scores)
            group_name = event_source if group == "source" else window_label
            grouped_scores[group_name].append(source_window_average)

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


def print_comparison_rankings(result: Mapping[str, Any]) -> None:
    """Print source and window rankings while keeping JSON stdout parseable."""

    print(
        "\nSource rankings (each asymmetric window weighted equally)",
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
        "\nWindow rankings (each event source weighted equally)",
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
            "Collect cue-target macro F1 and AUROC test-fold means for one model "
            "across every asymmetric window and the speaker, listener, and both "
            "event sources."
        )
    )
    parser.add_argument(
        "model",
        nargs="?",
        default=DEFAULT_MODEL,
        help=f"Exact model output directory name (default: {DEFAULT_MODEL}).",
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
            "fold_average_test_results_cue_<model>_"
            "asymmetric_source_temp_comparison.json)."
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
        / f"fold_average_test_results_{TARGET}_{args.model}_"
        "asymmetric_source_temp_comparison.json"
    )
    try:
        written_path = write_results(result, output_path)
    except OSError as error:
        print(f"error: could not write {output_path}: {error}", file=sys.stderr)
        return 2

    print(json.dumps(result, indent=2, allow_nan=False))
    print_comparison_rankings(result)
    print(
        f"Wrote {result['num_results']} results across {result['num_windows']} "
        f"asymmetric windows and {result['num_sources_with_results']} sources "
        f"to {written_path}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
