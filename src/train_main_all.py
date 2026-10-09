"""Run multiple ``train_main_*`` entry points concurrently.

Example::

    python src/train_main_all.py \
        --dataset-family target-cue_window-500_splits-10 \
        minirocket multirocket transformer

The names are the suffixes of files in ``src`` (``minirocket`` selects
``src/train_main_minirocket.py``). Each selected entry point runs in a separate
Python process and receives the requested dataset family through its module-level
``DATASET_PREFIX`` setting. Fixed-grid entry points also receive the time bounds
and actor scope encoded in the family name.
"""

from __future__ import annotations

import argparse
import importlib
import multiprocessing
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

try:
    from .main_utils.windowing import (
        SIGNED_MILLISECONDS_PATTERN,
        context_window_bounds_ms,
        decode_signed_milliseconds,
    )
except ImportError:  # Direct execution adds ``src`` to ``sys.path``.
    from main_utils.windowing import (
        SIGNED_MILLISECONDS_PATTERN,
        context_window_bounds_ms,
        decode_signed_milliseconds,
    )


SCRIPT_PATH = Path(__file__).resolve()
SOURCE_DIR = SCRIPT_PATH.parent
TRAIN_MAIN_PREFIX = "train_main_"
INTERNAL_RUN_ONE_FLAG = "--_run-one"
ORCHESTRATOR_FILENAMES = {
    "train_main_all.py",
    "train_main_all_one_model.py",
}

_SYMMETRIC_WINDOW_PATTERN = re.compile(
    r"(?:^|_)window-(?P<window_ms>\d+)(?:_|$)"
)
_ASYMMETRIC_WINDOW_PATTERNS = (
    re.compile(
        rf"(?:^|_)windowL-(?P<left_ms>{SIGNED_MILLISECONDS_PATTERN})_"
        rf"windowR-(?P<right_ms>{SIGNED_MILLISECONDS_PATTERN})(?:_|$)"
    ),
    re.compile(
        rf"(?:^|_)wL-(?P<left_ms>{SIGNED_MILLISECONDS_PATTERN})_"
        rf"wR-(?P<right_ms>{SIGNED_MILLISECONDS_PATTERN})(?:_|$)"
    ),
)
_EVENT_SOURCE_PATTERN = re.compile(
    r"(?:^|_)(?:src|source)-(?P<source>speaker|listener|both)(?:_|$)"
)
_ANY_EVENT_SOURCE_PATTERN = re.compile(
    r"(?:^|_)(?:src|source)-(?P<source>[^_]+)(?:_|$)"
)
_ACTOR_SCOPE_BY_EVENT_SOURCE = {
    "speaker": "anchor",
    "listener": "other",
    "both": "all",
}


@dataclass(frozen=True, slots=True)
class TrainingJob:
    suffix: str
    dataset_family: str


@dataclass(frozen=True, slots=True)
class TrainingJobResult:
    suffix: str
    return_code: int
    elapsed_seconds: float
    error: str | None = None


def available_suffixes() -> tuple[str, ...]:
    """Return the suffixes of runnable ``train_main_*`` files in ``src``."""

    suffixes = (
        path.stem.removeprefix(TRAIN_MAIN_PREFIX)
        for path in SOURCE_DIR.glob(f"{TRAIN_MAIN_PREFIX}*.py")
        if path.name not in ORCHESTRATOR_FILENAMES
    )
    return tuple(sorted(suffixes))


def _positive_integer(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return parsed


def _dataset_family(value: str) -> str:
    if not value or value in {".", ".."}:
        raise argparse.ArgumentTypeError("must be a non-empty dataset family name")
    if Path(value).name != value or "\\" in value:
        raise argparse.ArgumentTypeError("must be a name, not a path")
    return value


def _window_bounds_seconds(dataset_family: str) -> tuple[float, float] | None:
    """Infer fixed-grid bounds from a dataset family's millisecond window."""

    for pattern in _ASYMMETRIC_WINDOW_PATTERNS:
        match = pattern.search(dataset_family)
        if match is not None:
            left_ms = decode_signed_milliseconds(match.group("left_ms"))
            right_ms = decode_signed_milliseconds(match.group("right_ms"))
            if left_ms == 0 and right_ms == 0:
                raise ValueError("dataset window bounds cannot both be zero")
            window_start_ms, window_end_ms = context_window_bounds_ms(
                left_ms,
                right_ms,
            )
            return window_start_ms / 1_000.0, window_end_ms / 1_000.0

    match = _SYMMETRIC_WINDOW_PATTERN.search(dataset_family)
    if match is None:
        return None
    window_ms = int(match.group("window_ms"))
    if window_ms < 1:
        raise ValueError("dataset window must be positive")
    window_seconds = window_ms / 1_000.0
    return -window_seconds, window_seconds


def _event_source(dataset_family: str) -> str:
    """Infer the selected event actor, defaulting legacy names to speaker."""

    match = _EVENT_SOURCE_PATTERN.search(dataset_family)
    if match is not None:
        return match.group("source")

    unknown_source = _ANY_EVENT_SOURCE_PATTERN.search(dataset_family)
    if unknown_source is not None:
        raise ValueError(
            f"unsupported dataset event source {unknown_source.group('source')!r}; "
            "expected speaker, listener, or both"
        )

    # Dataset families created before the source field was introduced contain
    # speaker events only.
    return "speaker"


def _configure_dataset_family(module: object, dataset_family: str) -> None:
    """Apply the family, actor scope, and window bounds to a train-main module."""

    setattr(module, "DATASET_PREFIX", dataset_family)
    baseline = getattr(module, "baseline", None)
    actor_scope_targets = [
        target
        for target in (module, baseline)
        if target is not None and hasattr(target, "ACTOR_SCOPE")
    ]
    if actor_scope_targets:
        event_source = _event_source(dataset_family)
        actor_scope = _ACTOR_SCOPE_BY_EVENT_SOURCE[event_source]
        for target in actor_scope_targets:
            setattr(target, "ACTOR_SCOPE", actor_scope)
        print(
            f"Configured actor scope {actor_scope!r} for {event_source!r} events "
            f"from {dataset_family!r}.",
            flush=True,
        )

    has_start = hasattr(module, "WINDOW_START_SECONDS")
    has_end = hasattr(module, "WINDOW_END_SECONDS")
    if not has_start and not has_end:
        return
    if has_start != has_end:
        raise AttributeError(
            "train-main modules must define both WINDOW_START_SECONDS and "
            "WINDOW_END_SECONDS"
        )

    bounds = _window_bounds_seconds(dataset_family)
    if bounds is None:
        raise ValueError(
            f"Cannot infer fixed-grid window bounds from dataset family "
            f"{dataset_family!r}; expected 'window-MILLISECONDS', "
            "'windowL-LEFT_windowR-RIGHT', or 'wL-LEFT_wR-RIGHT' "
            "(use mN for a negative bound, for example wR-m500)"
        )
    window_start_seconds, window_end_seconds = bounds
    setattr(module, "WINDOW_START_SECONDS", window_start_seconds)
    setattr(module, "WINDOW_END_SECONDS", window_end_seconds)
    print(
        f"Configured fixed-grid window [{window_start_seconds:g}, "
        f"{window_end_seconds:g}] seconds from {dataset_family!r}.",
        flush=True,
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run selected src/train_main_<suffix>.py entry points concurrently "
            "for one dataset family."
        )
    )
    parser.add_argument(
        "suffixes",
        nargs="+",
        metavar="SUFFIX",
        help=(
            "train-main suffixes to run, for example 'minirocket transformer'"
        ),
    )
    parser.add_argument(
        "-d",
        "--dataset-family",
        required=True,
        type=_dataset_family,
        help="dataset prefix shared by all runs",
    )
    parser.add_argument(
        "-j",
        "--max-workers",
        type=_positive_integer,
        default=None,
        help="maximum concurrent training processes (default: one per suffix)",
    )
    return parser


def _validate_suffixes(
    parser: argparse.ArgumentParser,
    suffixes: Sequence[str],
) -> tuple[str, ...]:
    duplicates = sorted({suffix for suffix in suffixes if suffixes.count(suffix) > 1})
    if duplicates:
        parser.error(f"duplicate suffixes: {', '.join(duplicates)}")

    available = set(available_suffixes())
    unknown = sorted(set(suffixes) - available)
    if unknown:
        parser.error(
            f"unknown suffixes: {', '.join(unknown)}; "
            f"available suffixes: {', '.join(sorted(available))}"
        )
    return tuple(suffixes)


def _launch_training_job(job: TrainingJob) -> TrainingJobResult:
    """Launch one entry point from a pool worker.

    ``multiprocessing.Pool`` workers are daemonic and therefore cannot create the
    fold-level process pools used by some training entry points. Starting a fresh
    interpreter here keeps model-level and fold-level parallelism composable.
    """

    command = [
        sys.executable,
        os.fspath(SCRIPT_PATH),
        INTERNAL_RUN_ONE_FLAG,
        job.suffix,
        job.dataset_family,
    ]
    started_at = time.monotonic()
    try:
        completed = subprocess.run(command, check=False)
    except OSError as error:
        return TrainingJobResult(
            suffix=job.suffix,
            return_code=1,
            elapsed_seconds=time.monotonic() - started_at,
            error=str(error),
        )
    return TrainingJobResult(
        suffix=job.suffix,
        return_code=completed.returncode,
        elapsed_seconds=time.monotonic() - started_at,
    )


def _run_one(suffix: str, dataset_family: str) -> int:
    """Import and execute one training entry point in a non-daemonic process."""

    module_name = f"{TRAIN_MAIN_PREFIX}{suffix}"
    print(
        f"[{suffix}] starting for dataset family {dataset_family!r} "
        f"(pid={os.getpid()})",
        flush=True,
    )
    module = importlib.import_module(module_name)
    if not hasattr(module, "DATASET_PREFIX"):
        raise AttributeError(f"{module_name} does not define DATASET_PREFIX")
    entry_point = getattr(module, "main", None)
    if not callable(entry_point):
        raise AttributeError(f"{module_name} does not define a callable main()")

    _configure_dataset_family(module, dataset_family)
    entry_point()
    print(f"[{suffix}] finished successfully", flush=True)
    return 0


def _format_duration(elapsed_seconds: float) -> str:
    if elapsed_seconds < 60:
        return f"{elapsed_seconds:.1f}s"
    return f"{elapsed_seconds / 60:.1f}m"


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    arguments = parser.parse_args(argv)
    suffixes = _validate_suffixes(parser, arguments.suffixes)
    worker_count = min(arguments.max_workers or len(suffixes), len(suffixes))
    jobs = [
        TrainingJob(suffix=suffix, dataset_family=arguments.dataset_family)
        for suffix in suffixes
    ]

    print(
        f"Launching {len(jobs)} training entry point(s) with "
        f"{worker_count} worker process(es) for {arguments.dataset_family!r}.",
        flush=True,
    )
    results: list[TrainingJobResult] = []
    spawn_context = multiprocessing.get_context("spawn")
    try:
        with spawn_context.Pool(processes=worker_count) as pool:
            for result in pool.imap_unordered(_launch_training_job, jobs):
                results.append(result)
                state = "completed" if result.return_code == 0 else "failed"
                detail = f": {result.error}" if result.error else ""
                print(
                    f"[{result.suffix}] {state} in "
                    f"{_format_duration(result.elapsed_seconds)}{detail}",
                    flush=True,
                )
    except KeyboardInterrupt:
        print("Interrupted; terminating the training pool.", file=sys.stderr)
        return 130

    failures = [result for result in results if result.return_code != 0]
    if failures:
        failed_names = ", ".join(
            f"{result.suffix} (exit {result.return_code})" for result in failures
        )
        print(f"Training failed: {failed_names}", file=sys.stderr)
        return 1

    print("All requested training entry points completed successfully.", flush=True)
    return 0


def _entry_point(argv: Sequence[str]) -> int:
    if argv and argv[0] == INTERNAL_RUN_ONE_FLAG:
        if len(argv) != 3:
            raise SystemExit(
                f"internal usage: {INTERNAL_RUN_ONE_FLAG} SUFFIX DATASET_FAMILY"
            )
        return _run_one(suffix=argv[1], dataset_family=argv[2])
    return main(argv)


if __name__ == "__main__":
    raise SystemExit(_entry_point(sys.argv[1:]))
