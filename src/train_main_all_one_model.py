"""Run one ``train_main_*`` entry point on multiple datasets concurrently.

Example::

    python src/train_main_all_one_model.py \
        --model minirocket \
        target-cue_window-100_splits-10 \
        target-cue_window-500_splits-10

The model name is the suffix of a file in ``src`` (``minirocket`` selects
``src/train_main_minirocket.py``). Each dataset runs in a separate Python process
and reaches the selected entry point through its module-level ``DATASET_PREFIX``
setting. Fixed-grid entry points also receive the actor scope and time bounds encoded
in each family name.
"""

from __future__ import annotations

import argparse
import importlib
import multiprocessing
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

try:
    from .train_main_all import (
        INTERNAL_RUN_ONE_FLAG,
        TRAIN_MAIN_PREFIX,
        _configure_dataset_family,
        _dataset_family,
        _format_duration,
        _positive_integer,
        available_suffixes,
    )
except ImportError:  # Direct execution adds ``src`` rather than the project root.
    from train_main_all import (
        INTERNAL_RUN_ONE_FLAG,
        TRAIN_MAIN_PREFIX,
        _configure_dataset_family,
        _dataset_family,
        _format_duration,
        _positive_integer,
        available_suffixes,
    )


SCRIPT_PATH = Path(__file__).resolve()


@dataclass(frozen=True, slots=True)
class TrainingJob:
    suffix: str
    dataset_family: str


@dataclass(frozen=True, slots=True)
class TrainingJobResult:
    suffix: str
    dataset_family: str
    return_code: int
    elapsed_seconds: float
    error: str | None = None


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run one selected src/train_main_<suffix>.py entry point "
            "concurrently for multiple dataset families."
        )
    )
    parser.add_argument(
        "dataset_families",
        nargs="+",
        metavar="DATASET_FAMILY",
        type=_dataset_family,
        help="dataset prefixes to train the selected model on",
    )
    parser.add_argument(
        "-m",
        "--model",
        "--suffix",
        dest="suffix",
        required=True,
        help="train-main suffix to run, for example 'minirocket'",
    )
    parser.add_argument(
        "-j",
        "--max-workers",
        type=_positive_integer,
        default=None,
        help="maximum concurrent training processes (default: one per dataset)",
    )
    return parser


def _validate_suffix(parser: argparse.ArgumentParser, suffix: str) -> str:
    available = available_suffixes()
    if suffix not in available:
        parser.error(
            f"unknown suffix: {suffix}; available suffixes: {', '.join(available)}"
        )
    return suffix


def _validate_dataset_families(
    parser: argparse.ArgumentParser,
    dataset_families: Sequence[str],
) -> tuple[str, ...]:
    duplicates = sorted(
        {
            dataset_family
            for dataset_family in dataset_families
            if dataset_families.count(dataset_family) > 1
        }
    )
    if duplicates:
        parser.error(f"duplicate dataset families: {', '.join(duplicates)}")
    return tuple(dataset_families)


def _launch_training_job(job: TrainingJob) -> TrainingJobResult:
    """Launch one dataset run from a pool worker in a fresh interpreter."""

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
            dataset_family=job.dataset_family,
            return_code=1,
            elapsed_seconds=time.monotonic() - started_at,
            error=str(error),
        )
    return TrainingJobResult(
        suffix=job.suffix,
        dataset_family=job.dataset_family,
        return_code=completed.returncode,
        elapsed_seconds=time.monotonic() - started_at,
    )


def _run_one(suffix: str, dataset_family: str) -> int:
    """Import and execute the model for one dataset in a non-daemonic process."""

    module_name = f"{TRAIN_MAIN_PREFIX}{suffix}"
    print(
        f"[{suffix} @ {dataset_family}] starting (pid={os.getpid()})",
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
    print(f"[{suffix} @ {dataset_family}] finished successfully", flush=True)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    arguments = parser.parse_args(argv)
    suffix = _validate_suffix(parser, arguments.suffix)
    dataset_families = _validate_dataset_families(
        parser, arguments.dataset_families
    )
    worker_count = min(
        arguments.max_workers or len(dataset_families),
        len(dataset_families),
    )
    jobs = [
        TrainingJob(suffix=suffix, dataset_family=dataset_family)
        for dataset_family in dataset_families
    ]

    print(
        f"Launching {len(jobs)} training job(s) with {worker_count} worker "
        f"process(es) for model {suffix!r}.",
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
                    f"[{result.suffix} @ {result.dataset_family}] {state} in "
                    f"{_format_duration(result.elapsed_seconds)}{detail}",
                    flush=True,
                )
    except KeyboardInterrupt:
        print("Interrupted; terminating the training pool.", file=sys.stderr)
        return 130

    failures = [result for result in results if result.return_code != 0]
    if failures:
        failed_names = ", ".join(
            f"{result.dataset_family} (exit {result.return_code})"
            for result in failures
        )
        print(f"Training failed for model {suffix}: {failed_names}", file=sys.stderr)
        return 1

    print("All requested dataset runs completed successfully.", flush=True)
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
