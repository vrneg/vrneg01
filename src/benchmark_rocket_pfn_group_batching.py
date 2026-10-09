"""Compare sequential and batched TabPFN execution on fitted ROCKET groups.

Each timed mode runs in a fresh process so peak resident/GPU memory is attributable
to that mode. Feature transformation is reported separately from TabPFN inference.
The sampled query rows come from the saved training context and are excluded from the
benchmark context; their metrics are useful only for comparing the two execution
paths, not as an unbiased estimate of model quality.
"""

from __future__ import annotations

import argparse
import json
import resource
import subprocess
import sys
import tempfile
import time
from dataclasses import fields, replace
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import torch
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score

from rocket_pfn.config import RocketPFNConfig
from rocket_pfn.features import average_tabpfn_probabilities


def _balanced_indices(
    labels: np.ndarray,
    count: int,
    rng: np.random.Generator,
    excluded: set[int] | None = None,
) -> np.ndarray:
    if count < 2 or count % 2:
        raise ValueError("sample counts must be even and at least 2")
    excluded = excluded or set()
    selected: list[int] = []
    for label in (0, 1):
        candidates = np.asarray(
            [
                int(index)
                for index in np.flatnonzero(labels == label)
                if int(index) not in excluded
            ],
            dtype=np.int64,
        )
        if candidates.size < count // 2:
            raise ValueError(
                f"class {label} has only {candidates.size} available rows; "
                f"need {count // 2}"
            )
        rng.shuffle(candidates)
        selected.extend(candidates[: count // 2].tolist())
    return np.asarray(selected, dtype=np.int64)


def _current_model_config(saved: Any) -> RocketPFNConfig:
    """Fill fields absent from artifact-v1 slotted configs with current defaults."""

    defaults = RocketPFNConfig()
    values = {
        item.name: getattr(saved, item.name, getattr(defaults, item.name))
        for item in fields(RocketPFNConfig)
    }
    model_path = values["tabpfn_model_path"]
    if model_path is not None and not Path(model_path).is_file():
        relocated = (
            Path(__file__).resolve().parents[1]
            / "data"
            / "tabpfn"
            / Path(model_path).name
        )
        if not relocated.is_file():
            raise FileNotFoundError(
                f"saved TabPFN checkpoint does not exist at {model_path}, and no "
                f"relocated checkpoint was found at {relocated}"
            )
        values["tabpfn_model_path"] = relocated
    return RocketPFNConfig(**values)


def _worker(args: argparse.Namespace) -> None:
    artifact = joblib.load(args.artifact)
    fitted = artifact["model"]
    labels = np.asarray(fitted.train_labels, dtype=np.int64)
    if not np.array_equal(np.unique(labels), np.asarray([0, 1])):
        raise ValueError("benchmark artifact must contain binary labels 0 and 1")

    rng = np.random.default_rng(args.sample_seed)
    query_indices = _balanced_indices(labels, args.query_samples, rng)
    train_indices = _balanced_indices(
        labels,
        args.train_samples,
        rng,
        excluded=set(query_indices.tolist()),
    )
    train_values = np.asarray(fitted.train_values)[train_indices]
    query_values = np.asarray(fitted.train_values)[query_indices]
    train_labels = labels[train_indices]
    query_labels = labels[query_indices]

    transformers = fitted.feature_transformers[: args.groups]
    if len(transformers) != args.groups:
        raise ValueError(
            f"artifact has only {len(fitted.feature_transformers)} groups; "
            f"requested {args.groups}"
        )
    transform_start = time.perf_counter()
    feature_pairs = [
        (
            np.asarray(transformer.transform(train_values), dtype=np.float32),
            np.asarray(transformer.transform(query_values), dtype=np.float32),
        )
        for transformer in transformers
    ]
    transform_seconds = time.perf_counter() - transform_start

    model = replace(
        _current_model_config(fitted.model_config),
        tabpfn_device=args.device,
        tabpfn_n_estimators=args.estimators,
        tabpfn_show_progress_bar=False,
        tabpfn_batch_groups=args.worker_mode == "batched",
    )
    use_cuda_metrics = torch.cuda.is_available() and args.device != "cpu"
    if use_cuda_metrics:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    inference_start = time.perf_counter()
    probabilities, group_count = average_tabpfn_probabilities(
        feature_pairs,
        train_labels,
        model,
        fitted.seed,
    )
    if use_cuda_metrics:
        torch.cuda.synchronize()
    inference_seconds = time.perf_counter() - inference_start

    metadata = {
        "mode": args.worker_mode,
        "device": args.device,
        "num_train": int(train_indices.size),
        "num_query": int(query_indices.size),
        "num_groups": group_count,
        "num_features_per_group": [
            int(train_features.shape[1])
            for train_features, _ in feature_pairs
        ],
        "requested_tabpfn_estimators": args.estimators,
        "transform_seconds": transform_seconds,
        "inference_seconds": inference_seconds,
        "max_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "max_cuda_allocated_bytes": (
            int(torch.cuda.max_memory_allocated()) if use_cuda_metrics else None
        ),
        "max_cuda_reserved_bytes": (
            int(torch.cuda.max_memory_reserved()) if use_cuda_metrics else None
        ),
    }
    np.savez(
        args.worker_output,
        probabilities=probabilities,
        labels=query_labels,
        metadata=json.dumps(metadata),
    )


def _classification_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
) -> dict[str, float]:
    predictions = (probabilities >= threshold).astype(np.int64)
    return {
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "macro_f1": float(f1_score(labels, predictions, average="macro")),
        "roc_auc": float(roc_auc_score(labels, probabilities)),
    }


def _run_worker(
    args: argparse.Namespace,
    mode: str,
    output: Path,
) -> dict[str, Any]:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        str(args.artifact),
        "--worker-mode",
        mode,
        "--worker-output",
        str(output),
        "--device",
        args.device,
        "--train-samples",
        str(args.train_samples),
        "--query-samples",
        str(args.query_samples),
        "--groups",
        str(args.groups),
        "--estimators",
        str(args.estimators),
        "--sample-seed",
        str(args.sample_seed),
    ]
    completed = subprocess.run(command, check=False, capture_output=True, text=True)
    if completed.returncode:
        raise RuntimeError(
            f"{mode} benchmark failed with exit {completed.returncode}:\n"
            f"{completed.stdout}{completed.stderr}"
        )
    values = np.load(output)
    return {
        "probabilities": values["probabilities"],
        "labels": values["labels"],
        "metadata": json.loads(str(values["metadata"])),
    }


def _median(records: list[dict[str, Any]], name: str) -> float:
    return float(np.median([record["metadata"][name] for record in records]))


def _parent(args: argparse.Namespace) -> None:
    records: dict[str, list[dict[str, Any]]] = {"sequential": [], "batched": []}
    with tempfile.TemporaryDirectory(prefix="rocket-pfn-benchmark-") as temp:
        temp_dir = Path(temp)
        for repeat in range(args.repeats):
            order = (
                ("sequential", "batched")
                if repeat % 2 == 0
                else ("batched", "sequential")
            )
            for mode in order:
                print(
                    f"Benchmark repeat {repeat + 1}/{args.repeats}: {mode}",
                    flush=True,
                )
                output = temp_dir / f"{mode}-{repeat}.npz"
                records[mode].append(_run_worker(args, mode, output))

    sequential = records["sequential"][0]
    batched = records["batched"][0]
    np.testing.assert_array_equal(sequential["labels"], batched["labels"])
    labels = sequential["labels"]
    sequential_probabilities = sequential["probabilities"]
    batched_probabilities = batched["probabilities"]
    absolute_difference = np.abs(
        sequential_probabilities - batched_probabilities
    )
    threshold = args.threshold
    sequential_seconds = _median(records["sequential"], "inference_seconds")
    batched_seconds = _median(records["batched"], "inference_seconds")

    result = {
        "artifact": str(args.artifact),
        "repeats": args.repeats,
        "sample_seed": args.sample_seed,
        "threshold": threshold,
        "comparison_set_note": (
            "Query rows are sampled from and excluded from the saved training "
            "context; compare modes only, not absolute model quality."
        ),
        "sequential": {
            "median_inference_seconds": sequential_seconds,
            "inference_seconds": [
                record["metadata"]["inference_seconds"]
                for record in records["sequential"]
            ],
            "max_rss_kib": max(
                record["metadata"]["max_rss_kib"]
                for record in records["sequential"]
            ),
            "max_cuda_allocated_bytes": max(
                (
                    record["metadata"]["max_cuda_allocated_bytes"] or 0
                    for record in records["sequential"]
                )
            ),
            "metrics": _classification_metrics(
                labels, sequential_probabilities, threshold
            ),
        },
        "batched": {
            "median_inference_seconds": batched_seconds,
            "inference_seconds": [
                record["metadata"]["inference_seconds"]
                for record in records["batched"]
            ],
            "max_rss_kib": max(
                record["metadata"]["max_rss_kib"]
                for record in records["batched"]
            ),
            "max_cuda_allocated_bytes": max(
                (
                    record["metadata"]["max_cuda_allocated_bytes"] or 0
                    for record in records["batched"]
                )
            ),
            "metrics": _classification_metrics(
                labels, batched_probabilities, threshold
            ),
        },
        "comparison": {
            "speedup_sequential_over_batched": (
                sequential_seconds / batched_seconds
            ),
            "max_absolute_probability_difference": float(
                absolute_difference.max()
            ),
            "mean_absolute_probability_difference": float(
                absolute_difference.mean()
            ),
            "threshold_prediction_disagreements": int(
                np.count_nonzero(
                    (sequential_probabilities >= threshold)
                    != (batched_probabilities >= threshold)
                )
            ),
            "num_compared_predictions": int(labels.size),
        },
        "worker_configuration": sequential["metadata"],
    }
    rendered = json.dumps(result, indent=2) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact", type=Path, help="fitted RocketPFN model.joblib")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--train-samples", type=int, default=256)
    parser.add_argument("--query-samples", type=int, default=64)
    parser.add_argument("--groups", type=int, default=10)
    parser.add_argument("--estimators", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--sample-seed", type=int, default=20260821)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--worker-mode",
        choices=("sequential", "batched"),
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--worker-output", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.groups < 1 or args.estimators < 1 or args.repeats < 1:
        parser.error("groups, estimators, and repeats must be at least 1")
    if not 0.0 <= args.threshold <= 1.0:
        parser.error("threshold must be between 0 and 1")
    if args.worker_mode is not None and args.worker_output is None:
        parser.error("--worker-output is required in worker mode")
    return args


def main() -> None:
    args = _parse_args()
    if args.worker_mode is None:
        _parent(args)
    else:
        _worker(args)


if __name__ == "__main__":
    main()
