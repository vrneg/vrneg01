"""Resumable modality-only and leave-one-modality-out retraining experiments."""

from __future__ import annotations

import csv
import gc
import importlib
import json
import multiprocessing
import os
import time
from collections import deque
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import ExitStack
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Iterable

import joblib
import numpy as np
import torch
from threadpoolctl import threadpool_limits

try:
    from event_transformer.features import BASE_MODALITY_DIMS_BY_NAME, MODALITY_NAMES
except ModuleNotFoundError as error:
    if error.name != "event_transformer":
        raise
    from ..event_transformer.features import BASE_MODALITY_DIMS_BY_NAME, MODALITY_NAMES

try:
    from representation.facial_units import BLENDSHAPE_NAMES
except ModuleNotFoundError as error:
    if error.name != "representation":
        raise
    from ..representation.facial_units import BLENDSHAPE_NAMES

from .configuration import dataclass_from_dict, relocate_checkpoint_paths
from .metrics import METRIC_NAMES, binary_metrics, bootstrap_mean_interval


@dataclass(frozen=True, slots=True)
class TrainerSpec:
    config_module: str
    training_module: str
    train_function: str


@dataclass(frozen=True, slots=True)
class RetrainingResult:
    manifest_path: Path
    results_path: Path
    summary_path: Path
    summary: dict[str, Any]


@dataclass(frozen=True, slots=True)
class RetrainingJob:
    key: str
    checkpoint_type: str
    fold_name: str
    variant_name: str
    variant_type: str
    modality: str
    included_modalities: tuple[str, ...]
    masked_channels: tuple[str, ...]
    experiment_config: dict[str, Any]
    output_dir: Path
    device: str | None
    threads_per_job: int | None
    tabpfn_batch_groups: bool | None
    tabpfn_show_progress_bar: bool | None


@dataclass(frozen=True, slots=True)
class VariantSpec:
    variant_name: str
    variant_type: str
    modality: str
    included_modalities: tuple[str, ...]
    masked_channels: tuple[str, ...] = ()


VARIANT_TYPES = ("modality_only", "leave_one_out")
ROCKET_PFN_CHECKPOINT_TYPES = {
    "rocket_pfn_classifier",
    "multirocket_hydra_pfn_classifier",
}
LOWER_FACE_VARIANT = "lower_face"
LOWER_FACE_BLENDSHAPE_PREFIXES = (
    "Jaw_",
    "Lip_",
    "Lips_",
    "Lower_Lip_",
    "Upper_Lip_",
    "Mouth_",
)
LOWER_FACE_ADDITIONAL_BLENDSHAPES = frozenset(
    {
        "Cheek_Puff_L",
        "Cheek_Puff_R",
        "Cheek_Raiser_L",
        "Cheek_Raiser_R",
        "Cheek_Suck_L",
        "Cheek_Suck_R",
        "Chin_Raiser_B",
        "Chin_Raiser_T",
    }
)


TRAINERS: dict[str, TrainerSpec] = {
    "multivariate_minirocket_classifier": TrainerSpec(
        "minirocket.config", "minirocket.training", "train_minirocket"
    ),
    "multivariate_multirocket_classifier": TrainerSpec(
        "multirocket.config", "multirocket.training", "train_multirocket"
    ),
    "multivariate_multirocket_hydra_classifier": TrainerSpec(
        "multirocket.config", "multirocket.training", "train_multirocket"
    ),
    "rocket_pfn_classifier": TrainerSpec(
        "rocket_pfn.config", "rocket_pfn.training", "train_rocket_pfn"
    ),
    "multirocket_hydra_pfn_classifier": TrainerSpec(
        "rocket_pfn.config", "rocket_pfn.training", "train_rocket_pfn"
    ),
    "masht_tabpfn_v3_classifier": TrainerSpec(
        "masht.config", "masht.training", "train_masht"
    ),
    "adaptive_multi_representation_rocket_pfn_v3": TrainerSpec(
        "adaptive_rocket_pfn.config",
        "adaptive_rocket_pfn.training",
        "train_adaptive_rocket_pfn",
    ),
    "sparse_multirocket_hydra_classifier": TrainerSpec(
        "sparse_multirocket_hydra.config",
        "sparse_multirocket_hydra.training",
        "train_sparse_multirocket_hydra",
    ),
    "multivariate_selfrocket_classifier": TrainerSpec(
        "selfrocket.config", "selfrocket.training", "train_selfrocket"
    ),
    "multivariate_castor_classifier": TrainerSpec(
        "castor.config", "castor.training", "train_castor"
    ),
    "multivariate_weasel_v2_classifier": TrainerSpec(
        "weasel_v2.config", "weasel_v2.training", "train_weasel_v2"
    ),
    "multivariate_mrsqm_classifier": TrainerSpec(
        "mrsqm_model.config", "mrsqm_model.training", "train_mrsqm"
    ),
    "multivariate_drcif_classifier": TrainerSpec(
        "drcif.config", "drcif.training", "train_drcif"
    ),
    "multivariate_hivecote_v2_classifier": TrainerSpec(
        "hivecote_v2.config", "hivecote_v2.training", "train_hivecote_v2"
    ),
    "modality_aware_inception_tcn_classifier": TrainerSpec(
        "inception_tcn.config", "inception_tcn.training", "train_inception_tcn"
    ),
    "compact_cue_aware_fusion_tcn_classifier": TrainerSpec(
        "compact_fusion_tcn.config",
        "compact_fusion_tcn.training",
        "train_compact_fusion_tcn",
    ),
    "fine_tuned_classifier": TrainerSpec(
        "event_transformer.config",
        "event_transformer.training",
        "train_event_transformer",
    ),
}


def _load_payload(path: Path) -> dict[str, Any]:
    payload = (
        torch.load(path, map_location="cpu", weights_only=False)
        if path.suffix == ".pt"
        else joblib.load(path)
    )
    if not isinstance(payload, dict):
        raise ValueError(f"Unsupported checkpoint payload in {path}")
    return payload


def _checkpoint_type(payload: dict[str, Any]) -> str:
    return str(payload.get("artifact_type", payload.get("checkpoint_type", "")))


def _resolve_trainer(checkpoint_type: str) -> tuple[type, Callable[[Any], Any]]:
    try:
        spec = TRAINERS[checkpoint_type]
    except KeyError as error:
        raise ValueError(f"No retraining adapter for {checkpoint_type!r}") from error
    try:
        config_module = importlib.import_module(spec.config_module)
        training_module = importlib.import_module(spec.training_module)
    except ModuleNotFoundError as error:
        if error.name != spec.config_module.split(".", 1)[0]:
            raise
        config_module = importlib.import_module(f"src.{spec.config_module}")
        training_module = importlib.import_module(f"src.{spec.training_module}")
    return config_module.ExperimentConfig, getattr(training_module, spec.train_function)


def _normalize_variant_types(
    variant_types: Iterable[str] | None,
) -> tuple[str, ...]:
    if variant_types is None:
        return VARIANT_TYPES
    requested = set(variant_types)
    unknown = requested - set(VARIANT_TYPES)
    if unknown:
        raise ValueError(f"Unknown retraining variant types: {sorted(unknown)}")
    if not requested:
        raise ValueError("At least one retraining variant type must be requested")
    return tuple(
        variant_type for variant_type in VARIANT_TYPES if variant_type in requested
    )


def _variant_specs(
    variant_types: Iterable[str] | None = None,
) -> list[VariantSpec]:
    selected = set(_normalize_variant_types(variant_types))
    variants = []
    for modality in MODALITY_NAMES:
        if "modality_only" in selected:
            variants.append(
                VariantSpec(
                    f"only-{modality}",
                    "modality_only",
                    modality,
                    (modality,),
                )
            )
        if "leave_one_out" in selected:
            variants.append(
                VariantSpec(
                    f"without-{modality}",
                    "leave_one_out",
                    modality,
                    tuple(name for name in MODALITY_NAMES if name != modality),
                )
            )
    if "leave_one_out" in selected:
        variants.append(
            VariantSpec(
                "without-lower_face",
                "leave_one_out",
                LOWER_FACE_VARIANT,
                MODALITY_NAMES,
                _lower_face_masked_channels(),
            )
        )
    return variants


def _lower_face_blendshape_indices() -> tuple[int, ...]:
    return tuple(
        index
        for index, name in enumerate(BLENDSHAPE_NAMES)
        if name.startswith(LOWER_FACE_BLENDSHAPE_PREFIXES)
        or name in LOWER_FACE_ADDITIONAL_BLENDSHAPES
    )


def _lower_face_masked_channels() -> tuple[str, ...]:
    indices = _lower_face_blendshape_indices()
    velocity_offset = BASE_MODALITY_DIMS_BY_NAME["Facial"]
    return tuple(f"Facial.feature_{index}" for index in indices) + tuple(
        f"Facial.feature_{velocity_offset + index}" for index in indices
    )


def _build_retraining_config(
    config_cls: type,
    job: RetrainingJob,
) -> Any:
    base_config = dataclass_from_dict(config_cls, job.experiment_config)
    relocated_data = relocate_checkpoint_paths(base_config.data)
    relocated_model = relocate_checkpoint_paths(base_config.model)
    base_config = replace(base_config, data=relocated_data, model=relocated_model)
    data_changes: dict[str, Any] = {
        "included_modalities": job.included_modalities,
    }
    if hasattr(base_config.data, "masked_channels"):
        data_changes["masked_channels"] = job.masked_channels or None
    elif job.masked_channels:
        raise ValueError(
            f"{job.variant_name} requires fixed-grid data with masked_channels support"
        )
    data_config = replace(base_config.data, **data_changes)
    variant_root = job.output_dir / "models" / job.variant_name
    changes: dict[str, Any] = {
        "data": data_config,
        "output_dir": variant_root,
        "run_name": job.fold_name,
    }
    model_changes: dict[str, Any] = {}
    if job.device is not None and hasattr(base_config, "device"):
        changes["device"] = job.device
    if job.device is not None and hasattr(base_config.model, "tabpfn_device"):
        model_changes["tabpfn_device"] = job.device
    if job.tabpfn_batch_groups is not None:
        if not hasattr(base_config.model, "tabpfn_batch_groups"):
            raise ValueError(
                "--tabpfn-batch-groups is only supported by TabPFN retraining"
            )
        model_changes["tabpfn_batch_groups"] = job.tabpfn_batch_groups
    if job.tabpfn_show_progress_bar is not None:
        if not hasattr(base_config.model, "tabpfn_show_progress_bar"):
            raise ValueError(
                "--no-tabpfn-progress is only supported by TabPFN retraining"
            )
        model_changes["tabpfn_show_progress_bar"] = job.tabpfn_show_progress_bar
    if job.threads_per_job is not None:
        # Nested estimator/transform pools otherwise multiply outer process
        # parallelism. Capping these fields changes throughput, not model semantics.
        for field_name in (
            "transform_n_jobs",
            "n_jobs",
            "search_n_jobs",
            "tabpfn_preprocessing_jobs",
        ):
            if not hasattr(base_config.model, field_name):
                continue
            current = getattr(base_config.model, field_name)
            if isinstance(current, int) and (
                current < 0 or current > job.threads_per_job
            ):
                model_changes[field_name] = job.threads_per_job
    if model_changes:
        changes["model"] = replace(base_config.model, **model_changes)
    if hasattr(base_config, "feature_artifact_path"):
        # A full-modality feature artifact is not valid preprocessing for a
        # subset experiment; each variant must fit its own representation.
        changes["feature_artifact_path"] = None
    return replace(base_config, **changes)


_CACHED_ROCKET_DATA_CONFIG: Any | None = None
_CACHED_ROCKET_DATA: Any | None = None


def _cached_rocket_data(job: RetrainingJob, variant_config: Any) -> Any | None:
    """Prepare a fold once in each fold-affine worker and mask each variant."""

    if job.checkpoint_type not in ROCKET_PFN_CHECKPOINT_TYPES:
        return None
    try:
        from rocket_pfn.data import apply_fixed_grid_masks, prepare_data
    except ModuleNotFoundError as error:
        if error.name != "rocket_pfn":
            raise
        from ..rocket_pfn.data import apply_fixed_grid_masks, prepare_data

    data_changes: dict[str, Any] = {"included_modalities": None}
    if hasattr(variant_config.data, "masked_channels"):
        data_changes["masked_channels"] = None
    full_data_config = replace(variant_config.data, **data_changes)

    global _CACHED_ROCKET_DATA_CONFIG, _CACHED_ROCKET_DATA
    if _CACHED_ROCKET_DATA_CONFIG != full_data_config:
        started = time.monotonic()
        print(
            f"[{job.fold_name}] Preparing reusable full-modality dataset ...",
            flush=True,
        )
        _CACHED_ROCKET_DATA = prepare_data(full_data_config)
        _CACHED_ROCKET_DATA_CONFIG = full_data_config
        print(
            f"[{job.fold_name}] Prepared reusable dataset in "
            f"{time.monotonic() - started:.1f}s.",
            flush=True,
        )
    return apply_fixed_grid_masks(_CACHED_ROCKET_DATA, variant_config.data)


def _run_retraining_job_with_trainer(
    job: RetrainingJob,
    config_cls: type,
    trainer: Callable[[Any], Any],
) -> tuple[str, dict[str, Any]]:
    variant_config = _build_retraining_config(config_cls, job)
    print(f"[{job.key}] Starting subset retraining ...", flush=True)
    prepared_data = _cached_rocket_data(job, variant_config)
    with threadpool_limits(limits=job.threads_per_job):
        result = (
            trainer(variant_config)
            if prepared_data is None
            else trainer(variant_config, prepared_data=prepared_data)
        )
    print(f"[{job.key}] Completed subset retraining.", flush=True)
    return job.key, {
        "status": "complete",
        "fold": job.fold_name,
        "variant": job.variant_name,
        "variant_type": job.variant_type,
        "modality": job.modality,
        "included_modalities": list(job.included_modalities),
        "masked_channels": list(job.masked_channels),
        "checkpoint_path": str(result.checkpoint_path.resolve()),
        "predictions_path": str(result.predictions_path.resolve()),
        "validation_predictions_path": str(
            result.validation_predictions_path.resolve()
        ),
        "validation_metrics": result.validation_metrics,
        "test_metrics": result.test_metrics,
    }


def _run_retraining_job(job: RetrainingJob) -> tuple[str, dict[str, Any]]:
    config_cls, trainer = _resolve_trainer(job.checkpoint_type)
    return _run_retraining_job_with_trainer(job, config_cls, trainer)


def _initialize_retraining_worker(threads_per_job: int) -> None:
    """Apply a stable inner-thread budget inside every spawned worker."""

    torch.set_num_threads(threads_per_job)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        # PyTorch permits setting this only before inter-op work starts. A fresh
        # spawned process normally takes the first branch; keep initialization
        # robust if an imported dependency happened to initialize it earlier.
        pass


def _run_parallel_jobs(
    jobs: list[RetrainingJob],
    *,
    n_jobs: int,
) -> Iterable[tuple[str, dict[str, Any]]]:
    if not jobs:
        return
    threads_per_job = jobs[0].threads_per_job or 1
    context = multiprocessing.get_context("spawn")
    fold_jobs: dict[str, list[RetrainingJob]] = {}
    for job in jobs:
        fold_jobs.setdefault(job.fold_name, []).append(job)
    lane_queues = [deque() for _ in range(n_jobs)]
    for fold_index, grouped_jobs in enumerate(fold_jobs.values()):
        lane_queues[fold_index % n_jobs].extend(grouped_jobs)

    with ExitStack() as stack:
        executors = [
            stack.enter_context(
                ProcessPoolExecutor(
                    max_workers=1,
                    mp_context=context,
                    initializer=_initialize_retraining_worker,
                    initargs=(threads_per_job,),
                )
            )
            for _ in range(n_jobs)
        ]
        futures: dict[Any, tuple[int, RetrainingJob]] = {}
        for lane_index, (executor, queue) in enumerate(
            zip(executors, lane_queues, strict=True)
        ):
            if queue:
                job = queue.popleft()
                futures[executor.submit(_run_retraining_job, job)] = (
                    lane_index,
                    job,
                )

        while futures:
            future = next(as_completed(tuple(futures)))
            lane_index, job = futures.pop(future)
            try:
                yield future.result()
            except Exception as error:
                raise RuntimeError(
                    "Subset retraining worker exited while "
                    f"{job.key!r} was pending. Check Slurm MaxRSS/exit signal and "
                    "GPU memory; reduce --n-jobs if the device is oversubscribed."
                ) from error
            queue = lane_queues[lane_index]
            if queue:
                next_job = queue.popleft()
                next_future = executors[lane_index].submit(
                    _run_retraining_job,
                    next_job,
                )
                futures[next_future] = (lane_index, next_job)


def _available_cpu_count() -> int:
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except AttributeError:
        return max(1, os.cpu_count() or 1)


def _effective_threads_per_job(
    requested: int | None,
    *,
    worker_count: int,
) -> int:
    if requested is not None:
        if requested < 1:
            raise ValueError("threads_per_job must be at least 1")
        return requested
    return max(1, _available_cpu_count() // max(1, worker_count))


def _release_parent_cuda_cache(device: str | None) -> None:
    """Do not let completed post-hoc inference reserve worker GPU memory."""

    if device is None or not device.lower().startswith("cuda"):
        return
    gc.collect()
    if torch.cuda.is_initialized():
        torch.cuda.empty_cache()


def _pooled_prediction_metrics(paths: list[Path]) -> dict[str, float]:
    rows = [
        json.loads(line)
        for path in paths
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    labels = np.asarray([int(row["label"]) for row in rows])
    logits = np.asarray([float(row["logit"]) for row in rows])
    thresholds = np.asarray([float(row.get("threshold", 0.5)) for row in rows])
    return binary_metrics(labels, logits, thresholds)


def _write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def retrain_cross_validation(
    checkpoint_paths: list[str | Path],
    output_dir: str | Path,
    *,
    device: str | None = None,
    resume: bool = True,
    seed: int = 42,
    bootstrap_samples: int = 2_000,
    variant_types: Iterable[str] | None = None,
    n_jobs: int = 1,
    threads_per_job: int | None = None,
    tabpfn_batch_groups: bool | None = None,
    tabpfn_show_progress_bar: bool | None = None,
) -> RetrainingResult:
    """Retrain selected sufficiency/unique-value variants on the original folds."""

    paths = [Path(path) for path in checkpoint_paths]
    if len(paths) < 2:
        raise ValueError(
            "Cross-validation retraining requires at least two checkpoints"
        )
    if n_jobs < 1:
        raise ValueError("n_jobs must be at least 1")
    selected_variant_types = _normalize_variant_types(variant_types)
    variant_specs = _variant_specs(selected_variant_types)
    payloads = [_load_payload(path) for path in paths]
    checkpoint_types = {_checkpoint_type(payload) for payload in payloads}
    if len(checkpoint_types) != 1:
        raise ValueError("Retraining checkpoints must all use the same classifier type")
    checkpoint_type = next(iter(checkpoint_types))
    config_cls, trainer = _resolve_trainer(checkpoint_type)
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "retraining_manifest.json"
    manifest: dict[str, Any] = (
        json.loads(manifest_path.read_text(encoding="utf-8"))
        if resume and manifest_path.is_file()
        else {
            "retraining_version": 1,
            "checkpoint_type": checkpoint_type,
            "source_checkpoints": [str(path.resolve()) for path in paths],
            "inference_overrides": {
                "tabpfn_batch_groups": tabpfn_batch_groups,
            },
            "runs": {},
        }
    )
    if manifest.get("checkpoint_type") != checkpoint_type:
        raise ValueError("Existing retraining manifest belongs to another classifier")
    if manifest.get("source_checkpoints") != [str(path.resolve()) for path in paths]:
        raise ValueError(
            "Existing retraining manifest uses different source checkpoints"
        )
    requested_inference_overrides = {
        "tabpfn_batch_groups": tabpfn_batch_groups,
    }
    saved_inference_overrides = manifest.get("inference_overrides")
    if saved_inference_overrides is None:
        # Version-1 manifests created before runtime overrides always preserved the
        # checkpoint's original sequential group behavior.
        saved_inference_overrides = {"tabpfn_batch_groups": None}
    if (
        manifest.get("runs")
        and saved_inference_overrides != requested_inference_overrides
    ):
        raise ValueError(
            "Existing retraining runs use different TabPFN group-batching "
            "semantics; keep the original setting or rerun with --no-resume"
        )
    manifest["inference_overrides"] = requested_inference_overrides

    pending_jobs: list[RetrainingJob] = []
    for source_path, payload in zip(paths, payloads, strict=True):
        fold_name = source_path.parent.name
        for spec in variant_specs:
            key = f"{fold_name}/{spec.variant_name}"
            existing = manifest["runs"].get(key)
            if (
                resume
                and existing
                and existing.get("status") == "complete"
                and existing.get("variant_type") == spec.variant_type
                and existing.get("modality") == spec.modality
                and tuple(existing.get("included_modalities", ()))
                == spec.included_modalities
                and tuple(existing.get("masked_channels", ()))
                == spec.masked_channels
                and Path(existing["checkpoint_path"]).is_file()
                and Path(existing["predictions_path"]).is_file()
            ):
                continue
            pending_jobs.append(
                RetrainingJob(
                    key=key,
                    checkpoint_type=checkpoint_type,
                    fold_name=fold_name,
                    variant_name=spec.variant_name,
                    variant_type=spec.variant_type,
                    modality=spec.modality,
                    included_modalities=spec.included_modalities,
                    masked_channels=spec.masked_channels,
                    experiment_config=payload["experiment_config"],
                    output_dir=root,
                    device=device,
                    threads_per_job=None,
                    tabpfn_batch_groups=tabpfn_batch_groups,
                    tabpfn_show_progress_bar=tabpfn_show_progress_bar,
                )
            )

    pending_fold_count = len({job.fold_name for job in pending_jobs})
    worker_count = min(n_jobs, pending_fold_count) if pending_jobs else 1
    effective_threads = _effective_threads_per_job(
        threads_per_job,
        worker_count=worker_count,
    )
    pending_jobs = [
        replace(job, threads_per_job=effective_threads) for job in pending_jobs
    ]
    if pending_jobs:
        print(
            "Subset retraining resources: "
            f"workers={worker_count}, threads/job={effective_threads}, "
            f"available_cpus={_available_cpu_count()}, "
            f"start_method={'spawn' if worker_count > 1 else 'in-process'}.",
            flush=True,
        )

    if pending_jobs and n_jobs == 1:
        for job in pending_jobs:
            key, entry = _run_retraining_job_with_trainer(job, config_cls, trainer)
            manifest["runs"][key] = entry
            _write_manifest(manifest_path, manifest)
    elif pending_jobs:
        _release_parent_cuda_cache(device)
        for key, entry in _run_parallel_jobs(
            pending_jobs,
            n_jobs=worker_count,
        ):
            manifest["runs"][key] = entry
            _write_manifest(manifest_path, manifest)

    baseline_paths = [path.parent / "test_predictions.jsonl" for path in paths]
    missing_baselines = [path for path in baseline_paths if not path.is_file()]
    if missing_baselines:
        raise FileNotFoundError(
            "Baseline prediction files are required for paired reporting: "
            + ", ".join(str(path) for path in missing_baselines)
        )
    baseline_metrics = _pooled_prediction_metrics(baseline_paths)
    result_rows: list[dict[str, Any]] = []
    for variant_index, spec in enumerate(variant_specs):
        run_entries = [
            manifest["runs"][f"{path.parent.name}/{spec.variant_name}"]
            for path in paths
        ]
        prediction_paths = [Path(entry["predictions_path"]) for entry in run_entries]
        pooled = _pooled_prediction_metrics(prediction_paths)
        fold_macro = [float(entry["test_metrics"]["macro_f1"]) for entry in run_entries]
        baseline_fold_macro = [
            float(payload["test_metrics"]["macro_f1"]) for payload in payloads
        ]
        fold_changes = np.asarray(baseline_fold_macro) - np.asarray(fold_macro)
        ci_lower, ci_upper = bootstrap_mean_interval(
            fold_changes,
            seed=seed + variant_index,
            samples=bootstrap_samples,
        )
        row: dict[str, Any] = {
            "variant": spec.variant_name,
            "variant_type": spec.variant_type,
            "modality": spec.modality,
            "included_modalities": ";".join(spec.included_modalities),
            "masked_channels": ";".join(spec.masked_channels),
            "num_folds": len(paths),
            "fold_mean_macro_f1": float(np.mean(fold_macro)),
            "fold_mean_macro_f1_change_from_full": float(
                np.mean(fold_changes)
            ),
            "fold_macro_f1_change_ci_lower": ci_lower,
            "fold_macro_f1_change_ci_upper": ci_upper,
        }
        for metric in METRIC_NAMES:
            row[f"pooled_{metric}"] = pooled[metric]
            row[f"pooled_{metric}_change_from_full"] = (
                pooled[metric] - baseline_metrics[metric]
                if metric == "loss"
                else baseline_metrics[metric] - pooled[metric]
            )
        result_rows.append(row)

    results_path = root / "retraining_results.csv"
    with results_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(result_rows[0]))
        writer.writeheader()
        writer.writerows(result_rows)
    summary = {
        "baseline_pooled_metrics": baseline_metrics,
        "variant_types": list(selected_variant_types),
        "n_jobs": n_jobs,
        "threads_per_job": effective_threads,
        "parallel_start_method": "spawn" if worker_count > 1 else None,
        "fold_affine_workers": worker_count > 1,
        "reused_prepared_fold_data": checkpoint_type in ROCKET_PFN_CHECKPOINT_TYPES,
        "tabpfn_batch_groups_override": tabpfn_batch_groups,
        "interpretation": {
            "modality_only": "Predictive sufficiency of each modality by itself.",
            "leave_one_out": (
                "Unique value lost when one modality is omitted during fitting."
            ),
        },
    }
    if "modality_only" in selected_variant_types:
        summary["modality_only_ranking"] = sorted(
            (row for row in result_rows if row["variant_type"] == "modality_only"),
            key=lambda row: row["pooled_macro_f1"],
            reverse=True,
        )
    if "leave_one_out" in selected_variant_types:
        summary["leave_one_out_ranking"] = sorted(
            (row for row in result_rows if row["variant_type"] == "leave_one_out"),
            key=lambda row: row["pooled_macro_f1_change_from_full"],
            reverse=True,
        )
    summary_path = root / "retraining_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return RetrainingResult(manifest_path, results_path, summary_path, summary)
