"""Three-stage training for the GHTT-derived cascade: pose block, then action block.

Reuses the same generic helpers as ``t2m_gpt_v2.training`` (seeding, device resolution,
dict/JSON plumbing, prediction-row formatting, ``TrainingResult``/``EvaluationOutput``)
directly from ``t2m_gpt.training``, plus ``kl_divergence_loss`` from ``t2m_gpt_v2.vae``.
The pose stage has two modes (see ``ghtt.config.ExperimentConfig.pose_block_mode``):
``"vae"`` trains the native ``PoseBlock``; ``"vqvae"`` trains this project's existing
``MotionVQVAE`` per clip instead, as the discrete ablation. Either way, the action stage
is identical: it only ever sees a sequence of per-clip mid-level vectors, however they
were produced.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional as functional
from torch.utils.data import DataLoader

try:
    from event_transformer.metrics import (
        binary_classification_metrics,
        optimize_binary_threshold,
    )
    from t2m_gpt.vqvae import MotionVQVAE, reconstruction_losses
    from t2m_gpt.training import (
        CHECKPOINT_VERSION,
        SPLIT_NAMES,
        EvaluationOutput,
        ProgressCallback,
        TrainingResult,
        _config_signature,
        _detached_state,
        _initial_best_value,
        _is_improvement,
        _json_ready,
        _load_checkpoint,
        _prediction_rows,
        _report,
        _weighted_mean,
        _write_lines,
        resolve_device,
        set_reproducible_seed,
    )
    from t2m_gpt_v2.vae import kl_divergence_loss
except ModuleNotFoundError as error:  # pragma: no cover - import-path fallback
    if error.name not in {"event_transformer", "t2m_gpt", "t2m_gpt_v2"}:
        raise
    from ..event_transformer.metrics import (
        binary_classification_metrics,
        optimize_binary_threshold,
    )
    from ..t2m_gpt.vqvae import MotionVQVAE, reconstruction_losses
    from ..t2m_gpt.training import (
        CHECKPOINT_VERSION,
        SPLIT_NAMES,
        EvaluationOutput,
        ProgressCallback,
        TrainingResult,
        _config_signature,
        _detached_state,
        _initial_best_value,
        _is_improvement,
        _json_ready,
        _load_checkpoint,
        _prediction_rows,
        _report,
        _weighted_mean,
        _write_lines,
        resolve_device,
        set_reproducible_seed,
    )
    from ..t2m_gpt_v2.vae import kl_divergence_loss

from .config import ActionBlockConfig, ExperimentConfig, PoseBlockConfig, PoseTrainingConfig
from .data import (
    ClipDataset,
    DataBundle,
    FixedGridSplit,
    MotionLatentDataset,
    flatten_clips_for_pose_training,
    make_loader,
    prepare_data,
)
from .model import ActionBlock, PoseBlock, count_parameters

# The paper does not specify a commitment-loss weight for a per-clip VQ-VAE (it is not
# part of GHTT at all -- this is this project's own discrete ablation), so the discrete
# pose stage reuses T2M-GPT's own default rather than introducing an unvalidated new one.
_POSE_VQVAE_COMMITMENT_LOSS_WEIGHT = 0.02
_POSE_VQVAE_VELOCITY_LOSS_WEIGHT = 0.5


def _clip_length(config: ExperimentConfig) -> int:
    if config.pose_block_mode == "vae":
        return config.pose_block.clip_length
    return config.pose_vqvae.temporal_downsample_factor


# -----------------------------------------------------------------------------
# Pose stage: "vae" mode (native)
# -----------------------------------------------------------------------------


def _pose_batch_losses(
    model: PoseBlock,
    clip: torch.Tensor,
    pose_config: PoseBlockConfig,
    training_config: PoseTrainingConfig,
) -> tuple[torch.Tensor, dict[str, float]]:
    output = model(clip)
    # reconstruction_losses expects [batch, channels, length] (it diffs the last axis
    # for the velocity term); the pose block's tensors are [batch, length, channels].
    component_loss, component_velocity = reconstruction_losses(
        output.reconstruction.transpose(1, 2),
        output.context.transpose(1, 2),
        training_config.reconstruction_loss,
    )
    trajectory_loss, trajectory_velocity = reconstruction_losses(
        output.trajectory_prediction.transpose(1, 2),
        output.target.transpose(1, 2),
        training_config.reconstruction_loss,
    )
    kl = kl_divergence_loss(output.mean, output.log_variance, pose_config.free_bits)
    total = (
        pose_config.component_loss_weight * (component_loss + 0.5 * component_velocity)
        + pose_config.trajectory_loss_weight * (trajectory_loss + 0.5 * trajectory_velocity)
        + pose_config.kl_weight * kl
    )
    statistics = {
        "loss": float(total.item()),
        "component_loss": float(component_loss.item()),
        "trajectory_loss": float(trajectory_loss.item()),
        "kl_divergence": float(kl.item()),
    }
    return total, statistics


@torch.no_grad()
def evaluate_pose_block(
    model: PoseBlock,
    loader: DataLoader,
    pose_config: PoseBlockConfig,
    training_config: PoseTrainingConfig,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    accumulated: dict[str, list[tuple[float, int]]] = {}
    for clip in loader:
        clip = clip.to(device, non_blocking=True)
        _, statistics = _pose_batch_losses(model, clip, pose_config, training_config)
        weight = int(clip.shape[0])
        for key, value in statistics.items():
            accumulated.setdefault(key, []).append((value, weight))
    return {key: _weighted_mean(value) for key, value in accumulated.items()}


def train_pose_block(
    data: DataBundle,
    config: ExperimentConfig,
    device: torch.device,
    progress_callback: ProgressCallback | None = None,
) -> tuple[PoseBlock, list[dict[str, Any]], dict[str, float]]:
    pose_config = config.pose_block
    training_config = config.pose_training
    model = PoseBlock(data.num_channels, pose_config).to(device)
    pin_memory = device.type == "cuda"

    train_loader = make_loader(
        flatten_clips_for_pose_training(data.train, pose_config.clip_length),
        batch_size=training_config.batch_size,
        shuffle=True,
        seed=config.seed,
        pin_memory=pin_memory,
    )
    validation_loader = make_loader(
        flatten_clips_for_pose_training(data.validation, pose_config.clip_length),
        batch_size=training_config.evaluation_batch_size,
        shuffle=False,
        seed=config.seed,
        pin_memory=pin_memory,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=training_config.learning_rate, weight_decay=training_config.weight_decay
    )
    scheduler = (
        torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=training_config.lr_scheduler_factor,
            patience=training_config.lr_scheduler_patience,
            min_lr=training_config.minimum_learning_rate,
        )
        if training_config.lr_scheduler == "reduce_on_plateau"
        else None
    )

    history: list[dict[str, Any]] = []
    best_value = math.inf
    best_state = _detached_state(model)
    best_epoch = 0
    best_metrics: dict[str, float] = {}
    epochs_without_improvement = 0

    for epoch in range(1, training_config.max_epochs + 1):
        model.train()
        batch_statistics: dict[str, list[tuple[float, int]]] = {}
        for clip in train_loader:
            clip = clip.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            total, statistics = _pose_batch_losses(model, clip, pose_config, training_config)
            total.backward()
            if training_config.gradient_clip_norm is not None:
                nn.utils.clip_grad_norm_(model.parameters(), training_config.gradient_clip_norm)
            optimizer.step()
            weight = int(clip.shape[0])
            for key, value in statistics.items():
                batch_statistics.setdefault(key, []).append((value, weight))

        train_metrics = {key: _weighted_mean(value) for key, value in batch_statistics.items()}
        validation_metrics = evaluate_pose_block(
            model, validation_loader, pose_config, training_config, device
        )
        if scheduler is not None:
            scheduler.step(validation_metrics["loss"])

        record = {
            "stage": "pose",
            "epoch": epoch,
            **{f"train_{key}": value for key, value in train_metrics.items()},
            **{f"validation_{key}": value for key, value in validation_metrics.items()},
        }
        if _is_improvement(
            "loss", validation_metrics["loss"], best_value, training_config.early_stopping_min_delta
        ):
            best_value = validation_metrics["loss"]
            best_state = _detached_state(model)
            best_epoch = epoch
            best_metrics = dict(validation_metrics)
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        record["best_epoch"] = best_epoch
        history.append(record)
        _report(
            progress_callback,
            config.training.show_progress,
            record,
            f"[pose] epoch {epoch:3d}/{training_config.max_epochs} "
            f"train_loss={train_metrics['loss']:.4f} val_loss={validation_metrics['loss']:.4f} "
            f"best={best_epoch}",
        )
        if epochs_without_improvement >= training_config.early_stopping_patience:
            break

    if config.training.show_progress and progress_callback is None:
        print()
    model.load_state_dict(best_state)
    best_metrics["best_epoch"] = float(best_epoch)
    return model, history, best_metrics


# -----------------------------------------------------------------------------
# Pose stage: "vqvae" mode (discrete ablation)
# -----------------------------------------------------------------------------


@torch.no_grad()
def _evaluate_pose_vqvae(model: MotionVQVAE, loader: DataLoader, training_config: PoseTrainingConfig, device: torch.device) -> dict[str, float]:
    model.eval()
    accumulated: dict[str, list[tuple[float, int]]] = {}
    for clip in loader:
        clip = clip.to(device, non_blocking=True)
        output = model(clip)
        frame_loss, velocity_loss = reconstruction_losses(
            output.reconstruction, clip, training_config.reconstruction_loss
        )
        total = (
            frame_loss
            + _POSE_VQVAE_VELOCITY_LOSS_WEIGHT * velocity_loss
            + _POSE_VQVAE_COMMITMENT_LOSS_WEIGHT * output.commitment_loss
        )
        weight = int(clip.shape[0])
        statistics = {
            "loss": float(total.item()),
            "frame_loss": float(frame_loss.item()),
            "commitment_loss": float(output.commitment_loss.item()),
        }
        for key, value in statistics.items():
            accumulated.setdefault(key, []).append((value, weight))
    return {key: _weighted_mean(value) for key, value in accumulated.items()}


def train_pose_vqvae(
    data: DataBundle,
    config: ExperimentConfig,
    device: torch.device,
    progress_callback: ProgressCallback | None = None,
) -> tuple[MotionVQVAE, list[dict[str, Any]], dict[str, float]]:
    vqvae_config = config.pose_vqvae
    training_config = config.pose_training
    clip_length = vqvae_config.temporal_downsample_factor
    model = MotionVQVAE(data.num_channels, vqvae_config).to(device)
    pin_memory = device.type == "cuda"

    # MotionVQVAE expects channel-first [batch, channels, length]; ClipDataset stores
    # whatever it is given, so the flat clip pool is transposed once here.
    train_dataset = ClipDataset(
        flatten_clips_for_pose_training(data.train, clip_length).clips.transpose(1, 2)
    )
    validation_dataset = ClipDataset(
        flatten_clips_for_pose_training(data.validation, clip_length).clips.transpose(1, 2)
    )
    train_loader = make_loader(
        train_dataset, batch_size=training_config.batch_size, shuffle=True, seed=config.seed, pin_memory=pin_memory
    )
    validation_loader = make_loader(
        validation_dataset,
        batch_size=training_config.evaluation_batch_size,
        shuffle=False,
        seed=config.seed,
        pin_memory=pin_memory,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=training_config.learning_rate, weight_decay=training_config.weight_decay
    )
    scheduler = (
        torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=training_config.lr_scheduler_factor,
            patience=training_config.lr_scheduler_patience,
            min_lr=training_config.minimum_learning_rate,
        )
        if training_config.lr_scheduler == "reduce_on_plateau"
        else None
    )

    history: list[dict[str, Any]] = []
    best_value = math.inf
    best_state = _detached_state(model)
    best_epoch = 0
    best_metrics: dict[str, float] = {}
    epochs_without_improvement = 0

    for epoch in range(1, training_config.max_epochs + 1):
        model.train()
        batch_statistics: dict[str, list[tuple[float, int]]] = {}
        for clip in train_loader:
            clip = clip.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            output = model(clip)
            frame_loss, velocity_loss = reconstruction_losses(
                output.reconstruction, clip, training_config.reconstruction_loss
            )
            total = (
                frame_loss
                + _POSE_VQVAE_VELOCITY_LOSS_WEIGHT * velocity_loss
                + _POSE_VQVAE_COMMITMENT_LOSS_WEIGHT * output.commitment_loss
            )
            total.backward()
            if training_config.gradient_clip_norm is not None:
                nn.utils.clip_grad_norm_(model.parameters(), training_config.gradient_clip_norm)
            optimizer.step()
            weight = int(clip.shape[0])
            statistics = {
                "loss": float(total.item()),
                "frame_loss": float(frame_loss.item()),
                "commitment_loss": float(output.commitment_loss.item()),
            }
            for key, value in statistics.items():
                batch_statistics.setdefault(key, []).append((value, weight))

        train_metrics = {key: _weighted_mean(value) for key, value in batch_statistics.items()}
        validation_metrics = _evaluate_pose_vqvae(model, validation_loader, training_config, device)
        if scheduler is not None:
            scheduler.step(validation_metrics["loss"])

        record = {
            "stage": "pose_vqvae",
            "epoch": epoch,
            **{f"train_{key}": value for key, value in train_metrics.items()},
            **{f"validation_{key}": value for key, value in validation_metrics.items()},
        }
        if _is_improvement(
            "loss", validation_metrics["loss"], best_value, training_config.early_stopping_min_delta
        ):
            best_value = validation_metrics["loss"]
            best_state = _detached_state(model)
            best_epoch = epoch
            best_metrics = dict(validation_metrics)
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        record["best_epoch"] = best_epoch
        history.append(record)
        _report(
            progress_callback,
            config.training.show_progress,
            record,
            f"[pose-vqvae] epoch {epoch:3d}/{training_config.max_epochs} "
            f"train_loss={train_metrics['loss']:.4f} val_loss={validation_metrics['loss']:.4f} "
            f"best={best_epoch}",
        )
        if epochs_without_improvement >= training_config.early_stopping_patience:
            break

    if config.training.show_progress and progress_callback is None:
        print()
    model.load_state_dict(best_state)
    best_metrics["best_epoch"] = float(best_epoch)
    return model, history, best_metrics


def train_pose_stage(
    data: DataBundle,
    config: ExperimentConfig,
    device: torch.device,
    progress_callback: ProgressCallback | None = None,
) -> tuple[nn.Module, list[dict[str, Any]], dict[str, float]]:
    if config.pose_block_mode == "vae":
        return train_pose_block(data, config, device, progress_callback)
    return train_pose_vqvae(data, config, device, progress_callback)


# -----------------------------------------------------------------------------
# Mid-level sequence assembly (bridges pose stage output to action stage input)
# -----------------------------------------------------------------------------


@torch.no_grad()
def assemble_mid_level_sequences(
    pose_model: nn.Module,
    split: FixedGridSplit,
    config: ExperimentConfig,
    device: torch.device,
) -> MotionLatentDataset:
    from .data import split_into_clips

    clip_length = _clip_length(config)
    clips = split_into_clips(split.values, clip_length)
    windows, n_clips, actual_clip_length, channels = clips.shape
    flat = torch.from_numpy(clips.reshape(windows * n_clips, actual_clip_length, channels)).float()

    pose_model.eval()
    batch_size = config.pose_training.evaluation_batch_size
    batches: list[torch.Tensor] = []

    if config.pose_block_mode == "vae":
        for start in range(0, flat.shape[0], batch_size):
            batches.append(pose_model.encode_latent(flat[start : start + batch_size].to(device)).cpu())
        mid_dim = config.pose_block.latent_dim
    else:
        channel_first = flat.transpose(1, 2)
        for start in range(0, channel_first.shape[0], batch_size):
            chunk = channel_first[start : start + batch_size].to(device)
            indices = pose_model.encode_indices(chunk)
            embedded = pose_model.quantizer.lookup(indices).squeeze(-1)
            batches.append(embedded.cpu())
        mid_dim = config.pose_vqvae.code_dim

    mid = torch.cat(batches, dim=0) if batches else torch.empty((0, mid_dim))
    sequences = mid.reshape(windows, n_clips, mid_dim)
    return MotionLatentDataset(
        latents=sequences, labels=torch.from_numpy(split.labels).float(), sample_ids=split.sample_ids
    )


# -----------------------------------------------------------------------------
# Action stage
# -----------------------------------------------------------------------------


def _action_batch_losses(
    model: ActionBlock,
    mid_sequence: torch.Tensor,
    labels: torch.Tensor,
    config: ActionBlockConfig,
) -> tuple[torch.Tensor, dict[str, float], torch.Tensor]:
    output = model(mid_sequence)
    reconstruction_loss, _ = reconstruction_losses(
        output.reconstruction.transpose(1, 2), mid_sequence.transpose(1, 2), "smooth_l1"
    )
    kl = kl_divergence_loss(output.mean, output.log_variance, config.free_bits)
    classification_loss = functional.binary_cross_entropy_with_logits(output.logit, labels)
    total = (
        config.mid_reconstruction_weight * reconstruction_loss
        + config.classification_weight * classification_loss
        + config.kl_weight * kl
    )
    statistics = {
        "loss": float(total.item()),
        "reconstruction_loss": float(reconstruction_loss.item()),
        "classification_loss": float(classification_loss.item()),
        "kl_divergence": float(kl.item()),
    }
    return total, statistics, output.logit


@torch.no_grad()
def evaluate_action_block(
    model: ActionBlock,
    loader: DataLoader,
    config: ActionBlockConfig,
    device: torch.device,
    threshold: float,
) -> EvaluationOutput:
    model.eval()
    all_scores: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []
    sample_ids: list[str] = []
    losses: list[tuple[float, int]] = []

    for batch in loader:
        mid_sequence = batch["latents"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)
        loss, _, logit = _action_batch_losses(model, mid_sequence, labels, config)
        losses.append((float(loss.item()), int(labels.shape[0])))
        all_scores.append(logit.detach().float().cpu())
        all_labels.append(labels.detach().float().cpu())
        sample_ids.extend(batch["sample_id"])

    scores = torch.cat(all_scores) if all_scores else torch.empty(0, dtype=torch.float32)
    labels_out = torch.cat(all_labels) if all_labels else torch.empty(0, dtype=torch.float32)
    loss = _weighted_mean(losses)
    metrics = binary_classification_metrics(scores, labels_out, loss, threshold)
    return EvaluationOutput(metrics=metrics, sample_ids=sample_ids, labels=labels_out, logits=scores)


def train_action_block(
    mid_level_datasets: dict[str, MotionLatentDataset],
    config: ExperimentConfig,
    device: torch.device,
    sequence_length: int,
    mid_dim: int,
    progress_callback: ProgressCallback | None = None,
) -> tuple[ActionBlock, list[dict[str, Any]], dict[str, torch.Tensor], int]:
    action_config = config.action_block
    training_config = config.training
    model = ActionBlock(mid_dim, action_config, sequence_length).to(device)
    pin_memory = device.type == "cuda"

    train_loader = make_loader(
        mid_level_datasets["train"],
        batch_size=training_config.batch_size,
        shuffle=True,
        seed=config.seed,
        num_workers=training_config.num_workers,
        pin_memory=pin_memory,
    )
    validation_loader = make_loader(
        mid_level_datasets["validation"],
        batch_size=training_config.evaluation_batch_size,
        shuffle=False,
        seed=config.seed,
        num_workers=training_config.num_workers,
        pin_memory=pin_memory,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=training_config.learning_rate, weight_decay=training_config.weight_decay
    )
    scheduler = (
        torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min" if training_config.selection_metric == "loss" else "max",
            factor=training_config.lr_scheduler_factor,
            patience=training_config.lr_scheduler_patience,
            min_lr=training_config.minimum_learning_rate,
        )
        if training_config.lr_scheduler == "reduce_on_plateau"
        else None
    )

    history: list[dict[str, Any]] = []
    best_value = _initial_best_value(training_config.selection_metric)
    best_state = _detached_state(model)
    best_epoch = 0
    epochs_without_improvement = 0

    for epoch in range(1, training_config.max_epochs + 1):
        model.train()
        train_losses: list[tuple[float, int]] = []
        for batch in train_loader:
            mid_sequence = batch["latents"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss, _, _ = _action_batch_losses(model, mid_sequence, labels, action_config)
            loss.backward()
            if training_config.gradient_clip_norm is not None:
                nn.utils.clip_grad_norm_(model.parameters(), training_config.gradient_clip_norm)
            optimizer.step()
            train_losses.append((float(loss.item()), int(labels.shape[0])))

        validation_output = evaluate_action_block(
            model, validation_loader, action_config, device, threshold=training_config.threshold
        )
        selection_value = validation_output.metrics[training_config.selection_metric]
        if scheduler is not None:
            scheduler.step(selection_value)

        record = {
            "stage": "action",
            "epoch": epoch,
            "train_loss": _weighted_mean(train_losses),
            **{f"validation_{key}": value for key, value in validation_output.metrics.items()},
        }
        if _is_improvement(
            training_config.selection_metric, selection_value, best_value, training_config.early_stopping_min_delta
        ):
            best_value = selection_value
            best_state = _detached_state(model)
            best_epoch = epoch
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        record["best_epoch"] = best_epoch
        history.append(record)
        _report(
            progress_callback,
            training_config.show_progress,
            record,
            f"[action] epoch {epoch:3d}/{training_config.max_epochs} "
            f"train_loss={record['train_loss']:.4f} "
            f"val_loss={validation_output.metrics['loss']:.4f} "
            f"val_auc={validation_output.metrics['roc_auc']:.4f} "
            f"best={best_epoch}",
        )
        if epochs_without_improvement >= training_config.early_stopping_patience:
            break

    if training_config.show_progress and progress_callback is None:
        print()
    return model, history, best_state, best_epoch


# -----------------------------------------------------------------------------
# Orchestration
# -----------------------------------------------------------------------------


def train_ghtt(
    config: ExperimentConfig,
    progress_callback: ProgressCallback | None = None,
) -> TrainingResult:
    """Run pose then action stages for one fold and evaluate the test split once."""

    config.validate()
    set_reproducible_seed(config.seed, config.training.deterministic_algorithms)
    device = resolve_device(config.device)

    run_dir = Path(config.output_dir) / config.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    pose_path = run_dir / "pose_block.pt"
    checkpoint_path = run_dir / "best_model.pt"
    history_path = run_dir / "metrics.json"
    validation_predictions_path = run_dir / "validation_predictions.jsonl"
    test_predictions_path = run_dir / "test_predictions.jsonl"

    data = prepare_data(config.data)

    pose_model, pose_history, pose_metrics = train_pose_stage(data, config, device, progress_callback)
    torch.save(
        {
            "checkpoint_version": CHECKPOINT_VERSION,
            "checkpoint_type": f"ghtt_pose_{config.pose_block_mode}",
            "model_state_dict": pose_model.state_dict(),
            "pose_block_mode": config.pose_block_mode,
            "input_dim": data.num_channels,
            "history": pose_history,
            "validation_metrics": pose_metrics,
        },
        pose_path,
    )

    mid_level_datasets = {
        split: assemble_mid_level_sequences(pose_model, getattr(data, split), config, device)
        for split in SPLIT_NAMES
    }
    sequence_length = mid_level_datasets["train"].num_tokens
    mid_dim = mid_level_datasets["train"].latent_dim

    set_reproducible_seed(config.seed, config.training.deterministic_algorithms)
    action_model, action_history, best_state, best_epoch = train_action_block(
        mid_level_datasets, config, device, sequence_length, mid_dim, progress_callback
    )
    action_model.load_state_dict(best_state)

    pin_memory = device.type == "cuda"
    validation_loader = make_loader(
        mid_level_datasets["validation"],
        batch_size=config.training.evaluation_batch_size,
        shuffle=False,
        seed=config.seed,
        num_workers=config.training.num_workers,
        pin_memory=pin_memory,
    )
    test_loader = make_loader(
        mid_level_datasets["test"],
        batch_size=config.training.evaluation_batch_size,
        shuffle=False,
        seed=config.seed,
        num_workers=config.training.num_workers,
        pin_memory=pin_memory,
    )

    validation_output = evaluate_action_block(
        action_model, validation_loader, config.action_block, device, threshold=config.training.threshold
    )
    decision_threshold = config.training.threshold
    if config.training.calibrate_threshold_on_validation:
        decision_threshold = optimize_binary_threshold(
            validation_output.logits, validation_output.labels, config.training.threshold_metric
        )
        validation_output = EvaluationOutput(
            metrics=binary_classification_metrics(
                validation_output.logits,
                validation_output.labels,
                validation_output.metrics["loss"],
                decision_threshold,
            ),
            sample_ids=validation_output.sample_ids,
            labels=validation_output.labels,
            logits=validation_output.logits,
        )
    test_output = evaluate_action_block(
        action_model, test_loader, config.action_block, device, threshold=decision_threshold
    )

    _write_lines(validation_predictions_path, _prediction_rows(validation_output, decision_threshold))
    _write_lines(test_predictions_path, _prediction_rows(test_output, decision_threshold))

    model_parameter_count = count_parameters(action_model, trainable_only=False) + count_parameters(
        pose_model, trainable_only=False
    )
    torch.save(
        {
            "checkpoint_version": CHECKPOINT_VERSION,
            "checkpoint_type": "ghtt_action_classifier",
            "model_state_dict": best_state,
            "pose_block_mode": config.pose_block_mode,
            "mid_dim": mid_dim,
            "sequence_length": sequence_length,
            "experiment_config": _config_signature(config),
            "label_convention": {config.data.negative_label: 0, config.data.positive_label: 1},
            "best_epoch": best_epoch,
            "decision_threshold": decision_threshold,
            "validation_metrics": validation_output.metrics,
            "test_metrics": test_output.metrics,
            "model_parameter_count": model_parameter_count,
            "history": action_history,
            "pose_history": pose_history,
            "pose_metrics": pose_metrics,
        },
        checkpoint_path,
    )

    history_path.write_text(
        json.dumps(
            _json_ready(
                {
                    "run_name": config.run_name,
                    "experiment_config": _config_signature(config),
                    "num_channels": data.num_channels,
                    "sequence_length": sequence_length,
                    "mid_dim": mid_dim,
                    "best_epoch": best_epoch,
                    "decision_threshold": decision_threshold,
                    "model_parameter_count": model_parameter_count,
                    "pose_validation_metrics": pose_metrics,
                    "validation_metrics": validation_output.metrics,
                    "test_metrics": test_output.metrics,
                    "pose_history": pose_history,
                    "history": action_history,
                }
            ),
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    return TrainingResult(
        checkpoint_path=checkpoint_path,
        history_path=history_path,
        validation_predictions_path=validation_predictions_path,
        predictions_path=test_predictions_path,
        run_name=config.run_name,
        best_epoch=best_epoch,
        decision_threshold=decision_threshold,
        validation_metrics=validation_output.metrics,
        test_metrics=test_output.metrics,
        model_parameter_count=model_parameter_count,
        history=action_history,
        tokenizer_checkpoint_path=pose_path,
        tokenizer_history=pose_history,
        tokenizer_metrics=pose_metrics,
    )


def load_training_result_ghtt(config: ExperimentConfig) -> TrainingResult:
    """Reload a finished fold, refusing checkpoints from a different configuration."""

    config.validate()
    run_dir = Path(config.output_dir) / config.run_name
    checkpoint = _load_checkpoint(run_dir / "best_model.pt")
    if checkpoint.get("experiment_config") != _config_signature(config):
        raise ValueError(f"Checkpoint in {run_dir} was produced by a different configuration")
    for required in ("validation_predictions.jsonl", "test_predictions.jsonl", "metrics.json"):
        if not (run_dir / required).exists():
            raise FileNotFoundError(f"Missing {required} in {run_dir}")

    return TrainingResult(
        checkpoint_path=run_dir / "best_model.pt",
        history_path=run_dir / "metrics.json",
        validation_predictions_path=run_dir / "validation_predictions.jsonl",
        predictions_path=run_dir / "test_predictions.jsonl",
        run_name=config.run_name,
        best_epoch=int(checkpoint["best_epoch"]),
        decision_threshold=float(checkpoint["decision_threshold"]),
        validation_metrics=dict(checkpoint["validation_metrics"]),
        test_metrics=dict(checkpoint["test_metrics"]),
        model_parameter_count=int(checkpoint["model_parameter_count"]),
        history=list(checkpoint["history"]),
        tokenizer_checkpoint_path=run_dir / "pose_block.pt",
        tokenizer_history=list(checkpoint.get("pose_history", [])),
        tokenizer_metrics=dict(checkpoint.get("pose_metrics", {})),
    )


__all__ = [
    "EvaluationOutput",
    "TrainingResult",
    "assemble_mid_level_sequences",
    "evaluate_action_block",
    "evaluate_pose_block",
    "load_training_result_ghtt",
    "resolve_device",
    "set_reproducible_seed",
    "train_action_block",
    "train_ghtt",
    "train_pose_block",
    "train_pose_stage",
    "train_pose_vqvae",
]
