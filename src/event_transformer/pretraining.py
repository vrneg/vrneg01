"""Self-supervised pretraining for the event-transformer backbone.

The objective masks complete modality/time slots and reconstructs their normalized
continuous feature vectors.  Decoder heads are temporary: only the pretrained
``EventTransformer`` backbone is retained for supervised fine-tuning.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
from torch.nn import functional as functional
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm

from .config import PretrainingConfig
from .model import EventTransformer


ProgressCallback = Callable[[dict[str, Any]], None]


@dataclass(slots=True)
class PretrainingOutput:
    """Training history for one masked-modality pretraining phase."""

    history: list[dict[str, float | int]]


class MaskedModalityReconstructor(nn.Module):
    """Temporary reconstruction heads around a shared event-transformer backbone."""

    def __init__(self, backbone: EventTransformer) -> None:
        super().__init__()
        self.backbone = backbone
        hidden_dimension = backbone.config.modality_hidden_dim
        self.decoders = nn.ModuleDict(
            {
                str(modality_id): nn.Sequential(
                    nn.Linear(backbone.d_model, hidden_dimension),
                    nn.GELU(),
                    nn.Linear(hidden_dimension, input_dimension),
                )
                for modality_id, input_dimension in backbone.modality_dims.items()
            }
        )

    def forward(
        self,
        batch: Mapping[str, Any],
        modality_masks: Mapping[int, torch.Tensor],
    ) -> torch.Tensor:
        encoded = self.backbone.encode_sequence(
            modality_features=batch["modality_features"],
            finger_flags=batch["finger_flags"],
            modality_token_indices=batch["modality_token_indices"],
            modality_actor_relation_ids=batch["modality_actor_relation_ids"],
            time_features=batch["time_features"],
            event_mask=batch["event_mask"],
            anchor_mask=batch["anchor_mask"],
            padding_mask=batch["padding_mask"],
            modality_masks=modality_masks,
        )
        # Drop CLS and normalize temporal outputs with the same final normalization
        # that is later used by the classification representation.
        temporal_representations = self.backbone.final_norm(encoded[:, 1:])
        flat_representations = temporal_representations.reshape(
            -1, self.backbone.d_model
        )

        modality_losses: list[torch.Tensor] = []
        for modality_id in self.backbone.modality_dims:
            mask = modality_masks[modality_id]
            if not torch.any(mask):
                continue
            token_indices = batch["modality_token_indices"][modality_id][mask]
            targets = batch["modality_features"][modality_id][mask]
            predictions = self.decoders[str(modality_id)](
                flat_representations[token_indices]
            )
            modality_losses.append(functional.mse_loss(predictions, targets))

        if not modality_losses:
            raise ValueError("A pretraining batch must contain at least one masked event")
        # Equal modality weighting prevents the 226-dimensional finger target from
        # overwhelming smaller modalities merely because it has more dimensions.
        return torch.stack(modality_losses).mean()


def _move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        "modality_features": {
            modality_id: features.to(device, non_blocking=True)
            for modality_id, features in batch["modality_features"].items()
        },
        "finger_flags": {
            modality_id: flags.to(device, non_blocking=True)
            for modality_id, flags in batch["finger_flags"].items()
        },
        "modality_token_indices": {
            modality_id: indices.to(device, non_blocking=True)
            for modality_id, indices in batch["modality_token_indices"].items()
        },
        "modality_actor_relation_ids": {
            modality_id: actor_ids.to(device, non_blocking=True)
            for modality_id, actor_ids in batch[
                "modality_actor_relation_ids"
            ].items()
        },
        "time_features": batch["time_features"].to(device, non_blocking=True),
        "event_mask": batch["event_mask"].to(device, non_blocking=True),
        "anchor_mask": batch["anchor_mask"].to(device, non_blocking=True),
        "padding_mask": batch["padding_mask"].to(device, non_blocking=True),
    }


def sample_modality_masks(
    modality_token_indices: Mapping[int, torch.Tensor],
    mask_probability: float,
) -> dict[int, torch.Tensor]:
    """Sample one decision per modality/time slot, never per duplicate record."""

    masks: dict[int, torch.Tensor] = {}
    available_groups: list[tuple[int, torch.Tensor]] = []
    has_masked_group = False
    for modality_id, token_indices in modality_token_indices.items():
        if not token_indices.numel():
            masks[modality_id] = torch.zeros_like(token_indices, dtype=torch.bool)
            continue
        unique_tokens, inverse_indices = torch.unique(
            token_indices, sorted=False, return_inverse=True
        )
        group_masks = torch.rand(
            unique_tokens.shape[0], device=token_indices.device
        ) < mask_probability
        masks[modality_id] = group_masks[inverse_indices]
        available_groups.append((modality_id, unique_tokens))
        has_masked_group = has_masked_group or bool(torch.any(group_masks).item())

    if not available_groups:
        raise ValueError("Cannot pretrain on a batch without modality observations")
    if not has_masked_group:
        # Small batches can randomly receive no mask. Select exactly one available
        # modality/time slot so every optimization step has a valid objective.
        group_counts = [int(tokens.shape[0]) for _, tokens in available_groups]
        selected = int(
            torch.randint(
                sum(group_counts),
                (1,),
                device=available_groups[0][1].device,
            ).item()
        )
        group_offset = 0
        for (modality_id, unique_tokens), group_count in zip(
            available_groups, group_counts, strict=True
        ):
            if selected < group_offset + group_count:
                selected_token = unique_tokens[selected - group_offset]
                masks[modality_id] = masks[modality_id] | (
                    modality_token_indices[modality_id] == selected_token
                )
                break
            group_offset += group_count
    return masks


def _pretrain_epoch(
    reconstructor: MaskedModalityReconstructor,
    loader: DataLoader,
    optimizer: AdamW,
    device: torch.device,
    config: PretrainingConfig,
    mixed_precision: bool,
    scaler: torch.amp.GradScaler,
) -> tuple[float, int]:
    reconstructor.train()
    use_amp = mixed_precision and device.type == "cuda"
    loss_sum = 0.0
    num_batches = 0
    num_masked_observations = 0

    for raw_batch in loader:
        batch = _move_batch(raw_batch, device)
        modality_masks = sample_modality_masks(
            batch["modality_token_indices"], config.mask_probability
        )
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device.type, dtype=torch.float16, enabled=use_amp
        ):
            loss = reconstructor(batch, modality_masks)

        scaler.scale(loss).backward()
        if config.gradient_clip_norm is not None:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(
                reconstructor.parameters(), config.gradient_clip_norm
            )
        scaler.step(optimizer)
        scaler.update()

        loss_sum += float(loss.item())
        num_batches += 1
        num_masked_observations += sum(
            int(mask.sum().item()) for mask in modality_masks.values()
        )

    if not num_batches:
        raise ValueError("Cannot pretrain on an empty dataset")
    return loss_sum / num_batches, num_masked_observations


def pretrain_event_transformer(
    model: EventTransformer,
    train_loader: DataLoader,
    config: PretrainingConfig,
    device: torch.device,
    *,
    mixed_precision: bool,
    show_progress: bool,
    progress_callback: ProgressCallback | None = None,
) -> PretrainingOutput:
    """Pretrain ``model`` in place using only the supplied training loader."""

    config.validate()
    reconstructor = MaskedModalityReconstructor(model).to(device)
    optimizer = AdamW(
        reconstructor.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=config.lr_scheduler_factor,
        patience=config.lr_scheduler_patience,
        min_lr=config.minimum_learning_rate,
    )
    use_amp = mixed_precision and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    history: list[dict[str, float | int]] = []

    progress = tqdm(
        range(1, config.num_epochs + 1),
        desc="masked-modality pretraining",
        disable=not show_progress,
    )
    for epoch in progress:
        learning_rate = float(optimizer.param_groups[0]["lr"])
        reconstruction_loss, num_masked = _pretrain_epoch(
            reconstructor=reconstructor,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            config=config,
            mixed_precision=mixed_precision,
            scaler=scaler,
        )
        scheduler.step(reconstruction_loss)
        epoch_metrics: dict[str, float | int] = {
            "epoch": epoch,
            "learning_rate": learning_rate,
            "reconstruction_loss": reconstruction_loss,
            "num_masked_observations": num_masked,
        }
        history.append(epoch_metrics)
        progress.set_postfix(
            reconstruction_loss=f"{reconstruction_loss:.4f}",
            masked=num_masked,
        )
        if progress_callback is not None:
            progress_callback(
                {
                    "event": "pretraining_epoch_complete",
                    "epoch": epoch,
                    "max_epochs": config.num_epochs,
                    "reconstruction_loss": reconstruction_loss,
                    "num_masked_observations": num_masked,
                }
            )

    if progress_callback is not None:
        progress_callback(
            {
                "event": "pretraining_complete",
                "epochs_completed": len(history),
            }
        )
    return PretrainingOutput(history=history)
