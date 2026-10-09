"""Modality-aware Transformer for irregular VR event sequences."""

from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import nn

from .config import ModelConfig
from .data import TIME_FEATURE_DIM
from .features import (
    FINGER_FLAG_DIM,
    FINGER_MODALITY_IDS,
    MODALITY_DIMS,
    NUM_ACTOR_RELATIONS,
)


class MLPEncoder(nn.Module):
    """Project one modality's measurements into the shared token space."""

    def __init__(
        self,
        input_dim: int,
        d_model: int,
        hidden_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, d_model),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)


class TimeEncoder(nn.Module):
    """Encode relative time plus previous/next temporal-step gaps."""

    def __init__(self, d_model: int, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(TIME_FEATURE_DIM, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, d_model),
        )

    def forward(self, time_features: torch.Tensor) -> torch.Tensor:
        return self.net(time_features)


class EventTransformer(nn.Module):
    """Encode heterogeneous events and classify an anchored temporal window."""

    def __init__(
        self,
        config: ModelConfig,
        modality_dims: Mapping[int, int] = MODALITY_DIMS,
    ) -> None:
        super().__init__()
        config.validate()
        self.config = config
        self.d_model = config.d_model
        self.modality_dims = dict(modality_dims)
        self.num_modalities = len(self.modality_dims)
        if sorted(self.modality_dims) != list(range(self.num_modalities)):
            raise ValueError("Modality IDs must be contiguous and start at zero")

        self.modality_encoders = nn.ModuleDict(
            {
                str(modality_id): MLPEncoder(
                    input_dim=input_dim,
                    d_model=config.d_model,
                    hidden_dim=config.modality_hidden_dim,
                    dropout=config.dropout,
                )
                for modality_id, input_dim in self.modality_dims.items()
            }
        )
        self.modality_embedding = nn.Embedding(len(self.modality_dims), config.d_model)
        # One learned replacement vector per modality is used by self-supervised
        # pretraining. It replaces the complete continuous observation, rather than
        # leaking a partially masked raw vector into the encoder.
        self.modality_mask_embedding = nn.Embedding(
            len(self.modality_dims), config.d_model
        )
        self.actor_relation_embedding = nn.Embedding(NUM_ACTOR_RELATIONS, config.d_model)
        # A bias-free linear layer is equivalent to learning one embedding per flag
        # and summing embeddings for the active status/pinch bits.
        self.finger_flag_encoder = nn.Linear(
            FINGER_FLAG_DIM, config.d_model, bias=False
        )
        fusion_input_dimension = self.num_modalities * config.d_model + self.num_modalities
        self.modality_fusion = nn.Sequential(
            nn.Linear(fusion_input_dimension, config.modality_fusion_hidden_dim),
            nn.GELU(),
            nn.LayerNorm(config.modality_fusion_hidden_dim),
            nn.Dropout(config.dropout),
            nn.Linear(config.modality_fusion_hidden_dim, config.d_model),
        )
        self.time_encoder = TimeEncoder(config.d_model, config.time_hidden_dim)

        self.cls_token = nn.Parameter(torch.empty(1, 1, config.d_model))
        self.anchor_token = nn.Parameter(torch.empty(1, 1, config.d_model))
        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.anchor_token, std=0.02)

        self.input_norm = nn.LayerNorm(config.d_model)
        self.input_dropout = nn.Dropout(config.dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.nhead,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=config.norm_first,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.num_layers,
            enable_nested_tensor=False,
        )
        # TransformerEncoder clones the prototype layer, including its initial values.
        # Reinitialize matrix parameters so stacked layers do not start identically.
        for parameter in self.transformer.parameters():
            if parameter.dim() > 1:
                nn.init.xavier_uniform_(parameter)
        self.final_norm = nn.LayerNorm(config.d_model)
        self.classifier = nn.Sequential(
            nn.Linear(config.d_model, config.classifier_hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.classifier_hidden_dim, 1),
        )

    def encode_modalities(
        self,
        modality_features: Mapping[int, torch.Tensor],
        finger_flags: Mapping[int, torch.Tensor],
        modality_token_indices: Mapping[int, torch.Tensor],
        modality_actor_relation_ids: Mapping[int, torch.Tensor],
        event_mask: torch.Tensor,
        modality_masks: Mapping[int, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        batch_size, sequence_length = event_mask.shape
        device_type = self.cls_token.device.type
        output_dtype = (
            torch.get_autocast_dtype(device_type)
            if torch.is_autocast_enabled(device_type)
            else self.cls_token.dtype
        )
        num_tokens = batch_size * sequence_length
        num_slots = num_tokens * self.num_modalities
        slot_sums = torch.zeros(
            (num_slots, self.d_model),
            dtype=output_dtype,
            device=self.cls_token.device,
        )
        slot_counts = torch.zeros(
            (num_slots, 1),
            dtype=output_dtype,
            device=self.cls_token.device,
        )

        for modality_id, input_dimension in self.modality_dims.items():
            features = modality_features[modality_id]
            token_indices = modality_token_indices[modality_id]
            actor_ids = modality_actor_relation_ids[modality_id]
            expected_shape = (token_indices.shape[0], input_dimension)
            if tuple(features.shape) != expected_shape:
                raise ValueError(
                    f"Modality {modality_id} feature shape is {tuple(features.shape)}; "
                    f"expected {expected_shape}"
                )
            if actor_ids.shape != (features.shape[0],):
                raise ValueError("Actor-relation IDs must align with modality features")
            masked_observations = (
                torch.zeros(
                    features.shape[0],
                    dtype=torch.bool,
                    device=features.device,
                )
                if modality_masks is None
                else modality_masks[modality_id]
            )
            if masked_observations.shape != (features.shape[0],):
                raise ValueError("Modality masks must align with modality features")
            if masked_observations.dtype != torch.bool:
                raise ValueError("Modality masks must have boolean dtype")
            if not features.shape[0]:
                continue

            encoded_features = self.modality_encoders[str(modality_id)](features)
            mask_embedding = self.modality_mask_embedding.weight[modality_id].to(
                dtype=encoded_features.dtype
            )
            encoded_features = torch.where(
                masked_observations.unsqueeze(-1),
                mask_embedding,
                encoded_features,
            )
            encoded_features = encoded_features + self.modality_embedding.weight[
                modality_id
            ].to(dtype=encoded_features.dtype)
            encoded_features = encoded_features + self.actor_relation_embedding(
                actor_ids
            ).to(dtype=encoded_features.dtype)
            if modality_id in FINGER_MODALITY_IDS:
                modality_flags = finger_flags[modality_id]
                expected_finger_shape = (features.shape[0], FINGER_FLAG_DIM)
                if tuple(modality_flags.shape) != expected_finger_shape:
                    raise ValueError(
                        f"Finger flag shape is {tuple(modality_flags.shape)}; "
                        f"expected {expected_finger_shape}"
                    )
                encoded_flags = self.finger_flag_encoder(modality_flags).to(
                    dtype=encoded_features.dtype
                )
                # Finger status and pinch bits describe the masked target, so they
                # must be hidden alongside the continuous finger measurements.
                encoded_flags = encoded_flags * (~masked_observations).unsqueeze(-1)
                encoded_features = encoded_features + encoded_flags

            slot_indices = token_indices * self.num_modalities + modality_id
            encoded_features = encoded_features.to(dtype=slot_sums.dtype)
            slot_sums = slot_sums.index_add(0, slot_indices, encoded_features)
            slot_counts.index_add_(
                0,
                slot_indices,
                torch.ones(
                    (slot_indices.shape[0], 1),
                    dtype=slot_counts.dtype,
                    device=slot_counts.device,
                ),
            )

        slot_values = slot_sums / slot_counts.clamp_min(1.0)
        slot_values = slot_values.reshape(
            batch_size, sequence_length, self.num_modalities * self.d_model
        )
        modality_presence = (slot_counts > 0).to(dtype=slot_values.dtype).reshape(
            batch_size, sequence_length, self.num_modalities
        )
        fused = self.modality_fusion(
            torch.cat((slot_values, modality_presence), dim=-1)
        )
        return fused * event_mask.unsqueeze(-1)

    def encode_sequence(
        self,
        modality_features: Mapping[int, torch.Tensor],
        finger_flags: Mapping[int, torch.Tensor],
        modality_token_indices: Mapping[int, torch.Tensor],
        modality_actor_relation_ids: Mapping[int, torch.Tensor],
        time_features: torch.Tensor,
        event_mask: torch.Tensor,
        anchor_mask: torch.Tensor,
        padding_mask: torch.Tensor,
        modality_masks: Mapping[int, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """Return contextualized ``[CLS] + temporal tokens`` representations."""

        if event_mask.shape != anchor_mask.shape or event_mask.shape != padding_mask.shape:
            raise ValueError("event_mask, anchor_mask, and padding_mask must match")
        if time_features.shape[:2] != event_mask.shape:
            raise ValueError("time_features must start with [batch, sequence]")

        event_values = self.encode_modalities(
            modality_features,
            finger_flags,
            modality_token_indices,
            modality_actor_relation_ids,
            event_mask,
            modality_masks=modality_masks,
        )
        event_indicator = event_mask.unsqueeze(-1)
        valid_indicator = (~padding_mask).unsqueeze(-1)

        x = event_values
        x = x + self.time_encoder(time_features) * event_indicator
        x = x + self.anchor_token * anchor_mask.unsqueeze(-1)
        x = self.input_dropout(self.input_norm(x)) * valid_indicator

        batch_size = event_mask.shape[0]
        cls = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls, x), dim=1)
        cls_padding_mask = torch.zeros(
            (batch_size, 1), dtype=torch.bool, device=padding_mask.device
        )
        transformer_padding_mask = torch.cat((cls_padding_mask, padding_mask), dim=1)

        return self.transformer(x, src_key_padding_mask=transformer_padding_mask)

    def forward(
        self,
        modality_features: Mapping[int, torch.Tensor],
        finger_flags: Mapping[int, torch.Tensor],
        modality_token_indices: Mapping[int, torch.Tensor],
        modality_actor_relation_ids: Mapping[int, torch.Tensor],
        time_features: torch.Tensor,
        event_mask: torch.Tensor,
        anchor_mask: torch.Tensor,
        padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        encoded = self.encode_sequence(
            modality_features=modality_features,
            finger_flags=finger_flags,
            modality_token_indices=modality_token_indices,
            modality_actor_relation_ids=modality_actor_relation_ids,
            time_features=time_features,
            event_mask=event_mask,
            anchor_mask=anchor_mask,
            padding_mask=padding_mask,
        )
        pooled = self.final_norm(encoded[:, 0])
        return self.classifier(pooled).squeeze(-1)
