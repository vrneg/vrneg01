"""Compact cross-modal TCN with cue-aware temporal-pyramid pooling."""

from __future__ import annotations

import torch
from torch import nn

try:
    from event_transformer.features import MODALITY_NAMES
    from inception_tcn.model import ModalityChannelSpec, build_modality_channel_layout
except ModuleNotFoundError as error:
    if error.name not in {"event_transformer", "inception_tcn"}:
        raise
    from ..event_transformer.features import MODALITY_NAMES
    from ..inception_tcn.model import (
        ModalityChannelSpec,
        build_modality_channel_layout,
    )

from .config import ModelConfig


def _group_count(channels: int) -> int:
    for groups in (8, 4, 2):
        if channels % groups == 0:
            return groups
    return 1


class ModalityProjection(nn.Module):
    """Project one observed modality without fabricating missing values."""

    def __init__(self, input_channels: int, output_channels: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv1d(input_channels, output_channels, kernel_size=1, bias=False),
            nn.GroupNorm(_group_count(output_channels), output_channels),
            nn.GELU(),
        )

    def forward(
        self, features: torch.Tensor, presence: torch.Tensor
    ) -> torch.Tensor:
        return self.layers(features * presence) * presence


class ResidualTemporalBlock(nn.Module):
    """A low-parameter depthwise-separable residual temporal block."""

    def __init__(
        self,
        channels: int,
        kernel_size: int,
        dilation: int,
        num_convolutions: int,
        dropout: float,
    ):
        super().__init__()
        padding = dilation * (kernel_size // 2)
        layers: list[nn.Module] = []
        for _ in range(num_convolutions):
            layers.extend(
                (
                    nn.Conv1d(
                        channels,
                        channels,
                        kernel_size=kernel_size,
                        padding=padding,
                        dilation=dilation,
                        groups=channels,
                        bias=False,
                    ),
                    nn.Conv1d(channels, channels, kernel_size=1, bias=False),
                    nn.GroupNorm(_group_count(channels), channels),
                    nn.GELU(),
                    nn.Dropout(dropout),
                )
            )
        self.layers = nn.Sequential(*layers)
        self.activation = nn.GELU()

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.activation(values + self.layers(values))


def _masked_statistics(
    values: torch.Tensor,
    mask: torch.Tensor,
    statistics: tuple[str, ...],
) -> list[torch.Tensor]:
    """Pool ``[batch, channels, time]`` values under a broadcastable mask."""

    weights = mask.to(values.dtype)
    denominator = weights.sum(dim=2).clamp_min(1.0)
    mean = (values * weights).sum(dim=2) / denominator
    outputs: list[torch.Tensor] = []
    for statistic in statistics:
        if statistic == "mean":
            outputs.append(mean)
        elif statistic == "maximum":
            minimum = torch.finfo(values.dtype).min
            maximum = values.masked_fill(~mask, minimum).amax(dim=2)
            has_value = mask.any(dim=2)
            outputs.append(torch.where(has_value, maximum, torch.zeros_like(maximum)))
        elif statistic == "std":
            variance = ((values - mean.unsqueeze(2)).square() * weights).sum(
                dim=2
            ) / denominator
            outputs.append(variance.clamp_min(0.0).sqrt())
        else:  # ModelConfig validation prevents this branch.
            raise ValueError(f"Unknown pooling statistic: {statistic!r}")
    return outputs


class CompactFusionTCN(nn.Module):
    """Fuse projected modalities before a shared cue-aware temporal encoder."""

    def __init__(self, config: ModelConfig, channel_names: tuple[str, ...]):
        super().__init__()
        config.validate()
        self.config = config
        self.channel_names = tuple(channel_names)
        self.layout = build_modality_channel_layout(self.channel_names)
        if tuple(spec.name for spec in self.layout) != MODALITY_NAMES:
            raise ValueError("Fixed-grid modality order does not match MODALITY_NAMES")

        self.projections = nn.ModuleDict(
            {
                spec.name: ModalityProjection(spec.num_feature_channels, channels)
                for spec, channels in zip(
                    self.layout,
                    config.modality_projection_channels,
                    strict=True,
                )
            }
        )
        fusion_input_channels = sum(config.modality_projection_channels)
        fusion_input_channels += int(config.include_relative_time_channel)
        if config.include_modality_presence_channels:
            fusion_input_channels += len(self.layout)
        self.fusion_projection = nn.Sequential(
            nn.Conv1d(
                fusion_input_channels,
                config.fusion_channels,
                kernel_size=1,
                bias=False,
            ),
            nn.GroupNorm(_group_count(config.fusion_channels), config.fusion_channels),
            nn.GELU(),
            nn.Dropout(config.dropout),
        )
        self.temporal_encoder = nn.Sequential(
            *[
                ResidualTemporalBlock(
                    channels=config.fusion_channels,
                    kernel_size=config.temporal_kernel_size,
                    dilation=dilation,
                    num_convolutions=config.convolutions_per_tcn_block,
                    dropout=config.dropout,
                )
                for dilation in config.tcn_dilations
            ]
        )

        encoded_pool_dim = (
            config.fusion_channels
            * len(config.pooling_regions)
            * len(config.pooling_statistics)
        )
        presence_pool_dim = len(self.layout) * len(config.pooling_regions)
        classifier_input_dim = encoded_pool_dim + presence_pool_dim
        self.classifier = nn.Sequential(
            nn.LayerNorm(classifier_input_dim),
            nn.Linear(classifier_input_dim, config.classifier_hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.classifier_hidden_dim, 1),
        )

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def _time_region_mask(
        self,
        region: str,
        relative_time: torch.Tensor,
    ) -> torch.Tensor:
        if region == "global":
            return torch.ones_like(relative_time, dtype=torch.bool)
        if region == "pre":
            return relative_time < 0.0
        if region == "post":
            return relative_time >= 0.0
        raise ValueError(f"Unknown pooling region: {region!r}")

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        if values.ndim != 3:
            raise ValueError("values must have shape [batch, channels, time]")
        if values.shape[1] != len(self.channel_names):
            raise ValueError(
                f"Expected {len(self.channel_names)} channels, got {values.shape[1]}"
            )

        batch_size, _, time_points = values.shape
        relative_time = torch.linspace(
            -1.0,
            1.0,
            time_points,
            dtype=values.dtype,
            device=values.device,
        ).view(1, 1, time_points)
        relative_time = relative_time.expand(batch_size, -1, -1)

        presences: list[torch.Tensor] = []
        features_by_modality: list[torch.Tensor] = []
        for spec in self.layout:
            features_by_modality.append(values[:, spec.start : spec.feature_stop])
            presences.append(
                values[:, spec.presence_index : spec.presence_index + 1]
                if spec.presence_index is not None
                else torch.ones(
                    (batch_size, 1, time_points),
                    dtype=values.dtype,
                    device=values.device,
                )
            )

        if self.training and self.config.modality_dropout > 0.0:
            keep = (
                torch.rand(
                    (batch_size, len(self.layout), 1, 1),
                    device=values.device,
                )
                >= self.config.modality_dropout
            ).to(values.dtype)
            presences = [
                presence * keep[:, index]
                for index, presence in enumerate(presences)
            ]

        projected = [
            self.projections[spec.name](features, presence)
            for spec, features, presence in zip(
                self.layout,
                features_by_modality,
                presences,
                strict=True,
            )
        ]
        fusion_inputs = list(projected)
        if self.config.include_modality_presence_channels:
            fusion_inputs.extend(presences)
        if self.config.include_relative_time_channel:
            fusion_inputs.append(relative_time)
        encoded = self.temporal_encoder(
            self.fusion_projection(torch.cat(fusion_inputs, dim=1))
        )

        presence_tensor = torch.cat(presences, dim=1).clamp(0.0, 1.0)
        timeline_presence = presence_tensor.amax(dim=1, keepdim=True) > 0.0
        pooled: list[torch.Tensor] = []
        for region in self.config.pooling_regions:
            time_region_mask = self._time_region_mask(region, relative_time)
            mask = timeline_presence & time_region_mask
            pooled.extend(
                _masked_statistics(encoded, mask, self.config.pooling_statistics)
            )
            region_weights = time_region_mask.to(values.dtype)
            denominator = region_weights.sum(dim=2).clamp_min(1.0)
            pooled.append(
                (presence_tensor * region_weights).sum(dim=2) / denominator
            )
        return self.classifier(torch.cat(pooled, dim=1)).squeeze(1)


__all__ = [
    "CompactFusionTCN",
    "ModalityChannelSpec",
    "ResidualTemporalBlock",
    "build_modality_channel_layout",
]
