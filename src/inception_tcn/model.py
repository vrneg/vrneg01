"""Compact modality-aware InceptionTime and TCN hybrid."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

try:
    from event_transformer.features import MODALITY_NAMES
except ModuleNotFoundError as error:
    if error.name != "event_transformer":
        raise
    from ..event_transformer.features import MODALITY_NAMES

from .config import ModelConfig


def _group_count(channels: int) -> int:
    for groups in (8, 4, 2):
        if channels % groups == 0:
            return groups
    return 1


@dataclass(frozen=True, slots=True)
class ModalityChannelSpec:
    """Contiguous input slice and optional presence channel for one modality."""

    name: str
    start: int
    feature_stop: int
    stop: int
    presence_index: int | None

    @property
    def num_feature_channels(self) -> int:
        return self.feature_stop - self.start


def build_modality_channel_layout(
    channel_names: tuple[str, ...],
) -> tuple[ModalityChannelSpec, ...]:
    """Resolve deterministic modality slices from fixed-grid channel names."""

    layout: list[ModalityChannelSpec] = []
    for modality_name in MODALITY_NAMES:
        prefix = f"{modality_name}."
        indices = [
            index for index, name in enumerate(channel_names) if name.startswith(prefix)
        ]
        if not indices:
            raise ValueError(f"No fixed-grid channels found for {modality_name}")
        expected = list(range(indices[0], indices[-1] + 1))
        if indices != expected:
            raise ValueError(f"Channels for {modality_name} must be contiguous")

        presence_name = f"{modality_name}.present"
        presence_index = (
            channel_names.index(presence_name)
            if presence_name in channel_names
            else None
        )
        if presence_index is not None and presence_index != indices[-1]:
            raise ValueError(f"Presence channel for {modality_name} must be last")
        feature_stop = indices[-1] if presence_index is not None else indices[-1] + 1
        if feature_stop <= indices[0]:
            raise ValueError(f"{modality_name} has no feature channels")
        layout.append(
            ModalityChannelSpec(
                name=modality_name,
                start=indices[0],
                feature_stop=feature_stop,
                stop=indices[-1] + 1,
                presence_index=presence_index,
            )
        )
    return tuple(layout)


class InceptionTemporalBlock(nn.Module):
    """Residual multiscale convolution block inspired by InceptionTime."""

    def __init__(
        self,
        input_channels: int,
        bottleneck_channels: int,
        branch_channels: int,
        kernel_sizes: tuple[int, ...],
        dropout: float,
    ):
        super().__init__()
        self.bottleneck = nn.Conv1d(
            input_channels, bottleneck_channels, kernel_size=1, bias=False
        )
        self.branches = nn.ModuleList(
            [
                nn.Conv1d(
                    bottleneck_channels,
                    branch_channels,
                    kernel_size=kernel_size,
                    padding=kernel_size // 2,
                    bias=False,
                )
                for kernel_size in kernel_sizes
            ]
        )
        self.pool_branch = nn.Sequential(
            nn.MaxPool1d(kernel_size=3, stride=1, padding=1),
            nn.Conv1d(input_channels, branch_channels, kernel_size=1, bias=False),
        )
        output_channels = branch_channels * (len(kernel_sizes) + 1)
        self.normalization = nn.GroupNorm(
            _group_count(output_channels), output_channels
        )
        self.residual = (
            nn.Identity()
            if input_channels == output_channels
            else nn.Conv1d(input_channels, output_channels, kernel_size=1, bias=False)
        )
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.output_channels = output_channels

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        bottleneck = self.bottleneck(values)
        branches = [branch(bottleneck) for branch in self.branches]
        branches.append(self.pool_branch(values))
        merged = torch.cat(branches, dim=1)
        merged = self.dropout(self.normalization(merged))
        return self.activation(merged + self.residual(values))


class DepthwiseTCNBlock(nn.Module):
    """Residual depthwise-separable temporal block at one dilation."""

    def __init__(self, channels: int, dilation: int, dropout: float):
        super().__init__()
        layers: list[nn.Module] = []
        for _ in range(2):
            layers.extend(
                (
                    nn.Conv1d(
                        channels,
                        channels,
                        kernel_size=3,
                        padding=dilation,
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


class ModalityBranch(nn.Module):
    """Project, temporally encode, and mask-pool one sensor modality."""

    def __init__(self, input_channels: int, config: ModelConfig):
        super().__init__()
        projected_input_channels = input_channels + int(
            config.include_relative_time_channel
        )
        self.include_relative_time_channel = config.include_relative_time_channel
        self.input_projection = nn.Sequential(
            nn.Conv1d(
                projected_input_channels,
                config.projection_channels,
                kernel_size=1,
                bias=False,
            ),
            nn.GroupNorm(
                _group_count(config.projection_channels), config.projection_channels
            ),
            nn.GELU(),
        )

        inception_blocks: list[nn.Module] = []
        channels = config.projection_channels
        for _ in range(config.num_inception_blocks):
            block = InceptionTemporalBlock(
                input_channels=channels,
                bottleneck_channels=config.projection_channels,
                branch_channels=config.inception_branch_channels,
                kernel_sizes=config.inception_kernel_sizes,
                dropout=config.dropout,
            )
            inception_blocks.append(block)
            channels = block.output_channels
        self.inception_blocks = nn.Sequential(*inception_blocks)
        self.tcn_blocks = nn.Sequential(
            *[
                DepthwiseTCNBlock(channels, dilation, config.dropout)
                for dilation in config.tcn_dilations
            ]
        )
        self.output_channels = channels

    def forward(
        self,
        features: torch.Tensor,
        presence: torch.Tensor,
        relative_time: torch.Tensor,
    ) -> torch.Tensor:
        masked_features = features * presence
        inputs = (
            torch.cat((masked_features, relative_time * presence), dim=1)
            if self.include_relative_time_channel
            else masked_features
        )
        encoded = self.input_projection(inputs)
        encoded = self.inception_blocks(encoded)
        encoded = self.tcn_blocks(encoded)

        weight = presence.clamp(0.0, 1.0)
        denominator = weight.sum(dim=2).clamp_min(1.0)
        mean_pool = (encoded * weight).sum(dim=2) / denominator

        valid = weight > 0.0
        minimum = torch.finfo(encoded.dtype).min
        max_pool = encoded.masked_fill(~valid, minimum).amax(dim=2)
        has_observation = valid.any(dim=2)
        max_pool = torch.where(has_observation, max_pool, torch.zeros_like(max_pool))
        presence_fraction = weight.mean(dim=2)
        return torch.cat((mean_pool, max_pool, presence_fraction), dim=1)


class ModalityAwareInceptionTCN(nn.Module):
    """Eight-branch temporal classifier over aligned fixed-grid modalities."""

    def __init__(
        self,
        config: ModelConfig,
        channel_names: tuple[str, ...],
    ):
        super().__init__()
        config.validate()
        self.config = config
        self.channel_names = tuple(channel_names)
        self.layout = build_modality_channel_layout(self.channel_names)
        self.branches = nn.ModuleDict(
            {
                spec.name: ModalityBranch(spec.num_feature_channels, config)
                for spec in self.layout
            }
        )
        branch_embedding_dim = next(iter(self.branches.values())).output_channels * 2 + 1
        classifier_input_dim = len(self.layout) * branch_embedding_dim
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

        embeddings: list[torch.Tensor] = []
        for spec in self.layout:
            features = values[:, spec.start : spec.feature_stop]
            presence = (
                values[:, spec.presence_index : spec.presence_index + 1]
                if spec.presence_index is not None
                else torch.ones(
                    (batch_size, 1, time_points),
                    dtype=values.dtype,
                    device=values.device,
                )
            )
            embeddings.append(
                self.branches[spec.name](features, presence, relative_time)
            )
        return self.classifier(torch.cat(embeddings, dim=1)).squeeze(1)


__all__ = [
    "ModalityAwareInceptionTCN",
    "ModalityChannelSpec",
    "build_modality_channel_layout",
]
