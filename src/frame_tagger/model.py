"""Bidirectional encoders that emit per-frame BIO scores, with a CRF on top.

This is the direct approach to "find a temporal span in a sensor stream", and it skips
motion tokenization entirely: the encoder reads the continuous fixed-grid channels, one
step per frame, and the CRF turns per-frame scores into well-formed spans. Compared with
the T2M-GPT and MotionGPT stacks in this repository, nothing is discretized and nothing is
pretrained -- there is no codebook, so no reconstruction objective competes with the
supervised one, and no temporal downsampling blurs a span boundary.

Both encoders are deliberately bidirectional. A negation gesture can begin before the word
it belongs to, so a frame's label depends on frames after it as much as before it, and a
causal model would be handicapped for no reason.
"""

from __future__ import annotations

import torch
from torch import nn

from .crf import LinearChainCRF
from .labels import TAG_NAMES


class FrameEncoder(nn.Module):
    """Shared input projection and output head around one sequence encoder."""

    def __init__(
        self,
        num_channels: int,
        num_tags: int,
        encoder: str = "bilstm",
        d_model: int = 96,
        num_layers: int = 2,
        nhead: int = 4,
        dim_feedforward: int = 256,
        dropout: float = 0.3,
        max_frames: int = 256,
    ) -> None:
        super().__init__()
        if encoder not in {"bilstm", "transformer"}:
            raise ValueError(f"Unknown encoder: {encoder!r}")
        self.encoder_kind = encoder
        self.input_projection = nn.Linear(num_channels, d_model)
        self.input_dropout = nn.Dropout(dropout)

        if encoder == "bilstm":
            self.sequence = nn.LSTM(
                input_size=d_model,
                hidden_size=d_model // 2,
                num_layers=num_layers,
                batch_first=True,
                bidirectional=True,
                dropout=dropout if num_layers > 1 else 0.0,
            )
            self.positions = None
        else:
            self.positions = nn.Embedding(max_frames, d_model)
            layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                batch_first=True,
                norm_first=True,
            )
            self.sequence = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.emissions = nn.Linear(d_model, num_tags)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        """Map ``[batch, channels, frames]`` to ``[batch, frames, tags]`` emissions."""

        if values.dim() != 3:
            raise ValueError("values must have shape [batch, channels, frames]")
        hidden = self.input_dropout(self.input_projection(values.transpose(1, 2)))
        if self.encoder_kind == "bilstm":
            hidden, _ = self.sequence(hidden)
        else:
            frames = hidden.shape[1]
            index = torch.arange(frames, device=hidden.device)
            hidden = self.sequence(hidden + self.positions(index).unsqueeze(0))
        return self.emissions(self.norm(hidden))


class FrameTagger(nn.Module):
    """Encoder plus linear-chain CRF for frame-level negation tagging."""

    def __init__(
        self,
        num_channels: int,
        encoder: str = "bilstm",
        d_model: int = 96,
        num_layers: int = 2,
        nhead: int = 4,
        dim_feedforward: int = 256,
        dropout: float = 0.3,
        max_frames: int = 256,
        constrain_transitions: bool = True,
        tag_names: tuple[str, ...] = TAG_NAMES,
    ) -> None:
        super().__init__()
        self.tag_names = tag_names
        self.encoder = FrameEncoder(
            num_channels=num_channels,
            num_tags=len(tag_names),
            encoder=encoder,
            d_model=d_model,
            num_layers=num_layers,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            max_frames=max_frames,
        )
        self.crf = LinearChainCRF(
            num_tags=len(tag_names),
            constrain_transitions=constrain_transitions,
            tag_names=tag_names,
        )

    @property
    def positive_tag_ids(self) -> tuple[int, ...]:
        """Tag ids that mean "inside some negation span"."""

        return tuple(
            index for index, name in enumerate(self.tag_names) if name != "O"
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.encoder(values)

    def loss(
        self,
        values: torch.Tensor,
        tags: torch.Tensor,
        mask: torch.Tensor | None = None,
        reduction: str = "token_mean",
    ) -> torch.Tensor:
        return self.crf(self.encoder(values), tags, mask, reduction=reduction)

    @torch.no_grad()
    def predict(
        self, values: torch.Tensor, mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(viterbi_tags, per_frame_positive_probability)``."""

        emissions = self.encoder(values)
        tags = self.crf.decode(emissions, mask)
        probability = self.crf.marginal_positive_probability(
            emissions, self.positive_tag_ids, mask
        )
        return tags, probability


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


__all__ = ["FrameEncoder", "FrameTagger", "count_parameters"]
