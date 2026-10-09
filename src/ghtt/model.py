"""Pose block (short-span) and action block (long-span) for the GHTT-derived cascade.

Both blocks are Transformer-VAEs -- attention over a short sequence, pooled into a
single latent vector -- reusing ``t2m_gpt.model.TransformerBlock``/``SelfAttention``
directly for the attention layers (both are dtype/task-agnostic, operating purely on
``[batch, seq, d_model]`` hidden states) and ``t2m_gpt_v2.vae``'s ``reparameterize``/
``kl_divergence_loss`` directly for the VAE machinery (also generic, operating purely on
mean/log-variance tensors). Only the encoder/decoder wiring around those pieces --
attention over raw frames rather than motion tokens, non-causal rather than causal, and
a bottleneck feeding either a reconstruction head or a class label instead of a language-
model head -- is new.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

try:
    from t2m_gpt.config import GPTConfig
    from t2m_gpt.model import TransformerBlock
    from t2m_gpt_v2.vae import reparameterize
except ModuleNotFoundError as error:  # pragma: no cover - import-path fallback
    if error.name not in {"t2m_gpt", "t2m_gpt_v2"}:
        raise
    from ..t2m_gpt.config import GPTConfig
    from ..t2m_gpt.model import TransformerBlock
    from ..t2m_gpt_v2.vae import reparameterize

from .config import ActionBlockConfig, PoseBlockConfig


def _transformer_config(d_model: int, nhead: int, dim_feedforward: int, dropout: float) -> GPTConfig:
    """Vessel for reusing ``TransformerBlock``/``SelfAttention`` -- both blocks here are
    non-causal (a VAE encoder/decoder attends over its whole short input, it is not
    autoregressive), unlike the causal transformer T2M-GPT/``t2m_gpt_v2`` use."""

    return GPTConfig(
        d_model=d_model,
        nhead=nhead,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
        causal=False,
    )


@dataclass(slots=True)
class PoseBlockOutput:
    """Reconstruction, future prediction, and posterior parameters for one clip."""

    reconstruction: torch.Tensor
    trajectory_prediction: torch.Tensor
    mean: torch.Tensor
    log_variance: torch.Tensor
    context: torch.Tensor
    target: torch.Tensor


class PoseBlock(nn.Module):
    """Short-span Transformer-VAE: encode a clip's first half, predict its second."""

    def __init__(self, input_dim: int, config: PoseBlockConfig) -> None:
        super().__init__()
        config.validate()
        if input_dim < 1:
            raise ValueError("input_dim must be positive")
        self.config = config
        self.input_dim = input_dim

        transformer_config = _transformer_config(
            config.d_model, config.nhead, config.dim_feedforward, config.dropout
        )

        self.input_projection = nn.Linear(input_dim, config.d_model)
        self.encoder_position_embedding = nn.Parameter(
            torch.zeros(1, config.context_length, config.d_model)
        )
        self.encoder_blocks = nn.ModuleList(
            TransformerBlock(transformer_config) for _ in range(config.num_layers)
        )
        self.encoder_norm = nn.LayerNorm(config.d_model)
        self.mean_head = nn.Linear(config.d_model, config.latent_dim)
        self.logvar_head = nn.Linear(config.d_model, config.latent_dim)

        self.latent_projection = nn.Linear(config.latent_dim, config.d_model)

        self.reconstruction_position_embedding = nn.Parameter(
            torch.zeros(1, config.context_length, config.d_model)
        )
        self.reconstruction_blocks = nn.ModuleList(
            TransformerBlock(transformer_config) for _ in range(config.num_layers)
        )
        self.reconstruction_norm = nn.LayerNorm(config.d_model)
        self.reconstruction_head = nn.Linear(config.d_model, input_dim)

        self.trajectory_position_embedding = nn.Parameter(
            torch.zeros(1, config.target_length, config.d_model)
        )
        self.trajectory_blocks = nn.ModuleList(
            TransformerBlock(transformer_config) for _ in range(config.num_layers)
        )
        self.trajectory_norm = nn.LayerNorm(config.d_model)
        self.trajectory_head = nn.Linear(config.d_model, input_dim)

    def encode(self, context: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """``context``: ``[batch, context_length, input_dim]``."""

        hidden = self.input_projection(context) + self.encoder_position_embedding
        for block in self.encoder_blocks:
            hidden = block(hidden)
        hidden = self.encoder_norm(hidden)
        pooled = hidden.mean(dim=1)
        return self.mean_head(pooled), self.logvar_head(pooled)

    def _decode(
        self,
        latent: torch.Tensor,
        position_embedding: torch.Tensor,
        blocks: nn.ModuleList,
        norm: nn.LayerNorm,
        head: nn.Linear,
        length: int,
    ) -> torch.Tensor:
        broadcast = self.latent_projection(latent).unsqueeze(1).expand(-1, length, -1)
        hidden = broadcast + position_embedding
        for block in blocks:
            hidden = block(hidden)
        hidden = norm(hidden)
        return head(hidden)

    def reconstruct(self, latent: torch.Tensor) -> torch.Tensor:
        return self._decode(
            latent,
            self.reconstruction_position_embedding,
            self.reconstruction_blocks,
            self.reconstruction_norm,
            self.reconstruction_head,
            self.config.context_length,
        )

    def predict_trajectory(self, latent: torch.Tensor) -> torch.Tensor:
        return self._decode(
            latent,
            self.trajectory_position_embedding,
            self.trajectory_blocks,
            self.trajectory_norm,
            self.trajectory_head,
            self.config.target_length,
        )

    def forward(self, clip: torch.Tensor) -> PoseBlockOutput:
        """``clip``: ``[batch, clip_length, input_dim]``."""

        if clip.shape[1] != self.config.clip_length:
            raise ValueError(
                f"Expected clip_length {self.config.clip_length}, got {clip.shape[1]}"
            )
        context = clip[:, : self.config.context_length]
        target = clip[:, self.config.context_length :]
        mean, log_variance = self.encode(context)
        latent = reparameterize(mean, log_variance)
        reconstruction = self.reconstruct(latent)
        trajectory_prediction = self.predict_trajectory(latent)
        return PoseBlockOutput(
            reconstruction=reconstruction,
            trajectory_prediction=trajectory_prediction,
            mean=mean,
            log_variance=log_variance,
            context=context,
            target=target,
        )

    @torch.no_grad()
    def encode_latent(self, clip: torch.Tensor) -> torch.Tensor:
        """Deterministic posterior mean for one clip, for the action block's input."""

        was_training = self.training
        self.eval()
        try:
            context = clip[:, : self.config.context_length]
            mean, _ = self.encode(context)
            return mean
        finally:
            self.train(was_training)


@dataclass(slots=True)
class ActionBlockOutput:
    """Mid-level reconstruction, classification logit, and posterior parameters."""

    reconstruction: torch.Tensor
    logit: torch.Tensor
    mean: torch.Tensor
    log_variance: torch.Tensor


class ActionBlock(nn.Module):
    """Long-span Transformer-VAE over a sequence of per-clip mid-level features."""

    def __init__(self, mid_dim: int, config: ActionBlockConfig, sequence_length: int) -> None:
        super().__init__()
        config.validate()
        if mid_dim < 1:
            raise ValueError("mid_dim must be positive")
        if sequence_length < 2:
            raise ValueError("sequence_length must be at least 2")
        self.config = config
        self.mid_dim = mid_dim
        self.sequence_length = sequence_length

        transformer_config = _transformer_config(
            config.d_model, config.nhead, config.dim_feedforward, config.dropout
        )

        self.input_projection = nn.Linear(mid_dim, config.d_model)
        self.encoder_position_embedding = nn.Parameter(
            torch.zeros(1, sequence_length, config.d_model)
        )
        self.encoder_blocks = nn.ModuleList(
            TransformerBlock(transformer_config) for _ in range(config.num_layers)
        )
        self.encoder_norm = nn.LayerNorm(config.d_model)
        self.mean_head = nn.Linear(config.d_model, config.latent_dim)
        self.logvar_head = nn.Linear(config.d_model, config.latent_dim)

        self.latent_projection = nn.Linear(config.latent_dim, config.d_model)
        self.decoder_position_embedding = nn.Parameter(
            torch.zeros(1, sequence_length, config.d_model)
        )
        self.decoder_blocks = nn.ModuleList(
            TransformerBlock(transformer_config) for _ in range(config.num_layers)
        )
        self.decoder_norm = nn.LayerNorm(config.d_model)
        self.reconstruction_head = nn.Linear(config.d_model, mid_dim)

        self.classifier_head = nn.Sequential(
            nn.Linear(config.d_model, config.classifier_hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.classifier_hidden_dim, 1),
        )

    def encode(self, mid_sequence: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """``mid_sequence``: ``[batch, sequence_length, mid_dim]``."""

        if mid_sequence.shape[1] != self.sequence_length:
            raise ValueError(
                f"Expected sequence_length {self.sequence_length}, got {mid_sequence.shape[1]}"
            )
        hidden = self.input_projection(mid_sequence) + self.encoder_position_embedding
        for block in self.encoder_blocks:
            hidden = block(hidden)
        hidden = self.encoder_norm(hidden)
        pooled = hidden.mean(dim=1)
        return self.mean_head(pooled), self.logvar_head(pooled), pooled

    def reconstruct(self, latent: torch.Tensor) -> torch.Tensor:
        broadcast = self.latent_projection(latent).unsqueeze(1).expand(-1, self.sequence_length, -1)
        hidden = broadcast + self.decoder_position_embedding
        for block in self.decoder_blocks:
            hidden = block(hidden)
        hidden = self.decoder_norm(hidden)
        return self.reconstruction_head(hidden)

    def forward(self, mid_sequence: torch.Tensor) -> ActionBlockOutput:
        mean, log_variance, pooled = self.encode(mid_sequence)
        latent = reparameterize(mean, log_variance)
        reconstruction = self.reconstruct(latent)
        logit = self.classifier_head(pooled).squeeze(-1)
        return ActionBlockOutput(
            reconstruction=reconstruction, logit=logit, mean=mean, log_variance=log_variance
        )


def count_parameters(module: nn.Module, trainable_only: bool = True) -> int:
    return sum(
        parameter.numel()
        for parameter in module.parameters()
        if parameter.requires_grad or not trainable_only
    )


__all__ = ["ActionBlock", "ActionBlockOutput", "PoseBlock", "PoseBlockOutput", "count_parameters"]
