"""Pooling transformer and diffusion head for the MotionGPT3-derived classifier.

The pooling transformer reuses ``t2m_gpt.model.TransformerBlock`` directly (generic,
operates purely on ``[batch, seq, d_model]`` hidden states) and is parameterized by an
*injected* embedder module -- ``nn.Embedding`` for discrete VQ-VAE tokens or
``nn.Linear`` for continuous VAE latents -- so one class serves both tokenization
variants rather than forking the file, the same design already used for
``t2m_gpt_v2.model.MotionLatentGPT`` relative to ``t2m_gpt.model.MotionTokenGPT``.

The diffusion head is a small MLP+ResBlock denoiser trained with the standard DDPM
objective (predict the noise added to a target, conditioned on a projection of some
conditioning vector and the diffusion timestep), following the source paper's own
architecture at a scale appropriate for this project's data size.
"""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as functional

try:
    from t2m_gpt.config import GPTConfig
    from t2m_gpt.model import TransformerBlock
except ModuleNotFoundError as error:  # pragma: no cover - import-path fallback
    if error.name != "t2m_gpt":
        raise
    from ..t2m_gpt.config import GPTConfig
    from ..t2m_gpt.model import TransformerBlock

from .config import DiffusionHeadConfig


class MotionSummarizer(nn.Module):
    """Pool an already-observed motion-token/latent sequence into one global vector.

    Unlike ``MotionTokenGPT``/``MotionLatentGPT`` this is deliberately non-causal and
    has no classifier head of its own -- it plays the role of MotionGPT3's motion
    branch producing hidden states for the diffusion head to condition on, not a
    sequence generator.
    """

    def __init__(self, embedder: nn.Module, embedded_dim: int, num_tokens: int, config: GPTConfig) -> None:
        super().__init__()
        config.validate()
        if config.causal:
            raise ValueError("MotionSummarizer must be built with a non-causal GPTConfig")
        if num_tokens < 1:
            raise ValueError("num_tokens must be at least 1")
        self.embedder = embedder
        self.input_projection = (
            nn.Identity() if embedded_dim == config.d_model else nn.Linear(embedded_dim, config.d_model)
        )
        self.position_embedding = nn.Parameter(torch.zeros(1, num_tokens, config.d_model))
        self.blocks = nn.ModuleList(TransformerBlock(config) for _ in range(config.num_layers))
        self.final_norm = nn.LayerNorm(config.d_model)

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        """``sequence``: ``[batch, tokens]`` (discrete) or ``[batch, tokens, dim]``
        (continuous). Returns ``[batch, d_model]``."""

        embedded = self.input_projection(self.embedder(sequence))
        hidden = embedded + self.position_embedding
        for block in self.blocks:
            hidden = block(hidden)
        hidden = self.final_norm(hidden)
        return hidden.mean(dim=1)


class _SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        if dim % 2 != 0:
            raise ValueError("dim must be even for sin/cos pairing")
        self.dim = dim

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        exponents = torch.arange(half, device=timesteps.device, dtype=torch.float32) / half
        frequencies = torch.exp(-math.log(10000.0) * exponents)
        arguments = timesteps.float().unsqueeze(1) * frequencies.unsqueeze(0)
        return torch.cat([torch.sin(arguments), torch.cos(arguments)], dim=-1)


class _ResBlock(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return hidden + self.mlp(self.norm(hidden))


class DiffusionHead(nn.Module):
    """MLP+ResBlock denoiser: predicts the noise added to a continuous target,
    conditioned on a projection of ``condition`` and the diffusion timestep."""

    def __init__(self, motion_dim: int, condition_dim: int, config: DiffusionHeadConfig) -> None:
        super().__init__()
        config.validate()
        if motion_dim < 1 or condition_dim < 1:
            raise ValueError("motion_dim and condition_dim must be positive")
        self.config = config
        self.motion_dim = motion_dim

        self.input_projection = nn.Linear(motion_dim, config.hidden_dim)
        self.condition_projection = nn.Linear(condition_dim, config.hidden_dim)
        self.time_embedding = _SinusoidalTimeEmbedding(config.hidden_dim)
        self.time_projection = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
        )
        self.blocks = nn.ModuleList(
            _ResBlock(config.hidden_dim, config.dropout) for _ in range(config.num_resblocks)
        )
        self.output_projection = nn.Linear(config.hidden_dim, motion_dim)

        betas = torch.linspace(config.beta_start, config.beta_end, config.num_train_timesteps)
        alpha_cumulative_product = torch.cumprod(1.0 - betas, dim=0)
        self.register_buffer("alpha_cumulative_product", alpha_cumulative_product)

    def predict_noise(
        self, noisy_target: torch.Tensor, timesteps: torch.Tensor, condition: torch.Tensor
    ) -> torch.Tensor:
        hidden = (
            self.input_projection(noisy_target)
            + self.condition_projection(condition)
            + self.time_projection(self.time_embedding(timesteps))
        )
        for block in self.blocks:
            hidden = block(hidden)
        return self.output_projection(hidden)

    def diffusion_loss(self, target: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        """Per-window denoising loss, averaged over ``config.loss_samples`` random
        (timestep, noise) draws. A lower loss means ``target`` is more plausible under
        ``condition`` -- the diffusion analogue of a higher log-likelihood."""

        batch_size = target.shape[0]
        per_sample_losses = []
        for _ in range(self.config.loss_samples):
            timesteps = torch.randint(
                0, self.config.num_train_timesteps, (batch_size,), device=target.device
            )
            alpha_bar = self.alpha_cumulative_product[timesteps].unsqueeze(-1)
            noise = torch.randn_like(target)
            noisy_target = alpha_bar.sqrt() * target + (1.0 - alpha_bar).sqrt() * noise
            predicted_noise = self.predict_noise(noisy_target, timesteps, condition)
            per_sample_losses.append(
                functional.mse_loss(predicted_noise, noise, reduction="none").mean(dim=-1)
            )
        return torch.stack(per_sample_losses, dim=0).mean(dim=0)


class MotionGPT3Classifier(nn.Module):
    """Ties the summarizer and diffusion head together with class conditioning."""

    def __init__(self, summarizer: MotionSummarizer, d_model: int, diffusion_config: DiffusionHeadConfig) -> None:
        super().__init__()
        self.summarizer = summarizer
        self.class_embedding = nn.Embedding(2, d_model)
        self.diffusion_head = DiffusionHead(motion_dim=d_model, condition_dim=d_model, config=diffusion_config)

    def training_loss(self, sequence: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Trains ``p(motion | true class)`` only, mirroring how the existing
        generative head trains only on each window's true class prefix."""

        target = self.summarizer(sequence)
        condition = self.class_embedding(labels.long())
        return self.diffusion_head.diffusion_loss(target, condition).mean()

    def classification_score(self, sequence: torch.Tensor) -> torch.Tensor:
        """``negative_loss - positive_loss``: the diffusion analogue of
        ``log_likelihood_ratio`` (positive log-likelihood minus negative), since loss
        is a negative-log-likelihood-like quantity and the sign flips accordingly."""

        target = self.summarizer(sequence)
        batch_size = target.shape[0]
        positive_condition = self.class_embedding(
            torch.ones(batch_size, dtype=torch.long, device=target.device)
        )
        negative_condition = self.class_embedding(
            torch.zeros(batch_size, dtype=torch.long, device=target.device)
        )
        positive_loss = self.diffusion_head.diffusion_loss(target, positive_condition)
        negative_loss = self.diffusion_head.diffusion_loss(target, negative_condition)
        return negative_loss - positive_loss

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        return self.classification_score(sequence)


def count_parameters(module: nn.Module, trainable_only: bool = True) -> int:
    return sum(
        parameter.numel()
        for parameter in module.parameters()
        if parameter.requires_grad or not trainable_only
    )


__all__ = ["DiffusionHead", "MotionGPT3Classifier", "MotionSummarizer", "count_parameters"]
