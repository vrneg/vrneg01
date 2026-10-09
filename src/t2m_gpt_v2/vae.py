"""Motion VAE: a continuous-latent alternative to T2M-GPT's discrete VQ-VAE bottleneck.

Reuses ``t2m_gpt.vqvae.MotionEncoder``/``MotionDecoder`` directly -- both are generic
strided-conv backbones parameterized only by ``width``/``num_downsample_layers``/
``num_residual_blocks``/``dilation_growth_rate``/``activation``/``dropout``/``code_dim``,
with no vector-quantization logic of their own (that lives entirely in the separate
``QuantizeEMAReset`` class, which this module does not use). Swapping the discrete
codebook bottleneck for a continuous one only requires: (1) doubling the encoder's output
channels so half become the posterior mean and half the log-variance, (2) the standard
reparameterization trick, and (3) a KL-divergence term against a standard normal prior in
place of the commitment loss. Reusing ``MotionEncoder``/``MotionDecoder`` means the two
tokenizers share the exact same architecture up to the bottleneck, so a comparison between
them isolates the effect of discretization rather than confounding it with unrelated
architectural differences.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

try:
    from t2m_gpt.vqvae import MotionDecoder, MotionEncoder
    from t2m_gpt.config import VQVAEConfig
except ModuleNotFoundError as error:  # pragma: no cover - import-path fallback
    if error.name != "t2m_gpt":
        raise
    from ..t2m_gpt.vqvae import MotionDecoder, MotionEncoder
    from ..t2m_gpt.config import VQVAEConfig

from .config import VAEConfig


def _backbone_config(config: VAEConfig, code_dim: int) -> VQVAEConfig:
    """Build a ``VQVAEConfig`` vessel for reusing ``MotionEncoder``/``MotionDecoder``.

    Only the fields those two classes actually read (``code_dim``, ``width``,
    ``num_downsample_layers``, ``num_residual_blocks``, ``dilation_growth_rate``,
    ``activation``, ``dropout``) are meaningful here; ``num_codes`` and the
    codebook-specific fields are never read by either class and are left at their
    ``VQVAEConfig`` defaults.
    """

    return VQVAEConfig(
        code_dim=code_dim,
        width=config.width,
        num_downsample_layers=config.num_downsample_layers,
        num_residual_blocks=config.num_residual_blocks,
        dilation_growth_rate=config.dilation_growth_rate,
        activation=config.activation,
        dropout=config.dropout,
    )


def reparameterize(mean: torch.Tensor, log_variance: torch.Tensor) -> torch.Tensor:
    """Sample ``z ~ N(mean, exp(log_variance))`` via the reparameterization trick."""

    std = torch.exp(0.5 * log_variance)
    noise = torch.randn_like(std)
    return mean + noise * std


def kl_divergence_loss(
    mean: torch.Tensor,
    log_variance: torch.Tensor,
    free_bits: float = 0.0,
) -> torch.Tensor:
    """Mean KL(N(mean, exp(log_variance)) || N(0, 1)) per window.

    ``free_bits`` clamps each dimension's KL below this nats floor to zero before
    averaging -- a standard guard against posterior collapse (the decoder learning to
    ignore z) when ``kl_weight`` has not yet been tuned for the dataset.
    """

    per_dimension = 0.5 * (mean.pow(2) + log_variance.exp() - 1.0 - log_variance)
    if free_bits > 0.0:
        per_dimension = torch.clamp(per_dimension, min=free_bits)
    return per_dimension.mean()


@dataclass(slots=True)
class VAEOutput:
    """Reconstruction and posterior parameters for one batch."""

    reconstruction: torch.Tensor
    mean: torch.Tensor
    log_variance: torch.Tensor


class MotionVAE(nn.Module):
    """Continuous motion tokenizer over fixed-grid multimodal sensor windows."""

    def __init__(self, input_dim: int, config: VAEConfig) -> None:
        super().__init__()
        config.validate()
        if input_dim < 1:
            raise ValueError("input_dim must be positive")
        self.config = config
        self.input_dim = input_dim
        # The encoder emits mean and log-variance concatenated along the channel axis,
        # so its code_dim is twice the latent width.
        self.encoder = MotionEncoder(input_dim, _backbone_config(config, 2 * config.latent_dim))
        self.decoder = MotionDecoder(input_dim, _backbone_config(config, config.latent_dim))

    @property
    def temporal_downsample_factor(self) -> int:
        return self.config.temporal_downsample_factor

    def num_tokens(self, num_frames: int) -> int:
        factor = self.temporal_downsample_factor
        if num_frames % factor != 0:
            raise ValueError(
                f"{num_frames} frames is not divisible by the downsample factor {factor}"
            )
        return num_frames // factor

    def encode(self, values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``([batch, latent_dim, tokens], [batch, latent_dim, tokens])``."""

        encoded = self.encoder(values)
        mean, log_variance = encoded.chunk(2, dim=1)
        return mean, log_variance

    def decode_latents(self, latents: torch.Tensor) -> torch.Tensor:
        return self.decoder(latents)

    @torch.no_grad()
    def encode_latents(self, values: torch.Tensor) -> torch.Tensor:
        """Return the posterior mean, deterministically, without updating the model.

        Downstream stage-two training consumes this rather than a fresh reparameterized
        sample each time, exactly as the discrete tokenizer's ``encode_indices`` always
        returns the same code for the same input -- otherwise the transformer would be
        trained against a moving target.
        """

        was_training = self.training
        self.eval()
        try:
            mean, _ = self.encode(values)
            return mean
        finally:
            self.train(was_training)

    def forward(self, values: torch.Tensor) -> VAEOutput:
        mean, log_variance = self.encode(values)
        latents = reparameterize(mean, log_variance)
        reconstruction = self.decoder(latents)
        if reconstruction.shape != values.shape:
            raise RuntimeError(
                f"Reconstruction shape {tuple(reconstruction.shape)} does not match "
                f"input shape {tuple(values.shape)}"
            )
        return VAEOutput(reconstruction=reconstruction, mean=mean, log_variance=log_variance)


__all__ = ["MotionVAE", "VAEOutput", "kl_divergence_loss", "reparameterize"]
