"""Motion VQ-VAE: the stage-one tokenizer of T2M-GPT adapted to VR sensor windows.

The encoder is a strided 1D convolutional stack with dilated residual blocks, the
bottleneck is a codebook trained with exponential moving averages and dead-code reset,
and the decoder mirrors the encoder with nearest-neighbour upsampling.  The discretized
sequence is what the stage-two transformer consumes.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as functional

from .config import VQVAEConfig


def _activation(name: str) -> nn.Module:
    if name == "relu":
        return nn.ReLU()
    if name == "gelu":
        return nn.GELU()
    raise ValueError(f"Unsupported activation: {name!r}")


class ResidualConv1DBlock(nn.Module):
    """Dilated residual block: 3-tap dilated convolution followed by a 1-tap mixer."""

    def __init__(
        self,
        channels: int,
        dilation: int,
        activation: str = "relu",
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.activation_in = _activation(activation)
        self.dilated_convolution = nn.Conv1d(
            channels,
            channels,
            kernel_size=3,
            padding=dilation,
            dilation=dilation,
        )
        self.activation_hidden = _activation(activation)
        self.mixer = nn.Conv1d(channels, channels, kernel_size=1)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        hidden = self.dilated_convolution(self.activation_in(inputs))
        hidden = self.mixer(self.activation_hidden(hidden))
        return inputs + self.dropout(hidden)


class ResidualStack(nn.Module):
    """A stack of residual blocks whose dilation grows geometrically."""

    def __init__(
        self,
        channels: int,
        num_blocks: int,
        dilation_growth_rate: int,
        reverse_dilation: bool = False,
        activation: str = "relu",
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        dilations = [dilation_growth_rate**depth for depth in range(num_blocks)]
        if reverse_dilation:
            dilations = list(reversed(dilations))
        self.blocks = nn.Sequential(
            *(
                ResidualConv1DBlock(
                    channels,
                    dilation=dilation,
                    activation=activation,
                    dropout=dropout,
                )
                for dilation in dilations
            )
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.blocks(inputs)


class MotionEncoder(nn.Module):
    """Map ``[batch, channels, frames]`` windows to latent vectors per motion token."""

    def __init__(self, input_dim: int, config: VQVAEConfig) -> None:
        super().__init__()
        layers: list[nn.Module] = [
            nn.Conv1d(input_dim, config.width, kernel_size=3, padding=1),
            _activation(config.activation),
        ]
        for _ in range(config.num_downsample_layers):
            layers.extend(
                (
                    nn.Conv1d(
                        config.width,
                        config.width,
                        kernel_size=4,
                        stride=2,
                        padding=1,
                    ),
                    ResidualStack(
                        config.width,
                        num_blocks=config.num_residual_blocks,
                        dilation_growth_rate=config.dilation_growth_rate,
                        activation=config.activation,
                        dropout=config.dropout,
                    ),
                )
            )
        layers.append(
            nn.Conv1d(config.width, config.code_dim, kernel_size=3, padding=1)
        )
        self.layers = nn.Sequential(*layers)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.layers(inputs)


class MotionDecoder(nn.Module):
    """Reconstruct ``[batch, channels, frames]`` windows from quantized latents."""

    def __init__(self, output_dim: int, config: VQVAEConfig) -> None:
        super().__init__()
        layers: list[nn.Module] = [
            nn.Conv1d(config.code_dim, config.width, kernel_size=3, padding=1),
            _activation(config.activation),
        ]
        for _ in range(config.num_downsample_layers):
            layers.extend(
                (
                    ResidualStack(
                        config.width,
                        num_blocks=config.num_residual_blocks,
                        dilation_growth_rate=config.dilation_growth_rate,
                        reverse_dilation=True,
                        activation=config.activation,
                        dropout=config.dropout,
                    ),
                    nn.Upsample(scale_factor=2.0, mode="nearest"),
                    nn.Conv1d(config.width, config.width, kernel_size=3, padding=1),
                )
            )
        layers.extend(
            (
                _activation(config.activation),
                nn.Conv1d(config.width, output_dim, kernel_size=3, padding=1),
            )
        )
        self.layers = nn.Sequential(*layers)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.layers(inputs)


@dataclass(slots=True)
class QuantizerOutput:
    """Straight-through quantization result and codebook diagnostics."""

    quantized: torch.Tensor
    indices: torch.Tensor
    commitment_loss: torch.Tensor
    perplexity: torch.Tensor
    num_used_codes: int


class QuantizeEMAReset(nn.Module):
    """Codebook trained by exponential moving averages with dead-code reset.

    Gradients reach the encoder through the straight-through estimator, so the codebook
    itself is never updated by the optimizer.  Codes that attract fewer than
    ``code_reset_threshold`` vectors are re-seeded from current encoder outputs, which
    keeps a small dataset from collapsing onto a handful of entries.
    """

    def __init__(self, config: VQVAEConfig) -> None:
        super().__init__()
        self.num_codes = config.num_codes
        self.code_dim = config.code_dim
        self.decay = config.codebook_decay
        self.epsilon = config.codebook_epsilon
        self.reset_threshold = config.code_reset_threshold
        self.register_buffer("codebook", torch.randn(config.num_codes, config.code_dim))
        self.register_buffer("cluster_size", torch.ones(config.num_codes))
        self.register_buffer("code_sum", torch.zeros(config.num_codes, config.code_dim))
        self.register_buffer("initialized", torch.zeros((), dtype=torch.bool))

    def _sample_vectors(self, flat: torch.Tensor) -> torch.Tensor:
        """Draw ``num_codes`` encoder outputs, with replacement when necessary."""

        available = flat.shape[0]
        if available >= self.num_codes:
            selection = torch.randperm(available, device=flat.device)[: self.num_codes]
        else:
            selection = torch.randint(
                available, (self.num_codes,), device=flat.device
            )
        return flat[selection]

    @torch.no_grad()
    def _initialize(self, flat: torch.Tensor) -> None:
        sampled = self._sample_vectors(flat)
        self.codebook.copy_(sampled)
        self.code_sum.copy_(sampled)
        self.cluster_size.fill_(1.0)
        self.initialized.fill_(True)

    @torch.no_grad()
    def _update(self, flat: torch.Tensor, indices: torch.Tensor) -> None:
        one_hot = functional.one_hot(indices, self.num_codes).to(flat.dtype)
        batch_cluster_size = one_hot.sum(dim=0)
        batch_code_sum = one_hot.t() @ flat

        self.cluster_size.mul_(self.decay).add_(
            batch_cluster_size, alpha=1.0 - self.decay
        )
        self.code_sum.mul_(self.decay).add_(batch_code_sum, alpha=1.0 - self.decay)

        alive = (self.cluster_size >= self.reset_threshold).unsqueeze(1)
        updated = self.code_sum / self.cluster_size.clamp_min(self.epsilon).unsqueeze(1)
        replacements = self._sample_vectors(flat)
        self.codebook.copy_(torch.where(alive, updated, replacements))
        # Revived codes start from their new location so stale statistics cannot
        # immediately mark them dead again.
        self.code_sum.copy_(torch.where(alive, self.code_sum, replacements))
        self.cluster_size.copy_(
            torch.where(alive.squeeze(1), self.cluster_size, torch.ones_like(self.cluster_size))
        )

    def lookup(self, indices: torch.Tensor) -> torch.Tensor:
        """Return ``[batch, code_dim, tokens]`` latents for token indices."""

        embedded = functional.embedding(indices, self.codebook)
        return embedded.permute(0, 2, 1).contiguous()

    def forward(self, latents: torch.Tensor) -> QuantizerOutput:
        batch_size, code_dim, num_tokens = latents.shape
        if code_dim != self.code_dim:
            raise ValueError(
                f"Quantizer expected {self.code_dim} latent channels, got {code_dim}"
            )

        flattened = latents.permute(0, 2, 1).reshape(-1, code_dim).float()
        if self.training and not bool(self.initialized):
            self._initialize(flattened.detach())

        codebook = self.codebook
        distances = (
            flattened.pow(2).sum(dim=1, keepdim=True)
            - 2.0 * flattened @ codebook.t()
            + codebook.pow(2).sum(dim=1)
        )
        indices = distances.argmin(dim=1)
        selected = functional.embedding(indices, codebook)

        if self.training:
            self._update(flattened.detach(), indices)

        commitment_loss = functional.mse_loss(flattened, selected.detach())
        straight_through = flattened + (selected - flattened).detach()
        quantized = (
            straight_through.view(batch_size, num_tokens, code_dim)
            .permute(0, 2, 1)
            .contiguous()
        )

        with torch.no_grad():
            usage = torch.bincount(indices, minlength=self.num_codes).float()
            probabilities = usage / usage.sum().clamp_min(1.0)
            entropy = -(
                probabilities * torch.log(probabilities.clamp_min(1e-10))
            ).sum()
            perplexity = entropy.exp()
            num_used_codes = int((usage > 0).sum().item())

        return QuantizerOutput(
            quantized=quantized,
            indices=indices.view(batch_size, num_tokens),
            commitment_loss=commitment_loss,
            perplexity=perplexity,
            num_used_codes=num_used_codes,
        )


@dataclass(slots=True)
class VQVAEOutput:
    """Reconstruction and quantization diagnostics for one batch."""

    reconstruction: torch.Tensor
    indices: torch.Tensor
    commitment_loss: torch.Tensor
    perplexity: torch.Tensor
    num_used_codes: int


class MotionVQVAE(nn.Module):
    """Discrete motion tokenizer over fixed-grid multimodal sensor windows."""

    def __init__(self, input_dim: int, config: VQVAEConfig) -> None:
        super().__init__()
        config.validate()
        if input_dim < 1:
            raise ValueError("input_dim must be positive")
        self.config = config
        self.input_dim = input_dim
        self.encoder = MotionEncoder(input_dim, config)
        self.quantizer = QuantizeEMAReset(config)
        self.decoder = MotionDecoder(input_dim, config)

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

    def encode(self, values: torch.Tensor) -> torch.Tensor:
        return self.encoder(values)

    def decode_latents(self, quantized: torch.Tensor) -> torch.Tensor:
        return self.decoder(quantized)

    def decode_indices(self, indices: torch.Tensor) -> torch.Tensor:
        """Reconstruct windows directly from motion-token indices."""

        return self.decoder(self.quantizer.lookup(indices))

    @torch.no_grad()
    def encode_indices(self, values: torch.Tensor) -> torch.Tensor:
        """Return ``[batch, tokens]`` code indices without updating the codebook."""

        was_training = self.training
        self.eval()
        try:
            return self.quantizer(self.encoder(values)).indices
        finally:
            self.train(was_training)

    def forward(self, values: torch.Tensor) -> VQVAEOutput:
        quantization = self.quantizer(self.encoder(values))
        reconstruction = self.decoder(quantization.quantized)
        if reconstruction.shape != values.shape:
            raise RuntimeError(
                f"Reconstruction shape {tuple(reconstruction.shape)} does not match "
                f"input shape {tuple(values.shape)}"
            )
        return VQVAEOutput(
            reconstruction=reconstruction,
            indices=quantization.indices,
            commitment_loss=quantization.commitment_loss,
            perplexity=quantization.perplexity,
            num_used_codes=quantization.num_used_codes,
        )


def _elementwise_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    loss_name: str,
) -> torch.Tensor:
    if loss_name == "smooth_l1":
        return functional.smooth_l1_loss(prediction, target)
    if loss_name == "l1":
        return functional.l1_loss(prediction, target)
    if loss_name == "l2":
        return functional.mse_loss(prediction, target)
    raise ValueError(f"Unsupported reconstruction loss: {loss_name!r}")


def reconstruction_losses(
    reconstruction: torch.Tensor,
    target: torch.Tensor,
    loss_name: str = "smooth_l1",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the frame-wise and velocity reconstruction terms.

    The velocity term compares temporal first differences.  Without it the decoder is
    free to reproduce average poses and lose the movement that carries the signal.
    """

    frame_loss = _elementwise_loss(reconstruction, target, loss_name)
    if reconstruction.shape[-1] < 2:
        velocity_loss = reconstruction.new_zeros(())
    else:
        velocity_loss = _elementwise_loss(
            reconstruction.diff(dim=-1), target.diff(dim=-1), loss_name
        )
    return frame_loss, velocity_loss
