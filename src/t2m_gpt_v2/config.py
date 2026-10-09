"""Configuration objects for the continuous-latent VAE variant of T2M-GPT.

Mirrors ``t2m_gpt.config`` field-for-field wherever the discrete and continuous
tokenizers share a concept, so the two are comparable with only the bottleneck type
differing. ``DataConfig``, ``GPTConfig``, ``PretrainingConfig``, and ``TrainingConfig``
are reused directly from ``t2m_gpt.config`` -- they describe the fold, the stage-two
transformer, and its optimizer, none of which depend on whether stage one is discrete
or continuous. Only the stage-one architecture (``VAEConfig``) and its training loss
(``VAETrainingConfig``) are new, replacing ``VQVAEConfig``/``VQVAETrainingConfig``.

``GPTConfig.head`` must be ``"discriminative"``: the generative head's log-likelihood
ratio needs a discrete codebook to define ``p(tokens | class)`` over, which a continuous
latent has no equivalent for without a generative prior over z. That is out of scope
here -- see ``t2m_gpt_v2.model.MotionLatentGPT`` for the resulting restriction.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

try:
    from t2m_gpt.config import DataConfig, GPTConfig, PretrainingConfig, TrainingConfig
except ModuleNotFoundError as error:  # pragma: no cover - import-path fallback
    if error.name != "t2m_gpt":
        raise
    from ..t2m_gpt.config import DataConfig, GPTConfig, PretrainingConfig, TrainingConfig


@dataclass(frozen=True, slots=True)
class VAEConfig:
    """Motion VAE architecture: a continuous analogue of ``t2m_gpt.config.VQVAEConfig``.

    The encoder/decoder conv backbone is architecturally identical to the VQ-VAE's
    (same width, downsampling, residual stack) -- ``t2m_gpt_v2.vae.MotionVAE`` reuses
    ``t2m_gpt.vqvae.MotionEncoder``/``MotionDecoder`` directly. The only structural
    difference is the bottleneck: instead of a nearest-codebook lookup, the encoder's
    output channels are split into posterior mean and log-variance heads of width
    ``latent_dim`` each, and the decoder reads a ``latent_dim``-wide reparameterized
    sample (or, at inference, the posterior mean).
    """

    latent_dim: int = 64
    width: int = 256
    num_downsample_layers: int = 2
    num_residual_blocks: int = 2
    dilation_growth_rate: int = 3
    activation: Literal["relu", "gelu"] = "relu"
    dropout: float = 0.0

    @property
    def temporal_downsample_factor(self) -> int:
        return 2**self.num_downsample_layers

    def validate(self) -> None:
        for name, value in (
            ("latent_dim", self.latent_dim),
            ("width", self.width),
            ("num_residual_blocks", self.num_residual_blocks),
        ):
            if value < 1:
                raise ValueError(f"{name} must be positive")
        if self.num_downsample_layers < 0:
            raise ValueError("num_downsample_layers cannot be negative")
        if self.dilation_growth_rate < 1:
            raise ValueError("dilation_growth_rate must be at least 1")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")


@dataclass(frozen=True, slots=True)
class VAETrainingConfig:
    """Stage-one ELBO objective and optimization.

    The loss is the VQ-VAE's reconstruction + velocity terms (unchanged, so the two
    tokenizers are comparable on that half) plus a KL-divergence term weighted by
    ``kl_weight`` in place of the VQ-VAE's commitment loss. ``free_bits`` clamps the
    per-dimension KL below this floor to zero before weighting, a standard guard
    against posterior collapse (the decoder ignoring z entirely) when ``kl_weight``
    is not yet well-tuned; ``0.0`` disables it.
    """

    max_epochs: int = 200
    batch_size: int = 32
    evaluation_batch_size: int = 64
    num_workers: int = 0
    learning_rate: float = 2e-4
    weight_decay: float = 0.0
    gradient_clip_norm: float | None = 1.0
    reconstruction_loss: Literal["smooth_l1", "l1", "l2"] = "smooth_l1"
    velocity_loss_weight: float = 0.5
    kl_weight: float = 1e-4
    free_bits: float = 0.0
    early_stopping_patience: int = 20
    early_stopping_min_delta: float = 1e-5
    lr_scheduler: Literal["none", "reduce_on_plateau"] = "reduce_on_plateau"
    lr_scheduler_factor: float = 0.5
    lr_scheduler_patience: int = 8
    minimum_learning_rate: float = 1e-6

    def validate(self) -> None:
        if self.max_epochs < 1:
            raise ValueError("vae max_epochs must be at least 1")
        for name, value in (
            ("batch_size", self.batch_size),
            ("evaluation_batch_size", self.evaluation_batch_size),
        ):
            if value < 1:
                raise ValueError(f"vae {name} must be at least 1")
        if self.num_workers < 0:
            raise ValueError("vae num_workers cannot be negative")
        if self.learning_rate <= 0:
            raise ValueError("vae learning_rate must be positive")
        if self.weight_decay < 0:
            raise ValueError("vae weight_decay cannot be negative")
        if self.gradient_clip_norm is not None and self.gradient_clip_norm <= 0:
            raise ValueError("vae gradient_clip_norm must be positive or None")
        if self.velocity_loss_weight < 0:
            raise ValueError("velocity_loss_weight cannot be negative")
        if self.kl_weight < 0:
            raise ValueError("kl_weight cannot be negative")
        if self.free_bits < 0:
            raise ValueError("free_bits cannot be negative")
        if self.early_stopping_patience < 1:
            raise ValueError("vae early_stopping_patience must be at least 1")
        if self.early_stopping_min_delta < 0:
            raise ValueError("vae early_stopping_min_delta cannot be negative")
        if not 0.0 < self.lr_scheduler_factor < 1.0:
            raise ValueError("vae lr_scheduler_factor must be in (0, 1)")
        if self.lr_scheduler_patience < 1:
            raise ValueError("vae lr_scheduler_patience must be at least 1")
        if self.minimum_learning_rate < 0:
            raise ValueError("vae minimum_learning_rate cannot be negative")
        if self.minimum_learning_rate >= self.learning_rate:
            raise ValueError("vae minimum_learning_rate must be below learning_rate")


@dataclass(frozen=True, slots=True)
class ExperimentConfig:
    """Complete configuration for one fold and one random seed, continuous variant."""

    data: DataConfig
    output_dir: str
    run_name: str
    vae: VAEConfig = field(default_factory=VAEConfig)
    vae_training: VAETrainingConfig = field(default_factory=VAETrainingConfig)
    gpt: GPTConfig = field(default_factory=GPTConfig)
    pretraining: PretrainingConfig = field(default_factory=PretrainingConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    use_pretraining: bool = True
    seed: int = 42
    device: str = "auto"

    def validate(self) -> None:
        self.data.validate()
        self.vae.validate()
        self.vae_training.validate()
        self.gpt.validate()
        if self.gpt.head != "discriminative":
            raise ValueError(
                "t2m_gpt_v2 only supports gpt.head='discriminative': the generative "
                "head's class-conditional log-likelihood ratio needs a discrete "
                "codebook, which the continuous latent has no equivalent for"
            )
        self.pretraining.validate()
        self.training.validate()
        if not self.run_name.strip():
            raise ValueError("run_name cannot be empty")
        factor = self.vae.temporal_downsample_factor
        if self.data.num_time_points % factor != 0:
            raise ValueError(
                f"num_time_points ({self.data.num_time_points}) must be divisible by "
                f"the temporal downsample factor ({factor})"
            )
        if self.data.num_time_points // factor < 1:
            raise ValueError(
                "num_time_points is too small for the requested downsampling"
            )


__all__ = [
    "DataConfig",
    "ExperimentConfig",
    "GPTConfig",
    "PretrainingConfig",
    "TrainingConfig",
    "VAEConfig",
    "VAETrainingConfig",
]
