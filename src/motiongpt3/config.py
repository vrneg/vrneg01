"""Configuration objects for the MotionGPT3-derived diffusion-head classifier.

MotionGPT3 (arXiv:2506.24086) pools a whole motion sequence into a single continuous
latent and generates it with a diffusion head conditioned on a text branch's hidden
states, via cross-modal attention between a frozen text-language-model branch and a
from-scratch motion branch. This project has no captions, so the text branch and its
cross-modal attention -- MotionGPT3's actual "dual-stream" mechanism -- are dropped
entirely rather than adapted; what is faithfully portable is the diffusion head itself:
a small MLP+ResBlock denoiser predicting noise over a continuous target, conditioned via
a lightweight projection, trained with the standard DDPM objective.

For classification, the condition is the class label instead of text, mirroring how
``t2m_gpt.config.GPTConfig``'s existing ``head="generative"`` already trains
``p(tokens | class)`` and classifies by log-likelihood ratio: here the same idea is
implemented with a diffusion model's denoising loss (a proxy for negative log-likelihood)
in place of autoregressive cross-entropy, over a continuous, pooled motion summary
instead of a discrete token sequence.

``tokenization_mode`` selects the discrete/continuous ablation axis this project's other
new models also test: ``"discrete"`` embeds this project's existing VQ-VAE tokens before
pooling; ``"continuous"`` embeds ``t2m_gpt_v2``'s VAE latents instead. Either way the
pooling transformer and diffusion head are identical, isolating what the tokenization
choice itself contributes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

try:
    from t2m_gpt.config import DataConfig, GPTConfig, TrainingConfig, VQVAEConfig, VQVAETrainingConfig
    from t2m_gpt_v2.config import VAEConfig, VAETrainingConfig
except ModuleNotFoundError as error:  # pragma: no cover - import-path fallback
    if error.name not in {"t2m_gpt", "t2m_gpt_v2"}:
        raise
    from ..t2m_gpt.config import DataConfig, GPTConfig, TrainingConfig, VQVAEConfig, VQVAETrainingConfig
    from ..t2m_gpt_v2.config import VAEConfig, VAETrainingConfig


@dataclass(frozen=True, slots=True)
class DiffusionHeadConfig:
    """MLP+ResBlock denoiser and its DDPM noise schedule.

    ``hidden_dim``/``num_resblocks`` are scaled down from the paper's own
    hidden_dim=1024, 3-layer ResBlock MLP (sized for a 238M-parameter model on
    large-scale motion-text corpora) to match this project's existing small-model scale.
    ``num_train_timesteps`` is likewise scaled down from the paper's 1000: classification
    here only ever needs the denoising *loss*, never actual sample generation, so there
    is no separate inference-time step count to configure at all, unlike the source
    paper's 1000-train/100-inference split.
    """

    hidden_dim: int = 256
    num_resblocks: int = 2
    num_train_timesteps: int = 100
    beta_start: float = 1e-4
    beta_end: float = 0.02
    dropout: float = 0.1
    # How many random (timestep, noise) draws to average the denoising loss over per
    # window -- more samples reduce the score's variance at proportional extra compute.
    loss_samples: int = 4

    def validate(self) -> None:
        for name, value in (("hidden_dim", self.hidden_dim), ("num_resblocks", self.num_resblocks), ("num_train_timesteps", self.num_train_timesteps), ("loss_samples", self.loss_samples)):
            if value < 1:
                raise ValueError(f"{name} must be positive")
        if not 0.0 < self.beta_start < self.beta_end < 1.0:
            raise ValueError("beta_start must be positive and less than beta_end < 1")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")


@dataclass(frozen=True, slots=True)
class ExperimentConfig:
    """Complete configuration for one fold and one random seed."""

    data: DataConfig
    output_dir: str
    run_name: str
    tokenization_mode: Literal["discrete", "continuous"] = "continuous"
    vqvae: VQVAEConfig = field(default_factory=VQVAEConfig)
    vqvae_training: VQVAETrainingConfig = field(default_factory=VQVAETrainingConfig)
    vae: VAEConfig = field(default_factory=VAEConfig)
    vae_training: VAETrainingConfig = field(default_factory=VAETrainingConfig)
    # The pooling transformer's architecture; head/causal/pooling/token_corruption_rate
    # are unused (pooling always mean-pools the final hidden states of a non-causal
    # transformer -- this is a summarizer over an already-observed sequence, not a
    # generator), but GPTConfig is reused rather than duplicated for the fields that
    # matter: d_model, nhead, num_layers, dim_feedforward, dropout.
    summarizer: GPTConfig = field(default_factory=lambda: GPTConfig(causal=False))
    diffusion: DiffusionHeadConfig = field(default_factory=DiffusionHeadConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    seed: int = 42
    device: str = "auto"

    def validate(self) -> None:
        self.data.validate()
        if self.tokenization_mode == "discrete":
            self.vqvae.validate()
            self.vqvae_training.validate()
        elif self.tokenization_mode == "continuous":
            self.vae.validate()
            self.vae_training.validate()
        else:
            raise ValueError(f"Unknown tokenization_mode: {self.tokenization_mode!r}")
        if self.summarizer.causal:
            raise ValueError(
                "the summarizer must be non-causal: it pools an already-observed "
                "sequence, it does not generate autoregressively"
            )
        self.summarizer.validate()
        self.diffusion.validate()
        self.training.validate()
        if not self.run_name.strip():
            raise ValueError("run_name cannot be empty")

        factor = (
            self.vqvae.temporal_downsample_factor
            if self.tokenization_mode == "discrete"
            else self.vae.temporal_downsample_factor
        )
        if self.data.num_time_points % factor != 0:
            raise ValueError(
                f"num_time_points ({self.data.num_time_points}) must be divisible by "
                f"the temporal downsample factor ({factor})"
            )


__all__ = [
    "DataConfig",
    "DiffusionHeadConfig",
    "ExperimentConfig",
    "GPTConfig",
    "TrainingConfig",
    "VAEConfig",
    "VAETrainingConfig",
    "VQVAEConfig",
    "VQVAETrainingConfig",
]
