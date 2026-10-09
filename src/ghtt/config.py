"""Configuration objects for the GHTT-derived hierarchical short/long motion model.

Adapted from *Generative Hierarchical Temporal Transformer for Hand Pose and Action
Modeling* (arXiv:2311.17366): two cascaded Transformer-VAE blocks, a short-span
"pose" block (``PoseBlockConfig``) that encodes small clips of consecutive frames, and
a long-span "action" block (``ActionBlockConfig``) that consumes the sequence of the
pose block's per-clip latents. The pose block's whole purpose is preserving short-span
motion fidelity before any long-span abstraction happens, which is the closest
architectural match in this project's literature review to "negation from subtle
finger/hand/head motion."

Loss-term weights default to the paper's own reported values (``kl_weight=1e-5`` for
both blocks, ``classification_weight=0.1`` for the action block, mirroring its
contrastive-action-term weight) as a documented starting point, not a validated one --
see this package's README for the adaptations made and what should be re-checked.

``pose_block_mode`` selects the discrete/continuous ablation axis: ``"vae"`` is the
paper's own continuous per-clip bottleneck (this package's native form); ``"vqvae"``
replaces it with this project's existing discrete VQ-VAE applied per clip, feeding the
same action-block hierarchy, to test whether the cascade itself helps independent of
the pose block's bottleneck type.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

try:
    from t2m_gpt.config import DataConfig, TrainingConfig, VQVAEConfig
except ModuleNotFoundError as error:  # pragma: no cover - import-path fallback
    if error.name != "t2m_gpt":
        raise
    from ..t2m_gpt.config import DataConfig, TrainingConfig, VQVAEConfig


@dataclass(frozen=True, slots=True)
class PoseBlockConfig:
    """Short-span Transformer-VAE over one clip of consecutive frames.

    The encoder sees only the first half of each clip (``context_length`` frames) and
    is trained on two decoders: one reconstructs those same context frames
    (``component_loss_weight``, the paper's L_comp), and a separate one predicts the
    clip's second half -- frames the encoder never saw (``trajectory_loss_weight``, the
    paper's L_trj) -- which is what makes this "recognition encodes past, prediction
    decodes future" rather than a plain autoencoder.
    """

    clip_length: int = 8
    latent_dim: int = 32
    d_model: int = 64
    nhead: int = 4
    num_layers: int = 2
    dim_feedforward: int = 256
    dropout: float = 0.1
    component_loss_weight: float = 1.0
    trajectory_loss_weight: float = 1.0
    kl_weight: float = 1e-5
    free_bits: float = 0.0

    @property
    def context_length(self) -> int:
        return self.clip_length // 2

    @property
    def target_length(self) -> int:
        return self.clip_length - self.context_length

    def validate(self) -> None:
        if self.clip_length < 2:
            raise ValueError("clip_length must be at least 2")
        for name, value in (
            ("latent_dim", self.latent_dim),
            ("d_model", self.d_model),
            ("num_layers", self.num_layers),
            ("dim_feedforward", self.dim_feedforward),
        ):
            if value < 1:
                raise ValueError(f"{name} must be positive")
        if self.nhead < 1 or self.d_model % self.nhead != 0:
            raise ValueError("nhead must be positive and divide d_model exactly")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if self.component_loss_weight < 0:
            raise ValueError("component_loss_weight cannot be negative")
        if self.trajectory_loss_weight < 0:
            raise ValueError("trajectory_loss_weight cannot be negative")
        if self.kl_weight < 0:
            raise ValueError("kl_weight cannot be negative")
        if self.free_bits < 0:
            raise ValueError("free_bits cannot be negative")


@dataclass(frozen=True, slots=True)
class PoseTrainingConfig:
    """Optimization for the pose block, trained independently of the action block."""

    max_epochs: int = 100
    batch_size: int = 32
    evaluation_batch_size: int = 64
    learning_rate: float = 2e-4
    weight_decay: float = 0.0
    gradient_clip_norm: float | None = 1.0
    reconstruction_loss: Literal["smooth_l1", "l1", "l2"] = "smooth_l1"
    early_stopping_patience: int = 12
    early_stopping_min_delta: float = 1e-5
    lr_scheduler: Literal["none", "reduce_on_plateau"] = "reduce_on_plateau"
    lr_scheduler_factor: float = 0.5
    lr_scheduler_patience: int = 6
    minimum_learning_rate: float = 1e-6

    def validate(self) -> None:
        if self.max_epochs < 1:
            raise ValueError("pose max_epochs must be at least 1")
        for name, value in (("batch_size", self.batch_size), ("evaluation_batch_size", self.evaluation_batch_size)):
            if value < 1:
                raise ValueError(f"pose {name} must be at least 1")
        if self.learning_rate <= 0:
            raise ValueError("pose learning_rate must be positive")
        if self.weight_decay < 0:
            raise ValueError("pose weight_decay cannot be negative")
        if self.gradient_clip_norm is not None and self.gradient_clip_norm <= 0:
            raise ValueError("pose gradient_clip_norm must be positive or None")
        if self.early_stopping_patience < 1:
            raise ValueError("pose early_stopping_patience must be at least 1")
        if not 0.0 < self.lr_scheduler_factor < 1.0:
            raise ValueError("pose lr_scheduler_factor must be in (0, 1)")
        if self.lr_scheduler_patience < 1:
            raise ValueError("pose lr_scheduler_patience must be at least 1")
        if self.minimum_learning_rate < 0 or self.minimum_learning_rate >= self.learning_rate:
            raise ValueError("pose minimum_learning_rate must be non-negative and below learning_rate")


@dataclass(frozen=True, slots=True)
class ActionBlockConfig:
    """Long-span Transformer-VAE over the sequence of pose-block clip latents."""

    d_model: int = 64
    nhead: int = 4
    num_layers: int = 2
    dim_feedforward: int = 256
    dropout: float = 0.1
    latent_dim: int = 32
    classifier_hidden_dim: int = 32
    mid_reconstruction_weight: float = 1.0
    # The paper's contrastive action-recognition term, reweighted here as this
    # project's binary classification loss -- the paper's own endorsed classification
    # adaptation ("retrain a classification head with cross-entropy").
    classification_weight: float = 0.1
    kl_weight: float = 1e-5
    free_bits: float = 0.0

    def validate(self) -> None:
        for name, value in (
            ("d_model", self.d_model),
            ("num_layers", self.num_layers),
            ("dim_feedforward", self.dim_feedforward),
            ("latent_dim", self.latent_dim),
            ("classifier_hidden_dim", self.classifier_hidden_dim),
        ):
            if value < 1:
                raise ValueError(f"{name} must be positive")
        if self.nhead < 1 or self.d_model % self.nhead != 0:
            raise ValueError("nhead must be positive and divide d_model exactly")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        for name, value in (
            ("mid_reconstruction_weight", self.mid_reconstruction_weight),
            ("classification_weight", self.classification_weight),
            ("kl_weight", self.kl_weight),
            ("free_bits", self.free_bits),
        ):
            if value < 0:
                raise ValueError(f"{name} cannot be negative")


@dataclass(frozen=True, slots=True)
class ExperimentConfig:
    """Complete configuration for one fold and one random seed."""

    data: DataConfig
    output_dir: str
    run_name: str
    pose_block_mode: Literal["vae", "vqvae"] = "vae"
    pose_block: PoseBlockConfig = field(default_factory=PoseBlockConfig)
    pose_vqvae: VQVAEConfig = field(default_factory=VQVAEConfig)
    pose_training: PoseTrainingConfig = field(default_factory=PoseTrainingConfig)
    action_block: ActionBlockConfig = field(default_factory=ActionBlockConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    seed: int = 42
    device: str = "auto"

    def validate(self) -> None:
        self.data.validate()
        self.pose_block.validate()
        self.pose_vqvae.validate()
        self.pose_training.validate()
        self.action_block.validate()
        self.training.validate()
        if not self.run_name.strip():
            raise ValueError("run_name cannot be empty")

        if self.pose_block_mode == "vae":
            clip_length = self.pose_block.clip_length
        elif self.pose_block_mode == "vqvae":
            # A clip of exactly this many frames downsamples to precisely one VQ-VAE
            # token by construction (that is what temporal_downsample_factor means),
            # matching the "one mid-level vector per clip" convention the action block
            # expects regardless of which pose_block_mode produced it.
            clip_length = self.pose_vqvae.temporal_downsample_factor
        else:
            raise ValueError(f"Unknown pose_block_mode: {self.pose_block_mode!r}")

        if self.data.num_time_points % clip_length != 0:
            raise ValueError(
                f"num_time_points ({self.data.num_time_points}) must be divisible by "
                f"the clip length ({clip_length})"
            )
        if self.data.num_time_points // clip_length < 2:
            raise ValueError(
                "num_time_points must fit at least two clips, so the action block has "
                "a real sequence to attend over"
            )


__all__ = [
    "ActionBlockConfig",
    "DataConfig",
    "ExperimentConfig",
    "PoseBlockConfig",
    "PoseTrainingConfig",
    "TrainingConfig",
    "VQVAEConfig",
]
