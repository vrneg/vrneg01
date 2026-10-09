"""Configuration objects for T2M-GPT motion-token experiments.

The package follows the project convention of Python configuration objects instead of
a command-line interface.  Build these dataclasses in a script and pass the resulting
``ExperimentConfig`` to :func:`src.t2m_gpt.training.train_t2m_gpt`.

Two stages are configured separately because they optimize different objectives:
``VQVAEConfig``/``VQVAETrainingConfig`` describe the self-supervised motion tokenizer,
and ``GPTConfig``/``PretrainingConfig``/``TrainingConfig`` describe the transformer that
consumes its discrete codes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

try:
    from representation import IDENTITY_REPRESENTATION, RepresentationConfig
except ModuleNotFoundError as error:  # pragma: no cover - import-path fallback
    if error.name != "representation":
        raise
    from ..representation import IDENTITY_REPRESENTATION, RepresentationConfig


MODALITY_CHOICES: tuple[str, ...] = (
    "Eye",
    "Facial",
    "Head",
    "Body",
    "LeftHand",
    "RightHand",
    "LeftFinger",
    "RightFinger",
)


@dataclass(frozen=True, slots=True)
class DataConfig:
    """Fold source and fixed-grid motion representation.

    ``dataset`` is resolved in this order: a local ``save_to_disk`` directory, a local
    directory of parquet files, then a Hugging Face Hub dataset repository id.  Loading
    from the Hub uses the ordinary datasets cache, so no copy is written into the
    repository.

    The grid, event subsampling, actor scope, and presence channels match the
    representation used by the ROCKET and TCN baselines, so results stay comparable.
    ``modalities`` optionally restricts the tokenizer to a subset of the eight streams;
    ``None`` keeps all of them.
    """

    dataset: str
    num_time_points: int = 32
    window_start_seconds: float = -0.5
    window_end_seconds: float = 0.5
    max_events_per_modality: int | None = 128
    normalize_features: bool = True
    cache_in_memory: bool = True
    actor_scope: Literal["anchor", "other", "all"] = "anchor"
    include_presence_channels: bool = True
    modalities: tuple[str, ...] | None = None
    positive_label: str = "neg"
    negative_label: str = "none"
    representation: RepresentationConfig = IDENTITY_REPRESENTATION
    # Doubles the training split with a left-right mirrored copy of every window
    # (see representation.mirror); off by default so existing results stay
    # reproducible. Validation and test are never mirrored, so held-out metrics
    # stay comparable across configs with this on or off.
    mirror_augment_train: bool = False

    def validate(self) -> None:
        if not str(self.dataset).strip():
            raise ValueError("dataset cannot be empty")
        if self.num_time_points < 1:
            raise ValueError("num_time_points must be positive")
        if self.window_end_seconds <= self.window_start_seconds:
            raise ValueError("window_end_seconds must exceed window_start_seconds")
        if self.max_events_per_modality is not None and self.max_events_per_modality < 1:
            raise ValueError("max_events_per_modality must be at least 1 or None")
        if self.modalities is not None:
            if not self.modalities:
                raise ValueError("modalities must be None or a non-empty selection")
            unknown = sorted(set(self.modalities) - set(MODALITY_CHOICES))
            if unknown:
                raise ValueError(f"Unknown modalities: {unknown}")
            if len(set(self.modalities)) != len(self.modalities):
                raise ValueError("modalities must not contain duplicates")
        if self.positive_label == self.negative_label:
            raise ValueError("positive_label and negative_label must differ")
        self.representation.validate()
        if (
            self.representation.requires_presence_channels
            and not self.include_presence_channels
        ):
            raise ValueError(
                "Derived representation channels need include_presence_channels=True: "
                "interpolation fills unobserved frames with zeros, and without the "
                "presence channel those are indistinguishable from real observations "
                "sitting at the training mean"
            )
        if self.representation.root_relative_positions:
            reference = self.representation.root_reference
            if self.modalities is not None and reference not in self.modalities:
                raise ValueError(
                    f"root_relative_positions uses {reference!r} as its reference, so "
                    f"{reference!r} must be among the selected modalities"
                )
        if self.representation.append_action_units:
            if self.modalities is not None and "Facial" not in self.modalities:
                raise ValueError(
                    "append_action_units needs the Facial modality to be selected"
                )
        if self.representation.acceleration_modalities is not None:
            unknown = sorted(
                set(self.representation.acceleration_modalities) - set(MODALITY_CHOICES)
            )
            if unknown:
                raise ValueError(f"Unknown acceleration_modalities: {unknown}")
            if self.modalities is not None:
                missing = sorted(
                    set(self.representation.acceleration_modalities)
                    - set(self.modalities)
                )
                if missing:
                    raise ValueError(
                        f"acceleration_modalities not among selected modalities: {missing}"
                    )


@dataclass(frozen=True, slots=True)
class VQVAEConfig:
    """Motion VQ-VAE architecture.

    The layout follows T2M-GPT: a strided 1D convolutional encoder with dilated
    residual stacks, a discrete bottleneck, and a mirrored decoder.  Each downsampling
    layer halves the temporal resolution, so a window of ``num_time_points`` frames
    becomes ``num_time_points / 2 ** num_downsample_layers`` motion tokens.
    """

    num_codes: int = 256
    code_dim: int = 128
    width: int = 256
    num_downsample_layers: int = 2
    num_residual_blocks: int = 2
    dilation_growth_rate: int = 3
    activation: Literal["relu", "gelu"] = "relu"
    dropout: float = 0.0
    codebook_decay: float = 0.99
    codebook_epsilon: float = 1e-5
    # Codes used by fewer than this many vectors in a batch are re-seeded from the
    # encoder output.  This is the "code reset" half of T2M-GPT's EMA + reset recipe.
    code_reset_threshold: float = 1.0

    @property
    def temporal_downsample_factor(self) -> int:
        return 2**self.num_downsample_layers

    def validate(self) -> None:
        if self.num_codes < 2:
            raise ValueError("num_codes must be at least 2")
        for name, value in (
            ("code_dim", self.code_dim),
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
        if not 0.0 < self.codebook_decay < 1.0:
            raise ValueError("codebook_decay must be in (0, 1)")
        if self.codebook_epsilon <= 0:
            raise ValueError("codebook_epsilon must be positive")
        if self.code_reset_threshold < 0:
            raise ValueError("code_reset_threshold cannot be negative")


@dataclass(frozen=True, slots=True)
class VQVAETrainingConfig:
    """Stage-one reconstruction objective and optimization.

    The loss is T2M-GPT's: a smooth-L1 reconstruction term, a smooth-L1 term on the
    temporal first difference that penalizes over-smoothed velocity, and the commitment
    term of the discrete bottleneck.  Only the training split of the current fold is
    used, so the tokenizer never sees validation or test windows.
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
    commitment_loss_weight: float = 0.02
    early_stopping_patience: int = 20
    early_stopping_min_delta: float = 1e-5
    lr_scheduler: Literal["none", "reduce_on_plateau"] = "reduce_on_plateau"
    lr_scheduler_factor: float = 0.5
    lr_scheduler_patience: int = 8
    minimum_learning_rate: float = 1e-6

    def validate(self) -> None:
        if self.max_epochs < 1:
            raise ValueError("vqvae max_epochs must be at least 1")
        for name, value in (
            ("batch_size", self.batch_size),
            ("evaluation_batch_size", self.evaluation_batch_size),
        ):
            if value < 1:
                raise ValueError(f"vqvae {name} must be at least 1")
        if self.num_workers < 0:
            raise ValueError("vqvae num_workers cannot be negative")
        if self.learning_rate <= 0:
            raise ValueError("vqvae learning_rate must be positive")
        if self.weight_decay < 0:
            raise ValueError("vqvae weight_decay cannot be negative")
        if self.gradient_clip_norm is not None and self.gradient_clip_norm <= 0:
            raise ValueError("vqvae gradient_clip_norm must be positive or None")
        if self.velocity_loss_weight < 0:
            raise ValueError("velocity_loss_weight cannot be negative")
        if self.commitment_loss_weight < 0:
            raise ValueError("commitment_loss_weight cannot be negative")
        if self.early_stopping_patience < 1:
            raise ValueError("vqvae early_stopping_patience must be at least 1")
        if self.early_stopping_min_delta < 0:
            raise ValueError("vqvae early_stopping_min_delta cannot be negative")
        if not 0.0 < self.lr_scheduler_factor < 1.0:
            raise ValueError("vqvae lr_scheduler_factor must be in (0, 1)")
        if self.lr_scheduler_patience < 1:
            raise ValueError("vqvae lr_scheduler_patience must be at least 1")
        if self.minimum_learning_rate < 0:
            raise ValueError("vqvae minimum_learning_rate cannot be negative")
        if self.minimum_learning_rate >= self.learning_rate:
            raise ValueError("vqvae minimum_learning_rate must be below learning_rate")


@dataclass(frozen=True, slots=True)
class GPTConfig:
    """Stage-two transformer over motion tokens.

    ``head`` selects how the token sequence produces a decision:

    ``"discriminative"``
        A ``[BOS]`` token is prepended, the causal transformer reads the sequence, and
        the final position is pooled into one binary logit.  Autoregressive
        pretraining weights transfer directly into this model.

    ``"generative"``
        The class label replaces T2M-GPT's text condition: the model learns
        ``p(tokens | class)`` and classifies by the log-likelihood ratio between the
        two class prefixes.  This mirrors the original conditional formulation and
        needs no separate classifier head.
    """

    head: Literal["discriminative", "generative"] = "discriminative"
    d_model: int = 128
    nhead: int = 4
    num_layers: int = 3
    dim_feedforward: int = 512
    dropout: float = 0.2
    classifier_hidden_dim: int = 64
    causal: bool = True
    pooling: Literal["last", "mean"] = "last"
    # Fraction of input tokens replaced by uniformly random codes during stage-two
    # training. T2M-GPT calls this the corrupted-sequence strategy; it reduces the
    # mismatch between teacher forcing and inference on generated prefixes.
    token_corruption_rate: float = 0.1

    def validate(self) -> None:
        if self.d_model < 1:
            raise ValueError("d_model must be positive")
        if self.nhead < 1 or self.d_model % self.nhead != 0:
            raise ValueError("nhead must be positive and divide d_model exactly")
        if self.num_layers < 1:
            raise ValueError("num_layers must be at least 1")
        for name, value in (
            ("dim_feedforward", self.dim_feedforward),
            ("classifier_hidden_dim", self.classifier_hidden_dim),
        ):
            if value < 1:
                raise ValueError(f"{name} must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if not 0.0 <= self.token_corruption_rate < 1.0:
            raise ValueError("token_corruption_rate must be in [0, 1)")
        if self.head == "generative" and not self.causal:
            raise ValueError("the generative head requires causal attention")
        if self.head == "generative" and self.pooling != "last":
            raise ValueError("the generative head does not pool hidden states")


@dataclass(frozen=True, slots=True)
class PretrainingConfig:
    """Unconditional next-token pretraining for the stage-two transformer.

    Pretraining predicts each motion token from its predecessors using only the
    current fold's training split.  It gives the discriminative head an initialization
    that already models motion-token structure.  The generative head is trained
    autoregressively by definition, so pretraining is skipped for it.
    """

    num_epochs: int = 30
    batch_size: int = 32
    learning_rate: float = 3e-4
    weight_decay: float = 1e-2
    gradient_clip_norm: float | None = 1.0
    token_corruption_rate: float = 0.1

    def validate(self) -> None:
        if self.num_epochs < 1:
            raise ValueError("pretraining num_epochs must be at least 1")
        if self.batch_size < 1:
            raise ValueError("pretraining batch_size must be at least 1")
        if self.learning_rate <= 0:
            raise ValueError("pretraining learning_rate must be positive")
        if self.weight_decay < 0:
            raise ValueError("pretraining weight_decay cannot be negative")
        if self.gradient_clip_norm is not None and self.gradient_clip_norm <= 0:
            raise ValueError("pretraining gradient_clip_norm must be positive or None")
        if not 0.0 <= self.token_corruption_rate < 1.0:
            raise ValueError("pretraining token_corruption_rate must be in [0, 1)")


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    """Stage-two optimization, early stopping, and decision thresholds."""

    max_epochs: int = 100
    batch_size: int = 32
    evaluation_batch_size: int = 64
    num_workers: int = 0
    learning_rate: float = 3e-4
    classifier_learning_rate: float | None = None
    freeze_backbone_epochs: int = 0
    weight_decay: float = 1e-2
    gradient_clip_norm: float | None = 1.0
    early_stopping_patience: int = 15
    early_stopping_min_delta: float = 1e-4
    selection_metric: Literal[
        "loss",
        "accuracy",
        "balanced_accuracy",
        "precision",
        "recall",
        "specificity",
        "f1",
        "negative_f1",
        "macro_f1",
        "weighted_f1",
        "matthews_correlation_coefficient",
        "roc_auc",
        "average_precision",
    ] = "loss"
    positive_class_weight: float | None = None
    threshold: float = 0.5
    calibrate_threshold_on_validation: bool = False
    threshold_metric: Literal[
        "accuracy",
        "balanced_accuracy",
        "precision",
        "recall",
        "specificity",
        "f1",
        "negative_f1",
        "macro_f1",
        "weighted_f1",
        "matthews_correlation_coefficient",
    ] = "macro_f1"
    lr_scheduler: Literal["none", "reduce_on_plateau"] = "reduce_on_plateau"
    lr_scheduler_factor: float = 0.5
    lr_scheduler_patience: int = 5
    minimum_learning_rate: float = 1e-6
    mixed_precision: bool = False
    deterministic_algorithms: bool = False
    show_progress: bool = True

    def validate(self) -> None:
        if self.max_epochs < 1:
            raise ValueError("max_epochs must be at least 1")
        for name, value in (
            ("batch_size", self.batch_size),
            ("evaluation_batch_size", self.evaluation_batch_size),
        ):
            if value < 1:
                raise ValueError(f"{name} must be at least 1")
        if self.num_workers < 0:
            raise ValueError("num_workers cannot be negative")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if (
            self.classifier_learning_rate is not None
            and self.classifier_learning_rate <= 0
        ):
            raise ValueError("classifier_learning_rate must be positive or None")
        if self.freeze_backbone_epochs < 0:
            raise ValueError("freeze_backbone_epochs cannot be negative")
        if self.weight_decay < 0:
            raise ValueError("weight_decay cannot be negative")
        if self.gradient_clip_norm is not None and self.gradient_clip_norm <= 0:
            raise ValueError("gradient_clip_norm must be positive or None")
        if self.early_stopping_patience < 1:
            raise ValueError("early_stopping_patience must be at least 1")
        if self.early_stopping_min_delta < 0:
            raise ValueError("early_stopping_min_delta cannot be negative")
        if self.positive_class_weight is not None and self.positive_class_weight <= 0:
            raise ValueError("positive_class_weight must be positive or None")
        if not 0.0 < self.threshold < 1.0:
            raise ValueError("threshold must be strictly between 0 and 1")
        if not 0.0 < self.lr_scheduler_factor < 1.0:
            raise ValueError("lr_scheduler_factor must be in (0, 1)")
        if self.lr_scheduler_patience < 1:
            raise ValueError("lr_scheduler_patience must be at least 1")
        if self.minimum_learning_rate < 0:
            raise ValueError("minimum_learning_rate cannot be negative")
        if self.minimum_learning_rate >= self.learning_rate:
            raise ValueError("minimum_learning_rate must be below learning_rate")
        if (
            self.classifier_learning_rate is not None
            and self.minimum_learning_rate >= self.classifier_learning_rate
        ):
            raise ValueError(
                "minimum_learning_rate must be below classifier_learning_rate"
            )


@dataclass(frozen=True, slots=True)
class ExperimentConfig:
    """Complete configuration for one fold and one random seed."""

    data: DataConfig
    output_dir: str
    run_name: str
    vqvae: VQVAEConfig = field(default_factory=VQVAEConfig)
    vqvae_training: VQVAETrainingConfig = field(default_factory=VQVAETrainingConfig)
    gpt: GPTConfig = field(default_factory=GPTConfig)
    pretraining: PretrainingConfig = field(default_factory=PretrainingConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    use_pretraining: bool = True
    seed: int = 42
    device: str = "auto"

    def validate(self) -> None:
        self.data.validate()
        self.vqvae.validate()
        self.vqvae_training.validate()
        self.gpt.validate()
        self.pretraining.validate()
        self.training.validate()
        if not self.run_name.strip():
            raise ValueError("run_name cannot be empty")
        factor = self.vqvae.temporal_downsample_factor
        if self.data.num_time_points % factor != 0:
            raise ValueError(
                f"num_time_points ({self.data.num_time_points}) must be divisible by "
                f"the temporal downsample factor ({factor})"
            )
        if self.data.num_time_points // factor < 1:
            raise ValueError(
                "num_time_points is too small for the requested downsampling"
            )
