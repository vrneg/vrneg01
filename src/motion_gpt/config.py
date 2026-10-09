"""Configuration objects for MotionGPT experiments.

MotionGPT treats motion as a language: a VQ-VAE turns windows into discrete tokens, those
tokens share one vocabulary with text, and an encoder-decoder transformer is trained on a
mixture of motion-language tasks before being instruction-tuned on the target task.

Stage one is shared with the T2M-GPT package, so ``DataConfig``, ``VQVAEConfig``, and
``VQVAETrainingConfig`` are re-exported rather than redefined.  ``TrainingConfig`` is also
shared, which keeps optimization, early stopping, and thresholds identical between the two
models.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

try:
    from t2m_gpt.config import (
        MODALITY_CHOICES,
        DataConfig,
        TrainingConfig,
        VQVAEConfig,
        VQVAETrainingConfig,
    )
except ModuleNotFoundError as error:
    if error.name != "t2m_gpt":
        raise
    from ..t2m_gpt.config import (
        MODALITY_CHOICES,
        DataConfig,
        TrainingConfig,
        VQVAEConfig,
        VQVAETrainingConfig,
    )


# The self-supervised motion-language tasks MotionGPT mixes during pretraining.
PRETRAINING_TASKS: tuple[str, ...] = ("denoise", "predict", "inbetween")
# The supervised task. Motion-to-text answers with a label word; the classification task
# reads a pooled encoder state instead.
SUPERVISED_TASK = "classify"
ALL_TASKS: tuple[str, ...] = (*PRETRAINING_TASKS, SUPERVISED_TASK)
# Answer words indexed by label: position 0 is the negative class.
ANSWER_WORDS: tuple[str, str] = ("none", "negation")


@dataclass(frozen=True, slots=True)
class MotionLanguageConfig:
    """Encoder-decoder architecture over the unified motion/text vocabulary.

    ``head`` selects how a decision is produced:

    ``"motion_to_text"``
        MotionGPT's motion-to-text task with a two-word caption.  The decoder answers
        with a label word and the score is the log-odds between the two answer tokens.

    ``"discriminative"``
        A mean-pooled encoder state feeds a binary classifier.  The encoder is
        bidirectional, which suits classification better than a causal stack, at the cost
        of leaving MotionGPT's generative formulation behind.
    """

    head: Literal["motion_to_text", "discriminative"] = "motion_to_text"
    d_model: int = 128
    nhead: int = 4
    num_encoder_layers: int = 2
    num_decoder_layers: int = 2
    dim_feedforward: int = 512
    dropout: float = 0.2
    classifier_hidden_dim: int = 64
    tie_word_embeddings: bool = True
    max_sequence_length: int = 64
    num_sentinels: int = 8

    def validate(self) -> None:
        if self.d_model < 1:
            raise ValueError("d_model must be positive")
        if self.nhead < 1 or self.d_model % self.nhead != 0:
            raise ValueError("nhead must be positive and divide d_model exactly")
        for name, value in (
            ("num_encoder_layers", self.num_encoder_layers),
            ("num_decoder_layers", self.num_decoder_layers),
            ("dim_feedforward", self.dim_feedforward),
            ("classifier_hidden_dim", self.classifier_hidden_dim),
            ("max_sequence_length", self.max_sequence_length),
        ):
            if value < 1:
                raise ValueError(f"{name} must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if self.num_sentinels < 1:
            raise ValueError("num_sentinels must be at least 1")


@dataclass(frozen=True, slots=True)
class InstructionPretrainingConfig:
    """Self-supervised motion-language pretraining on the training split only.

    Every task is expressed as one sequence-to-sequence problem over motion tokens, which
    is what lets a single model absorb all of them:

    ``denoise``
        Spans of motion tokens are replaced by sentinels and the decoder restores them.
        This is T5-style span corruption applied to motion.

    ``predict``
        The decoder forecasts the tail of the window from its head.

    ``inbetween``
        The decoder fills a removed middle segment from both surrounding contexts.

    On a dataset of a few hundred windows these tasks matter more than they do at the
    paper's scale: they multiply the supervision extracted from each window without
    needing any label.
    """

    num_epochs: int = 40
    batch_size: int = 32
    learning_rate: float = 3e-4
    weight_decay: float = 1e-2
    gradient_clip_norm: float | None = 1.0
    tasks: tuple[str, ...] = PRETRAINING_TASKS
    # Fraction of motion tokens hidden by span corruption.
    span_corruption_rate: float = 0.25
    mean_span_length: float = 2.0
    # Fraction of the window the ``predict`` task keeps as visible context.
    prediction_context_fraction: float = 0.5
    # Fraction of the window the ``inbetween`` task removes from the middle.
    inbetween_span_fraction: float = 0.25

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
        if not self.tasks:
            raise ValueError("at least one pretraining task is required")
        unknown = sorted(set(self.tasks) - set(PRETRAINING_TASKS))
        if unknown:
            raise ValueError(f"Unknown pretraining tasks: {unknown}")
        if not 0.0 < self.span_corruption_rate < 1.0:
            raise ValueError("span_corruption_rate must be in (0, 1)")
        if self.mean_span_length < 1.0:
            raise ValueError("mean_span_length must be at least 1")
        if not 0.0 < self.prediction_context_fraction < 1.0:
            raise ValueError("prediction_context_fraction must be in (0, 1)")
        if not 0.0 < self.inbetween_span_fraction < 1.0:
            raise ValueError("inbetween_span_fraction must be in (0, 1)")


@dataclass(frozen=True, slots=True)
class ExperimentConfig:
    """Complete configuration for one fold and one random seed."""

    data: DataConfig
    output_dir: str
    run_name: str
    vqvae: VQVAEConfig = field(default_factory=VQVAEConfig)
    vqvae_training: VQVAETrainingConfig = field(default_factory=VQVAETrainingConfig)
    model: MotionLanguageConfig = field(default_factory=MotionLanguageConfig)
    pretraining: InstructionPretrainingConfig = field(
        default_factory=InstructionPretrainingConfig
    )
    training: TrainingConfig = field(default_factory=TrainingConfig)
    use_pretraining: bool = True
    seed: int = 42
    device: str = "auto"

    def validate(self) -> None:
        self.data.validate()
        self.vqvae.validate()
        self.vqvae_training.validate()
        self.model.validate()
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
        num_tokens = self.data.num_time_points // factor
        if num_tokens < 2:
            raise ValueError(
                "MotionGPT needs at least two motion tokens per window; increase "
                "num_time_points or reduce num_downsample_layers"
            )
        # Encoder sequences are one task token, the motion span, and two boundary markers.
        if num_tokens + 3 > self.model.max_sequence_length:
            raise ValueError(
                f"max_sequence_length ({self.model.max_sequence_length}) is too small "
                f"for {num_tokens} motion tokens"
            )


__all__ = [
    "ALL_TASKS",
    "ANSWER_WORDS",
    "MODALITY_CHOICES",
    "PRETRAINING_TASKS",
    "SUPERVISED_TASK",
    "DataConfig",
    "ExperimentConfig",
    "InstructionPretrainingConfig",
    "MotionLanguageConfig",
    "TrainingConfig",
    "VQVAEConfig",
    "VQVAETrainingConfig",
]
