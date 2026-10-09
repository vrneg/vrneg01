"""Motion-language task construction: every task is one sequence-to-sequence problem.

MotionGPT trains one model on several motion-language tasks by phrasing each as an
instruction-prefixed sequence pair.  This module builds those pairs, batches them with the
padding and label masking a seq2seq objective needs, and resamples the self-supervised
corruptions each epoch.
"""

from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch
from torch.utils.data import Dataset

from .config import InstructionPretrainingConfig
from .vocabulary import MotionLanguageVocabulary

# Positions excluded from the sequence-to-sequence loss.
LABEL_IGNORE_INDEX = -100


@dataclass(slots=True)
class TaskExample:
    """One instruction-prefixed sequence pair."""

    encoder_input: list[int]
    decoder_target: list[int]
    task: str
    label: float
    sample_id: str


def _motion_block(
    vocabulary: MotionLanguageVocabulary,
    task: str,
    body: Sequence[int],
) -> list[int]:
    """Wrap a motion body in its instruction and boundary markers."""

    return [
        vocabulary.task_id(task),
        vocabulary.start_of_motion_id,
        *body,
        vocabulary.end_of_motion_id,
    ]


def build_classify_example(
    vocabulary: MotionLanguageVocabulary,
    tokens: Sequence[int],
    label: float,
    sample_id: str,
) -> TaskExample:
    """Motion-to-text with a two-word caption: the decoder answers with a label word."""

    return TaskExample(
        encoder_input=_motion_block(vocabulary, "classify", tokens),
        decoder_target=[vocabulary.answer_id(int(label)), vocabulary.eos_id],
        task="classify",
        label=float(label),
        sample_id=sample_id,
    )


def sample_corruption_spans(
    num_tokens: int,
    corruption_rate: float,
    mean_span_length: float,
    num_sentinels: int,
    rng: random.Random,
) -> list[tuple[int, int]]:
    """Choose non-adjacent ``(start, length)`` spans to hide.

    At least one token stays visible and at least one span is always produced, so the
    encoder never receives an empty motion body and the decoder always has a target.
    """

    if num_tokens < 2:
        raise ValueError("span corruption needs at least two motion tokens")

    num_masked = max(1, min(num_tokens - 1, round(num_tokens * corruption_rate)))
    num_spans = max(
        1, min(num_sentinels, num_masked, round(num_masked / mean_span_length) or 1)
    )
    base, extra = divmod(num_masked, num_spans)
    lengths = [base + (1 if index < extra else 0) for index in range(num_spans)]
    rng.shuffle(lengths)

    spans: list[tuple[int, int]] = []
    occupied: set[int] = set()
    for length in lengths:
        candidates = [
            start
            for start in range(num_tokens - length + 1)
            # Spans stay disjoint and non-adjacent so each sentinel marks a real gap.
            if not any(
                position in occupied for position in range(start - 1, start + length + 1)
            )
        ]
        if not candidates:
            continue
        start = rng.choice(candidates)
        spans.append((start, length))
        occupied.update(range(start, start + length))

    if not spans:
        # Every placement was blocked; fall back to hiding a single token.
        start = rng.randrange(num_tokens)
        spans = [(start, 1)]
    return sorted(spans)


def build_denoise_example(
    vocabulary: MotionLanguageVocabulary,
    tokens: Sequence[int],
    label: float,
    sample_id: str,
    config: InstructionPretrainingConfig,
    rng: random.Random,
) -> TaskExample:
    """T5-style span corruption over motion tokens."""

    spans = sample_corruption_spans(
        len(tokens),
        config.span_corruption_rate,
        config.mean_span_length,
        vocabulary.num_sentinels,
        rng,
    )
    starts = {start: (index, length) for index, (start, length) in enumerate(spans)}
    masked_positions = {
        position
        for start, length in spans
        for position in range(start, start + length)
    }

    body: list[int] = []
    target: list[int] = []
    for position, token in enumerate(tokens):
        if position in starts:
            sentinel_index, length = starts[position]
            body.append(vocabulary.sentinel_id(sentinel_index))
            target.append(vocabulary.sentinel_id(sentinel_index))
            target.extend(tokens[position : position + length])
        elif position not in masked_positions:
            body.append(token)
    target.append(vocabulary.eos_id)

    return TaskExample(
        encoder_input=_motion_block(vocabulary, "denoise", body),
        decoder_target=target,
        task="denoise",
        label=float(label),
        sample_id=sample_id,
    )


def build_predict_example(
    vocabulary: MotionLanguageVocabulary,
    tokens: Sequence[int],
    label: float,
    sample_id: str,
    config: InstructionPretrainingConfig,
) -> TaskExample:
    """Forecast the tail of the window from its head."""

    split = max(1, min(len(tokens) - 1, round(len(tokens) * config.prediction_context_fraction)))
    return TaskExample(
        encoder_input=_motion_block(vocabulary, "predict", tokens[:split]),
        decoder_target=[*tokens[split:], vocabulary.eos_id],
        task="predict",
        label=float(label),
        sample_id=sample_id,
    )


def build_inbetween_example(
    vocabulary: MotionLanguageVocabulary,
    tokens: Sequence[int],
    label: float,
    sample_id: str,
    config: InstructionPretrainingConfig,
    rng: random.Random,
) -> TaskExample:
    """Fill a removed middle segment from both surrounding contexts."""

    num_tokens = len(tokens)
    span_length = max(1, min(num_tokens - 2, round(num_tokens * config.inbetween_span_fraction)))
    # Keep at least one token of context on each side.
    start = rng.randrange(1, num_tokens - span_length)
    body = [
        *tokens[:start],
        vocabulary.sentinel_id(0),
        *tokens[start + span_length :],
    ]
    return TaskExample(
        encoder_input=_motion_block(vocabulary, "inbetween", body),
        decoder_target=[*tokens[start : start + span_length], vocabulary.eos_id],
        task="inbetween",
        label=float(label),
        sample_id=sample_id,
    )


def build_pretraining_example(
    task: str,
    vocabulary: MotionLanguageVocabulary,
    tokens: Sequence[int],
    label: float,
    sample_id: str,
    config: InstructionPretrainingConfig,
    rng: random.Random,
) -> TaskExample:
    if task == "denoise":
        return build_denoise_example(vocabulary, tokens, label, sample_id, config, rng)
    if task == "predict":
        return build_predict_example(vocabulary, tokens, label, sample_id, config)
    if task == "inbetween":
        return build_inbetween_example(vocabulary, tokens, label, sample_id, config, rng)
    raise ValueError(f"Unknown pretraining task: {task!r}")


class ClassificationTaskDataset(Dataset[TaskExample]):
    """Deterministic motion-to-text examples for the supervised task."""

    def __init__(
        self,
        indices: torch.Tensor,
        labels: torch.Tensor,
        sample_ids: Sequence[str],
        vocabulary: MotionLanguageVocabulary,
    ) -> None:
        if indices.ndim != 2:
            raise ValueError("indices must have shape [windows, tokens]")
        self.examples = [
            build_classify_example(
                vocabulary,
                indices[position].tolist(),
                float(labels[position].item()),
                sample_ids[position],
            )
            for position in range(indices.shape[0])
        ]

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> TaskExample:
        return self.examples[index]


class PretrainingTaskDataset(Dataset[TaskExample]):
    """Self-supervised examples whose corruption is resampled every epoch.

    ``set_epoch`` mixes the epoch into the per-item seed, so each window is corrupted
    differently across epochs while the whole run stays reproducible from ``seed``.
    """

    def __init__(
        self,
        indices: torch.Tensor,
        labels: torch.Tensor,
        sample_ids: Sequence[str],
        vocabulary: MotionLanguageVocabulary,
        config: InstructionPretrainingConfig,
        seed: int = 42,
    ) -> None:
        if indices.ndim != 2:
            raise ValueError("indices must have shape [windows, tokens]")
        config.validate()
        self.indices = indices
        self.labels = labels
        self.sample_ids = list(sample_ids)
        self.vocabulary = vocabulary
        self.config = config
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return int(self.indices.shape[0])

    def __getitem__(self, index: int) -> TaskExample:
        rng = random.Random(self.seed * 1_000_003 + self.epoch * 10_007 + index)
        task = rng.choice(self.config.tasks)
        return build_pretraining_example(
            task,
            self.vocabulary,
            self.indices[index].tolist(),
            float(self.labels[index].item()),
            self.sample_ids[index],
            self.config,
            rng,
        )


class TaskBatchCollator:
    """Pad encoder and decoder sequences and mask padded label positions."""

    def __init__(self, vocabulary: MotionLanguageVocabulary) -> None:
        self.vocabulary = vocabulary

    def __call__(self, examples: Sequence[TaskExample]) -> dict[str, Any]:
        if not examples:
            raise ValueError("Cannot collate an empty batch")

        pad_id = self.vocabulary.pad_id
        encoder_length = max(len(example.encoder_input) for example in examples)
        decoder_length = max(len(example.decoder_target) for example in examples)

        batch_size = len(examples)
        encoder_input = torch.full((batch_size, encoder_length), pad_id, dtype=torch.long)
        encoder_mask = torch.zeros((batch_size, encoder_length), dtype=torch.bool)
        decoder_input = torch.full((batch_size, decoder_length), pad_id, dtype=torch.long)
        decoder_mask = torch.zeros((batch_size, decoder_length), dtype=torch.bool)
        labels = torch.full(
            (batch_size, decoder_length), LABEL_IGNORE_INDEX, dtype=torch.long
        )

        for position, example in enumerate(examples):
            source = example.encoder_input
            encoder_input[position, : len(source)] = torch.tensor(source, dtype=torch.long)
            encoder_mask[position, : len(source)] = True

            target = example.decoder_target
            # Teacher forcing shifts the target right behind the start token.
            shifted = [self.vocabulary.bos_id, *target[:-1]]
            decoder_input[position, : len(shifted)] = torch.tensor(
                shifted, dtype=torch.long
            )
            decoder_mask[position, : len(shifted)] = True
            labels[position, : len(target)] = torch.tensor(target, dtype=torch.long)

        return {
            "encoder_input": encoder_input,
            "encoder_mask": encoder_mask,
            "decoder_input": decoder_input,
            "decoder_mask": decoder_mask,
            "labels": labels,
            "label": torch.tensor(
                [example.label for example in examples], dtype=torch.float32
            ),
            "sample_id": [example.sample_id for example in examples],
            "task": [example.task for example in examples],
        }


__all__ = [
    "LABEL_IGNORE_INDEX",
    "ClassificationTaskDataset",
    "PretrainingTaskDataset",
    "TaskBatchCollator",
    "TaskExample",
    "build_classify_example",
    "build_denoise_example",
    "build_inbetween_example",
    "build_predict_example",
    "build_pretraining_example",
    "sample_corruption_spans",
]
