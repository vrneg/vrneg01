"""Span-level and frame-level scoring for the frame tagger.

Frame accuracy is close to useless here: roughly 80% of frames are ``O``, so predicting
``O`` everywhere already scores 0.8 while finding nothing. The metrics that matter are
span-level.

Exact versus overlap matching
-----------------------------
Exact span matching demands identical start and end frames. That is the standard NER
criterion, but it is too harsh for *these* labels: three quarters of the annotated spans
have an assumed rather than a measured offset (see :mod:`.labels`), and the grid quantizes
time to ~32ms, so a perfect prediction can still miss the reference end frame. Overlap
matching -- a prediction counts when it overlaps a same-type reference by at least
``minimum_overlap`` intersection-over-union -- measures what the model can actually be held
to. Both are reported so the gap between them is visible rather than hidden by a choice of
criterion.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from .labels import OUTSIDE, SPAN_TYPES, decode_spans


@dataclass(slots=True)
class SpanScore:
    """Precision, recall, and F1 for one span type or one micro-average."""

    true_positives: int = 0
    false_positives: int = 0
    false_negatives: int = 0

    @property
    def precision(self) -> float:
        denominator = self.true_positives + self.false_positives
        return self.true_positives / denominator if denominator else 0.0

    @property
    def recall(self) -> float:
        denominator = self.true_positives + self.false_negatives
        return self.true_positives / denominator if denominator else 0.0

    @property
    def f1(self) -> float:
        precision, recall = self.precision, self.recall
        return 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    def as_dict(self) -> dict[str, float]:
        return {
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "true_positives": self.true_positives,
            "false_positives": self.false_positives,
            "false_negatives": self.false_negatives,
        }


def _intersection_over_union(
    first: tuple[str, int, int], second: tuple[str, int, int]
) -> float:
    start = max(first[1], second[1])
    stop = min(first[2], second[2])
    intersection = max(0, stop - start)
    if intersection == 0:
        return 0.0
    union = max(first[2], second[2]) - min(first[1], second[1])
    return intersection / union if union else 0.0


def score_spans(
    predicted: Sequence[Sequence[tuple[str, int, int]]],
    reference: Sequence[Sequence[tuple[str, int, int]]],
    minimum_overlap: float | None = None,
) -> dict[str, SpanScore]:
    """Match predicted spans to reference spans, per type and micro-averaged.

    ``minimum_overlap`` of ``None`` requires exact boundary equality; a float in ``(0, 1]``
    accepts the best-overlapping unmatched reference span of the same type when their
    intersection-over-union reaches it. Matching is greedy by descending overlap, and each
    reference span can be claimed only once, so duplicate predictions covering one
    reference are counted as false positives rather than silently forgiven.
    """

    if len(predicted) != len(reference):
        raise ValueError("predicted and reference must describe the same windows")
    scores = {span_type: SpanScore() for span_type in SPAN_TYPES}

    for predicted_spans, reference_spans in zip(predicted, reference, strict=True):
        unmatched = list(reference_spans)
        for span in predicted_spans:
            span_type = span[0]
            if span_type not in scores:
                scores[span_type] = SpanScore()
            best_index, best_overlap = None, 0.0
            for index, candidate in enumerate(unmatched):
                if candidate[0] != span_type:
                    continue
                if minimum_overlap is None:
                    if candidate[1] == span[1] and candidate[2] == span[2]:
                        best_index, best_overlap = index, 1.0
                        break
                    continue
                overlap = _intersection_over_union(span, candidate)
                if overlap >= minimum_overlap and overlap > best_overlap:
                    best_index, best_overlap = index, overlap
            if best_index is None:
                scores[span_type].false_positives += 1
            else:
                scores[span_type].true_positives += 1
                unmatched.pop(best_index)
        for candidate in unmatched:
            if candidate[0] not in scores:
                scores[candidate[0]] = SpanScore()
            scores[candidate[0]].false_negatives += 1

    micro = SpanScore()
    for score in scores.values():
        micro.true_positives += score.true_positives
        micro.false_positives += score.false_positives
        micro.false_negatives += score.false_negatives
    scores["micro"] = micro
    return scores


def frame_metrics(predicted: np.ndarray, reference: np.ndarray) -> dict[str, float]:
    """Binary in-span/out-of-span frame metrics, plus exact tag accuracy."""

    if predicted.shape != reference.shape:
        raise ValueError("predicted and reference must have the same shape")
    predicted_positive = predicted != OUTSIDE
    reference_positive = reference != OUTSIDE

    true_positive = int((predicted_positive & reference_positive).sum())
    false_positive = int((predicted_positive & ~reference_positive).sum())
    false_negative = int((~predicted_positive & reference_positive).sum())
    true_negative = int((~predicted_positive & ~reference_positive).sum())

    precision = (
        true_positive / (true_positive + false_positive)
        if true_positive + false_positive
        else 0.0
    )
    recall = (
        true_positive / (true_positive + false_negative)
        if true_positive + false_negative
        else 0.0
    )
    positive_support = true_positive + false_negative
    negative_support = true_negative + false_positive
    negative_recall = true_negative / negative_support if negative_support else 0.0
    return {
        "frame_tag_accuracy": float((predicted == reference).mean()),
        "frame_precision": precision,
        "frame_recall": recall,
        "frame_f1": (
            2 * precision * recall / (precision + recall) if precision + recall else 0.0
        ),
        "frame_balanced_accuracy": (recall + negative_recall) / 2.0,
        "frame_positive_support": positive_support,
    }


@dataclass(slots=True)
class TaggerMetrics:
    """The full metric bundle reported for one split."""

    frame: dict[str, float] = field(default_factory=dict)
    exact: dict[str, dict[str, float]] = field(default_factory=dict)
    overlap: dict[str, dict[str, float]] = field(default_factory=dict)
    minimum_overlap: float = 0.5

    def as_dict(self) -> dict[str, object]:
        return {
            "frame": self.frame,
            "span_exact": self.exact,
            "span_overlap": self.overlap,
            "minimum_overlap": self.minimum_overlap,
        }

    @property
    def primary(self) -> float:
        """Overlap micro F1 -- the metric used for model selection."""

        return float(self.overlap.get("micro", {}).get("f1", 0.0))


def evaluate_tagging(
    predicted: np.ndarray,
    reference: np.ndarray,
    minimum_overlap: float = 0.5,
) -> TaggerMetrics:
    """Score decoded tag arrays of shape ``[windows, frames]``."""

    predicted_spans = [decode_spans(row) for row in predicted]
    reference_spans = [decode_spans(row) for row in reference]
    return TaggerMetrics(
        frame=frame_metrics(predicted, reference),
        exact={
            name: score.as_dict()
            for name, score in score_spans(predicted_spans, reference_spans, None).items()
        },
        overlap={
            name: score.as_dict()
            for name, score in score_spans(
                predicted_spans, reference_spans, minimum_overlap
            ).items()
        },
        minimum_overlap=minimum_overlap,
    )


__all__ = [
    "SpanScore",
    "TaggerMetrics",
    "evaluate_tagging",
    "frame_metrics",
    "score_spans",
]
