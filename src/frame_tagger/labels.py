"""Per-frame BIO labels for negation cue and scope spans.

The window-level datasets carry one binary label per window, but the annotation behind
them is finer: each ``word`` row keeps its negation record in ``word["neg"]``, and that
record lists ``cue_tokens`` and ``scope_tokens`` as SurrealDB ``Word`` references. Those
references are the useful part, because a ``Word`` record id is
``[timeMs, Player, index]`` -- the onset timestamp is literally its first element. Frame
labels can therefore be derived from the fold datasets alone, with no database access.

What is exact and what is assumed
---------------------------------
*Exact*: every token's **onset**, read from its record id, and the anchor word's
**offset**, since the anchor row carries ``duration``/``endTime`` (verified consistent:
``endTime - timeMs == duration * 1000``).

*Assumed*: the **offset of non-anchor tokens**. The token references carry no duration, so
each non-anchor token is given an assumed duration -- by default the median anchor-word
duration of the split, which is estimated from training rows only. This is the one
approximation in this module and it is why :class:`FrameLabelStats` reports how many
tokens used it: a scope span built from three assumed-duration tokens is a coarser target
than a cue span anchored on a measured word.

Tag scheme
----------
Standard BIO over two span types, five tags total: ``O``, ``B-CUE``, ``I-CUE``,
``B-SCOPE``, ``I-SCOPE``. Where a frame falls inside both a cue and a scope span, cue
wins -- cue tokens are the narrower, better-attested annotation, and the datasets in this
repository are built around them.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

TAG_NAMES: tuple[str, ...] = ("O", "B-CUE", "I-CUE", "B-SCOPE", "I-SCOPE")
TAG_TO_ID: dict[str, int] = {name: index for index, name in enumerate(TAG_NAMES)}
OUTSIDE = TAG_TO_ID["O"]

SPAN_TYPES: tuple[str, ...] = ("CUE", "SCOPE")
#: Cue takes precedence where spans overlap.
SPAN_PRIORITY: tuple[str, ...] = ("SCOPE", "CUE")

DEFAULT_ASSUMED_DURATION_MS = 300.0


@dataclass(slots=True)
class TokenSpan:
    """One annotated word occupying ``[start_ms, end_ms)``."""

    span_type: str
    start_ms: float
    end_ms: float
    duration_is_measured: bool


@dataclass(slots=True)
class FrameLabelStats:
    """How the derived frame labels were produced, for reporting alongside results."""

    num_windows: int = 0
    num_windows_with_spans: int = 0
    num_spans: int = 0
    num_measured_durations: int = 0
    num_assumed_durations: int = 0
    num_frames: int = 0
    num_labeled_frames: int = 0
    assumed_duration_ms: float = DEFAULT_ASSUMED_DURATION_MS

    @property
    def positive_frame_fraction(self) -> float:
        return self.num_labeled_frames / self.num_frames if self.num_frames else 0.0

    def as_dict(self) -> dict[str, float]:
        return {
            "num_windows": self.num_windows,
            "num_windows_with_spans": self.num_windows_with_spans,
            "num_spans": self.num_spans,
            "num_measured_durations": self.num_measured_durations,
            "num_assumed_durations": self.num_assumed_durations,
            "num_frames": self.num_frames,
            "num_labeled_frames": self.num_labeled_frames,
            "positive_frame_fraction": self.positive_frame_fraction,
            "assumed_duration_ms": self.assumed_duration_ms,
        }


def _as_mapping(value: Any) -> Mapping[str, Any]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return {}
    return value if isinstance(value, Mapping) else {}


def record_onset_ms(record: Any) -> float | None:
    """Read the onset timestamp out of a SurrealDB ``Word`` record reference.

    A ``Word`` id is the composite ``[timeMs, Player, index]``, so the onset is its first
    element. Anything that does not look like that shape yields ``None`` rather than a
    guess.
    """

    mapping = _as_mapping(record)
    identifier = mapping.get("id")
    if isinstance(identifier, Sequence) and not isinstance(identifier, (str, bytes)):
        if not identifier:
            return None
        candidate = identifier[0]
        if isinstance(candidate, bool):
            return None
        if isinstance(candidate, (int, float)):
            return float(candidate)
    return None


def anchor_duration_ms(word: Mapping[str, Any]) -> float | None:
    """Measured duration of the anchor word, in milliseconds, if it is recorded."""

    duration = word.get("duration")
    if isinstance(duration, (int, float)) and not isinstance(duration, bool):
        if float(duration) > 0.0:
            return float(duration) * 1000.0
    start, end = word.get("timeMs"), word.get("endTime")
    if isinstance(start, (int, float)) and isinstance(end, (int, float)):
        if float(end) > float(start):
            return float(end) - float(start)
    return None


def median_anchor_duration_ms(
    words: Iterable[Mapping[str, Any]],
    default: float = DEFAULT_ASSUMED_DURATION_MS,
) -> float:
    """Median measured anchor duration, used as the assumed non-anchor token duration.

    Fit this on the training split only and pass the result to
    :func:`window_frame_labels` for every split, so the assumption does not vary with the
    data being evaluated.
    """

    durations = [
        duration
        for word in words
        if (duration := anchor_duration_ms(_as_mapping(word))) is not None
    ]
    return float(np.median(durations)) if durations else default


def negation_records(word: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    """Non-null negation records attached to one word.

    ``neg`` is a list that can contain ``None`` padding, and a word can in principle
    belong to more than one negation, so every record is collected rather than only the
    first.
    """

    raw = word.get("neg")
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        return []
    records = []
    for entry in raw:
        mapping = _as_mapping(entry)
        if mapping:
            records.append(mapping)
    return records


def token_spans(
    word: Mapping[str, Any], assumed_duration_ms: float
) -> list[TokenSpan]:
    """Every cue and scope span attached to one anchor word.

    The anchor's own span uses its measured duration; other tokens use
    ``assumed_duration_ms``. Spans are deduplicated by ``(type, onset)`` because a word
    can appear in several negation records.
    """

    anchor_onset = word.get("timeMs")
    anchor_onset = (
        float(anchor_onset)
        if isinstance(anchor_onset, (int, float)) and not isinstance(anchor_onset, bool)
        else None
    )
    measured = anchor_duration_ms(word)

    seen: set[tuple[str, float]] = set()
    spans: list[TokenSpan] = []
    for record in negation_records(word):
        for span_type, key in (("CUE", "cue_tokens"), ("SCOPE", "scope_tokens")):
            tokens = record.get(key)
            if not isinstance(tokens, Sequence) or isinstance(tokens, (str, bytes)):
                continue
            for token in tokens:
                onset = record_onset_ms(token)
                if onset is None:
                    continue
                if (span_type, onset) in seen:
                    continue
                seen.add((span_type, onset))
                is_anchor = (
                    anchor_onset is not None
                    and measured is not None
                    and abs(onset - anchor_onset) < 1e-6
                )
                duration = measured if is_anchor else assumed_duration_ms
                spans.append(
                    TokenSpan(
                        span_type=span_type,
                        start_ms=onset,
                        end_ms=onset + float(duration),
                        duration_is_measured=bool(is_anchor),
                    )
                )
    return spans


def window_frame_labels(
    word: Mapping[str, Any],
    time_grid_seconds: np.ndarray,
    assumed_duration_ms: float = DEFAULT_ASSUMED_DURATION_MS,
) -> np.ndarray:
    """BIO tag ids for one window's frames.

    ``time_grid_seconds`` holds each frame's offset from the anchor word's onset, which is
    exactly the grid the fixed-grid feature arrays use, so tags align frame for frame with
    the channels a model reads.
    """

    word = _as_mapping(word)
    grid = np.asarray(time_grid_seconds, dtype=np.float64)
    if grid.ndim != 1:
        raise ValueError("time_grid_seconds must be one-dimensional")
    anchor_onset = word.get("timeMs")
    tags = np.full(grid.shape[0], OUTSIDE, dtype=np.int64)
    if not isinstance(anchor_onset, (int, float)) or isinstance(anchor_onset, bool):
        return tags

    frame_ms = float(anchor_onset) + grid * 1000.0
    spans = token_spans(word, assumed_duration_ms)
    # Lower-priority types are written first so cue overwrites scope on overlap.
    ordered = sorted(spans, key=lambda span: SPAN_PRIORITY.index(span.span_type))
    for span in ordered:
        inside = (frame_ms >= span.start_ms) & (frame_ms < span.end_ms)
        if not inside.any():
            continue
        positions = np.flatnonzero(inside)
        tags[positions] = TAG_TO_ID[f"I-{span.span_type}"]
        tags[positions[0]] = TAG_TO_ID[f"B-{span.span_type}"]
    return tags


def split_frame_labels(
    words: Sequence[Mapping[str, Any]],
    time_grid_seconds: np.ndarray,
    assumed_duration_ms: float = DEFAULT_ASSUMED_DURATION_MS,
) -> tuple[np.ndarray, FrameLabelStats]:
    """Frame tags for a whole split, plus statistics describing how they were built."""

    grid = np.asarray(time_grid_seconds, dtype=np.float64)
    tags = np.stack(
        [window_frame_labels(word, grid, assumed_duration_ms) for word in words]
    ) if words else np.zeros((0, grid.shape[0]), dtype=np.int64)

    stats = FrameLabelStats(
        num_windows=len(words),
        num_frames=int(tags.size),
        num_labeled_frames=int((tags != OUTSIDE).sum()),
        assumed_duration_ms=assumed_duration_ms,
    )
    for word in words:
        spans = token_spans(_as_mapping(word), assumed_duration_ms)
        if spans:
            stats.num_windows_with_spans += 1
        stats.num_spans += len(spans)
        stats.num_measured_durations += sum(
            1 for span in spans if span.duration_is_measured
        )
        stats.num_assumed_durations += sum(
            1 for span in spans if not span.duration_is_measured
        )
    return tags, stats


def decode_spans(tags: Sequence[int]) -> list[tuple[str, int, int]]:
    """Convert a BIO tag sequence to ``(type, start_frame, end_frame_exclusive)`` spans.

    ``I-X`` without a preceding ``B-X`` starts a new span, which is the conventional
    lenient decoding and keeps a model's raw output from being silently discarded.
    """

    spans: list[tuple[str, int, int]] = []
    current_type: str | None = None
    current_start = 0
    for index, tag_id in enumerate(tags):
        name = TAG_NAMES[int(tag_id)]
        if name == "O":
            if current_type is not None:
                spans.append((current_type, current_start, index))
                current_type = None
            continue
        prefix, _, span_type = name.partition("-")
        if prefix == "B" or current_type != span_type:
            if current_type is not None:
                spans.append((current_type, current_start, index))
            current_type = span_type
            current_start = index
    if current_type is not None:
        spans.append((current_type, current_start, len(tags)))
    return spans


__all__ = [
    "DEFAULT_ASSUMED_DURATION_MS",
    "OUTSIDE",
    "SPAN_TYPES",
    "TAG_NAMES",
    "TAG_TO_ID",
    "FrameLabelStats",
    "TokenSpan",
    "anchor_duration_ms",
    "decode_spans",
    "median_anchor_duration_ms",
    "negation_records",
    "record_onset_ms",
    "split_frame_labels",
    "token_spans",
    "window_frame_labels",
]
