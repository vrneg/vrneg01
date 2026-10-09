"""Frame-level negation tagging: BIO sequence labelling over the VR sensor grid.

Where :mod:`src.t2m_gpt` and :mod:`src.motion_gpt` ask "does this window contain a negation
cue", this package asks "which frames are inside the cue and which inside its scope" -- the
NER-shaped formulation of the same annotation. It needs no motion tokenizer and no
pretraining: a bidirectional encoder reads the continuous fixed-grid channels directly and
a linear-chain CRF turns per-frame scores into well-formed spans.

The targets come from :mod:`.labels`, which recovers word-level cue and scope spans from the
existing fold datasets -- a ``Word`` record id carries its onset timestamp as its first
element, so no database access is required. Read that module's docstring before
interpreting results: token *onsets* are exact but most token *offsets* are assumed, which
is why :mod:`.metrics` reports overlap-based span F1 alongside the exact-match figure.

Every run also writes a window-level score in the schema
:mod:`src.comparison_stats` consumes, so a tagger can be compared head to head with the
window classifiers on the same folds.
"""

from .config import (
    DataConfig,
    LabelConfig,
    TaggerConfig,
    TaggerExperimentConfig,
    TaggerTrainingConfig,
)
from .crf import LinearChainCRF, bio_start_mask, bio_transition_mask
from .labels import (
    DEFAULT_ASSUMED_DURATION_MS,
    OUTSIDE,
    SPAN_TYPES,
    TAG_NAMES,
    TAG_TO_ID,
    FrameLabelStats,
    TokenSpan,
    anchor_duration_ms,
    decode_spans,
    median_anchor_duration_ms,
    record_onset_ms,
    split_frame_labels,
    token_spans,
    window_frame_labels,
)
from .metrics import (
    SpanScore,
    TaggerMetrics,
    evaluate_tagging,
    frame_metrics,
    score_spans,
)
from .model import FrameEncoder, FrameTagger, count_parameters
from .training import (
    FrameTaggingDataset,
    SplitPredictions,
    TaggerTrainingResult,
    resolve_device,
    seed_everything,
    train_frame_tagger,
)

__all__ = [
    "DEFAULT_ASSUMED_DURATION_MS",
    "OUTSIDE",
    "SPAN_TYPES",
    "TAG_NAMES",
    "TAG_TO_ID",
    "DataConfig",
    "FrameEncoder",
    "FrameLabelStats",
    "FrameTagger",
    "FrameTaggingDataset",
    "LabelConfig",
    "LinearChainCRF",
    "SpanScore",
    "SplitPredictions",
    "TaggerConfig",
    "TaggerExperimentConfig",
    "TaggerMetrics",
    "TaggerTrainingConfig",
    "TaggerTrainingResult",
    "TokenSpan",
    "anchor_duration_ms",
    "bio_start_mask",
    "bio_transition_mask",
    "count_parameters",
    "decode_spans",
    "evaluate_tagging",
    "frame_metrics",
    "median_anchor_duration_ms",
    "record_onset_ms",
    "resolve_device",
    "score_spans",
    "seed_everything",
    "split_frame_labels",
    "token_spans",
    "train_frame_tagger",
    "window_frame_labels",
]
