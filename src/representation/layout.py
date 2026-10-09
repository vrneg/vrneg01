"""Structured view of the fixed grid's flat channel axis.

The grid names channels ``<Modality>.feature_<i>``, ``<Modality>.flag_<i>``, and
``<Modality>.present``, which is enough to reconstruct where each modality's raw
measurements, first derivatives, and status bits live. This module turns that flat naming
back into slices, plus the position sub-slices inside each modality's raw block, so the
derived-channel transforms can address "the left hand's position" without hard-coding
offsets at every call site.

Offsets come from the feature extractor's own layout, mirroring
``event_transformer.features``:

======================  ==========================================================
Modality                Raw block layout
======================  ==========================================================
``Head`` ``Body``       ``position[3] quaternion[4]``
``LeftHand``            same as ``Head``
``RightHand``           same as ``Head``
``Eye``                 per eye: ``confidence valid position[3] quaternion[4]``
``Facial``              ``blendshapes[63] confidences[2] validity[2]``
``LeftFinger``          ``confidence scale pointerPose[7] rootPose[7] bones[104] ...``
``RightFinger``         same as ``LeftFinger``
======================  ==========================================================
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

try:
    from event_transformer.features import (
        BASE_MODALITY_DIMS_BY_NAME,
        MODALITY_DIMS_BY_NAME,
        MODALITY_NAMES,
        MODALITY_TO_ID,
        MOTION_DIMS_BY_NAME,
    )
except ModuleNotFoundError as error:  # pragma: no cover - import-path fallback
    if error.name != "event_transformer":
        raise
    from ..event_transformer.features import (
        BASE_MODALITY_DIMS_BY_NAME,
        MODALITY_DIMS_BY_NAME,
        MODALITY_NAMES,
        MODALITY_TO_ID,
        MOTION_DIMS_BY_NAME,
    )


#: Offsets of ``xyz`` position triples inside each modality's raw block, with a label
#: describing which rigid body the triple belongs to.
POSITION_OFFSETS: dict[str, tuple[tuple[str, int], ...]] = {
    "Head": (("pose", 0),),
    "Body": (("pose", 0),),
    "LeftHand": (("pose", 0),),
    "RightHand": (("pose", 0),),
    "Eye": (("left", 2), ("right", 11)),
    "LeftFinger": (("pointer", 2), ("root", 9)),
    "RightFinger": (("pointer", 2), ("root", 9)),
    "Facial": (),
}

#: Offset of the 63 blendshape weights inside the ``Facial`` raw block.
BLENDSHAPE_OFFSET = 0


@dataclass(slots=True)
class ModalityLayout:
    """Where one modality's channels sit on the grid's channel axis."""

    name: str
    raw_start: int
    raw_stop: int
    motion_start: int
    motion_stop: int
    present_index: int | None
    flag_start: int | None = None
    flag_stop: int | None = None

    def position_indices(self, offset: int) -> np.ndarray:
        """Absolute channel indices of the ``xyz`` triple at a raw-block offset."""

        start = self.raw_start + offset
        if start + 3 > self.raw_stop:
            raise ValueError(
                f"{self.name} position offset {offset} exceeds its raw block"
            )
        return np.arange(start, start + 3, dtype=np.int64)

    def raw_indices(self, offset: int, length: int) -> np.ndarray:
        """Absolute channel indices of a sub-block of the raw measurements."""

        start = self.raw_start + offset
        if start + length > self.raw_stop:
            raise ValueError(
                f"{self.name} raw offset {offset}+{length} exceeds its raw block"
            )
        return np.arange(start, start + length, dtype=np.int64)

    def motion_indices(self, offset: int, length: int) -> np.ndarray:
        """Absolute channel indices of a sub-block of the first derivatives."""

        start = self.motion_start + offset
        if start + length > self.motion_stop:
            raise ValueError(
                f"{self.name} motion offset {offset}+{length} exceeds its motion block"
            )
        return np.arange(start, start + length, dtype=np.int64)


@dataclass(slots=True)
class ChannelLayout:
    """Per-modality slices over one fixed grid's channel axis."""

    channel_names: tuple[str, ...]
    modalities: dict[str, ModalityLayout] = field(default_factory=dict)

    @property
    def num_channels(self) -> int:
        return len(self.channel_names)

    def require(self, modality: str) -> ModalityLayout:
        if modality not in self.modalities:
            raise ValueError(
                f"Modality {modality!r} is not present in this grid; "
                f"available: {sorted(self.modalities)}"
            )
        return self.modalities[modality]


def _parse_suffix(suffix: str) -> tuple[str, int | None]:
    if suffix == "present":
        return "present", None
    kind, _, index = suffix.partition("_")
    if kind not in {"feature", "flag"} or not index.isdigit():
        raise ValueError(f"Unrecognized channel suffix: {suffix!r}")
    return kind, int(index)


def build_channel_layout(channel_names: Sequence[str]) -> ChannelLayout:
    """Recover per-modality slices from the grid's channel names.

    The names are the single source of truth here rather than a recomputed expectation,
    so a grid built with a modality subset or without presence channels is described
    correctly instead of being silently mis-sliced.
    """

    names = tuple(channel_names)
    positions: dict[str, dict[str, list[int]]] = {}
    for index, name in enumerate(names):
        modality, _, suffix = name.partition(".")
        if modality not in MODALITY_TO_ID:
            raise ValueError(f"Unknown modality in channel name {name!r}")
        kind, _ = _parse_suffix(suffix)
        positions.setdefault(modality, {"feature": [], "flag": [], "present": []})
        positions[modality][kind].append(index)

    layout = ChannelLayout(channel_names=names)
    for modality in MODALITY_NAMES:
        if modality not in positions:
            continue
        feature_indices = positions[modality]["feature"]
        flag_indices = positions[modality]["flag"]
        present_indices = positions[modality]["present"]

        expected_features = MODALITY_DIMS_BY_NAME[modality]
        if len(feature_indices) != expected_features:
            raise ValueError(
                f"{modality} has {len(feature_indices)} feature channels; "
                f"expected {expected_features}"
            )
        if feature_indices != list(
            range(feature_indices[0], feature_indices[0] + expected_features)
        ):
            raise ValueError(f"{modality} feature channels are not contiguous")
        if len(present_indices) > 1:
            raise ValueError(f"{modality} has more than one presence channel")

        raw_dimension = BASE_MODALITY_DIMS_BY_NAME[modality]
        motion_dimension = MOTION_DIMS_BY_NAME[modality]
        raw_start = feature_indices[0]
        layout.modalities[modality] = ModalityLayout(
            name=modality,
            raw_start=raw_start,
            raw_stop=raw_start + raw_dimension,
            motion_start=raw_start + raw_dimension,
            motion_stop=raw_start + raw_dimension + motion_dimension,
            present_index=present_indices[0] if present_indices else None,
            flag_start=flag_indices[0] if flag_indices else None,
            flag_stop=flag_indices[-1] + 1 if flag_indices else None,
        )
    return layout


def channel_normalization_arrays(
    channel_names: Sequence[str],
    normalizer: Any | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return per-channel ``(mean, scale)`` for inverting the fitted normalizer.

    ``normalizer`` is duck-typed on ``event_transformer.features.FeatureNormalizer``:
    ``means`` and ``standard_deviations`` are mappings from modality id to a per-feature
    tensor. Only ``feature_*`` channels are standardized by that class, so flag and
    presence channels get the identity ``(0, 1)`` and pass through untouched. Passing
    ``None`` (a grid built with ``normalize_features=False``) likewise yields the
    identity, which makes the inversion a no-op instead of a special case.
    """

    names = tuple(channel_names)
    means = np.zeros(len(names), dtype=np.float64)
    scales = np.ones(len(names), dtype=np.float64)
    if normalizer is None:
        return means, scales

    normalizer_means: Mapping[int, Any] = normalizer.means
    normalizer_scales: Mapping[int, Any] = normalizer.standard_deviations
    for index, name in enumerate(names):
        modality, _, suffix = name.partition(".")
        kind, feature_index = _parse_suffix(suffix)
        if kind != "feature" or feature_index is None:
            continue
        modality_id = MODALITY_TO_ID[modality]
        means[index] = float(np.asarray(normalizer_means[modality_id])[feature_index])
        scales[index] = float(np.asarray(normalizer_scales[modality_id])[feature_index])
    return means, scales


__all__ = [
    "BLENDSHAPE_OFFSET",
    "POSITION_OFFSETS",
    "ChannelLayout",
    "ModalityLayout",
    "build_channel_layout",
    "channel_normalization_arrays",
]
