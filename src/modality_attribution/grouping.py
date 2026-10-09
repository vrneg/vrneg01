"""Resolve and perturb complete input modalities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

try:
    from event_transformer.features import MODALITY_NAMES
except ModuleNotFoundError as error:
    if error.name != "event_transformer":
        raise
    from ..event_transformer.features import MODALITY_NAMES


@dataclass(frozen=True, slots=True)
class ModalityGroup:
    """One modality and all fixed-grid channels belonging to it."""

    name: str
    indices: tuple[int, ...]


def normalize_modalities(modalities: Iterable[str]) -> tuple[str, ...]:
    """Validate a subset and return it in canonical sensor order."""

    requested = tuple(modalities)
    unknown = set(requested) - set(MODALITY_NAMES)
    if unknown:
        raise ValueError(f"Unknown modalities: {sorted(unknown)}")
    if len(set(requested)) != len(requested):
        raise ValueError("Modalities must not contain duplicates")
    return tuple(name for name in MODALITY_NAMES if name in requested)


def build_channel_groups(
    channel_names: tuple[str, ...],
) -> tuple[ModalityGroup, ...]:
    """Group a complete fixed-grid channel schema by modality prefix."""

    groups: list[ModalityGroup] = []
    covered: list[int] = []
    for modality in MODALITY_NAMES:
        prefix = f"{modality}."
        indices = tuple(
            index for index, channel in enumerate(channel_names) if channel.startswith(prefix)
        )
        if not indices:
            raise ValueError(f"No fixed-grid channels found for modality {modality}")
        if indices != tuple(range(indices[0], indices[-1] + 1)):
            raise ValueError(f"Channels for modality {modality} must be contiguous")
        groups.append(ModalityGroup(modality, indices))
        covered.extend(indices)

    expected = list(range(len(channel_names)))
    if sorted(covered) != expected or len(covered) != len(set(covered)):
        ungrouped = [channel_names[index] for index in set(expected) - set(covered)]
        raise ValueError(
            "Every input channel must belong to exactly one modality; "
            f"ungrouped={ungrouped}"
        )
    return tuple(groups)


def mask_fixed_grid(
    values: np.ndarray,
    groups: tuple[ModalityGroup, ...],
    included_modalities: Iterable[str],
) -> np.ndarray:
    """Return a copy with excluded modalities, including presence flags, zeroed."""

    if values.ndim != 3:
        raise ValueError("Fixed-grid values must have shape [cases, channels, time]")
    included = set(normalize_modalities(included_modalities))
    masked = np.array(values, copy=True)
    for group in groups:
        if group.name not in included:
            masked[:, group.indices, :] = 0.0
    return masked
