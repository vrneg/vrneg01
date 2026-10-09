"""Derived-channel transforms fitted on a training split and applied to every split.

The flow for every option is the same three steps, and the order matters:

1. **Invert the fitted normalizer.** The grid arrives standardized per channel, so a
   difference of two position channels, or a mean of several blendshape channels, would
   otherwise mix incompatible units. Everything derived here is computed in raw sensor
   units.
2. **Compute the derived block, respecting presence.** Interpolation fills frames outside
   a modality's observed range with zeros, which un-normalize to the training mean rather
   than to "missing". Derived channels are therefore computed only where the source
   frames are real, and are left at zero elsewhere -- the same convention the rest of the
   grid uses, and the reason presence channels are mandatory for these options.
3. **Standardize with training statistics only.** Statistics are accumulated over present
   frames of the training split alone, matching how the upstream normalizer is fitted
   over observed events rather than over interpolated filler.

Original channels are passed through untouched in their already-normalized form, so any
option that only appends leaves previously recorded results reproducible channel for
channel.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from .config import RepresentationConfig
from .facial_units import ACTION_UNIT_MEMBERS, NUM_BLENDSHAPES, action_unit_names
from .layout import (
    BLENDSHAPE_OFFSET,
    POSITION_OFFSETS,
    ChannelLayout,
    build_channel_layout,
    channel_normalization_arrays,
)

AXIS_NAMES = ("x", "y", "z")
MINIMUM_SCALE = 1e-6


def masked_time_derivative(
    values: np.ndarray, present: np.ndarray, delta_seconds: float
) -> np.ndarray:
    """Differentiate along time using only frames the sensor actually observed.

    Central differences are used where both neighbours are present, one-sided differences
    at the edge of a contiguous present run, and zero where the frame itself is absent or
    has no present neighbour. Differencing straight through a presence boundary would
    manufacture a large spike from interpolation filler, which is exactly the artifact a
    model would learn to key on.
    """

    if values.ndim != 3:
        raise ValueError("values must have shape [windows, channels, frames]")
    if present.shape != (values.shape[0], values.shape[2]):
        raise ValueError("present must have shape [windows, frames]")
    if delta_seconds <= 0.0:
        raise ValueError("delta_seconds must be positive")

    observed = present.astype(bool)
    forward = np.zeros_like(values)
    forward[:, :, :-1] = values[:, :, 1:] - values[:, :, :-1]
    backward = np.zeros_like(values)
    backward[:, :, 1:] = values[:, :, 1:] - values[:, :, :-1]

    pair = observed[:, :-1] & observed[:, 1:]
    has_next = np.zeros_like(observed)
    has_next[:, :-1] = pair
    has_previous = np.zeros_like(observed)
    has_previous[:, 1:] = pair

    central = (forward + backward) / (2.0 * delta_seconds)
    one_sided_forward = forward / delta_seconds
    one_sided_backward = backward / delta_seconds

    use_central = (has_previous & has_next)[:, None, :]
    use_forward = (has_next & ~has_previous)[:, None, :]
    use_backward = (has_previous & ~has_next)[:, None, :]

    derivative = np.where(use_central, central, 0.0)
    derivative = np.where(use_forward, one_sided_forward, derivative)
    derivative = np.where(use_backward, one_sided_backward, derivative)
    return derivative


@dataclass(slots=True)
class DerivedBlock:
    """One group of derived channels, with the frames on which it is defined."""

    names: tuple[str, ...]
    values: np.ndarray
    present: np.ndarray


def _presence_mask(
    values: np.ndarray, layout: ChannelLayout, modality: str
) -> np.ndarray:
    modality_layout = layout.require(modality)
    if modality_layout.present_index is None:
        raise ValueError(
            f"{modality} has no presence channel; derived channels need "
            "include_presence_channels=True to distinguish missing frames from "
            "frames that happen to sit at the training mean"
        )
    return values[:, modality_layout.present_index, :] > 0.5


def _root_relative_block(
    raw: np.ndarray, layout: ChannelLayout, config: RepresentationConfig
) -> list[DerivedBlock]:
    reference = config.root_reference
    reference_layout = layout.require(reference)
    reference_present = _presence_mask(raw, layout, reference)
    reference_position = raw[:, reference_layout.position_indices(0), :]

    blocks: list[DerivedBlock] = []
    for modality in sorted(layout.modalities):
        if modality == reference:
            continue
        for label, offset in POSITION_OFFSETS[modality]:
            modality_layout = layout.require(modality)
            position = raw[:, modality_layout.position_indices(offset), :]
            present = _presence_mask(raw, layout, modality) & reference_present
            blocks.append(
                DerivedBlock(
                    names=tuple(
                        f"{modality}.relpos_{label}_{axis}" for axis in AXIS_NAMES
                    ),
                    values=position - reference_position,
                    present=present,
                )
            )
    return blocks


def _acceleration_block(
    raw: np.ndarray,
    layout: ChannelLayout,
    config: RepresentationConfig,
    delta_seconds: float,
) -> list[DerivedBlock]:
    selected = (
        sorted(layout.modalities)
        if config.acceleration_modalities is None
        else [
            modality
            for modality in sorted(layout.modalities)
            if modality in config.acceleration_modalities
        ]
    )
    blocks: list[DerivedBlock] = []
    for modality in selected:
        modality_layout = layout.require(modality)
        width = modality_layout.motion_stop - modality_layout.motion_start
        velocity = raw[:, modality_layout.motion_start : modality_layout.motion_stop, :]
        present = _presence_mask(raw, layout, modality)
        blocks.append(
            DerivedBlock(
                names=tuple(f"{modality}.accel_{index}" for index in range(width)),
                values=masked_time_derivative(velocity, present, delta_seconds),
                present=present,
            )
        )
    return blocks


def _action_unit_block(
    raw: np.ndarray, layout: ChannelLayout, config: RepresentationConfig
) -> list[DerivedBlock]:
    facial = layout.require("Facial")
    present = _presence_mask(raw, layout, "Facial")
    blendshapes = raw[:, facial.raw_indices(BLENDSHAPE_OFFSET, NUM_BLENDSHAPES), :]
    units = action_unit_names(config.action_unit_subset)

    intensity = np.stack(
        [blendshapes[:, ACTION_UNIT_MEMBERS[unit], :].mean(axis=1) for unit in units],
        axis=1,
    )
    blocks = [
        DerivedBlock(
            names=tuple(f"Facial.{unit}" for unit in units),
            values=intensity,
            present=present,
        )
    ]
    if config.append_action_unit_velocity:
        # The Facial motion block is exactly the 63 blendshape rates, in blendshape
        # order, so the same member indices pool velocity as pool intensity.
        rates = raw[:, facial.motion_indices(0, NUM_BLENDSHAPES), :]
        velocity = np.stack(
            [rates[:, ACTION_UNIT_MEMBERS[unit], :].mean(axis=1) for unit in units],
            axis=1,
        )
        blocks.append(
            DerivedBlock(
                names=tuple(f"Facial.{unit}_velocity" for unit in units),
                values=velocity,
                present=present,
            )
        )
    return blocks


def _dropped_channels(
    layout: ChannelLayout, config: RepresentationConfig
) -> set[int]:
    dropped: set[int] = set()
    if config.drop_absolute_positions:
        reference = config.root_reference
        for modality in layout.modalities:
            modality_layout = layout.require(modality)
            for _label, offset in POSITION_OFFSETS[modality]:
                indices = modality_layout.position_indices(offset)
                if modality == reference and config.keep_reference_height:
                    # Vertical position is posture (leaning, slouching); horizontal
                    # position is only where the participant stood.
                    dropped.update(int(index) for index in indices[[0, 2]])
                else:
                    dropped.update(int(index) for index in indices)
    if config.drop_raw_blendshapes:
        facial = layout.require("Facial")
        dropped.update(
            int(index) for index in facial.raw_indices(BLENDSHAPE_OFFSET, NUM_BLENDSHAPES)
        )
        dropped.update(int(index) for index in facial.motion_indices(0, NUM_BLENDSHAPES))
    return dropped


@dataclass(slots=True)
class RepresentationTransform:
    """A representation fitted on one training split, applicable to any split.

    ``channel_names`` is the resulting channel order: retained original channels first,
    in their original order, then each derived block.
    """

    config: RepresentationConfig
    source_channel_names: tuple[str, ...]
    channel_names: tuple[str, ...]
    keep_indices: np.ndarray
    channel_means: np.ndarray
    channel_scales: np.ndarray
    derived_means: np.ndarray
    derived_scales: np.ndarray
    delta_seconds: float

    @property
    def num_derived_channels(self) -> int:
        return int(self.derived_means.shape[0])

    def _raw(self, values: np.ndarray) -> np.ndarray:
        return values.astype(np.float64) * self.channel_scales[None, :, None] + (
            self.channel_means[None, :, None]
        )

    def _blocks(self, values: np.ndarray) -> list[DerivedBlock]:
        layout = build_channel_layout(self.source_channel_names)
        raw = self._raw(values)
        blocks: list[DerivedBlock] = []
        if self.config.root_relative_positions:
            blocks.extend(_root_relative_block(raw, layout, self.config))
        if self.config.append_acceleration:
            blocks.extend(
                _acceleration_block(raw, layout, self.config, self.delta_seconds)
            )
        if self.config.append_action_units:
            blocks.extend(_action_unit_block(raw, layout, self.config))
        return blocks

    def apply(self, values: np.ndarray) -> np.ndarray:
        """Return ``values`` with derived channels appended and dropped ones removed."""

        if values.ndim != 3:
            raise ValueError("values must have shape [windows, channels, frames]")
        if values.shape[1] != len(self.source_channel_names):
            raise ValueError(
                f"values has {values.shape[1]} channels; this transform was fitted on "
                f"{len(self.source_channel_names)}"
            )
        if self.config.is_identity:
            return values

        blocks = self._blocks(values)
        retained = values[:, self.keep_indices, :]
        if not blocks:
            return np.ascontiguousarray(retained)

        stacked = np.concatenate([block.values for block in blocks], axis=1)
        presence = np.concatenate(
            [
                np.repeat(block.present[:, None, :], len(block.names), axis=1)
                for block in blocks
            ],
            axis=1,
        )
        standardized = (stacked - self.derived_means[None, :, None]) / (
            self.derived_scales[None, :, None]
        )
        standardized = np.where(presence, standardized, 0.0)
        return np.ascontiguousarray(
            np.concatenate(
                (retained, standardized.astype(values.dtype, copy=False)), axis=1
            )
        )


def fit_representation(
    train_values: np.ndarray,
    channel_names: Sequence[str],
    normalizer: object | None,
    config: RepresentationConfig,
    delta_seconds: float,
) -> RepresentationTransform:
    """Fit derived-channel statistics on a training split.

    ``normalizer`` is the fitted :class:`event_transformer.features.FeatureNormalizer`
    used to build the grid, or ``None`` when the grid was built unnormalized; it is
    inverted so derived channels are computed in raw sensor units.
    """

    config.validate()
    names = tuple(channel_names)
    if train_values.ndim != 3 or train_values.shape[1] != len(names):
        raise ValueError("train_values must have shape [windows, len(channel_names), frames]")

    means, scales = channel_normalization_arrays(names, normalizer)
    if config.is_identity:
        return RepresentationTransform(
            config=config,
            source_channel_names=names,
            channel_names=names,
            keep_indices=np.arange(len(names), dtype=np.int64),
            channel_means=means,
            channel_scales=scales,
            derived_means=np.zeros(0, dtype=np.float64),
            derived_scales=np.ones(0, dtype=np.float64),
            delta_seconds=delta_seconds,
        )

    layout = build_channel_layout(names)
    dropped = _dropped_channels(layout, config)
    keep_indices = np.asarray(
        [index for index in range(len(names)) if index not in dropped], dtype=np.int64
    )

    probe = RepresentationTransform(
        config=config,
        source_channel_names=names,
        channel_names=names,
        keep_indices=keep_indices,
        channel_means=means,
        channel_scales=scales,
        derived_means=np.zeros(0, dtype=np.float64),
        derived_scales=np.ones(0, dtype=np.float64),
        delta_seconds=delta_seconds,
    )
    blocks = probe._blocks(train_values)

    derived_names: list[str] = []
    derived_means: list[float] = []
    derived_scales: list[float] = []
    for block in blocks:
        observed = block.present
        for offset, name in enumerate(block.names):
            channel = block.values[:, offset, :]
            selected = channel[observed]
            if selected.size:
                mean = float(selected.mean())
                scale = float(selected.std())
            else:
                mean, scale = 0.0, 1.0
            derived_names.append(name)
            derived_means.append(mean)
            derived_scales.append(max(scale, MINIMUM_SCALE))

    return RepresentationTransform(
        config=config,
        source_channel_names=names,
        channel_names=tuple(names[index] for index in keep_indices) + tuple(derived_names),
        keep_indices=keep_indices,
        channel_means=means,
        channel_scales=scales,
        derived_means=np.asarray(derived_means, dtype=np.float64),
        derived_scales=np.asarray(derived_scales, dtype=np.float64),
        delta_seconds=delta_seconds,
    )


def grid_delta_seconds(time_grid: np.ndarray) -> float:
    """Frame spacing of a uniform time grid, in seconds."""

    grid = np.asarray(time_grid, dtype=np.float64)
    if grid.ndim != 1 or grid.shape[0] < 2:
        raise ValueError("time_grid must be one-dimensional with at least two frames")
    return float((grid[-1] - grid[0]) / (grid.shape[0] - 1))


__all__ = [
    "DerivedBlock",
    "RepresentationTransform",
    "fit_representation",
    "grid_delta_seconds",
    "masked_time_derivative",
]
