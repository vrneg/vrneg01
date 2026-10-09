"""Toggles for optional derived channels on top of the shared fixed grid.

Every field defaults to the behaviour the repository already had, so a default
:class:`RepresentationConfig` is the identity transform: the channel array and channel
names come out byte-identical to a run that never touched this package. That matters
because the ROCKET, TCN, T2M-GPT, and MotionGPT numbers already recorded in this
repository were produced without it, and they have to stay reproducible while new
representations are compared against them.

The three groups of options correspond to three distinct concerns:

``root_relative_positions``
    Absolute positions let a model key on *where in the room* a participant stood, which
    is a recording confound rather than a communicative signal. Expressing each
    modality's position relative to a body reference removes it.

``append_acceleration``
    The pipeline already computes first derivatives (the "motion" half of every
    modality's channels). Second derivatives are the natural next term, and in gesture
    research the sharp onset of a movement often carries more than its velocity.

``append_action_units``
    Raw Quest Pro blendshapes are 63 uninterpreted floats. Pooling them into FACS Action
    Units gives a much smaller, linguistically grounded feature space that connects to
    the facial-expression literature on negation and disagreement.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from .facial_units import ACTION_UNIT_SUBSETS

RootReference = Literal["Head", "Body"]
ActionUnitSubset = Literal["all", "negation"]


@dataclass(frozen=True, slots=True)
class RepresentationConfig:
    """Optional derived-channel selection applied after the fixed grid is built.

    All derived channels are computed in raw sensor units (the fitted feature
    normalizer is inverted first) and then standardized using training-split
    statistics only, so turning an option on never leaks validation or test
    information into the representation.
    """

    root_relative_positions: bool = False
    root_reference: RootReference = "Head"
    drop_absolute_positions: bool = False
    keep_reference_height: bool = True

    append_acceleration: bool = False
    acceleration_modalities: tuple[str, ...] | None = None

    append_action_units: bool = False
    action_unit_subset: ActionUnitSubset = "all"
    append_action_unit_velocity: bool = True
    drop_raw_blendshapes: bool = False

    @property
    def is_identity(self) -> bool:
        """Whether this configuration leaves the fixed grid completely unchanged."""

        return not (
            self.root_relative_positions
            or self.append_acceleration
            or self.append_action_units
        )

    @property
    def requires_presence_channels(self) -> bool:
        """Whether any enabled option needs presence channels to be correct.

        Interpolation writes zeros outside a modality's observed range, which after
        normalization is indistinguishable from "the value happened to equal the
        training mean". Every derived channel here has to know which frames are real,
        so the presence channels are a hard requirement rather than a nicety.
        """

        return not self.is_identity

    def validate(self) -> None:
        if self.root_reference not in ("Head", "Body"):
            raise ValueError(
                f"root_reference must be 'Head' or 'Body'; got {self.root_reference!r}"
            )
        if self.action_unit_subset not in ACTION_UNIT_SUBSETS:
            raise ValueError(
                f"Unknown action_unit_subset {self.action_unit_subset!r}; "
                f"expected one of {sorted(ACTION_UNIT_SUBSETS)}"
            )
        if self.acceleration_modalities is not None:
            if not self.acceleration_modalities:
                raise ValueError(
                    "acceleration_modalities must be None or a non-empty selection"
                )
            if len(set(self.acceleration_modalities)) != len(self.acceleration_modalities):
                raise ValueError("acceleration_modalities must not contain duplicates")
        if self.drop_absolute_positions and not self.root_relative_positions:
            raise ValueError(
                "drop_absolute_positions requires root_relative_positions=True; "
                "dropping positions without a relative replacement would discard "
                "all positional information"
            )
        if self.drop_raw_blendshapes and not self.append_action_units:
            raise ValueError(
                "drop_raw_blendshapes requires append_action_units=True; dropping the "
                "blendshapes without the pooled units would discard all facial signal"
            )
        if self.drop_raw_blendshapes and not self.append_action_unit_velocity:
            raise ValueError(
                "drop_raw_blendshapes also drops the 63 blendshape velocity channels, "
                "so it requires append_action_unit_velocity=True to retain any facial "
                "dynamics"
            )

    @property
    def signature(self) -> str:
        """Short deterministic description, for checkpoint configuration signatures."""

        if self.is_identity:
            return "identity"
        parts: list[str] = []
        if self.root_relative_positions:
            reference = self.root_reference.lower()
            flags = "".join(
                (
                    "d" if self.drop_absolute_positions else "",
                    "h" if self.keep_reference_height else "",
                )
            )
            parts.append(f"rootrel-{reference}{'-' + flags if flags else ''}")
        if self.append_acceleration:
            scope = (
                "all"
                if self.acceleration_modalities is None
                else "+".join(sorted(self.acceleration_modalities))
            )
            parts.append(f"accel-{scope}")
        if self.append_action_units:
            flags = "".join(
                (
                    "v" if self.append_action_unit_velocity else "",
                    "d" if self.drop_raw_blendshapes else "",
                )
            )
            parts.append(
                f"au-{self.action_unit_subset}{'-' + flags if flags else ''}"
            )
        return "_".join(parts)


IDENTITY_REPRESENTATION = RepresentationConfig()


__all__ = [
    "ActionUnitSubset",
    "IDENTITY_REPRESENTATION",
    "RepresentationConfig",
    "RootReference",
]
