"""Optional derived-channel representations layered on the shared fixed grid.

This package adds three switchable feature groups -- body-relative positions, second
derivatives, and FACS Action Unit pooling of the face blendshapes -- without changing the
existing pipeline. A default :class:`RepresentationConfig` is the identity, so every model
in this repository keeps its current representation until a configuration explicitly asks
for something else. That is what makes representations comparable: the same folds, the
same tokenizer, and the same training code, with only the channel axis varying.

Usage follows the fit-on-train discipline the rest of the pipeline uses::

    transform = fit_representation(
        bundle.train.values, bundle.channel_names, bundle.normalizer, config,
        grid_delta_seconds(bundle.time_grid),
    )
    train_values = transform.apply(bundle.train.values)
    test_values = transform.apply(bundle.test.values)
    names = transform.channel_names
"""

from .config import (
    IDENTITY_REPRESENTATION,
    ActionUnitSubset,
    RepresentationConfig,
    RootReference,
)
from .facial_units import (
    ACTION_UNIT_MEMBERS,
    ACTION_UNIT_SUBSETS,
    BLENDSHAPE_NAMES,
    NEGATION_ACTION_UNITS,
    NUM_BLENDSHAPES,
    action_unit_names,
)
from .layout import (
    POSITION_OFFSETS,
    ChannelLayout,
    ModalityLayout,
    build_channel_layout,
    channel_normalization_arrays,
)
from .mirror import MirrorTransform, apply_mirror, build_mirror_transform
from .transforms import (
    DerivedBlock,
    RepresentationTransform,
    fit_representation,
    grid_delta_seconds,
    masked_time_derivative,
)

__all__ = [
    "ACTION_UNIT_MEMBERS",
    "ACTION_UNIT_SUBSETS",
    "BLENDSHAPE_NAMES",
    "IDENTITY_REPRESENTATION",
    "NEGATION_ACTION_UNITS",
    "NUM_BLENDSHAPES",
    "POSITION_OFFSETS",
    "ActionUnitSubset",
    "ChannelLayout",
    "DerivedBlock",
    "MirrorTransform",
    "ModalityLayout",
    "RepresentationConfig",
    "RepresentationTransform",
    "RootReference",
    "action_unit_names",
    "apply_mirror",
    "build_channel_layout",
    "build_mirror_transform",
    "channel_normalization_arrays",
    "fit_representation",
    "grid_delta_seconds",
    "masked_time_derivative",
]
