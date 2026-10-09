"""Left-right mirror augmentation for the fixed grid.

A negation cue has no lateral handedness, so relabeling every left-side measurement as
right-side (and vice versa) and geometrically mirroring the scene should leave the label
unchanged. That makes "mirror the window" a label-preserving augmentation: this module
builds a channel permutation + sign flip that turns one fixed-grid window into its
left-right mirror image, and :func:`apply_mirror` runs it. The caller (``t2m_gpt.data``)
decides when to use it -- normally by concatenating the mirrored copy onto the training
split only, which is what doubles the effective training set without touching validation
or test.

Why one rule covers every modality
-----------------------------------
``Head``/``Body``/``LeftHand``/``RightHand`` are Unity world-space poses; ``Eye`` and
``LeftFinger``/``RightFinger`` are raw ``OVRPlugin`` structs. The two frames disagree
only about *Z*: ``OVRCommon.ToOVRPose`` converts Unity -> OVR as position ``(x, y, -z)``,
rotation ``(-x, -y, z, w)`` (see the ``tracking-coordinate-frames`` project notes), which
is exactly the mirror-about-*Z* rule this module derives independently for mirroring
about *X* -- i.e. both frames agree that *X* is the lateral axis, so "mirror left-right"
means the same thing everywhere in this dataset: negate *X* on every position, and negate
the (*y*, *z*) pair on every quaternion (stored ``x, y, z, w``; see
``event_transformer.features._quaternion``).

Where that quaternion rule comes from: mirroring position by ``F = diag(-1, 1, 1)``
turns a rotation matrix ``R`` into ``F R F``, which is itself a proper rotation (its
determinant is ``det(F)^2 det(R) = det(R)``) equal to conjugating ``R`` by a 180-degree
rotation about *X*. Working that conjugation through the quaternion product gives
``(w, x, y, z) -> (w, x, -y, -z)`` -- verified against a concrete 90-degree-about-*Z*
example and against the ``ToOVRPose`` rule above (which is the same derivation for a
*Z*-mirror instead of an *X*-mirror, and matches digit for digit).

Angular velocity is a different story: it is an axial vector (defined through a cross
product), so it picks up mirroring's extra ``det(F) = -1`` and transforms like a
quaternion's vector part, *not* like a position -- also verified by conjugating the
angular-velocity skew matrix. In practice this means every "quaternion-shaped" xyz
triple in this grid, whether it is really a quaternion or an angular velocity, mirrors
the same way: keep *x*, negate *y* and *z*. Every "position-shaped" xyz triple (position
itself or linear velocity) mirrors the opposite way: negate *x*, keep *y* and *z*.

Six things also change identity, not just sign, because they are bilateral pairs:
``LeftHand``/``RightHand`` and ``LeftFinger``/``RightFinger`` swap wholesale (raw block,
motion block, finger flags, presence bit), ``Eye``'s two per-eye sub-blocks swap with
each other, and ``Facial``'s left/right blendshape pairs (and the three directional
shapes -- jaw sideways, mouth corner -- that flip identity under a mirror without being
named "_L"/"_R") swap with their partner. Everything else (``Head``, ``Body``, and the
non-lateral blendshapes: jaw drop/thrust, lips toward, chin raiser top/bottom) mirrors in
place.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from .facial_units import BLENDSHAPE_NAMES, NUM_BLENDSHAPES
from .layout import BLENDSHAPE_OFFSET, ChannelLayout, ModalityLayout, build_channel_layout

#: Each blendshape's mirror partner. Entries that map to themselves (``Jaw_Drop``,
#: ``Jaw_Thrust``, ``Lips_Toward``, ``Chin_Raiser_B``, ``Chin_Raiser_T``) have no lateral
#: component. The eight gaze shapes swap both the eye *and* the "left"/"right" direction
#: token (``Eyes_Look_Left_L`` <-> ``Eyes_Look_Right_R``), unlike every other pair, which
#: only swaps one token.
_BLENDSHAPE_MIRROR_MAP: dict[str, str] = {
    "Brow_Lowerer_L": "Brow_Lowerer_R",
    "Brow_Lowerer_R": "Brow_Lowerer_L",
    "Cheek_Puff_L": "Cheek_Puff_R",
    "Cheek_Puff_R": "Cheek_Puff_L",
    "Cheek_Raiser_L": "Cheek_Raiser_R",
    "Cheek_Raiser_R": "Cheek_Raiser_L",
    "Cheek_Suck_L": "Cheek_Suck_R",
    "Cheek_Suck_R": "Cheek_Suck_L",
    "Chin_Raiser_B": "Chin_Raiser_B",
    "Chin_Raiser_T": "Chin_Raiser_T",
    "Dimpler_L": "Dimpler_R",
    "Dimpler_R": "Dimpler_L",
    "Eyes_Closed_L": "Eyes_Closed_R",
    "Eyes_Closed_R": "Eyes_Closed_L",
    "Eyes_Look_Down_L": "Eyes_Look_Down_R",
    "Eyes_Look_Down_R": "Eyes_Look_Down_L",
    "Eyes_Look_Left_L": "Eyes_Look_Right_R",
    "Eyes_Look_Left_R": "Eyes_Look_Right_L",
    "Eyes_Look_Right_L": "Eyes_Look_Left_R",
    "Eyes_Look_Right_R": "Eyes_Look_Left_L",
    "Eyes_Look_Up_L": "Eyes_Look_Up_R",
    "Eyes_Look_Up_R": "Eyes_Look_Up_L",
    "Inner_Brow_Raiser_L": "Inner_Brow_Raiser_R",
    "Inner_Brow_Raiser_R": "Inner_Brow_Raiser_L",
    "Jaw_Drop": "Jaw_Drop",
    "Jaw_Sideways_Left": "Jaw_Sideways_Right",
    "Jaw_Sideways_Right": "Jaw_Sideways_Left",
    "Jaw_Thrust": "Jaw_Thrust",
    "Lid_Tightener_L": "Lid_Tightener_R",
    "Lid_Tightener_R": "Lid_Tightener_L",
    "Lip_Corner_Depressor_L": "Lip_Corner_Depressor_R",
    "Lip_Corner_Depressor_R": "Lip_Corner_Depressor_L",
    "Lip_Corner_Puller_L": "Lip_Corner_Puller_R",
    "Lip_Corner_Puller_R": "Lip_Corner_Puller_L",
    "Lip_Funneler_LB": "Lip_Funneler_RB",
    "Lip_Funneler_LT": "Lip_Funneler_RT",
    "Lip_Funneler_RB": "Lip_Funneler_LB",
    "Lip_Funneler_RT": "Lip_Funneler_LT",
    "Lip_Pucker_L": "Lip_Pucker_R",
    "Lip_Pucker_R": "Lip_Pucker_L",
    "Lip_Stretcher_L": "Lip_Stretcher_R",
    "Lip_Stretcher_R": "Lip_Stretcher_L",
    "Lip_Suck_LB": "Lip_Suck_RB",
    "Lip_Suck_LT": "Lip_Suck_RT",
    "Lip_Suck_RB": "Lip_Suck_LB",
    "Lip_Suck_RT": "Lip_Suck_LT",
    "Lip_Pressor_L": "Lip_Pressor_R",
    "Lip_Pressor_R": "Lip_Pressor_L",
    "Lip_Tightener_L": "Lip_Tightener_R",
    "Lip_Tightener_R": "Lip_Tightener_L",
    "Lips_Toward": "Lips_Toward",
    "Lower_Lip_Depressor_L": "Lower_Lip_Depressor_R",
    "Lower_Lip_Depressor_R": "Lower_Lip_Depressor_L",
    "Mouth_Left": "Mouth_Right",
    "Mouth_Right": "Mouth_Left",
    "Nose_Wrinkler_L": "Nose_Wrinkler_R",
    "Nose_Wrinkler_R": "Nose_Wrinkler_L",
    "Outer_Brow_Raiser_L": "Outer_Brow_Raiser_R",
    "Outer_Brow_Raiser_R": "Outer_Brow_Raiser_L",
    "Upper_Lid_Raiser_L": "Upper_Lid_Raiser_R",
    "Upper_Lid_Raiser_R": "Upper_Lid_Raiser_L",
    "Upper_Lip_Raiser_L": "Upper_Lip_Raiser_R",
    "Upper_Lip_Raiser_R": "Upper_Lip_Raiser_L",
}


def _validate_blendshape_mirror_map() -> None:
    names = set(BLENDSHAPE_NAMES)
    if set(_BLENDSHAPE_MIRROR_MAP) != names:
        raise RuntimeError("_BLENDSHAPE_MIRROR_MAP does not cover exactly BLENDSHAPE_NAMES")
    for name, partner in _BLENDSHAPE_MIRROR_MAP.items():
        if partner not in names:
            raise RuntimeError(f"{name!r} mirrors to unknown blendshape {partner!r}")
        if _BLENDSHAPE_MIRROR_MAP[partner] != name:
            raise RuntimeError(f"{name!r} <-> {partner!r} mirror mapping is not involutive")


_validate_blendshape_mirror_map()


def _blendshape_permutation() -> np.ndarray:
    """Local (0..62) permutation implementing the blendshape mirror map."""

    name_to_index = {name: index for index, name in enumerate(BLENDSHAPE_NAMES)}
    return np.asarray(
        [name_to_index[_BLENDSHAPE_MIRROR_MAP[name]] for name in BLENDSHAPE_NAMES],
        dtype=np.int64,
    )


@dataclass(slots=True)
class MirrorTransform:
    """Channel permutation + sign flip implementing a left-right mirror of the grid.

    ``mirrored[:, i, :] = sign[i] * source[:, permutation[i], :]``, applied in raw
    (un-normalized) sensor units -- see :func:`apply_mirror`.
    """

    permutation: np.ndarray
    sign: np.ndarray


def _swap_ranges(
    permutation: np.ndarray, range_a: tuple[int, int], range_b: tuple[int, int]
) -> None:
    a_start, a_stop = range_a
    b_start, b_stop = range_b
    if (a_stop - a_start) != (b_stop - b_start):
        raise ValueError("mirrored modality pair has mismatched channel widths")
    a_indices = np.arange(a_start, a_stop, dtype=np.int64)
    b_indices = np.arange(b_start, b_stop, dtype=np.int64)
    permutation[a_indices] = b_indices
    permutation[b_indices] = a_indices


def _swap_single(permutation: np.ndarray, index_a: int, index_b: int) -> None:
    permutation[index_a], permutation[index_b] = index_b, index_a


def _mirror_pose_block(modality: ModalityLayout, sign: np.ndarray) -> None:
    """``Head``/``Body``/``LeftHand``/``RightHand``-shaped raw(7) + motion(6) block."""

    # Raw: position xyz at +0..2, quaternion xyzw at +3..6.
    sign[modality.raw_start + 0] *= -1.0  # position x
    sign[modality.raw_start + 4] *= -1.0  # quaternion y
    sign[modality.raw_start + 5] *= -1.0  # quaternion z
    # Motion: linear velocity xyz at +0..2, angular velocity xyz at +3..5.
    sign[modality.motion_start + 0] *= -1.0  # linear velocity x
    sign[modality.motion_start + 4] *= -1.0  # angular velocity y
    sign[modality.motion_start + 5] *= -1.0  # angular velocity z


def _mirror_eye(eye: ModalityLayout, permutation: np.ndarray, sign: np.ndarray) -> None:
    # Raw: two 9-wide sub-blocks (confidence, valid, position xyz, quaternion xyzw).
    _swap_ranges(
        permutation, (eye.raw_start, eye.raw_start + 9), (eye.raw_start + 9, eye.raw_start + 18)
    )
    for sub_start in (eye.raw_start, eye.raw_start + 9):
        sign[sub_start + 2] *= -1.0  # position x
        sign[sub_start + 6] *= -1.0  # quaternion y
        sign[sub_start + 7] *= -1.0  # quaternion z
    # Motion: two 6-wide sub-blocks (linear velocity xyz, angular velocity xyz).
    _swap_ranges(
        permutation,
        (eye.motion_start, eye.motion_start + 6),
        (eye.motion_start + 6, eye.motion_start + 12),
    )
    for sub_start in (eye.motion_start, eye.motion_start + 6):
        sign[sub_start + 0] *= -1.0  # linear velocity x
        sign[sub_start + 4] *= -1.0  # angular velocity y
        sign[sub_start + 5] *= -1.0  # angular velocity z


def _mirror_facial(facial: ModalityLayout, permutation: np.ndarray) -> None:
    blendshape_permutation = _blendshape_permutation()
    raw_indices = facial.raw_indices(BLENDSHAPE_OFFSET, NUM_BLENDSHAPES)
    permutation[raw_indices] = raw_indices[blendshape_permutation]
    motion_indices = facial.motion_indices(0, NUM_BLENDSHAPES)
    permutation[motion_indices] = motion_indices[blendshape_permutation]
    # The two confidences and two validity flags after the blendshapes are not
    # per-side and stay untouched.


def _mirror_finger_in_place(finger: ModalityLayout, sign: np.ndarray) -> None:
    """Sign flips for one ``LeftFinger``/``RightFinger`` block, post-swap.

    Both hands share the identical internal layout, so the same relative offsets apply
    whichever physical hand's data now sits here.
    """

    base = finger.raw_start
    sign[base + 2] *= -1.0  # pointer position x
    sign[base + 6] *= -1.0  # pointer quaternion y
    sign[base + 7] *= -1.0  # pointer quaternion z
    sign[base + 9] *= -1.0  # root position x
    sign[base + 13] *= -1.0  # root quaternion y
    sign[base + 14] *= -1.0  # root quaternion z
    for bone in range(26):
        bone_start = base + 16 + 4 * bone
        sign[bone_start + 1] *= -1.0  # bone quaternion y
        sign[bone_start + 2] *= -1.0  # bone quaternion z

    motion_base = finger.motion_start
    sign[motion_base + 1] *= -1.0  # pointer linear velocity x
    sign[motion_base + 5] *= -1.0  # pointer angular velocity y
    sign[motion_base + 6] *= -1.0  # pointer angular velocity z
    sign[motion_base + 7] *= -1.0  # root linear velocity x
    sign[motion_base + 11] *= -1.0  # root angular velocity y
    sign[motion_base + 12] *= -1.0  # root angular velocity z
    for bone in range(26):
        bone_start = motion_base + 13 + 3 * bone
        sign[bone_start + 1] *= -1.0
        sign[bone_start + 2] *= -1.0
    # handConfidence, handScale, the five fingerConfidences, and the five
    # pinchStrengths are not per-side and stay untouched; the scale-rate and
    # pinch-rate motion entries likewise.


def build_mirror_transform(channel_names: Sequence[str]) -> MirrorTransform:
    """Build the permutation + sign flip mirroring one fixed grid left-right.

    Safe to call with a modality subset (``DataConfig.modalities``): a bilateral pair
    with only one side selected mirrors that side in place instead of swapping, since
    there is no partner channel to swap with.
    """

    names = tuple(channel_names)
    layout = build_channel_layout(names)
    permutation = np.arange(len(names), dtype=np.int64)
    sign = np.ones(len(names), dtype=np.float64)

    for modality in ("Head", "Body"):
        if modality in layout.modalities:
            _mirror_pose_block(layout.require(modality), sign)

    has_left_hand = "LeftHand" in layout.modalities
    has_right_hand = "RightHand" in layout.modalities
    if has_left_hand and has_right_hand:
        left, right = layout.require("LeftHand"), layout.require("RightHand")
        _swap_ranges(permutation, (left.raw_start, left.raw_stop), (right.raw_start, right.raw_stop))
        _swap_ranges(
            permutation,
            (left.motion_start, left.motion_stop),
            (right.motion_start, right.motion_stop),
        )
        if left.present_index is not None and right.present_index is not None:
            _swap_single(permutation, left.present_index, right.present_index)
        _mirror_pose_block(left, sign)
        _mirror_pose_block(right, sign)
    elif has_left_hand:
        _mirror_pose_block(layout.require("LeftHand"), sign)
    elif has_right_hand:
        _mirror_pose_block(layout.require("RightHand"), sign)

    if "Eye" in layout.modalities:
        _mirror_eye(layout.require("Eye"), permutation, sign)

    if "Facial" in layout.modalities:
        _mirror_facial(layout.require("Facial"), permutation)

    has_left_finger = "LeftFinger" in layout.modalities
    has_right_finger = "RightFinger" in layout.modalities
    if has_left_finger and has_right_finger:
        left, right = layout.require("LeftFinger"), layout.require("RightFinger")
        _swap_ranges(permutation, (left.raw_start, left.raw_stop), (right.raw_start, right.raw_stop))
        _swap_ranges(
            permutation,
            (left.motion_start, left.motion_stop),
            (right.motion_start, right.motion_stop),
        )
        if left.flag_start is not None and right.flag_start is not None:
            _swap_ranges(
                permutation, (left.flag_start, left.flag_stop), (right.flag_start, right.flag_stop)
            )
        if left.present_index is not None and right.present_index is not None:
            _swap_single(permutation, left.present_index, right.present_index)
        _mirror_finger_in_place(left, sign)
        _mirror_finger_in_place(right, sign)
    elif has_left_finger:
        _mirror_finger_in_place(layout.require("LeftFinger"), sign)
    elif has_right_finger:
        _mirror_finger_in_place(layout.require("RightFinger"), sign)

    return MirrorTransform(permutation=permutation, sign=sign)


def apply_mirror(
    values: np.ndarray,
    transform: MirrorTransform,
    means: np.ndarray,
    scales: np.ndarray,
) -> np.ndarray:
    """Mirror one ``[windows, channels, frames]`` array left-right.

    ``means``/``scales`` are the per-channel ``(mean, standard deviation)`` from
    :func:`representation.layout.channel_normalization_arrays`; ``(0, 1)`` for a grid
    built with ``normalize_features=False`` makes the normalization handling below a
    no-op.

    Each mirrored value is re-standardized with the statistics of the channel it *came
    from*, not the channel it lands in. That choice matters, and measurably: the two
    sides of a bilateral pair are not standardized identically, because the participants
    were not symmetric. Measured on ``target-cue_window-1000`` fold 0, 12 of the 654
    swapped pairs have fitted scales differing by more than 2x (worst 3.8x), and
    ``LeftFinger.flag_7`` is identically zero while ``RightFinger.flag_7`` is live with
    mean 0.35 -- the two hands genuinely report different OVR status bits here.
    Re-standardizing with the destination channel's statistics inflated the mirrored
    copy's overall standard deviation by 1.92x, manufacturing exactly the kind of
    distribution shift a model can separate the real windows from the augmented ones on.
    Using the source statistics keeps every value at the standardized position it held
    in its own channel, and reproduces the raw-space mirror's distribution (which is
    distribution-preserving to 1.0001x) after standardization.
    """

    if values.ndim != 3:
        raise ValueError("values must have shape [windows, channels, frames]")
    num_channels = values.shape[1]
    if transform.permutation.shape != (num_channels,):
        raise ValueError("transform does not match this grid's channel count")

    source_means = means[transform.permutation]
    source_scales = scales[transform.permutation]
    source = values[:, transform.permutation, :].astype(np.float64)
    raw = source * source_scales[None, :, None] + source_means[None, :, None]
    mirrored_raw = transform.sign[None, :, None] * raw
    # Sign-flipping a channel negates its deviation from its own mean; the mean itself
    # is a property of the source channel, so it is re-added rather than mirrored.
    mirrored = (mirrored_raw - transform.sign[None, :, None] * source_means[None, :, None]) / (
        source_scales[None, :, None]
    )
    return mirrored.astype(values.dtype, copy=False)


__all__ = [
    "MirrorTransform",
    "apply_mirror",
    "build_mirror_transform",
]
