"""FACS Action Unit pooling for the 63 Quest Pro face-tracking blendshapes.

The ``Facial`` modality stores ``expressionWeights`` as 63 unnamed floats
(``event_transformer.features.FACIAL_DIM = 67`` is those 63 plus two confidences and two
validity flags), and the fixed grid names them ``Facial.feature_0 .. Facial.feature_62``.
Those indices are ``OVRPlugin.FaceExpression`` order, which this module names and then
pools into Action Units.

Why the ordering below is trustworthy
-------------------------------------
The enum order is not guessed. ``neo/replay/export_replay.py`` remaps the recorded
63-long block onto the avatar SDK's 72-long ``ovrAvatar2FaceExpression`` array, and its
``FACE_MAP`` pins the alignment at three independent points:

- indices ``0..49`` map straight through, so the head of the list is fixed;
- index ``50`` expands into SDK slots ``50..53``, identifying it as ``Lips_Toward``
  (the SDK splits that one shape into four);
- the tail maps ``[54, 55, 56, 57, 60, 61, 66, 67, 68, 69, 70, 71]``, and ``neo/README.md``
  independently states that SDK slots ``68/69`` are ``UpperLidRaiser L/R`` — which lands
  ``Upper_Lid_Raiser_L/R`` at OVR indices ``59/60`` exactly as listed here.

The gaps that ``FACE_MAP`` skips (SDK ``58/59`` NasiolabialFurrow, ``62..65`` Nostril*)
are shapes with no OVRPlugin counterpart, which is consistent with this list having no
entry for them.

What pooling does, and what it does not
---------------------------------------
Each Action Unit is the **mean of its member blendshapes**, which is the ordinary way to
collapse a bilateral pair into one intensity. Pooling is deliberately lossy: it trades
left/right detail for a compact, interpretable space. It is *not* a validated FACS
coder -- Meta's shapes are named after AUs but are not calibrated to FACS intensity
scales, so treat an "AU4" channel as "the brow-lowering shape pair" rather than as a
certified AU4 measurement.

The ``negation`` subset collects the units with reported links to negation, rejection,
and disagreement in the facial-expression literature: brow lowering, lip-corner
depression, nose wrinkling and upper-lip raising (the disgust/rejection cluster), the
brow-raise pair associated with disbelief, and the lip tightening/pressing/stretching
group associated with withholding or refusal.
"""

from __future__ import annotations

NUM_BLENDSHAPES = 63

#: ``OVRPlugin.FaceExpression`` order; index i names ``Facial.feature_i``.
BLENDSHAPE_NAMES: tuple[str, ...] = (
    "Brow_Lowerer_L",  # 0
    "Brow_Lowerer_R",  # 1
    "Cheek_Puff_L",  # 2
    "Cheek_Puff_R",  # 3
    "Cheek_Raiser_L",  # 4
    "Cheek_Raiser_R",  # 5
    "Cheek_Suck_L",  # 6
    "Cheek_Suck_R",  # 7
    "Chin_Raiser_B",  # 8
    "Chin_Raiser_T",  # 9
    "Dimpler_L",  # 10
    "Dimpler_R",  # 11
    "Eyes_Closed_L",  # 12
    "Eyes_Closed_R",  # 13
    "Eyes_Look_Down_L",  # 14
    "Eyes_Look_Down_R",  # 15
    "Eyes_Look_Left_L",  # 16
    "Eyes_Look_Left_R",  # 17
    "Eyes_Look_Right_L",  # 18
    "Eyes_Look_Right_R",  # 19
    "Eyes_Look_Up_L",  # 20
    "Eyes_Look_Up_R",  # 21
    "Inner_Brow_Raiser_L",  # 22
    "Inner_Brow_Raiser_R",  # 23
    "Jaw_Drop",  # 24
    "Jaw_Sideways_Left",  # 25
    "Jaw_Sideways_Right",  # 26
    "Jaw_Thrust",  # 27
    "Lid_Tightener_L",  # 28
    "Lid_Tightener_R",  # 29
    "Lip_Corner_Depressor_L",  # 30
    "Lip_Corner_Depressor_R",  # 31
    "Lip_Corner_Puller_L",  # 32
    "Lip_Corner_Puller_R",  # 33
    "Lip_Funneler_LB",  # 34
    "Lip_Funneler_LT",  # 35
    "Lip_Funneler_RB",  # 36
    "Lip_Funneler_RT",  # 37
    "Lip_Pucker_L",  # 38
    "Lip_Pucker_R",  # 39
    "Lip_Stretcher_L",  # 40
    "Lip_Stretcher_R",  # 41
    "Lip_Suck_LB",  # 42
    "Lip_Suck_LT",  # 43
    "Lip_Suck_RB",  # 44
    "Lip_Suck_RT",  # 45
    "Lip_Pressor_L",  # 46
    "Lip_Pressor_R",  # 47
    "Lip_Tightener_L",  # 48
    "Lip_Tightener_R",  # 49
    "Lips_Toward",  # 50  -- FACE_MAP expands this one into SDK slots 50..53
    "Lower_Lip_Depressor_L",  # 51
    "Lower_Lip_Depressor_R",  # 52
    "Mouth_Left",  # 53
    "Mouth_Right",  # 54
    "Nose_Wrinkler_L",  # 55
    "Nose_Wrinkler_R",  # 56
    "Outer_Brow_Raiser_L",  # 57
    "Outer_Brow_Raiser_R",  # 58
    "Upper_Lid_Raiser_L",  # 59  -- SDK 68, per neo/README.md
    "Upper_Lid_Raiser_R",  # 60  -- SDK 69, per neo/README.md
    "Upper_Lip_Raiser_L",  # 61
    "Upper_Lip_Raiser_R",  # 62
)

#: Action Unit name -> the blendshape indices averaged to form it.
#:
#: The eight gaze shapes (indices 14..21) are deliberately excluded: they report eye
#: direction rather than facial muscle action, and the ``Eye`` modality already carries
#: gaze as poses and velocities.
ACTION_UNIT_MEMBERS: dict[str, tuple[int, ...]] = {
    "AU1_inner_brow_raiser": (22, 23),
    "AU2_outer_brow_raiser": (57, 58),
    "AU4_brow_lowerer": (0, 1),
    "AU5_upper_lid_raiser": (59, 60),
    "AU6_cheek_raiser": (4, 5),
    "AU7_lid_tightener": (28, 29),
    "AU8_lips_toward": (50,),
    "AU9_nose_wrinkler": (55, 56),
    "AU10_upper_lip_raiser": (61, 62),
    "AU12_lip_corner_puller": (32, 33),
    "AU14_dimpler": (10, 11),
    "AU15_lip_corner_depressor": (30, 31),
    "AU16_lower_lip_depressor": (51, 52),
    "AU17_chin_raiser": (8, 9),
    "AU18_lip_pucker": (38, 39),
    "AU20_lip_stretcher": (40, 41),
    "AU22_lip_funneler": (34, 35, 36, 37),
    "AU23_lip_tightener": (48, 49),
    "AU24_lip_pressor": (46, 47),
    "AU26_jaw_drop": (24,),
    "AU28_lip_suck": (42, 43, 44, 45),
    "AU29_jaw_thrust": (27,),
    "AU30_jaw_sideways": (25, 26),
    "AU33_cheek_puff": (2, 3),
    "AU35_cheek_suck": (6, 7),
    "AU43_eyes_closed": (12, 13),
    "mouth_sideways": (53, 54),
}

#: Units with reported associations to negation, rejection, and disagreement.
NEGATION_ACTION_UNITS: tuple[str, ...] = (
    "AU1_inner_brow_raiser",
    "AU2_outer_brow_raiser",
    "AU4_brow_lowerer",
    "AU9_nose_wrinkler",
    "AU10_upper_lip_raiser",
    "AU14_dimpler",
    "AU15_lip_corner_depressor",
    "AU17_chin_raiser",
    "AU20_lip_stretcher",
    "AU23_lip_tightener",
    "AU24_lip_pressor",
)

ACTION_UNIT_SUBSETS: dict[str, tuple[str, ...]] = {
    "all": tuple(ACTION_UNIT_MEMBERS),
    "negation": NEGATION_ACTION_UNITS,
}


def action_unit_names(subset: str = "all") -> tuple[str, ...]:
    """Return the ordered unit names for one subset."""

    if subset not in ACTION_UNIT_SUBSETS:
        raise ValueError(
            f"Unknown action unit subset {subset!r}; "
            f"expected one of {sorted(ACTION_UNIT_SUBSETS)}"
        )
    return ACTION_UNIT_SUBSETS[subset]


def _validate_tables() -> None:
    """Guard the hand-written tables at import time.

    A silent off-by-one in these indices would mislabel every Action Unit channel while
    still producing plausible-looking numbers, so the cheap checks run eagerly.
    """

    if len(BLENDSHAPE_NAMES) != NUM_BLENDSHAPES:
        raise RuntimeError(
            f"BLENDSHAPE_NAMES has {len(BLENDSHAPE_NAMES)} entries; "
            f"expected {NUM_BLENDSHAPES}"
        )
    if len(set(BLENDSHAPE_NAMES)) != NUM_BLENDSHAPES:
        raise RuntimeError("BLENDSHAPE_NAMES contains duplicates")
    if BLENDSHAPE_NAMES[50] != "Lips_Toward":
        raise RuntimeError("Blendshape 50 must be Lips_Toward to match FACE_MAP")
    if BLENDSHAPE_NAMES[59:61] != ("Upper_Lid_Raiser_L", "Upper_Lid_Raiser_R"):
        raise RuntimeError(
            "Blendshapes 59/60 must be Upper_Lid_Raiser_L/R to match SDK slots 68/69"
        )
    for unit, members in ACTION_UNIT_MEMBERS.items():
        if not members:
            raise RuntimeError(f"Action unit {unit!r} has no members")
        if len(set(members)) != len(members):
            raise RuntimeError(f"Action unit {unit!r} repeats a blendshape")
        for index in members:
            if not 0 <= index < NUM_BLENDSHAPES:
                raise RuntimeError(
                    f"Action unit {unit!r} references blendshape {index}, out of range"
                )
    gaze = set(range(14, 22))
    pooled = {index for members in ACTION_UNIT_MEMBERS.values() for index in members}
    if pooled & gaze:
        raise RuntimeError("Gaze blendshapes must not be pooled into Action Units")
    missing = set(range(NUM_BLENDSHAPES)) - pooled - gaze
    if missing:
        raise RuntimeError(
            f"Non-gaze blendshapes missing from every Action Unit: {sorted(missing)}"
        )
    for subset, names in ACTION_UNIT_SUBSETS.items():
        unknown = sorted(set(names) - set(ACTION_UNIT_MEMBERS))
        if unknown:
            raise RuntimeError(f"Subset {subset!r} names unknown units: {unknown}")


_validate_tables()


__all__ = [
    "ACTION_UNIT_MEMBERS",
    "ACTION_UNIT_SUBSETS",
    "BLENDSHAPE_NAMES",
    "NEGATION_ACTION_UNITS",
    "NUM_BLENDSHAPES",
    "action_unit_names",
]
