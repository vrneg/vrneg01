"""Fixed-size feature extraction for the eight VR event modalities.

Only measurements that are available at inference time are used.  Database IDs,
absolute timestamps, counters, message IDs, and raw participant IDs are deliberately
excluded from the continuous feature vectors.
"""

from __future__ import annotations

import math
import struct
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import torch


FEATURE_SCHEMA_VERSION = 3

MODALITY_NAMES: tuple[str, ...] = (
    "Eye",
    "Facial",
    "Head",
    "Body",
    "LeftHand",
    "RightHand",
    "LeftFinger",
    "RightFinger",
)
MODALITY_TO_ID: dict[str, int] = {
    name: modality_id for modality_id, name in enumerate(MODALITY_NAMES)
}

POSE_DIM = 7  # xyz position + xyzw quaternion
EYE_DIM = 18  # two eyes * (confidence, valid, position, quaternion)
FACIAL_DIM = 67  # 63 blendshapes + 2 confidences + 2 validity flags
FINGER_DIM = 130  # continuous tracked-hand measurements only
FINGER_FLAG_DIM = 13  # 8 status bits + 5 pinch bits, encoded separately

BASE_MODALITY_DIMS_BY_NAME: dict[str, int] = {
    "Eye": EYE_DIM,
    "Facial": FACIAL_DIM,
    "Head": POSE_DIM,
    "Body": POSE_DIM,
    "LeftHand": POSE_DIM,
    "RightHand": POSE_DIM,
    "LeftFinger": FINGER_DIM,
    "RightFinger": FINGER_DIM,
}
MOTION_DIMS_BY_NAME: dict[str, int] = {
    "Eye": 12,  # linear + angular velocity for both eyes
    "Facial": 63,  # blendshape velocity
    "Head": 6,  # linear + angular velocity
    "Body": 6,
    "LeftHand": 6,
    "RightHand": 6,
    # scale, pointer/root poses, 26 bone angular velocities, pinch velocity
    "LeftFinger": 96,
    "RightFinger": 96,
}
MODALITY_DIMS_BY_NAME: dict[str, int] = {
    name: BASE_MODALITY_DIMS_BY_NAME[name] + MOTION_DIMS_BY_NAME[name]
    for name in MODALITY_NAMES
}
BASE_MODALITY_DIMS: dict[int, int] = {
    MODALITY_TO_ID[name]: dimension
    for name, dimension in BASE_MODALITY_DIMS_BY_NAME.items()
}
MODALITY_DIMS: dict[int, int] = {
    MODALITY_TO_ID[name]: dimension
    for name, dimension in MODALITY_DIMS_BY_NAME.items()
}

ACTOR_SAME_AS_ANCHOR = 0
ACTOR_DIFFERENT_FROM_ANCHOR = 1
ACTOR_UNKNOWN = 2
NUM_ACTOR_RELATIONS = 3
FINGER_MODALITY_IDS = frozenset(
    {MODALITY_TO_ID["LeftFinger"], MODALITY_TO_ID["RightFinger"]}
)


def _finite_float(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _fixed_values(values: Any, size: int) -> list[float]:
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        values = []
    result = [_finite_float(value) for value in values[:size]]
    result.extend([0.0] * (size - len(result)))
    return result


def _position(value: Any) -> list[float]:
    value = _mapping(value)
    return [_finite_float(value.get(axis)) for axis in ("x", "y", "z")]


def _quaternion(value: Any) -> list[float]:
    value = _mapping(value)
    quaternion = [
        _finite_float(value.get(axis)) for axis in ("x", "y", "z", "w")
    ]
    norm = math.sqrt(sum(component * component for component in quaternion))
    if norm < 1e-8:
        return [0.0, 0.0, 0.0, 0.0]
    quaternion = [component / norm for component in quaternion]
    if quaternion[3] < 0.0:
        quaternion = [-component for component in quaternion]
    return quaternion


def _pose(value: Any) -> list[float]:
    """Extract a pose with either OpenXR or lower-case field names."""

    value = _mapping(value)
    position = value.get("Position", value.get("position"))
    orientation = value.get("Orientation", value.get("rotation"))
    return _position(position) + _quaternion(orientation)


def _event_pose(event: Mapping[str, Any]) -> list[float]:
    return _position(event.get("position")) + _quaternion(event.get("rotation"))


def _decoded_confidence(value: Any) -> float:
    """Decode confidence floats that were serialized as their uint32 bit pattern."""

    if isinstance(value, int) and value not in (0, 1) and 0 <= value <= 0xFFFFFFFF:
        decoded = struct.unpack("<f", struct.pack("<I", value))[0]
        if math.isfinite(decoded):
            return float(decoded)
    return _finite_float(value)


def _bit_features(value: Any, width: int) -> list[float]:
    try:
        integer = int(value)
    except (TypeError, ValueError):
        integer = 0
    return [float(bool(integer & (1 << bit))) for bit in range(width)]


def has_detailed_finger_tracking(event: Mapping[str, Any]) -> bool:
    """Return whether a finger record contains an actual tracked-hand sample.

    In the stored data, records without the 26 bone rotations also have status zero,
    zero confidence/scale, and zero pointer/root poses. They are sensor heartbeats that
    report unavailable tracking rather than observed hand events.
    """

    bone_rotations = event.get("boneRotations")
    return (
        isinstance(bone_rotations, Sequence)
        and not isinstance(bone_rotations, (str, bytes))
        and len(bone_rotations) >= 26
    )


def _extract_eye(event: Mapping[str, Any]) -> list[float]:
    gazes = event.get("eyeGazes")
    if not isinstance(gazes, Sequence) or isinstance(gazes, (str, bytes)):
        gazes = []

    features: list[float] = []
    for gaze_index in range(2):
        gaze = _mapping(gazes[gaze_index]) if gaze_index < len(gazes) else {}
        features.extend(
            [
                _finite_float(gaze.get("Confidence")),
                float(bool(gaze.get("IsValid", False))),
                *_pose(gaze.get("Pose")),
            ]
        )
    return features


def _extract_facial(event: Mapping[str, Any]) -> list[float]:
    status = _mapping(event.get("status"))
    return [
        *_fixed_values(event.get("expressionWeights"), 63),
        *_fixed_values(event.get("expressionWeightConfidences"), 2),
        float(bool(status.get("IsValid", False))),
        float(bool(status.get("IsEyeFollowingBlendshapesValid", False))),
    ]


def _extract_finger(event: Mapping[str, Any]) -> list[float]:
    bone_rotations = event.get("boneRotations")
    if not has_detailed_finger_tracking(event):
        raise ValueError("Finger event does not contain detailed hand tracking")

    flattened_bones: list[float] = []
    for bone_index in range(26):
        bone = bone_rotations[bone_index] if bone_index < len(bone_rotations) else {}
        flattened_bones.extend(_quaternion(bone))

    finger_confidences = event.get("fingerConfidences")
    if not isinstance(finger_confidences, Sequence) or isinstance(
        finger_confidences, (str, bytes)
    ):
        finger_confidences = []
    decoded_finger_confidences = [
        _decoded_confidence(finger_confidences[index])
        if index < len(finger_confidences)
        else 0.0
        for index in range(5)
    ]

    return [
        _decoded_confidence(event.get("handConfidence")),
        _finite_float(event.get("handScale")),
        *_pose(event.get("pointerPose")),
        *_pose(event.get("rootPose")),
        *flattened_bones,
        *decoded_finger_confidences,
        *_fixed_values(event.get("pinchStrength"), 5),
    ]


def extract_finger_flags(event: Mapping[str, Any]) -> torch.Tensor:
    """Decode categorical status/pinch bitmasks into independently learnable flags."""

    values = _bit_features(event.get("status"), 8) + _bit_features(
        event.get("pinches"), 5
    )
    return torch.tensor(values, dtype=torch.float32)


def _rate(
    current: torch.Tensor,
    previous: torch.Tensor,
    delta_seconds: float,
) -> torch.Tensor:
    if delta_seconds <= 0.0:
        return torch.zeros_like(current)
    return torch.nan_to_num((current - previous) / delta_seconds)


def _angular_velocity(
    current: torch.Tensor,
    previous: torch.Tensor,
    delta_seconds: float,
) -> torch.Tensor:
    """Shortest-path quaternion angular velocity for xyzw quaternions."""

    if delta_seconds <= 0.0:
        return current.new_zeros(3)
    current = current / current.norm().clamp_min(1e-8)
    previous = previous / previous.norm().clamp_min(1e-8)
    current_vector, current_w = current[:3], current[3]
    previous_vector, previous_w = -previous[:3], previous[3]
    relative_vector = (
        current_w * previous_vector
        + previous_w * current_vector
        + torch.linalg.cross(current_vector, previous_vector)
    )
    relative_w = current_w * previous_w - torch.dot(
        current_vector, previous_vector
    )
    if relative_w < 0:
        relative_vector = -relative_vector
        relative_w = -relative_w
    vector_norm = relative_vector.norm()
    if vector_norm < 1e-8:
        return current.new_zeros(3)
    angle = 2.0 * torch.atan2(vector_norm, relative_w.clamp(-1.0, 1.0))
    return torch.nan_to_num(relative_vector / vector_norm * angle / delta_seconds)


def _pose_velocity(
    current: torch.Tensor,
    previous: torch.Tensor,
    position_start: int,
    quaternion_start: int,
    delta_seconds: float,
) -> torch.Tensor:
    return torch.cat(
        (
            _rate(
                current[position_start : position_start + 3],
                previous[position_start : position_start + 3],
                delta_seconds,
            ),
            _angular_velocity(
                current[quaternion_start : quaternion_start + 4],
                previous[quaternion_start : quaternion_start + 4],
                delta_seconds,
            ),
        )
    )


def motion_features(
    modality_id: int,
    current: torch.Tensor,
    previous: torch.Tensor | None,
    delta_seconds: float,
) -> torch.Tensor:
    """Derive physically meaningful first-order motion features per modality."""

    modality_name = MODALITY_NAMES[modality_id]
    expected_base_dimension = BASE_MODALITY_DIMS[modality_id]
    if current.shape != (expected_base_dimension,):
        raise ValueError(
            f"{modality_name} base feature shape is {tuple(current.shape)}; "
            f"expected {(expected_base_dimension,)}"
        )
    motion_dimension = MOTION_DIMS_BY_NAME[modality_name]
    if previous is None:
        return current.new_zeros(motion_dimension)

    if modality_name == "Eye":
        motion = torch.cat(
            (
                _pose_velocity(current, previous, 2, 5, delta_seconds),
                _pose_velocity(current, previous, 11, 14, delta_seconds),
            )
        )
    elif modality_name == "Facial":
        motion = _rate(current[:63], previous[:63], delta_seconds)
    elif modality_name in {"Head", "Body", "LeftHand", "RightHand"}:
        motion = _pose_velocity(current, previous, 0, 3, delta_seconds)
    else:
        components = [
            _rate(current[1:2], previous[1:2], delta_seconds),
            _pose_velocity(current, previous, 2, 5, delta_seconds),
            _pose_velocity(current, previous, 9, 12, delta_seconds),
        ]
        components.extend(
            _angular_velocity(
                current[16 + bone * 4 : 20 + bone * 4],
                previous[16 + bone * 4 : 20 + bone * 4],
                delta_seconds,
            )
            for bone in range(26)
        )
        components.append(_rate(current[125:130], previous[125:130], delta_seconds))
        motion = torch.cat(components)

    if motion.shape != (motion_dimension,):
        raise RuntimeError(
            f"{modality_name} motion feature shape is {tuple(motion.shape)}; "
            f"expected {(motion_dimension,)}"
        )
    return motion


def augment_with_motion(
    modality_id: int,
    current: torch.Tensor,
    previous: torch.Tensor | None,
    delta_seconds: float,
) -> torch.Tensor:
    """Append motion derivatives to one raw modality measurement."""

    result = torch.cat(
        (current, motion_features(modality_id, current, previous, delta_seconds))
    )
    expected_dimension = MODALITY_DIMS[modality_id]
    if result.shape != (expected_dimension,):
        raise RuntimeError(
            f"Augmented feature shape is {tuple(result.shape)}; "
            f"expected {(expected_dimension,)}"
        )
    return result


@dataclass(frozen=True, slots=True)
class EventFeatureExtractor:
    """Convert one raw database event into its modality-specific vector."""

    modality_dims: Mapping[int, int] = field(
        default_factory=lambda: dict(BASE_MODALITY_DIMS)
    )

    def should_keep(self, event: Mapping[str, Any]) -> bool:
        """Discard finger heartbeats for which no hand measurement was observed."""

        modality_name = str(event.get("event_type", ""))
        if modality_name in {"LeftFinger", "RightFinger"}:
            return has_detailed_finger_tracking(event)
        return True

    def extract(self, event: Mapping[str, Any]) -> tuple[int, torch.Tensor]:
        modality_name = str(event.get("event_type", ""))
        if modality_name not in MODALITY_TO_ID:
            raise ValueError(f"Unknown event modality: {modality_name!r}")

        if modality_name == "Eye":
            values = _extract_eye(event)
        elif modality_name == "Facial":
            values = _extract_facial(event)
        elif modality_name in {"Head", "Body", "LeftHand", "RightHand"}:
            values = _event_pose(event)
        else:
            values = _extract_finger(event)

        modality_id = MODALITY_TO_ID[modality_name]
        expected_dimension = self.modality_dims[modality_id]
        if len(values) != expected_dimension:
            raise RuntimeError(
                f"{modality_name} extractor produced {len(values)} values; "
                f"expected {expected_dimension}"
            )
        return modality_id, torch.tensor(values, dtype=torch.float32)


class FeatureNormalizer:
    """Per-modality standardization statistics fitted on training events only."""

    def __init__(
        self,
        means: Mapping[int, torch.Tensor],
        standard_deviations: Mapping[int, torch.Tensor],
    ) -> None:
        if means.keys() != standard_deviations.keys():
            raise ValueError("means and standard_deviations must have identical modalities")
        self.means = {key: value.detach().cpu().float() for key, value in means.items()}
        self.standard_deviations = {
            key: value.detach().cpu().float() for key, value in standard_deviations.items()
        }

    def normalize(self, modality_id: int, features: torch.Tensor) -> torch.Tensor:
        return (features - self.means[modality_id]) / self.standard_deviations[modality_id]

    def state_dict(self) -> dict[str, dict[str, torch.Tensor]]:
        return {
            str(modality_id): {
                "mean": self.means[modality_id],
                "standard_deviation": self.standard_deviations[modality_id],
            }
            for modality_id in sorted(self.means)
        }

    @classmethod
    def from_state_dict(
        cls, state_dict: Mapping[str, Mapping[str, torch.Tensor]]
    ) -> "FeatureNormalizer":
        return cls(
            means={int(key): value["mean"] for key, value in state_dict.items()},
            standard_deviations={
                int(key): value["standard_deviation"] for key, value in state_dict.items()
            },
        )
