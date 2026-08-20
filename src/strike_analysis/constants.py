from __future__ import annotations

from typing import Literal

Side = Literal["left", "right"]

Stance = Literal["orthodox", "southpaw"]

# Order sets the row order of the detection mask
STRIKE_TYPES: tuple[str, ...] = (
    "straight",
    "hook",
    "uppercut",
    "kick",
)

SIDES: tuple[Side, ...] = ("left", "right")

KEYPOINT_NAMES: tuple[str, ...] = (
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
)
KEYPOINT_INDEX: dict[str, int] = {name: i for i, name in enumerate(KEYPOINT_NAMES)}
N_KEYPOINTS = len(KEYPOINT_NAMES)

# Joints that exist on both sides, ordered to match KEYPOINT_NAMES.
# Combine with SIDES as f"{side}_{joint}" to rebuild the sided keypoint
# names. "nose" is excluded since it has no side.
JOINT_NAMES: tuple[str, ...] = (
    "eye",
    "ear",
    "shoulder",
    "elbow",
    "wrist",
    "hip",
    "knee",
    "ankle",
)

SKELETON_EDGES: tuple[tuple[str, str], ...] = (
    ("left_shoulder", "right_shoulder"),
    ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_wrist"),
    ("left_shoulder", "left_hip"),
    ("right_shoulder", "right_hip"),
    ("left_hip", "right_hip"),
    ("left_hip", "left_knee"),
    ("left_knee", "left_ankle"),
    ("right_hip", "right_knee"),
    ("right_knee", "right_ankle"),
    ("nose", "left_shoulder"),
    ("nose", "right_shoulder"),
)
