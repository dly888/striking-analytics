from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    """Configuration parameters controlling detection thresholds and tracking behaviour."""

    keypoint_conf: float = 0.4
    max_hold: int = 6
    min_punch_frames: int = 1
    keypoint_smoothing_window: int = 3


@dataclass(frozen=True)
class StrikeConfig:
    # Straight punches
    straight_angle_threshold: float = 150.0  # Minimum angle to be considered a straight
    straight_speed_mps: float = 5.0  # Minimum wrist speed in meters per second

    # Hooks
    hook_sweep_speed_threshold: float = 450.0  # Minimum sweep speed in degrees per second
    hook_elbow_angle_threshold: float = 100  # Maximum angle to be considered a hook
    hook_wrist_shoulder_line_angle_threshold: float = 120

    # Arm body/ Armpit
    arm_body_angle_threshold: float = (
        65.0  # Minimum angle to not be considered an arm swing down
    )

    # Kicks
    angle_between_shins_threshold: float = 90  # Minimum angle
    kick_speed_mps: float = 4.0  # Minimum ankle speed in meters per second

    # Other
    max_hold: int = 6
    min_punch_frames: int = 1
