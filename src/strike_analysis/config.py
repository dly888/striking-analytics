from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    """Configuration parameters controlling detection thresholds and tracking behaviour."""

    keypoint_conf: float = 0.4
    velocity_percentile: float = 95.0
    max_hold: int = 6
    min_punch_frames: int = 1


@dataclass(frozen=True)
class StrikeConfig:
    straight_angle_threshold: float = 150.0  # Minimum angle to be considered a hook
    hook_elbow_angle_threshold: float = 100  # Maximum angle to be considered a hook
    hook_wrist_shoulder_line_angle_threshold: float= 90  # Maximum angle to be considered a hook
    arm_body_angle_threshold: float = 35.0  # Minimum angle to not be considered an arm swing down
    straight_speed_percentile: float = 95.0
    hook_angle_speed_percentile: float = 95.0
    max_hold: int = 6
    min_punch_frames: int = 1