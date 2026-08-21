from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    keypoint_conf: float = 0.4
    max_hold: int = 6
    min_punch_frames: int = 1
    keypoint_smoothing_window: int = 3


@dataclass(frozen=True)
class DefenseConfig:
    # How far below the shoulder the wrist has to fall for the guard to
    # count as dropped
    guard_drop_torso_fraction: float = 0.75


@dataclass(frozen=True)
class StrikeConfig:
    # Straight punches
    straight_angle_threshold: float = 150.0  # Minimum angle to be considered a straight
    min_straight_speed_mps: float = 3.5  # Minimum wrist speed in meters per second
    max_straight_speed_mps: float = 15.0

    # Hooks
    hook_sweep_speed_threshold: float = (
        450.0  # Minimum sweep speed in degrees per second
    )
    max_hook_sweep_speed: float = 2000.0  # Maximum sweep speed in degrees per second
    hook_elbow_angle_threshold: float = 100  # Maximum angle to be considered a hook
    hook_wrist_shoulder_line_angle_threshold: float = 120

    # Uppercuts
    min_uppercut_speed_mps: float = (
        3.5  # Minimum upward wrist speed in meters per second
    )
    max_uppercut_speed_mps: float = (
        15.0  # Maximum upward wrist speed in meters per second
    )
    min_uppercut_rise_m: float = 0.15  # Minimum upward wrist travel in meters
    uppercut_arm_body_angle_threshold: float = 60.0  # Maximum angle, arm stays tucked
    uppercut_elbow_angle_threshold: float = 90.0  # Maximum angle, tighter than a hook

    # Arm body/ Armpit
    arm_body_angle_threshold: float = (
        65.0  # Minimum angle to not be considered an arm swing down
    )

    # Kicks
    angle_between_shins_threshold: float = 90  # Minimum angle
    min_kick_speed: float = 3.0  # Minimum ankle speed in meters per second
    max_kick_speed: float = 15.0  # Maximum ankle speed in meters per second
    min_kick_ankle_raise_m: float = (
        0.25  # Minimum height of the kicking ankle above the pivot ankle in meters
    )

    # Other
    max_hold: int = 6
    min_punch_frames: int = 1
