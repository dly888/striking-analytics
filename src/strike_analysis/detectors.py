from __future__ import annotations

import numpy as np

from .config import StrikeConfig
from .constants import Side
from .detections import Detections, Strike
from .features import (
    get_joint_angle,
    get_joint_speed,
    get_punch_speed_threshold,
    get_relative_speed_threshold,
    get_wrist_angular_speed, get_arm_sweep_speed
)
from .tracking import PersonState


def detect_straight(track: PersonState, strike: Strike, config: StrikeConfig) -> np.ndarray:
    """
    Detects when a straight punch occurs.

    Args:
        track: PersonTrack object
        strike: Strike object to be detected
        config: Config object

    Returns:
        Numpy array which is a boolean mask where which detects if a punch occurred for
        each frame.
    """
    side = strike.side
    strike_type = strike.strike_type

    speed = get_joint_speed(track, f"{side}_wrist", config)
    thresholds = get_relative_speed_threshold(speed, config, strike_type)

    arm_lifted = (
            get_joint_angle(track, f"{side}_hip", f"{side}_shoulder", f"{side}_elbow")
            > config.arm_body_angle_threshold
    )

    arm_extended = (
            get_joint_angle(track, f"{side}_shoulder", f"{side}_elbow", f"{side}_wrist")
            > config.straight_angle_threshold
    )
    return arm_extended & arm_lifted & (speed > thresholds)


def detect_hook(track: PersonState, strike: Strike, config: StrikeConfig) -> np.ndarray:
    side = strike.side
    opposite = "left" if side == "right" else "right"
    strike_type = strike.strike_type

    speed = get_arm_sweep_speed(track, side, config)
    thresholds = get_relative_speed_threshold(speed, config, strike_type)

    # Threshold is a minimum
    arm_lifted = (
            get_joint_angle(track, f"{side}_hip", f"{side}_shoulder", f"{side}_elbow")
            > config.arm_body_angle_threshold
    )

    # Threshold is a maximum
    arm_bent = (
            get_joint_angle(track, f"{side}_shoulder", f"{side}_elbow", f"{side}_wrist")
            < config.hook_elbow_angle_threshold
    )

    # Threshold is a maximum
    shoulder_rotated = (
            get_joint_angle(track, f"{opposite}_shoulder", f"{side}_shoulder", f"{side}_wrist")
            < config.hook_wrist_shoulder_line_angle_threshold
    )

    return arm_bent & arm_lifted & (speed > thresholds) & shoulder_rotated


DETECTORS = {
    "straight": detect_straight,
    "hook": detect_hook,
    # Add other strikes later
}


class MoveAnalyser:
    def __init__(self, track: PersonState, config: StrikeConfig = StrikeConfig()):
        self.track = track
        self.config = config
        self.thresholds = get_punch_speed_threshold(track)

    def get_detections(self) -> Detections:
        """
        Detects strikes for each strike type.

        Returns:
            Detection object containing information on when a strike occurs.
        """
        strikes = tuple(
            Strike(technique, side)
            for technique in DETECTORS
            for side in ("left", "right")
        )
        mask = np.stack(
            [
                DETECTORS[strike.strike_type](self.track, strike, self.config)
                for strike in strikes
            ]
        ).astype(bool)
        return Detections(strikes, mask)