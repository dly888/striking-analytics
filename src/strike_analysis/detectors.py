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
)
from .tracking import PersonTrack


def detect_straight(track: PersonTrack, side: Side, config: StrikeConfig) -> np.ndarray:
    """
    Detects when a straight punch occurs.

    Args:
        track: PersonTrack object
        side: Side which the strike is, left or right
        config: Config object

    Returns:
        Numpy array which is a boolean mask where which detects if a punch occurred for
        each frame.
    """

    speed = get_joint_speed(track, f"{side}_wrist", config)
    thresholds = get_relative_speed_threshold(speed, config)

    arm_lifted = (
            get_joint_angle(track, f"{side}_hip", f"{side}_shoulder", f"{side}_elbow")
            > config.arm_body_angle_threshold
    )

    arm_extended = (
            get_joint_angle(track, f"{side}_shoulder", f"{side}_elbow", f"{side}_wrist")
            > config.straight_angle_threshold
    )
    return arm_extended & arm_lifted & (speed > thresholds)


def detect_hook(track: PersonTrack, side: Side, config: StrikeConfig) -> np.ndarray:
    speed = get_joint_speed(track, f"{side}_wrist", config)
    thresholds = get_relative_speed_threshold(speed, config)

    arm_lifted = (
            get_joint_angle(track, f"{side}_hip", f"{side}_shoulder", f"{side}_elbow")
            > config.arm_body_angle_threshold
    )

    # Use hook angle threshold as maximum not minimum here
    arm_extended = (
            get_joint_angle(track, f"{side}_shoulder", f"{side}_elbow", f"{side}_wrist")
            < config.hook_angle_threshold
    )

    return arm_extended & arm_lifted & (speed > thresholds)


DETECTORS = {
    "straight": detect_straight,
    "hook": detect_hook,
    # Add other strikes later
}


class MoveAnalyser:
    def __init__(self, track: PersonTrack, config: StrikeConfig = StrikeConfig()):
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
                DETECTORS[strike.strike_type](self.track, strike.side, self.config)
                for strike in strikes
            ]
        ).astype(bool)
        return Detections(strikes, mask)