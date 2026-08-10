from __future__ import annotations

import numpy as np

from .config import StrikeConfig
from .detections import Detections, Strike
from .features import (
    get_arm_sweep_speed,
    get_joint_angle,
    get_joint_speed,
    get_punch_speed_threshold,
    get_relative_speed_threshold, get_joint_speed_peaks,
)
from .tracking import PersonState


def detect_straight(
    state: PersonState, strike: Strike, strike_config: StrikeConfig
) -> np.ndarray:
    """
    Detects when a straight punch occurs.

    Args:
        state: PersonState object
        strike: Strike object to be detected
        strike_config: Config object

    Returns:
        Numpy array which is a boolean mask where which detects if a punch occurred for
        each frame.
    """
    side = strike.side
    strike_type = strike.strike_type

    speed = get_joint_speed(state, f"{side}_wrist", strike_config)
    thresholds = get_relative_speed_threshold(speed, strike_config, strike_type)

    peaks = get_joint_speed_peaks(speed, thresholds)
    detections = np.full(shape=len(speed), fill_value=False)

    arm_lifted = (
        get_joint_angle(state, f"{side}_hip", f"{side}_shoulder", f"{side}_elbow")
        > strike_config.arm_body_angle_threshold
    )

    arm_extended = (
        get_joint_angle(state, f"{side}_shoulder", f"{side}_elbow", f"{side}_wrist")
        > strike_config.straight_angle_threshold
    )

    for peak in peaks:
        window = slice(
            max(0, peak - 10),
            min(len(speed), peak + 11),
        )

        if np.any(arm_lifted[window] & arm_extended[window]):
            detections[peak] = True

    return detections

def detect_hook(
    state: PersonState, strike: Strike, strike_strike_config: StrikeConfig
) -> np.ndarray:
    """
    Detects whether a hook occurs.

    Use three conditions:
        - Arm is lifted high enough
        - Elbow is bent enough
        - The angular speed relative to the shoulder line is fast enough

    Args:
        state: PersonState object to detect from
        strike: Information on the strike
        strike_strike_config: Strike strike_config values

    Returns:
        Numpy array which contains whether a hook is detected at each frame
    """
    side = strike.side
    opposite = "left" if side == "right" else "right"
    strike_type = strike.strike_type

    arm_sweep_speed = get_arm_sweep_speed(state, side, strike_strike_config)
    arm_sweep_thresholds = get_relative_speed_threshold(
        arm_sweep_speed, strike_strike_config, strike_type
    )

    wrist_speed = get_joint_speed(state, f"{side}_wrist", strike_strike_config)

    # Use "straight" as the strike type here
    wrist_speed_threshold = get_relative_speed_threshold(
        wrist_speed, strike_strike_config, "straight"
    )

    # Threshold is a minimum
    arm_lifted = (
        get_joint_angle(state, f"{side}_hip", f"{side}_shoulder", f"{side}_elbow")
        > strike_strike_config.arm_body_angle_threshold
    )

    # Threshold is a maximum
    arm_bent = (
        get_joint_angle(state, f"{side}_shoulder", f"{side}_elbow", f"{side}_wrist")
        < strike_strike_config.hook_elbow_angle_threshold
    )

    # Threshold is a maximum
    shoulder_rotated = (
        get_joint_angle(
            state, f"{opposite}_shoulder", f"{side}_shoulder", f"{side}_wrist"
        )
        < strike_strike_config.hook_wrist_shoulder_line_angle_threshold
    )

    return (
        arm_bent
        & arm_lifted
        & (arm_sweep_speed > arm_sweep_thresholds)
        & shoulder_rotated
        & (wrist_speed > wrist_speed_threshold)
    )


def detect_kick(
    state: PersonState, strike: Strike, strike_config: StrikeConfig
) -> np.ndarray:
    side = strike.side
    opposite = "left" if side == "right" else "right"

    pivot_foot_xy = state.positions(f"{opposite}_ankle")
    pivot_knee_xy = state.positions(f"{opposite}_knee")
    pivot_shin_vector = pivot_foot_xy - pivot_knee_xy

    strike_foot_xy = state.positions(f"{side}_ankle")
    strike_knee_xy = state.positions(f"{side}_knee")
    strike_shin_vector = strike_foot_xy - strike_knee_xy

    strike_foot_speed = get_joint_speed(state, f"{side}_ankle", strike_config)

    # Angle between shins
    dot = np.sum(strike_shin_vector * pivot_shin_vector, axis=1)
    norms = np.linalg.norm(strike_shin_vector, axis=1) * np.linalg.norm(
        pivot_shin_vector, axis=1
    )
    angle_between_shins = np.degrees(np.arccos(np.clip(dot / norms, -1.0, 1.0)))

    # Angle of pivot foot
    pivot_angle = np.degrees(
        np.arctan2(pivot_shin_vector[:, 0], -pivot_shin_vector[:, 1])
    )

    strike_foot_speed_threshold = get_relative_speed_threshold(
        strike_foot_speed, strike_config, "kick"
    )

    detections = (
        (angle_between_shins > strike_config.angle_between_shins_threshold)
        & (pivot_angle >= -5)
        & (pivot_angle <= 5)
        & (strike_foot_speed > strike_foot_speed_threshold)
    )

    return detections


DETECTORS = {"straight": detect_straight, "hook": detect_hook, "kick": detect_kick}


class MoveAnalyser:
    def __init__(
        self, track: PersonState, strike_config: StrikeConfig = StrikeConfig()
    ):
        self.track = track
        self.strike_config = strike_config
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
                DETECTORS[strike.strike_type](self.track, strike, self.strike_config)
                for strike in strikes
            ]
        ).astype(bool)
        return Detections(strikes, mask)
