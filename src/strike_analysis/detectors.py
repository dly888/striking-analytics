from __future__ import annotations

import numpy as np

from .config import StrikeConfig
from .detections import Detections, Strike
from .features import (
    get_arm_sweep_speed,
    get_joint_angle,
    get_joint_rise_speed,
    get_joint_speed,
    get_pixel_to_meter_ratio,
    get_speed_threshold, get_joint_speed_peaks,
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

    speed = get_joint_speed(state, f"{side}_wrist", strike_config)
    thresholds = get_speed_threshold(state, strike_config.min_straight_speed_mps)
    max_thresholds = get_speed_threshold(state, strike_config.max_straight_speed_mps)

    peaks = get_joint_speed_peaks(speed, thresholds, max_thresholds)
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
        state: PersonState, strike: Strike, strike_strike_config: StrikeConfig,
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

    arm_sweep_speed = get_arm_sweep_speed(state, side, strike_strike_config)
    arm_sweep_threshold = strike_strike_config.hook_sweep_speed_threshold
    arm_sweep_speed_peaks = get_joint_speed_peaks(
        arm_sweep_speed,
        arm_sweep_threshold,
        strike_strike_config.max_hook_sweep_speed,
    )

    wrist_speed = get_joint_speed(state, f"{side}_wrist", strike_strike_config)
    wrist_speed_threshold = get_speed_threshold(
        state, strike_strike_config.min_straight_speed_mps  # Use the straight speed here
    )

    arm_lifted = (
            get_joint_angle(state, f"{side}_hip", f"{side}_shoulder", f"{side}_elbow")
            > strike_strike_config.arm_body_angle_threshold
    )

    arm_bent = (
            get_joint_angle(state, f"{side}_shoulder", f"{side}_elbow", f"{side}_wrist")
            < strike_strike_config.hook_elbow_angle_threshold  # Threshold is a maximum
    )

    # Checks if the wrist is inward enough
    shoulder_rotated = (
            get_joint_angle(
                state, f"{opposite}_shoulder", f"{side}_shoulder", f"{side}_wrist"
            )
            < strike_strike_config.hook_wrist_shoulder_line_angle_threshold  # Threshold is a maximum
    )

    detections = np.full(len(wrist_speed), fill_value=False)

    for peak in arm_sweep_speed_peaks:
        window = slice(
            max(0, peak - 10),
            min(len(wrist_speed), peak + 11),
        )

        if np.any(arm_lifted[window] & arm_bent[window] & shoulder_rotated[window] & (
                wrist_speed[window] > wrist_speed_threshold[window])):
            detections[peak] = True

    return detections


def detect_uppercut(
        state: PersonState, strike: Strike, strike_config: StrikeConfig
) -> np.ndarray:
    """
    Detects whether an uppercut occurs.

    Use three conditions:
        - The upward wrist speed is fast enough
        - The wrist travels far enough upward, a guard bounce only rises a
          few centimeters while an uppercut carries the fist to the chin
        - Arm stays tucked with the elbow bent tighter than a hook

    Args:
        state: PersonState object to detect from
        strike: Information on the strike
        strike_config: Strike config values

    Returns:
        Numpy array which contains whether an uppercut is detected at each frame
    """
    side = strike.side

    # Peaks found on upward speed
    wrist_rise_speed = get_joint_rise_speed(state, f"{side}_wrist", strike_config)
    wrist_rise_speed_threshold = get_speed_threshold(
        state, strike_config.min_uppercut_speed_mps
    )
    wrist_rise_speed_max_threshold = get_speed_threshold(
        state, strike_config.max_uppercut_speed_mps
    )
    wrist_rise_speed_peaks = get_joint_speed_peaks(
        wrist_rise_speed,
        wrist_rise_speed_threshold,
        wrist_rise_speed_max_threshold,
    )

    arm_tucked = (
            get_joint_angle(state, f"{side}_hip", f"{side}_shoulder", f"{side}_elbow")
            < strike_config.uppercut_arm_body_angle_threshold  # Threshold is a maximum
    )

    arm_bent = (
            get_joint_angle(state, f"{side}_shoulder", f"{side}_elbow", f"{side}_wrist")
            < strike_config.uppercut_elbow_angle_threshold  # Threshold is a maximum
    )

    # A hand returning to guard after a punch also rises fast with a bent
    # arm, but it is preceded by an extended arm while an uppercut starts
    # from a compact guard
    arm_extended = (
            get_joint_angle(state, f"{side}_shoulder", f"{side}_elbow", f"{side}_wrist")
            > strike_config.straight_angle_threshold
    )

    wrist_y = state.positions(f"{side}_wrist")[:, 1]
    pixel_to_m_ratio = get_pixel_to_meter_ratio(state)

    detections = np.full(len(wrist_rise_speed), fill_value=False)

    for peak in wrist_rise_speed_peaks:
        before = slice(max(0, peak - 10), peak + 1)
        after = slice(peak, min(len(wrist_y), peak + 11))

        # Net upward travel from the lowest point before the peak to the
        # highest point after it, y decreases upward
        start_y = np.nanmax(wrist_y[before])
        end_y = np.nanmin(wrist_y[after])
        rise_m = (start_y - end_y) * pixel_to_m_ratio[peak]

        if (
                rise_m >= strike_config.min_uppercut_rise_m
                and arm_tucked[peak]
                and arm_bent[peak]
                and not np.any(arm_extended[before])
        ):
            detections[peak] = True

    return detections


def detect_kick(
        state: PersonState, strike: Strike, strike_config: StrikeConfig
) -> np.ndarray:
    """
    Detects whether a kick occurs.

    Use three conditions:
        - The angle between the shins is wide enough and the striking foot
          is faster than the pivot foot
        - The striking foot speed is fast enough
        - The striking ankle lifts high enough above the pivot ankle, a kick
          lifts the foot off the ground while footwork keeps both feet near
          ground level

    Args:
        state: PersonState object to detect from
        strike: Information on the strike
        strike_config: Strike config values

    Returns:
        Numpy array which contains whether a kick is detected at each frame
    """
    side = strike.side
    opposite = "left" if side == "right" else "right"

    # Pivot side features
    pivot_foot_xy = state.positions(f"{opposite}_ankle")
    pivot_knee_xy = state.positions(f"{opposite}_knee")
    pivot_shin_vector = pivot_foot_xy - pivot_knee_xy
    pivot_foot_speed = get_joint_speed(state, f"{opposite}_ankle", strike_config)

    # Strike side features
    strike_foot_xy = state.positions(f"{side}_ankle")
    strike_knee_xy = state.positions(f"{side}_knee")
    strike_shin_vector = strike_foot_xy - strike_knee_xy
    strike_foot_speed = get_joint_speed(state, f"{side}_ankle", strike_config)
    strike_foot_speed_threshold = get_speed_threshold(
        state, strike_config.min_kick_speed
    )
    strike_foot_speed_max_threshold = get_speed_threshold(
        state, strike_config.max_kick_speed
    )
    strike_foot_speed_peaks = get_joint_speed_peaks(
        strike_foot_speed,
        strike_foot_speed_threshold,
        strike_foot_speed_max_threshold,
    )

    # Angle between shins
    dot = np.sum(strike_shin_vector * pivot_shin_vector, axis=1)
    norms = np.linalg.norm(strike_shin_vector, axis=1) * np.linalg.norm(
        pivot_shin_vector, axis=1
    )
    angle_between_shins = np.degrees(np.arccos(np.clip(dot / norms, -1.0, 1.0)))

    # Height of the kicking ankle above the pivot ankle in meters.
    # Differentiates between a kick and footwork
    pixel_to_m_ratio = get_pixel_to_meter_ratio(state)
    ankle_raise = (pivot_foot_xy[:, 1] - strike_foot_xy[:, 1]) * pixel_to_m_ratio

    foot_lifted = ankle_raise > strike_config.min_kick_ankle_raise_m

    detections = np.full(len(strike_foot_speed), fill_value=False)

    for peak in strike_foot_speed_peaks:
        window = slice(
            max(0, peak - 10),
            min(len(strike_foot_speed), peak + 11),
        )

        if np.any(
                (angle_between_shins[window] > strike_config.angle_between_shins_threshold)
                & (strike_foot_speed[peak] > pivot_foot_speed[peak])) and np.any(
                foot_lifted[window]):

            detections[peak] = True

    return detections
