from __future__ import annotations

import numpy as np
from numpy import ndarray
from scipy.signal import find_peaks

from . import Side
from .config import StrikeConfig
from .geometry import calculate_angles
from .tracking import PersonState


def get_joint_speed(state: PersonState, joint_name: str, strike_config: StrikeConfig) -> np.ndarray:
    """Calculate the speed of a joint in pixels per second.

    Missing keypoint detections handled by using the most recent valid
    position, as long as the number of skipped frames stays under the
    maximum hold distance. Frames where speed  can't be calculated
    accurately return NaN.

    Args:
        state: PersonTrack object
        joint_name: Name of the joint/keypoint
        strike_config: Config object

    Returns:
        Numpy array containing the joint speed for each frame.
        Frames where the speed can't be calculated have NaN values.
    """

    xy = state.positions(joint_name)

    # Assume if x is NaN then y is NaN.
    visible = ~np.isnan(xy[:, 0])

    # Each frames[i] represents the frame index.
    frames = np.arange(len(xy))

    # For each frame store the most recent valid frame at or before it
    prefix = np.maximum.accumulate(np.where(visible, frames, -1))

    # Shift by one so that prefix[i] is strictly before frame i
    prefix = np.roll(prefix, 1)
    prefix[0] = -1

    previous = prefix

    gap = frames - previous

    measurable = visible & (previous >= 0) & (gap <= strike_config.max_hold + 1)

    speed = np.full(len(xy), np.nan)

    distance = np.linalg.norm(
        xy[measurable] - xy[previous[measurable]],
        axis=1,
    )

    speed[measurable] = distance * state.fps / gap[measurable]

    return speed

def get_joint_speed_peaks(speed: np.ndarray, threshold: np.ndarray, max_speed: np.ndarray) -> ndarray:
    """
    Gets the peak values of the joint speeds tracked.

    Args:
        speed: Speed of the joint at each frame
        threshold: Threshold to be considered a peak at each frame
        max_speed: Maximum speed at which a detection will be considered a glitch
    """
    peaks, properties = find_peaks(
        speed,
        height=threshold,
        distance=10,
    )

    # Filter out glitch spikes
    valid = properties["peak_heights"] <= (
        max_speed[peaks] if np.ndim(max_speed) else max_speed
    )

    return peaks[valid]


def get_joint_rise_speed(state: PersonState, name: str, config: StrikeConfig) -> np.ndarray:
    """Calculate the upward speed of a joint in pixels per second.

    Upward movement is positive since image y grows downward. Missing
    keypoint detections handled by using the most recent valid position,
    as long as the number of skipped frames stays under the maximum hold
    distance. Frames where speed can't be calculated accurately return NaN.

    Args:
        state: PersonTrack object
        name: Name of the joint/keypoint
        config: Config object

    Returns:
        Numpy array containing the upward joint speed for each frame.
        Frames where the speed can't be calculated have NaN values.
    """

    xy = state.positions(name)

    # Assume if x is NaN then y is NaN.
    visible = ~np.isnan(xy[:, 0])

    # Each frames[i] represents the frame index.
    frames = np.arange(len(xy))

    # For each frame store the most recent valid frame at or before it
    prefix = np.maximum.accumulate(np.where(visible, frames, -1))

    # Shift by one so that prefix[i] is strictly before frame i
    prefix = np.roll(prefix, 1)
    prefix[0] = -1

    previous = prefix

    gap = frames - previous

    measurable = visible & (previous >= 0) & (gap <= config.max_hold + 1)

    rise_speed = np.full(len(xy), np.nan)

    # Upward movement decreases y, so previous minus current is positive
    rise = xy[previous[measurable], 1] - xy[measurable, 1]

    rise_speed[measurable] = rise * state.fps / gap[measurable]

    return rise_speed


def get_wrist_angular_speed(
    state: PersonState,
    side: Side,
    config: StrikeConfig,
) -> np.ndarray:
    """Calculate wrist angular speed relative to the corresponding shoulder.

    Missing keypoint detections are handled by using the most recent valid
    angle, as long as the number of skipped frames stays under the
    maximum hold distance. Frames where angular speed can't be calculated
    accurately return NaN.

    Args:
        state: PersonState object.
        side: The side which the wrist is on.
        config: StrikeConfig object.

    Returns:
        Numpy array containing the angular speed for each frame in
        degrees per second. Frames where the angular speed can't be
        calculated have NaN values.
    """

    wrist_xy = state.positions(f"{side}_wrist")
    shoulder_xy = state.positions(f"{side}_shoulder")

    delta = wrist_xy - shoulder_xy
    angle = np.degrees(np.arctan2(delta[:, 1], delta[:, 0]))

    # Assume if the angle is NaN then the wrist/shoulder position is invalid
    visible = ~np.isnan(angle)

    # Each frames[i] represents the frame index.
    frames = np.arange(len(angle))

    # For each frame store the most recent valid frame at or before it
    prefix = np.maximum.accumulate(np.where(visible, frames, -1))

    # Shift by one so that prefix[i] is strictly before frame i
    prefix = np.roll(prefix, 1)
    prefix[0] = -1

    previous = prefix

    gap = frames - previous

    measurable = visible & (previous >= 0) & (gap <= config.max_hold + 1)

    # Signed change, wrapped into [-180, 180)
    d_angle = angle[measurable] - angle[previous[measurable]]

    d_angle = (d_angle + 180.0) % 360.0 - 180.0

    angular_speed = np.full(len(angle), np.nan)

    angular_speed[measurable] = np.abs(d_angle) * state.fps / gap[measurable]

    return angular_speed


def get_arm_sweep_speed(
    state: PersonState,
    side: Side,
    config: StrikeConfig,
) -> np.ndarray:
    """Calculate arm sweep speed relative to the shoulder line.

    Missing keypoint detections are handled by using the most recent valid
    angle, as long as the number of skipped frames stays under the
    maximum hold distance. Frames where sweep speed can't be calculated
    accurately return NaN.

    Only inward wrist movement towards the opposite shoulder is measured.

    Args:
        state: PersonState object.
        side: The side which the wrist is on.
        config: StrikeConfig object.

    Returns:
        Numpy array containing the arm sweep speed for each frame in
        degrees per second. Frames where the sweep speed can't be
        calculated have NaN values.
    """

    opposite = "left" if side == "right" else "right"

    striking_shoulder = state.positions(f"{side}_shoulder")
    opposite_shoulder = state.positions(f"{opposite}_shoulder")
    wrist = state.positions(f"{side}_wrist")

    # Vector from striking shoulder to opposite shoulder
    u = opposite_shoulder - striking_shoulder

    # Vector from striking shoulder to wrist
    v = wrist - striking_shoulder

    cross_product = u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0]

    dot_product = u[:, 0] * v[:, 0] + u[:, 1] * v[:, 1]

    angle = np.degrees(np.arctan2(cross_product, dot_product))

    # Assume if the angle is NaN then the required keypoints are invalid
    visible = ~np.isnan(angle)

    # Each frames[i] represents the frame index
    frames = np.arange(len(angle))

    # For each frame store the most recent valid frame at or before it
    prefix = np.maximum.accumulate(np.where(visible, frames, -1))

    # Shift by one so that prefix[i] is strictly before frame i
    prefix = np.roll(prefix, 1)
    prefix[0] = -1

    previous = prefix

    gap = frames - previous

    measurable = visible & (previous >= 0) & (gap <= config.max_hold + 1)

    # Signed change, wrapped into [-180, 180)
    d_angle = angle[measurable] - angle[previous[measurable]]

    d_angle = (d_angle + 180.0) % 360.0 - 180.0

    # Only detect inward wrist movement towards the opposite shoulder
    closing = -np.sign(angle[previous[measurable]]) * d_angle

    sweep = np.full(len(angle), np.nan)

    sweep[measurable] = closing * state.fps / gap[measurable]

    return sweep


def get_joint_angle(track: PersonState, a: str, b: str, c: str) -> np.ndarray:
    """
    Gets the angle between three joints.

    Args:
        track: PersonTrack object.
        a: Position vector of the first keypoint/joint
        b: Position vector of the second keypoint/joint, the joint where the angle is actually at.
        c: Position vector of the third keypoint/joint

    Returns:
        Angle ABC in degrees.
    """
    return calculate_angles(
        track.positions(a),
        track.positions(b),
        track.positions(c),
    )


def get_pixel_to_meter_ratio(
    track: PersonState,
) -> np.ndarray:
    """
    Calculates pixel-to-meter conversion ratio for each frame.

    Conversion estimated using user's height to approximate torso
    length .

    Args:
        track: PersonTrack object

    Returns:
        Numpy array containing the pixel to meter ratio for each frame.
    """
    shoulder_centre = (
        track.positions("left_shoulder") + track.positions("right_shoulder")
    ) / 2
    hip_centre = (track.positions("left_hip") + track.positions("right_hip")) / 2

    torso_pixels = np.linalg.norm(
        shoulder_centre - hip_centre,
        axis=1,
    )

    # Torso length in pixels shrinks when the person crouches or bends over,
    # so floor it at half the video median to stop the ratio exploding
    torso_floor = np.nanmedian(torso_pixels) * 0.5
    torso_pixels = np.maximum(torso_pixels, torso_floor)

    height_m = track.person.height_m
    torso_length_m = height_m * 0.3
    ratio = torso_length_m / torso_pixels

    return ratio


def get_speed_threshold(track: PersonState, speed_mps: float) -> np.ndarray:
    """
    Calculates a speed threshold based on a fixed real life speed.

    Args:
        track: PersonTrack object
        speed_mps: Speed threshold in meters per second, set at config.

    Returns:
        Numpy array containing the threshold at each frame in pixels per second.
        :rtype: np.ndarray
    """
    pixel_to_m_ratio = get_pixel_to_meter_ratio(track)
    thresholds = speed_mps / pixel_to_m_ratio
    return thresholds
