from __future__ import annotations

import numpy as np

from . import Strike, Side
from .config import StrikeConfig
from .geometry import calculate_angles
from .tracking import PersonState


def get_joint_speed(track: PersonState, name: str, config: StrikeConfig) -> np.ndarray:
    """Calculate the speed of a joint in pixels per second.

    Missing keypoint detections handled by using the most recent valid
    position, as long as the number of skipped frames stays under the
    maximum hold distance. Frames where speed  can't be calculated
    accurately return NaN.

    Args:
        track: PersonTrack object
        name: Name of the joint/keypoint
        config: Config object

    Returns:
        Numpy array containing the joint speed for each frame.
        Frames where the speed can't be calculated have NaN values.
    """

    xy = track.positions(name)
    visible = ~np.isnan(xy[:, 0])
    frames = np.arange(len(xy))

    seen_at_or_before = np.maximum.accumulate(np.where(visible, frames, -1))
    previous = np.roll(seen_at_or_before, 1)
    previous[0] = -1

    gap = frames - previous
    measurable = visible & (previous >= 0) & (gap <= config.max_hold + 1)

    speed = np.full(len(xy), np.nan)
    distance = np.linalg.norm(xy[measurable] - xy[previous[measurable]], axis=1)
    speed[measurable] = distance * track.fps / gap[measurable]

    return speed

def get_wrist_angular_speed(track: PersonState, side: Side, config: StrikeConfig) -> np.ndarray:
    """
    Calculates wrist angular speed relative to corresponding shoulder in degrees per second.

    Args:
        track: PersonTrack object
        side: The side which the wrist is on
        config: Strike config

    Returns:
        Numpy array containing current angular speed at each frame
    """
    wrist_xy = track.positions(f"{side}_wrist")
    shoulder_xy = track.positions(f"{side}_shoulder")

    delta = wrist_xy - shoulder_xy
    angle = np.degrees(np.arctan2(delta[:, 1], delta[:, 0]))

    visible = ~np.isnan(angle)
    frames = np.arange(len(angle))

    seen_at_or_before = np.maximum.accumulate(np.where(visible, frames, -1))
    previous = np.roll(seen_at_or_before, 1)
    previous[0] = -1
    gap = frames - previous
    measurable = visible & (previous >= 0) & (gap <= config.max_hold + 1)

    # Signed change, wrapped into [-180, 180) to handle the atan2 seam
    d_angle = angle[measurable] - angle[previous[measurable]]
    d_angle = (d_angle + 180.0) % 360.0 - 180.0

    angular_speed = np.full(len(angle), np.nan)
    angular_speed[measurable] = np.abs(d_angle) * track.fps / gap[measurable]

    return angular_speed

def get_arm_sweep_speed(track: PersonState, side: Side, config: StrikeConfig) -> np.ndarray:
    """
    Calculates wrist angular speed relative to shoulder line in degrees per second.
    Prevents false detections when fighter spins around with arm in air.

    Args:
        track: PersonTrack object
        side: The side which the wrist is on
        config: Strike config

    Returns:
        Numpy array containing current angular speed at each frame
    """
    opposite = "left" if side == "right" else "right"

    b = track.positions(f"{side}_shoulder")     # striking shoulder
    a = track.positions(f"{opposite}_shoulder") # opposite shoulder
    c = track.positions(f"{side}_wrist")

    u = a - b  # shoulder line/vector
    v = c - b  # arm line/vector

    cross_product = u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0]
    dot_product = u[:, 0] * v[:, 0] + u[:, 1] * v[:, 1]
    angle = np.degrees(np.arctan2(cross_product, dot_product))  # signed, (-180, 180], NaN propagates

    visible = ~np.isnan(angle)
    frames = np.arange(len(angle))

    seen_at_or_before = np.maximum.accumulate(np.where(visible, frames, -1))
    previous = np.roll(seen_at_or_before, 1)
    previous[0] = -1

    gap = frames - previous
    measurable = visible & (previous >= 0) & (gap <= config.max_hold + 1)

    d_angle = angle[measurable] - angle[previous[measurable]]
    d_angle = (d_angle + 180.0) % 360.0 - 180.0  # shortest rotation between frames

    # Makes sure to only detect wrist movement inwards(towards the opposite shoulder)
    closing = -np.sign(angle[previous[measurable]]) * d_angle

    sweep = np.full(len(angle), np.nan)
    sweep[measurable] = closing * track.fps / gap[measurable]

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


def get_relative_speed_threshold(speed: np.ndarray, config: StrikeConfig, strike_type: str) -> float:
    """
    Calculates the speed threshold using a top n percentile based on the entire video, set at config.

    Args:
        speed: Numpy array containing speeds for each frame.
        config: Config object

    Returns:
        The calculated speed threshold. Returns infinity if no valid speed
        values are available.
    """
    strike_percentile_dict = {
        "straight": config.straight_speed_percentile,
        "hook": config.hook_angle_speed_percentile
    }

    if not np.any(~np.isnan(speed)):
        return np.inf
    return float(np.nanpercentile(speed, strike_percentile_dict[strike_type]))


def get_pixel_to_meter_ratio(
        track: PersonState,
) -> np.ndarray:
    """
    Calculates pixel-to-meter conversion ratio for each frame.

    Conversion estimated using the user's wingspan to approximate shoulder
    width in real-world units.

    Args:
        track: PersonTrack object

    Returns:
        Numpy array containing the pixel to meter ratio for each frame.
    """
    left_shoulder = track.positions("left_shoulder")
    right_shoulder = track.positions("right_shoulder")

    shoulder_pixels = np.linalg.norm(
        right_shoulder - left_shoulder,
        axis=1,
    )

    wingspan_m = track.person.wingspan_m
    shoulder_width_m = wingspan_m * 0.25
    ratio = shoulder_width_m / shoulder_pixels

    return ratio


def get_punch_speed_threshold(track: PersonState):
    """
    Calculates punch speed threshold based on fixed real life value.

    Args:
        track: PersonTrack object

    Returns:
        Numpy array containing the threshold at each frame in pixels per second.
    """
    pixel_to_m_ratio = get_pixel_to_meter_ratio(track)
    thresholds = 5 / pixel_to_m_ratio
    return thresholds