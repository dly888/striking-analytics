from __future__ import annotations

import numpy as np
import pytest

from strike_analysis import StrikeConfig
from strike_analysis.constants import KEYPOINT_INDEX
from strike_analysis.detections import Strike
from strike_analysis.detectors import (
    DETECTORS,
    MoveAnalyser,
    detect_hook,
    detect_kick,
    detect_straight,
    detect_uppercut,
)

N_FRAMES = 60

# A fighter standing in guard. Torso length is 200 pixels so with a
# 1.80m person the pixel to meter ratio is 0.0027
GUARD_POSE: dict[str, tuple[float, float]] = {
    "nose": (500, 100),
    "left_eye": (495, 95),
    "right_eye": (505, 95),
    "left_ear": (490, 100),
    "right_ear": (510, 100),
    "left_shoulder": (480, 200),
    "right_shoulder": (520, 200),
    "left_elbow": (470, 280),
    "right_elbow": (530, 280),
    "left_wrist": (465, 250),
    "right_wrist": (535, 250),
    "left_hip": (485, 400),
    "right_hip": (515, 400),
    "left_knee": (480, 550),
    "right_knee": (520, 550),
    "left_ankle": (480, 700),
    "right_ankle": (520, 700),
}


@pytest.fixture()
def guard_keypoints():
    keypoints = np.zeros(shape=(N_FRAMES, 17, 3))

    for name, (x, y) in GUARD_POSE.items():
        keypoints[:, KEYPOINT_INDEX[name], 0] = x
        keypoints[:, KEYPOINT_INDEX[name], 1] = y

    keypoints[:, :, 2] = 1
    return keypoints


def move_keypoint(keypoints, name, start_frame, target, steps):
    """
    Moves a keypoint from its current position towards a target, one
    interpolation step per frame, then holds it at the final position.

    Args:
        keypoints: Keypoints array to be modified in place
        name: Name of the keypoint to move
        start_frame: Frame index where the movement starts
        target: The position the keypoint moves to
        steps: Interpolation fractions between 0 and 1, one per frame
    """
    idx = KEYPOINT_INDEX[name]
    origin = keypoints[start_frame - 1, idx, :2].copy()
    direction = np.asarray(target, dtype=float) - origin

    for step, t in enumerate(steps):
        keypoints[start_frame + step:, idx, :2] = origin + t * direction


def test_detect_straight_fast_extension(guard_keypoints, make_person_state):
    move_keypoint(guard_keypoints, "right_elbow", 30, (600, 210), [0.3, 0.75, 1.0])
    move_keypoint(guard_keypoints, "right_wrist", 30, (695, 215), [0.3, 0.75, 1.0])

    state = make_person_state(keypoints=guard_keypoints)

    detections = detect_straight(state, Strike("straight", "right"), StrikeConfig())

    assert detections.any()


def test_detect_straight_slow_extension_ignored(guard_keypoints, make_person_state):
    # Same trajectory but spread over 20 frames, too slow to be a punch
    steps = np.linspace(0.05, 1.0, 20)
    move_keypoint(guard_keypoints, "right_elbow", 30, (600, 210), steps)
    move_keypoint(guard_keypoints, "right_wrist", 30, (695, 215), steps)

    state = make_person_state(keypoints=guard_keypoints)

    detections = detect_straight(state, Strike("straight", "right"), StrikeConfig())

    assert not detections.any()


def set_hook_arc(keypoints, alphas):
    """
    Places the right wrist on a circle around the right shoulder, one
    angle per frame. Angle 0 points along the shoulder line towards the
    opposite shoulder, matching the sweep angle in get_arm_sweep_speed.

    Args:
        keypoints: Keypoints array to be modified in place
        alphas: Wrist angle in degrees for each frame
    """
    radius = np.hypot(90, 90)
    a = np.radians(alphas)

    keypoints[:, KEYPOINT_INDEX["right_wrist"], 0] = 520 - radius * np.cos(a)
    keypoints[:, KEYPOINT_INDEX["right_wrist"], 1] = 200 - radius * np.sin(a)


def test_detect_hook_fast_swing(guard_keypoints, make_person_state):
    # The arm starts with the elbow lifted then the wrist sweeps
    # inward
    alphas = np.full(N_FRAMES, 135.0)
    alphas[30:33] = (110, 75, 45)
    alphas[33:] = 45
    set_hook_arc(guard_keypoints, alphas)

    guard_keypoints[:30, KEYPOINT_INDEX["right_elbow"], :2] = (610, 200)
    move_keypoint(guard_keypoints, "right_elbow", 30, (520, 110), [0.3, 0.65, 1.0])

    state = make_person_state(keypoints=guard_keypoints)

    detections = detect_hook(state, Strike("hook", "right"), StrikeConfig())

    assert detections.any()


def test_detect_hook_slow_swing_ignored(guard_keypoints, make_person_state):
    # The same arc spread over 12 frames sweeps too slowly to be a hook
    alphas = np.full(N_FRAMES, 135.0)
    alphas[30:42] = np.linspace(127.5, 45, 12)
    alphas[42:] = 45
    set_hook_arc(guard_keypoints, alphas)

    guard_keypoints[:30, KEYPOINT_INDEX["right_elbow"], :2] = (610, 200)
    move_keypoint(guard_keypoints, "right_elbow", 30, (520, 110), np.linspace(0.1, 1.0, 12))

    state = make_person_state(keypoints=guard_keypoints)

    detections = detect_hook(state, Strike("hook", "right"), StrikeConfig())

    assert not detections.any()


def test_detect_kick_chambered_kick(guard_keypoints, make_person_state):
    move_keypoint(guard_keypoints, "right_knee", 30, (560, 450), [0.25, 0.6, 0.9, 1.0])
    move_keypoint(guard_keypoints, "right_ankle", 30, (700, 430), [0.25, 0.6, 0.9, 1.0])

    state = make_person_state(keypoints=guard_keypoints)

    detections = detect_kick(state, Strike("kick", "right"), StrikeConfig())

    assert detections.any()


def test_detect_kick_footwork_ignored(guard_keypoints, make_person_state):
    move_keypoint(guard_keypoints, "right_knee", 30, (700, 550), [0.25, 0.6, 0.9, 1.0])
    move_keypoint(guard_keypoints, "right_ankle", 30, (700, 700), [0.25, 0.6, 0.9, 1.0])

    state = make_person_state(keypoints=guard_keypoints)

    detections = detect_kick(state, Strike("kick", "right"), StrikeConfig())

    assert not detections.any()


def test_detect_uppercut_rising_from_guard(guard_keypoints, make_person_state):
    move_keypoint(guard_keypoints, "right_elbow", 30, (535, 200), [0.3, 0.75, 1.0])
    move_keypoint(guard_keypoints, "right_wrist", 30, (545, 120), [0.3, 0.75, 1.0])

    state = make_person_state(keypoints=guard_keypoints)

    detections = detect_uppercut(state, Strike("uppercut", "right"), StrikeConfig())

    assert detections.any()


def test_detect_uppercut_guard_return_ignored(guard_keypoints, make_person_state):
    # The same rise preceded by an extended arm is a hand returning to
    # guard after a punch, not an uppercut
    wrist_idx = KEYPOINT_INDEX["right_wrist"]
    elbow_idx = KEYPOINT_INDEX["right_elbow"]
    guard_keypoints[15:25, elbow_idx, :2] = (600, 210)
    guard_keypoints[15:25, wrist_idx, :2] = (695, 215)

    move_keypoint(guard_keypoints, "right_elbow", 30, (535, 200), [0.3, 0.75, 1.0])
    move_keypoint(guard_keypoints, "right_wrist", 30, (545, 120), [0.3, 0.75, 1.0])

    state = make_person_state(keypoints=guard_keypoints)

    detections = detect_uppercut(state, Strike("uppercut", "right"), StrikeConfig())

    assert not detections.any()


def test_straight_takes_priority_over_hook(guard_keypoints, make_person_state, monkeypatch):
    state = make_person_state(keypoints=guard_keypoints)

    scripted = {
        ("straight", "left"): (30,),
        ("hook", "left"): (35,),
        ("hook", "right"): (35,),
    }

    def scripted_detector(track, strike, config):
        mask = np.full(N_FRAMES, False)
        for frame in scripted.get((strike.strike_type, strike.side), ()):
            mask[frame] = True
        return mask

    for name in DETECTORS:
        monkeypatch.setitem(DETECTORS, name, scripted_detector)

    detections = MoveAnalyser(state).get_detections()

    # The left hook is within 10 frames of the left straight so it is
    # the same punch, the right hook has no straight to clash with
    assert detections[Strike("straight", "left")].any()
    assert not detections[Strike("hook", "left")].any()
    assert detections[Strike("hook", "right")].any()
