from collections.abc import Callable

import pytest
import numpy as np

from strike_analysis import StrikeConfig, calculate_angles, get_joint_speed, get_pixel_to_meter_ratio, get_speed_threshold
from strike_analysis.constants import KEYPOINT_INDEX
from strike_analysis.features import get_joint_rise_speed, get_joint_speed_peaks


@pytest.fixture()
def all_nan_conf_keypoints():
    keypoints = np.full((100, 17, 3), np.nan)
    keypoints[:, :, :2] = 0.0
    return keypoints


@pytest.fixture()
def all_nan_xy_keypoints():
    keypoints = np.full((100, 17, 3), 1.0)
    keypoints[:, :, :2] = np.nan
    return keypoints


@pytest.fixture()
def constant_speed_keypoints():
    keypoints = np.zeros(shape=(100, 17, 3))
    keypoints[:, :, 0] = np.tile(np.arange(100)[:, None], (1, 17))  # x coordinate
    keypoints[:, :, 1] = 0  # y coordinate
    keypoints[:, :, 2] = 1  # confidence

    return keypoints


@pytest.fixture()
def constant_boxes():
    boxes = np.full(shape=(100, 4), fill_value=400)
    boxes[:, :2] = 100
    return boxes


@pytest.fixture()
def full_confidence_box():
    return np.full(shape=100, fill_value=1)


@pytest.fixture()
def standing_keypoints():
    # Torso length is 200 pixels, shoulder centre (500, 200) to hip
    # centre (500, 400), so with a 1.80m person the pixel to meter
    # ratio is 0.54 / 200 = 0.0027
    keypoints = np.zeros(shape=(100, 17, 3))
    keypoints[:, KEYPOINT_INDEX["left_shoulder"], :2] = (480, 200)
    keypoints[:, KEYPOINT_INDEX["right_shoulder"], :2] = (520, 200)
    keypoints[:, KEYPOINT_INDEX["left_hip"], :2] = (485, 400)
    keypoints[:, KEYPOINT_INDEX["right_hip"], :2] = (515, 400)
    keypoints[:, :, 2] = 1
    return keypoints


def test_get_joint_speed_constant_velocity(make_person_state: Callable,
                                           constant_boxes: np.ndarray,
                                           full_confidence_box: np.ndarray,
                                           constant_speed_keypoints: np.ndarray):

    state = make_person_state(
        keypoints=constant_speed_keypoints,
        boxes=constant_boxes,
        conf=full_confidence_box
    )

    speeds = get_joint_speed(state, "right_wrist", config=StrikeConfig())

    np.testing.assert_allclose(speeds[1:], 30)


def test_get_joint_speed_all_nan_xy(make_person_state: Callable,
                                    all_nan_xy_keypoints: np.ndarray):

    state = make_person_state(keypoints=all_nan_xy_keypoints)

    speeds = get_joint_speed(state, "right_wrist", config=StrikeConfig())

    assert np.isnan(speeds).all()


def test_get_joint_speed_ignores_confidence(make_person_state: Callable,
                                            all_nan_conf_keypoints: np.ndarray):
    # Confidence filtering happens in _densify, get_joint_speed only
    # looks at the xy positions
    state = make_person_state(keypoints=all_nan_conf_keypoints)

    speeds = get_joint_speed(state, "right_wrist", config=StrikeConfig())

    np.testing.assert_allclose(speeds[1:], 0)


def test_get_joint_speed_bridges_short_gap(make_person_state: Callable,
                                           constant_speed_keypoints: np.ndarray):
    # A gap within max_hold uses the most recent valid position, so the
    # speed averaged over the gap matches the constant velocity
    keypoints = constant_speed_keypoints.copy()
    keypoints[10:13, :, :2] = np.nan

    state = make_person_state(keypoints=keypoints)

    speeds = get_joint_speed(state, "right_wrist", config=StrikeConfig())

    assert np.isnan(speeds[10:13]).all()
    np.testing.assert_allclose(speeds[13], 30)


def test_get_joint_speed_long_gap_returns_nan(make_person_state: Callable,
                                              constant_speed_keypoints: np.ndarray):
    # A gap longer than max_hold can't be measured accurately
    keypoints = constant_speed_keypoints.copy()
    keypoints[20:28, :, :2] = np.nan

    state = make_person_state(keypoints=keypoints)

    speeds = get_joint_speed(state, "right_wrist", config=StrikeConfig())

    assert np.isnan(speeds[28])
    np.testing.assert_allclose(speeds[29], 30)


def test_get_joint_speed_peaks_finds_peak():
    speed = np.zeros(100)
    speed[20] = 5

    peaks = get_joint_speed_peaks(speed, threshold=3, max_speed=10)

    np.testing.assert_array_equal(peaks, [20])


def test_get_joint_speed_peaks_rejects_glitch_spike():
    speed = np.zeros(100)
    speed[20] = 50

    peaks = get_joint_speed_peaks(speed, threshold=3, max_speed=10)

    assert len(peaks) == 0


def test_get_joint_speed_peaks_spike_does_not_admit_neighbour():
    # The glitch spike must still occupy its distance window, otherwise
    # removing it lets the smaller glitchy peaks next to it through
    speed = np.zeros(100)
    speed[20] = 50
    speed[25] = 5

    peaks = get_joint_speed_peaks(speed, threshold=3, max_speed=10)

    assert len(peaks) == 0


def test_get_joint_speed_peaks_array_max():
    speed = np.zeros(100)
    speed[20] = 5
    speed[60] = 5

    max_speed = np.full(100, 10.0)
    max_speed[60] = 4.0

    peaks = get_joint_speed_peaks(speed, threshold=3, max_speed=max_speed)

    np.testing.assert_array_equal(peaks, [20])


def test_get_pixel_to_meter_ratio(make_person_state: Callable,
                                  standing_keypoints: np.ndarray):
    state = make_person_state(keypoints=standing_keypoints)

    ratio = get_pixel_to_meter_ratio(state)

    np.testing.assert_allclose(ratio, 0.0027)


def test_get_speed_threshold_scales_with_speed(make_person_state: Callable,
                                               standing_keypoints: np.ndarray):
    state = make_person_state(keypoints=standing_keypoints)

    thresholds = get_speed_threshold(state, 3.5)
    doubled = get_speed_threshold(state, 7.0)

    np.testing.assert_allclose(thresholds, 3.5 / 0.0027)
    np.testing.assert_allclose(doubled, thresholds * 2)


def test_calculate_angles_right_angle():
    a = np.array([[0.0, 1.0]])
    b = np.array([[0.0, 0.0]])
    c = np.array([[1.0, 0.0]])

    np.testing.assert_allclose(calculate_angles(a, b, c), 90)


def test_calculate_angles_straight_line():
    a = np.array([[0.0, 0.0]])
    b = np.array([[1.0, 0.0]])
    c = np.array([[2.0, 0.0]])

    np.testing.assert_allclose(calculate_angles(a, b, c), 180)
