from collections.abc import Callable

import pytest
import numpy as np
from numpy import ndarray
from pygments.lexers import make
from scipy.signal import find_peaks

from strike_analysis import Side, StrikeConfig, calculate_angles, PersonState, Person, get_joint_speed


@pytest.fixture()
def all_nan_conf_keypoints():
    keypoints = np.full((100, 17, 3), np.nan)
    keypoints[:, :, :2] = 0.0
    return keypoints


@pytest.fixture()
def all_nan_xy_keypoints():
    keypoints = np.full((100, 17, 3), 1)
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


@pytest.fixture
def make_person_state():
    def _make(keypoints, boxes, conf):
        person = Person(
            name="test", wingspan_m=1.80, weight=71,
            stance="left", height_m=1.80,
        )
        return PersonState(
            track_id=1, keypoints=keypoints, boxes=boxes,
            box_conf=conf, fps=30, person=person,
        )

    return _make


def test_get_joint_speed_constant_velocity(make_person_state: Callable ,
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
