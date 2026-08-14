import pytest
import numpy as np

from strike_analysis import StrikeConfig, get_joint_speed
from strike_analysis.constants import KEYPOINT_INDEX


@pytest.fixture()
def get_wrist_velocities_test_data() -> list[tuple]:
    TEST_WRIST_TRACK = [
        (100, 200, 0.95),
        (105, 205, 0.90),
        (110, 210, 0.70),
        (115, 213, 0.80),
        (120, 211, 0.84),
        (125, 210, 0.84),
        (125, 210, 0.84),
        (125, 210, 0.84),
        (130, 210, 0.2),
        (130, 210, 0.2),
        (130, 210, 0.2),
        (130, 210, 0.2),
        (130, 210, 0.2),
        (130, 210, 0.2),
        (130, 210, 0.2),
        (130, 210, 0.2),
        (145, 210, 0.90),
        (345, 410, 0.90),
        (345, 410, 0.10),
        (345, 410, 0.10),
        (350, 411, 0.96),
    ]

    return TEST_WRIST_TRACK


def test_get_joint_speed_realistic_track(get_wrist_velocities_test_data,
                                         make_person_state):
    # Low confidence detections are NaN'd like _densify does, leaving an
    # 8 frame gap that is longer than max_hold and a 2 frame gap that isn't
    track = np.array(get_wrist_velocities_test_data)
    n_frames = len(track)

    keypoints = np.zeros(shape=(n_frames, 17, 3))
    keypoints[:, KEYPOINT_INDEX["right_wrist"], :] = track
    keypoints[track[:, 2] < 0.4, KEYPOINT_INDEX["right_wrist"], :2] = np.nan
    keypoints[:, :, 2] = 1

    state = make_person_state(keypoints=keypoints)

    speeds = get_joint_speed(state, "right_wrist", config=StrikeConfig())

    # Visible frames with a valid previous frame
    np.testing.assert_allclose(speeds[1], np.hypot(5, 5) * 30)
    np.testing.assert_allclose(speeds[17], np.hypot(200, 200) * 30)

    # The low confidence frames can't be measured
    assert np.isnan(speeds[8:16]).all()

    # Frame 16 follows the 8 frame gap, longer than max_hold
    assert np.isnan(speeds[16])

    # Frame 20 follows a 2 frame gap, bridged from frame 17
    np.testing.assert_allclose(speeds[20], np.hypot(5, 1) * 30 / 3)
