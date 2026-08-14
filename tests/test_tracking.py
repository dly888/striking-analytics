from __future__ import annotations

import numpy as np
import pytest

from strike_analysis.tracking import smooth_keypoints


@pytest.fixture()
def constant_keypoints():
    keypoints = np.full(shape=(100, 17, 3), fill_value=100.0)
    keypoints[:, :, 2] = 1
    return keypoints


def test_smooth_keypoints_constant_unchanged(constant_keypoints):
    smoothed = smooth_keypoints(constant_keypoints, window=3)

    np.testing.assert_allclose(smoothed, constant_keypoints)


def test_smooth_keypoints_window_one_unchanged(constant_keypoints):
    smoothed = smooth_keypoints(constant_keypoints, window=1)

    np.testing.assert_allclose(smoothed, constant_keypoints)


def test_smooth_keypoints_even_window_raises(constant_keypoints):
    with pytest.raises(ValueError):
        smooth_keypoints(constant_keypoints, window=4)


def test_smooth_keypoints_reduces_jitter(constant_keypoints):
    rng = np.random.default_rng(0)
    keypoints = constant_keypoints.copy()
    keypoints[:, :, :2] += rng.normal(scale=3, size=(100, 17, 2))

    smoothed = smooth_keypoints(keypoints, window=5)

    assert np.std(smoothed[:, 0, 0]) < np.std(keypoints[:, 0, 0])


def test_smooth_keypoints_nan_stays_nan(constant_keypoints):
    keypoints = constant_keypoints.copy()
    keypoints[50, :, :2] = np.nan

    smoothed = smooth_keypoints(keypoints, window=3)

    # The missing frame stays missing and its neighbours average over
    # the remaining valid values only
    assert np.isnan(smoothed[50, 0, 0])
    np.testing.assert_allclose(smoothed[49, 0, 0], 100)
    np.testing.assert_allclose(smoothed[51, 0, 0], 100)
