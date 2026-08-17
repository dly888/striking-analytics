from __future__ import annotations

import numpy as np
import pytest

from strike_analysis import Person, PersonState


@pytest.fixture
def make_person_state():
    def _make(keypoints, boxes=None, conf=None):
        n_frames = len(keypoints)

        if boxes is None:
            boxes = np.full(shape=(n_frames, 4), fill_value=400.0)
            boxes[:, :2] = 100.0

        if conf is None:
            conf = np.full(shape=n_frames, fill_value=1.0)

        person = Person(
            name="test", wingspan_m=1.80, weight=71,
            stance="orthodox", height_m=1.80,
        )
        return PersonState(
            track_id=1, keypoints=keypoints, boxes=boxes,
            box_conf=conf, fps=30, person=person,
        )

    return _make
