import numpy as np

from .config import DefenseConfig
from .features import get_torso_length
from .tracking import PersonState


def detect_guard_drop(
    person_state: PersonState, defense_config: DefenseConfig
) -> np.ndarray:
    """
    Detects when the fighter's hands fall well below their shoulders.

    Each wrist is compared against the shoulder on its own side. The gap is
    measured in torso lengths.

    Args:
        person_state: PersonState object
        defense_config: DefenseConfig object

    Returns:
        Numpy boolean array of shape (3, frames_processed) holding the
        left, right and both hands dropped masks, in that order.
    """
    torso_length = get_torso_length(person_state)

    left_wrist_y = person_state.positions("left_wrist")[:, 1]
    right_wrist_y = person_state.positions("right_wrist")[:, 1]

    left_shoulder_y = person_state.positions("left_shoulder")[:, 1]
    right_shoulder_y = person_state.positions("right_shoulder")[:, 1]

    # y grows downward, so wrist below threshold is positive
    left_drop = (left_wrist_y - left_shoulder_y) / torso_length
    right_drop = (right_wrist_y - right_shoulder_y) / torso_length

    threshold = defense_config.guard_drop_torso_fraction

    left_guard_dropped = left_drop > threshold
    right_guard_dropped = right_drop > threshold
    both_guard_dropped = left_guard_dropped & right_guard_dropped

    guard_dropped = np.stack(
        [
            left_guard_dropped,
            right_guard_dropped,
            both_guard_dropped,
        ]
    )

    return guard_dropped
