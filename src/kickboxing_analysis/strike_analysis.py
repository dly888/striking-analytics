from __future__ import annotations

import numpy as np

from .config import StrikeConfig
from .constants import SIDES, STRIKE_TYPES
from .features import get_speed_threshold
from .strike_detections import Strike, StrikeDetections
from .strike_detectors import detect_hook, detect_kick, detect_straight, detect_uppercut
from .tracking import PersonState

STRIKE_DETECTORS = {
    "straight": detect_straight,
    "hook": detect_hook,
    "uppercut": detect_uppercut,
    "kick": detect_kick,
}


class StrikeAnalyser:
    def __init__(self, person_state: PersonState, strike_config: StrikeConfig):
        self.track = person_state
        self.strike_config = strike_config
        self.thresholds = get_speed_threshold(
            person_state, strike_config.min_straight_speed_mps
        )

    def get_strike_detections(self) -> StrikeDetections:
        """
        Detects strikes for each strike type.

        Returns:
            Detection object containing information on when a strike occurs.
        """
        strikes = tuple(
            Strike(technique, side) for technique in STRIKE_TYPES for side in SIDES
        )
        mask = np.stack(
            [
                STRIKE_DETECTORS[strike.strike_type](
                    self.track, strike, self.strike_config
                )
                for strike in strikes
            ]
        ).astype(bool)

        # If both a hook and straight hare detected, the straight will take priority
        for side in SIDES:
            straight_row = mask[strikes.index(Strike("straight", side))]
            hook_row = mask[strikes.index(Strike("hook", side))]

            for frame in np.flatnonzero(straight_row):
                window = slice(
                    max(0, frame - 10),
                    min(len(hook_row), frame + 11),
                )
                hook_row[window] = False

        self.remove_punch_detections_during_kicks(mask, strikes)

        return StrikeDetections(strikes, mask)

    def remove_punch_detections_during_kicks(
        self, mask: np.ndarray, strikes: tuple[Strike, ...]
    ) -> None:
        """
        Remove punch detections that coincide with a kick.

        Used to get rid of arms swings detected as punches.
        May get rid of actual punches as well.

        Args:
            mask: The detection mask, one boolean row per strike.
            strikes: The strikes in the same order as the rows of the mask.
        """
        before = self.strike_config.punch_suppression_before_kick_window
        after = self.strike_config.punch_suppression_after_kick_window

        punch_rows = [
            mask[strikes.index(strike)]
            for strike in strikes
            if strike.strike_type != "kick"
        ]

        for kick_side in SIDES:
            kick_row = mask[strikes.index(Strike("kick", kick_side))]

            for frame in np.flatnonzero(kick_row):
                window = slice(
                    max(0, frame - before),
                    min(mask.shape[1], frame + after + 1),
                )
                for punch_row in punch_rows:
                    punch_row[window] = False
