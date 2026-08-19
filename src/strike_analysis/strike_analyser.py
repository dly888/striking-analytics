from __future__ import annotations

import numpy as np

from .config import StrikeConfig
from .constants import SIDES, STRIKE_TYPES
from .detections import Detections, Strike
from .features import get_speed_threshold
from .tracking import PersonState
from .detectors import (
    detect_hook,
    detect_kick,
    detect_straight,
    detect_uppercut
)

DETECTORS = {
    "straight": detect_straight,
    "hook": detect_hook,
    "uppercut": detect_uppercut,
    "kick": detect_kick,
}

class StrikeAnalyser:
    def __init__(
            self, track: PersonState, strike_config: StrikeConfig = StrikeConfig()
    ):
        self.track = track
        self.strike_config = strike_config
        self.thresholds = get_speed_threshold(track, strike_config.min_straight_speed_mps)

    def get_detections(self) -> Detections:
        """
        Detects strikes for each strike type.

        Returns:
            Detection object containing information on when a strike occurs.
        """
        strikes = tuple(
            Strike(technique, side)
            for technique in STRIKE_TYPES
            for side in SIDES
        )
        mask = np.stack(
            [
                DETECTORS[strike.strike_type](self.track, strike, self.strike_config)
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

        return Detections(strikes, mask)
