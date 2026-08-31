from dataclasses import dataclass

import numpy as np

from strike_analysis import PersonState, count_segments
from strike_analysis.defense_detections import GuardDetections

@dataclass(frozen=True)
class GuardStats:
    guard_up_time: float
    guard_up_time_percentage: float
    guard_drop_count: int


class DefenseStatsCalculator:
    def __init__(self, person_state : PersonState, guard_detections: GuardDetections):
        self.person_state = person_state
        self.fps = person_state.fps
        self.guard_detections = guard_detections

    def calculate_guard_stats(self) -> GuardStats:
        """
        Get all stats about the guard.

        Returns:
            GuardStats object storing stats on the guard.
        """
        return GuardStats(
            guard_up_time=self.get_guard_up_time(),
            guard_up_time_percentage=self.get_guard_up_time_percentage(),
            guard_drop_count=self.get_guard_dropped_count()
        )

    def get_guard_up_time(self) -> float:
        """
        Get the total time the guard is held up.

        Returns:
            The total time the guard is held up
        """
        guard_up_time = (len(self.guard_detections.mask) - np.sum(self.guard_detections.mask)) / self.fps
        return guard_up_time

    def get_guard_up_time_percentage(self) -> float:
        """
        Get the percentage of the video where the guard is being held up.

        Returns:
            The percentage of the video where the guard is being held up
        """
        guard_up_time = (len(self.guard_detections.mask) - np.sum(self.guard_detections.mask))
        guard_up_time_percentage = guard_up_time / len(self.guard_detections.mask)
        return guard_up_time_percentage

    def get_guard_dropped_count(self) -> int:
        """
        Get the number of times the guard is dropped.

        Returns:
            The number of times the guard is dropped.
        :return:
        """
        return count_segments(self.guard_detections.mask)
