from dataclasses import dataclass

import numpy as np

from .footwork_projection import FootworkProjector
from .tracking import PersonState


@dataclass(frozen=True)
class FootworkStats:
    floor_coverage: float


class FootworkStatsCalculator:
    def __init__(
        self,
        person_state: PersonState,
        projector: FootworkProjector,
        bins: int = 40,
    ):
        self.person_state = person_state
        self.projector = projector
        self.bins = bins

        self.floor_size = projector.get_floor_size()
        self.left = projector.mapped_left_ankle_keypoints
        self.right = projector.mapped_right_ankle_keypoints

    def calculate_footwork_stats(self) -> FootworkStats:
        """
        Get every footwork statistic for the video in one object.

        Returns:
            FootworkStats object holding the footwork statistics for the
            video.
        """
        return FootworkStats(
            floor_coverage=self.get_floor_coverage(),
        )

    def get_floor_counts(self) -> np.ndarray:
        """
        Counts how many frames the fighter's feet spend in each floor cell.

        Returns:
            Frame counts for each cell, indexed by x bin then y bin.
        """
        both = np.vstack([self.left, self.right])
        points = both[np.isfinite(both).all(axis=1)]

        width, depth = self.floor_size

        counts, _, _ = np.histogram2d(
            points[:, 0],
            points[:, 1],
            bins=self.bins,
            range=[[0, width], [0, depth]],
        )

        return counts

    def get_floor_coverage(self) -> float:
        """
        Get the fraction of the floor covered by the fighter.

        Returns:
            The fraction of floor cells the fighter's feet visited.
        """
        counts = self.get_floor_counts()
        coverage = np.count_nonzero(counts) / counts.size

        return float(coverage)
