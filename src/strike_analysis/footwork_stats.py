from dataclasses import dataclass

import numpy as np

from .footwork_projection import FootworkProjector
from .tracking import PersonState


@dataclass(frozen=True)
class FootworkStats:
    floor_coverage: float
    stance_width_mean: float
    stance_width_std_dev: float


class FootworkStatsCalculator:
    def __init__(
        self,
        person_state: PersonState,
        projector: FootworkProjector,
        left_ankle_keypoints,
        right_ankle_keypoints,
        bins: int = 40,
    ):
        self.person_state = person_state
        self.projector = projector
        self.bins = bins

        self.floor_size = projector.get_floor_size()
        self.left_ankle_keypoints = left_ankle_keypoints
        self.right_ankle_keypoints = right_ankle_keypoints

    def calculate_footwork_stats(self) -> FootworkStats:
        """
        Get every footwork statistic for the video in one object.

        Returns:
            FootworkStats object holding the footwork statistics for the
            video.
        """
        return FootworkStats(
            floor_coverage=self.get_floor_coverage(),
            stance_width_mean=self.get_stance_width_mean(),
            stance_width_std_dev=self.get_stance_width_standard_deviation()
        )

    def get_floor_counts(self) -> np.ndarray:
        """
        Counts how many frames the fighter's feet spend in each floor cell.

        Returns:
            Frame counts for each cell, indexed by x bin then y bin.
        """
        both = np.vstack([self.left_ankle_keypoints, self.right_ankle_keypoints])
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

    def get_distance_travelled(self) -> tuple[float, float]:
        """
        Get the total distance each ankle travels across the video.

        Skips steps where a position is missing.

        Returns:
            Tuple of (left, right) distance travelled.
        """
        left_steps = np.linalg.norm(
            np.diff(self.left_ankle_keypoints[:, :2], axis=0), axis=1
        )
        right_steps = np.linalg.norm(
            np.diff(self.right_ankle_keypoints[:, :2], axis=0), axis=1
        )

        return float(np.nansum(left_steps)), float(np.nansum(right_steps))

    def get_stance_width_mean(self) -> float:
        """
        Get the mean distance between the fighter's ankles.

        Returns:
            The mean distance between the two ankles across all frames
            where both are visible.
        """
        diff = self.left_ankle_keypoints[:, :2] - self.right_ankle_keypoints[:, :2]
        widths = np.linalg.norm(diff, axis=1)

        return float(np.nanmean(widths))

    def get_stance_width_standard_deviation(self) -> float:
        """
        Get the standard deviation of the distance between the fighter's ankles
        in meters.

        Returns:
            The standard deviation of the distance between the two ankles across all frames
            where both are visible.
        """
        diff = self.left_ankle_keypoints[:, :2] - self.right_ankle_keypoints[:, :2]
        widths = np.linalg.norm(diff, axis=1)

        return  float(np.nanstd(widths))
