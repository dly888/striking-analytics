from dataclasses import dataclass

import numpy as np

from .config import FootworkConfig
from .footwork_projection import FootworkProjector
from .tracking import PersonState


@dataclass(frozen=True)
class FootworkStats:
    floor_coverage: float
    stance_width_mean: float
    stance_width_std_dev: float
    distance_travelled: float
    distance_travelled_cumsum: np.ndarray


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
            stance_width_std_dev=self.get_stance_width_standard_deviation(),
            distance_travelled=self.get_total_distance_travelled(),
            distance_travelled_cumsum=self.get_distance_travelled_cumsum(),
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

    def get_total_distance_travelled(self) -> float:
        """
        Get the final distance travelled throughout the video.

        Returns:
            The final total distance travelled by the fighter
        """
        return self.get_distance_travelled_cumsum()[-1]

    def get_distance_travelled_cumsum(self) -> np.ndarray:
        """
        Cumulative distance travelled by the fighter at each frame.

        Travel is measured from the midpoint of the two ankles,
        A threshold is applied to midpoint movement to ignore
        jitters being considered as travel.

        Returns:
            Array of the cumulative distance travelled, one entry per frame.
        """
        left = self.left_ankle_keypoints[:, :2]
        right = self.right_ankle_keypoints[:, :2]

        # Use midpoint to track distance travel, need both ankles
        # to be non NaN
        centre = (left + right) / 2

        steps = np.linalg.norm(np.diff(centre, axis=0), axis=1)

        # Jitters of less than 5cm per frame are ignored
        steps = np.where(steps >= FootworkConfig.step_threshold, steps, 0.0)
        steps = np.nan_to_num(steps)

        # Match with the video frame count
        return np.concatenate([[0.0], np.cumsum(steps)])

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

        return float(np.nanstd(widths))
