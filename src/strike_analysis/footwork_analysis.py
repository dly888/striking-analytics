import numpy as np

from .config import FootworkConfig
from .features import get_ankle_raise
from .footwork_plotting import FootworkPlotter
from .footwork_projection import FootworkProjector
from .footwork_stats import FootworkStats, FootworkStatsCalculator
from .tracking import PersonState


class FootworkAnalyser:
    """
    Orchestrates footwork analysis for one fighter.

    Projects the ankle keypoints onto the floor, drops the frames where a
    foot is airborne, and holds the filtered keypoints for the stats and
    plotting to use.
    """

    def __init__(
        self,
        person_state: PersonState,
        config: FootworkConfig | None = None,
    ):
        self.person_state = person_state
        self.config = config or FootworkConfig()
        self.projector = FootworkProjector(person_state)

        self.left_ankle_keypoints = None
        self.right_ankle_keypoints = None

    def select_floor(self, edge1, edge2):
        """
        Sets the floor region by tracing two opposite edges of the floor.

        Args:
            edge1: Two (x, y) points along one floor edge, near end first
            edge2: Two (x, y) points along the opposite edge, near end first
        """
        self.projector.select_floor(edge1=edge1, edge2=edge2)

    def select_floor_edge_lengths_m(self, length1: float, length2: float):
        """
        Select the lengths in meters of the floor edges selected

        Args:
            length1: Length of edge 1
            length2: Length of edge 2
        """
        self.projector.select_floor_edge_lengths_m(length1, length2)

    def get_foot_in_air_mask(self, min_raise_m=0.15):
        """
        Get mask to determine whether each foot is in the air.

        Args:
            min_raise_m: Minimum ankle raise, in metres

        Returns:
            Tuple of (left_airborne, right_airborne) boolean arrays.
        """
        left_raise = get_ankle_raise(self.person_state, "left")
        right_raise = get_ankle_raise(self.person_state, "right")

        return left_raise > min_raise_m, right_raise > min_raise_m

    def filter_kicks(self):
        """
        Store the mapped ankle keypoints with kick frames removed.
        """
        window = self.config.kick_filter_window
        left_airborne, right_airborne = self.get_foot_in_air_mask()

        left = self.projector.mapped_left_ankle_keypoints.copy()
        right = self.projector.mapped_right_ankle_keypoints.copy()

        left[self.grow_mask(left_airborne, window)] = np.nan
        right[self.grow_mask(right_airborne, window)] = np.nan

        self.left_ankle_keypoints = left
        self.right_ankle_keypoints = right

    @staticmethod
    def grow_mask(mask: np.ndarray, window: int) -> np.ndarray:
        """
        Grow a boolean frame mask by window frames on both side.

        Args:
            mask: Boolean array, one entry per frame.
            window: Number of frames to expand each True by on each side.

        Returns:
            Boolean mask array to indicate which frames to ignore as footwork movement.
        """
        grown = mask.copy()

        for idx in np.flatnonzero(mask):
            start = max(0, idx - window)
            end = min(len(mask), idx + window + 1)
            grown[start:end] = True

        return grown

    def get_footwork_stats(self, bins: int = 40) -> FootworkStats:
        """
        Get the footwork statistics for the video.

        Args:
            bins: Number of bins along each axis of the floor

        Returns:
            FootworkStats object holding the footwork statistics.
        """
        self.projector.project_true_scale()
        self.filter_kicks()

        return FootworkStatsCalculator(
            self.person_state,
            self.projector,
            self.left_ankle_keypoints,
            self.right_ankle_keypoints,
            bins=bins,
        ).calculate_footwork_stats()

    def get_plot_figure(self, bins: int = 40, smoothing: float = 1.5):
        """
        Plots a heatmap of where the person's feet spend time on the floor.

        Args:
            bins: Number of bins along each axis of the floor
            smoothing: Standard deviation of the smoothing, in cells

        Returns:
            Figure of the plot
        """
        self.projector.project_true_scale()
        self.filter_kicks()

        return FootworkPlotter(
            self.projector,
            self.left_ankle_keypoints,
            self.right_ankle_keypoints,
        ).get_heatmap_figure(bins, smoothing)
