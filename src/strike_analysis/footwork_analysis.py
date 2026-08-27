from .footwork_plotting import FootworkPlotter
from .footwork_projection import FootworkProjector
from .footwork_stats import FootworkStats, FootworkStatsCalculator
from .tracking import PersonState


class FootworkAnalyser:
    """
    Orchestrates footwork analysis for one fighter.
    """

    def __init__(self, person_state: PersonState):
        self.person_state = person_state
        self.projector = FootworkProjector(person_state)

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

    def get_footwork_stats(self, bins: int = 40) -> FootworkStats:
        """
        Get the footwork statistics for the video.

        Args:
            bins: Number of bins along each axis of the floor

        Returns:
            FootworkStats object holding the footwork statistics.
        """
        self.projector.project_true_scale()

        return FootworkStatsCalculator(
            self.person_state, self.projector, bins=bins
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

        return FootworkPlotter(self.projector).get_heatmap_figure(bins, smoothing)
