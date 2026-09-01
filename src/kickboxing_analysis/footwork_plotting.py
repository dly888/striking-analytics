import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from scipy.ndimage import gaussian_filter

from .footwork_projection import FootworkProjector

LEFT_FOOT_COLOUR = "#63b5ff"
RIGHT_FOOT_COLOUR = "#d7ecff"
FLOOR_HEATMAP_COLOUR_MAP = LinearSegmentedColormap.from_list(
    "strike_room_blue",
    ["#17231f", "#194a70", "#2d82c5", "#9ed5ff"],
)


class FootworkPlotter:
    def __init__(
        self,
        projector: FootworkProjector,
        left_ankle_keypoints,
        right_ankle_keypoints,
    ):
        self.projector = projector
        self.left_ankle_keypoints = left_ankle_keypoints
        self.right_ankle_keypoints = right_ankle_keypoints

    @staticmethod
    def floor_counts(points, bins, floor_size) -> np.ndarray:
        """
        Counts how many frames a foot spends in each cell of the floor.

        Args:
            points: Array of (x, y) floor positions, one per frame
            bins: Number of bins along each axis of the floor
            floor_size: The floor's (width, depth)

        Returns:
            Frame counts for each cell, indexed by x bin then y bin.
        """
        points = np.asarray(points)
        points = points[np.isfinite(points).all(axis=1)]

        width, depth = floor_size

        counts, _, _ = np.histogram2d(
            points[:, 0],
            points[:, 1],
            bins=bins,
            range=[[0, width], [0, depth]],
        )

        return counts

    def outline_feet(self, ax, left, right, bins, smoothing):
        """
        Draws where each foot spends significant time in.

        Args:
            ax: Axes the heatmap is drawn on
            left: Array of (x, y) floor positions of the left ankle
            right: Array of (x, y) floor positions of the right ankle
            bins: Number of bins along each axis of the floor
            smoothing: Standard deviation of the smoothing, in cells
        """
        # Contours are placed at the centre of each cell
        width, depth = self.projector.get_floor_size()
        x_centres = (np.arange(bins) + 0.5) / bins * width
        y_centres = (np.arange(bins) + 0.5) / bins * depth

        feet = (
            ("Left foot", left, LEFT_FOOT_COLOUR),
            ("Right foot", right, RIGHT_FOOT_COLOUR),
        )

        for _, points, colour in feet:
            density = gaussian_filter(
                self.floor_counts(points, bins, self.projector.get_floor_size()),
                sigma=smoothing,
            )

            if not density.max():
                continue

            # Half the maximum keeps the outline around the most significant area
            ax.contour(
                x_centres,
                y_centres,
                density.T,
                levels=[density.max() / 2],
                colors=colour,
                linewidths=1.8,
            )

        ax.legend(
            handles=[
                Line2D([], [], color=colour, linewidth=1.8, label=label)
                for label, _, colour in feet
            ],
            loc="upper right",
            fontsize=9,
            framealpha=0.85,
        )

    def get_heatmap_figure(self, bins=40, smoothing=1.5):
        """
        Plots a heatmap of where the person's feet spend time on the floor.

        Assumes the projector has already run its projection and that airborne
        frames have been filtered out of the mapped keypoints.

        Args:
            bins: Number of bins along each axis of the floor
            smoothing: Standard deviation of the smoothing, in cells

        Returns:
            Figure of the plot
        """
        projector = self.projector

        # Airborne frames were already dropped by FootworkAnalyser.filter_kicks
        left = self.left_ankle_keypoints
        right = self.right_ankle_keypoints

        both = np.vstack([left, right])

        counts = self.floor_counts(both, bins, projector.get_floor_size())

        # Scaled to its maximum, since the smoothed counts are no longer
        # a number of frames
        density = gaussian_filter(counts, sigma=smoothing)
        density = density / density.max() if density.max() else density

        sns.set_theme(
            style="dark",
            rc={
                "axes.facecolor": "#171a18",
                "axes.edgecolor": "#566056",
                "axes.labelcolor": "#e7e4dd",
                "font.family": "Arial",
                "figure.facecolor": "#121513",
                "text.color": "#e7e4dd",
                "xtick.color": "#c7cec7",
                "ytick.color": "#c7cec7",
            },
        )

        fig, ax = plt.subplots(figsize=(9, 8))

        width, depth = projector.get_floor_size()

        # Transposed so the floor isn't drawn upside down
        image = ax.imshow(
            density.T,
            origin="lower",
            extent=(0, width, 0, depth),
            cmap=FLOOR_HEATMAP_COLOUR_MAP,
            interpolation="bilinear",
        )

        self.outline_feet(ax, left, right, bins, smoothing)

        colour_bar = fig.colorbar(image, ax=ax, shrink=0.82, pad=0.03)
        colour_bar.set_label("Time spent (relative)", fontsize=10)


        if self.projector.true_scale_homography is None:
            # Unit square default
            ax.set_xlabel("Across the front", fontsize=10)
            ax.set_ylabel("Towards the back", fontsize=10)
        else:
            ax.set_xlabel("Across the front (m)", fontsize=10)
            ax.set_ylabel("Towards the back (m)", fontsize=10)

        ax.set_title(
            f"Footwork heatmap: {projector.person_state.person.name}",
            fontsize=13,
            pad=16,
        )

        fig.tight_layout()

        return fig
