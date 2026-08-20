import cv2
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.lines import Line2D
from scipy.ndimage import gaussian_filter

from .tracking import PersonState

LEFT_FOOT_COLOUR = "#4cc9f0"
RIGHT_FOOT_COLOUR = "#f72585"


class FootworkAnalyser:
    def __init__(self, person_state: PersonState):
        self.mapped_right_ankle_keypoints = None
        self.mapped_left_ankle_keypoints = None
        self.homography = None
        self.homography_mask = None
        self.person_state = person_state
        self.left_ankle_keypoints = person_state.positions("left_ankle")
        self.right_ankle_keypoints = person_state.positions("right_ankle")
        self.floor_edges = None
        self.floor_corners = None

    def select_floor(self, edge1, edge2):
        """
        Sets the floor region by tracing two opposite edges of the floor.

        Args:
            edge1: Two (x, y) points along one floor edge, near end first
            edge2: Two (x, y) points along the opposite edge, near end first
        """
        edges = np.asarray([edge1, edge2], dtype=np.float32)

        if edges.shape != (2, 2, 2):
            raise ValueError("Each floor edge needs two points along it.")

        self.floor_edges = edges

    def get_corners(self):
        """
        Builds the floor's four corners from the two traced edges.

        The corners are ordered to match the unit square they map onto:
        near edge1, near edge2, far edge2, far edge1.
        """
        if self.floor_edges is None:
            raise ValueError("Please select the floor first.")

        near1, far1 = self.floor_edges[0]
        near2, far2 = self.floor_edges[1]

        self.floor_corners = np.stack([near1, near2, far2, far1])

    def get_homography(self):
        if self.floor_corners is None:
            raise ValueError("Please select the floor first.")

        unit_corners = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])

        H, H_mask = cv2.findHomography(
            srcPoints=self.floor_corners, dstPoints=unit_corners
        )
        self.homography = H
        self.homography_mask = H_mask

    def map_joint_to_unit_square(self, points):
        """
        Maps image positions onto the floor's unit square.

        Args:
            points: Array of (x, y) image positions, one per frame

        Returns:
            Array of (x, y) floor positions, one per frame.
        """
        if self.homography is None:
            raise ValueError("Homography not found.")

        points = np.asarray(points, dtype=np.float32)[:, :2]
        mapped = np.full(points.shape, np.nan, dtype=np.float32)

        visible = np.isfinite(points).all(axis=1)

        # Ignore keypoints that are NaN, perspectiveTransform cant take NaN values
        if visible.any():
            mapped[visible] = cv2.perspectiveTransform(
                points[visible][None], self.homography
            )[0]

        return mapped

    def transform_keypoints(self):
        self.mapped_left_ankle_keypoints = self.map_joint_to_unit_square(
            self.left_ankle_keypoints
        )
        self.mapped_right_ankle_keypoints = self.map_joint_to_unit_square(
            self.right_ankle_keypoints
        )

    @staticmethod
    def get_floor_counts(points, bins):
        """
        Counts how many frames a foot spends in each cell of the floor.

        Args:
            points: Array of (x, y) floor positions, one per frame
            bins: Number of bins along each axis of the floor

        Returns:
            Frame counts for each cell, indexed by x bin then y bin.
        """
        points = np.asarray(points)
        points = points[np.isfinite(points).all(axis=1)]

        counts, _, _ = np.histogram2d(
            points[:, 0], points[:, 1], bins=bins, range=[[0, 1], [0, 1]]
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
        centres = (np.arange(bins) + 0.5) / bins

        feet = (
            ("Left foot", left, LEFT_FOOT_COLOUR),
            ("Right foot", right, RIGHT_FOOT_COLOUR),
        )

        for _, points, colour in feet:
            density = gaussian_filter(
                self.get_floor_counts(points, bins), sigma=smoothing
            )

            if not density.max():
                continue

            # Half the maximum keeps the outline around the most significant area
            ax.contour(
                centres,
                centres,
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

    def get_plot_figure(self, bins=40, smoothing=1.5):
        """
        Plots a heatmap of where the person's feet spend time on the floor.

        Args:
            bins: Number of bins along each axis of the floor
            smoothing: Standard deviation of the smoothing, in cells

        Returns:
            Figure of the plot
        """
        self.get_corners()
        self.get_homography()
        self.transform_keypoints()

        left = self.mapped_left_ankle_keypoints
        right = self.mapped_right_ankle_keypoints

        both = np.vstack([left, right])
        tracked = int(np.isfinite(both).all(axis=1).sum())

        counts = self.get_floor_counts(both, bins)

        # Scaled to its maximum, since the smoothed counts are no longer
        # a number of frames
        density = gaussian_filter(counts, sigma=smoothing)
        density = density / density.max() if density.max() else density

        sns.set_theme(style="white")

        fig, ax = plt.subplots(figsize=(9, 8))

        # Transposed so the floor isn't drawn upside down
        image = ax.imshow(
            density.T,
            origin="lower",
            extent=(0, 1, 0, 1),
            cmap="inferno",
            interpolation="bilinear",
        )

        self.outline_feet(ax, left, right, bins, smoothing)

        colour_bar = fig.colorbar(image, ax=ax, shrink=0.82, pad=0.03)
        colour_bar.set_label("Time spent (relative)", fontsize=10)

        ax.set_xlabel("Across the floor", fontsize=10)
        ax.set_ylabel("Towards the wall", fontsize=10)
        ax.set_title(
            f"Footwork heatmap: {self.person_state.person.name}",
            fontsize=13,
            pad=16,
        )

        # How much of the floor the fighter set foot on at all
        coverage = np.count_nonzero(counts) / counts.size

        ax.text(
            0.0,
            1.015,
            f"{tracked / 2 / self.person_state.fps:.0f}s tracked"
            f"    |    {coverage:.0%} of floor used"
            f"    |    hover for time per cell",
            transform=ax.transAxes,
            fontsize=9,
            color="dimgrey",
        )

        fig.tight_layout()

        return fig
