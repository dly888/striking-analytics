import cv2
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.lines import Line2D
from scipy.ndimage import gaussian_filter

from .features import get_ankle_raise
from .tracking import PersonState

LEFT_FOOT_COLOUR = "#4cc9f0"
RIGHT_FOOT_COLOUR = "#f72585"


class FootworkAnalyser:
    def __init__(self, person_state: PersonState):
        # Person data
        self.person_state = person_state
        self.mapped_right_ankle_keypoints = None
        self.mapped_left_ankle_keypoints = None
        self.left_ankle_keypoints = person_state.positions("left_ankle")
        self.right_ankle_keypoints = person_state.positions("right_ankle")

        # Homographies
        self.unit_square_homography = None
        self.unit_square_homography_mask = None
        self.true_scale_homography = None
        self.true_scale_homography_mask = None

        # Floor data
        self.floor_edges = None
        self.floor_corners = None
        self.floor_edge_lengths = None
        self.true_scale_corners = None

    def select_floor(self, edge1, edge2):
        """
        Sets the floor region by tracing two opposite edges of the floor.

        Args:
            edge1: Two (x, y) points along one floor edge, near end first
            edge2: Two (x, y) points along the opposite edge, near end first
        """
        edges = np.array([edge1, edge2])

        self.floor_edges = edges

    def select_floor_edge_lengths_m(self, length1: float, length2: float):
        """
        Select the lengths in meters of the floor edges selected

        Args:
            length1: Length of edge 1
            length2: Length of edge 2
        """
        self.floor_edge_lengths = np.array([length1, length2])


    def get_selected_corners(self):
        """
        Builds the floor's four corners from the two traced edges.

        The corners are ordered to match the unit square they map onto:
        near edge1, near edge2, far edge2, far edge1.
        """
        if self.floor_edges is None:
            raise ValueError("Please select the floor first.")

        near1, far1 = self.floor_edges[0]
        near2, far2 = self.floor_edges[1]

        self.floor_corners = np.array([near1, near2, far2, far1])

    def get_unit_square_homography(self):
        """
        Get the homography for a unit quare floor projection.
        """
        if self.floor_corners is None:
            raise ValueError("Please select the floor first.")

        unit_corners = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])

        H, H_mask = cv2.findHomography(
            srcPoints=self.floor_corners, dstPoints=unit_corners
        )
        self.unit_square_homography = H
        self.unit_square_homography_mask = H_mask

    def get_true_scale_corners(self):
        """
        Builds the floor's four corners in metres from the traced edges.
        """
        if self.floor_corners is None:
            raise ValueError("Please select the floor first.")

        if self.floor_edge_lengths is None:
            raise ValueError("Please select the floor edge lengths first.")

        near1, near2, far2, far1 = self.floor_corners
        length1, length2 = self.floor_edge_lengths

        # Pixel lengths of the selected edges
        depth1_pixel_length = np.linalg.norm(far1 - near1)
        depth2_pixel_length = np.linalg.norm(far2 - near2)
        near_pixel_length = np.linalg.norm(near2 - near1)
        far_pixel_length = np.linalg.norm(far2 - far1)

        # Scale using the real length inputs
        metres_per_pixel = (length1 + length2) / (depth1_pixel_length + depth2_pixel_length)

        depth_m = (length1 + length2) / 2
        width_m = (near_pixel_length + far_pixel_length) / 2 * metres_per_pixel

        self.true_scale_corners = np.array(
            [[0, 0], [width_m, 0], [width_m, depth_m], [0, depth_m]],
            dtype=np.float64,
        )

    def get_true_scale_homography(self):
        """
        Get the homography for a true scale floor projection.

        Defaults to a unit square if edge lengths are unknown.
        """
        if self.floor_corners is None:
            raise ValueError("Please select the floor first.")

        if self.floor_edge_lengths is None:
            self.get_unit_square_homography()
            return

        self.get_true_scale_corners()

        H, mask = cv2.findHomography(self.floor_corners, self.true_scale_corners)
        self.true_scale_homography = H
        self.true_scale_homography_mask = mask

    def get_floor_size(self) -> np.ndarray:
        """
        Get the floor's (width, depth).

        Metres when a true-scale projection is used, otherwise the (1, 1)
        unit square.
        """
        if self.true_scale_corners is None:
            return np.array([1.0, 1.0])

        return self.true_scale_corners[2]

    def map_joint_to_unit_square(self, points) -> np.ndarray:
        """
        Maps image positions onto the floor's unit square.

        Args:
            points: Array of (x, y) image positions, one per frame

        Returns:
            Array of (x, y) floor positions, one per frame.
        """
        if self.unit_square_homography is None:
            raise ValueError("Homography not found.")

        points = np.array(points)[:, :2]
        mapped = np.full(points.shape, np.nan)

        visible = np.isfinite(points).all(axis=1)

        # Ignore keypoints that are NaN, perspectiveTransform cant take NaN values
        if visible.any():
            mapped[visible] = cv2.perspectiveTransform(
                points[visible][None], self.unit_square_homography
            )[0]

        return mapped


    def map_joint_to_true_scale(self, points) -> np.ndarray:
        """
        Maps image positions onto the floor's unit square.

        Args:
            points: Array of (x, y) image positions, one per frame

        Returns:
            Array of (x, y) floor positions, one per frame.
        """
        if self.true_scale_homography is None:
            return self.map_joint_to_unit_square(points)

        else:
            points = np.array(points)[:, :2]
            mapped = np.full(points.shape, np.nan)

            visible = np.isfinite(points).all(axis=1)

            # Ignore keypoints that are NaN, perspectiveTransform cant take NaN values
            if visible.any():
                mapped[visible] = cv2.perspectiveTransform(
                    points[visible][None], self.true_scale_homography
                )[0]

            return mapped


    def transform_keypoints_to_unit_scale(self):
        """
        Transform the keypoints onto the unit square floor projection.
        """
        self.mapped_left_ankle_keypoints = self.map_joint_to_unit_square(
            self.left_ankle_keypoints
        )
        self.mapped_right_ankle_keypoints = self.map_joint_to_unit_square(
            self.right_ankle_keypoints
        )

    def transform_keypoints_to_true_scale(self):
        """
        Transform the keypoints onto the true scale floor projection

        Defaults to the unit square projection if the edge lengths are not
        known, defaulting handled in map_joint_to_true_scale.
        :return:
        """
        self.mapped_left_ankle_keypoints = self.map_joint_to_true_scale(
            self.left_ankle_keypoints
        )
        self.mapped_right_ankle_keypoints = self.map_joint_to_true_scale(
            self.right_ankle_keypoints
        )


    def get_floor_counts(self, points, bins):
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

        width, depth = self.get_floor_size()

        counts, _, _ = np.histogram2d(
            points[:, 0],
            points[:, 1],
            bins=bins,
            range=[[0, width], [0, depth]],
        )

        return counts

    def get_floor_coverage(self, bins) -> float:
        """
        Get the percentage of the floor covered by the fighter.

        Args:
            bins: Number of bins along each axis of the floor.
        """
        left = self.mapped_left_ankle_keypoints
        right = self.mapped_right_ankle_keypoints
        both = np.vstack([left, right])

        counts = self.get_floor_counts(both, bins)
        coverage = np.count_nonzero(counts) / counts.size

        return coverage

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
        width, depth = self.get_floor_size()
        x_centres = (np.arange(bins) + 0.5) / bins * width
        y_centres = (np.arange(bins) + 0.5) / bins * depth

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

    def get_plot_figure(self, bins=40, smoothing=1.5):
        """
        Plots a heatmap of where the person's feet spend time on the floor.

        Args:
            bins: Number of bins along each axis of the floor
            smoothing: Standard deviation of the smoothing, in cells

        Returns:
            Figure of the plot
        """
        self.get_selected_corners()
        self.get_true_scale_homography()
        self.transform_keypoints_to_true_scale()

        # Drop frames where a foot is airborne
        left_airborne, right_airborne = self.get_foot_in_air_mask()

        left = self.mapped_left_ankle_keypoints.copy()
        right = self.mapped_right_ankle_keypoints.copy()
        left[left_airborne] = np.nan
        right[right_airborne] = np.nan

        both = np.vstack([left, right])

        counts = self.get_floor_counts(both, bins)

        # Scaled to its maximum, since the smoothed counts are no longer
        # a number of frames
        density = gaussian_filter(counts, sigma=smoothing)
        density = density / density.max() if density.max() else density

        sns.set_theme(style="white")

        fig, ax = plt.subplots(figsize=(9, 8))

        width, depth = self.get_floor_size()

        # Transposed so the floor isn't drawn upside down
        image = ax.imshow(
            density.T,
            origin="lower",
            extent=(0, width, 0, depth),
            cmap="inferno",
            interpolation="bilinear",
        )

        self.outline_feet(ax, left, right, bins, smoothing)

        colour_bar = fig.colorbar(image, ax=ax, shrink=0.82, pad=0.03)
        colour_bar.set_label("Time spent (relative)", fontsize=10)

        ax.set_xlabel("Across the front", fontsize=10)
        ax.set_ylabel("Towards the back", fontsize=10)
        ax.set_title(
            f"Footwork heatmap: {self.person_state.person.name}",
            fontsize=13,
            pad=16,
        )

        fig.tight_layout()

        return fig
