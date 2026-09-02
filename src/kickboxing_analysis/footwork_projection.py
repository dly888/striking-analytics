import cv2
import numpy as np

from .tracking import PersonState


class FootworkProjector:
    def __init__(self, person_state: PersonState, principal_point=None):
        # Person data
        self.person_state = person_state

        self.principal_point = (
            None
            if principal_point is None
            else np.asarray(principal_point, dtype=np.float64)
        )
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
        self.floor_width_m = None
        self.true_scale_corners = None

    # ----------------------------------------------------
    # Floor selection
    # ----------------------------------------------------

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

    def select_floor_width_m(self, width: float):
        """
        Set the floor's real width in metres, measured across the near/far
        edges.

        Args:
            width: Real width of the floor in metres.
        """
        self.floor_width_m = width

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

    # ----------------------------------------------------
    # Homography
    # ----------------------------------------------------

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

        length1, length2 = self.floor_edge_lengths

        depth_m = (length1 + length2) / 2

        # Default to width estimate if real width length is not known
        if self.floor_width_m is not None:
            width_m = self.floor_width_m
        else:
            width_m = self.recover_width_m(depth_m)

        self.true_scale_corners = np.array(
            [[0, 0], [width_m, 0], [width_m, depth_m], [0, depth_m]],
            dtype=np.float64,
        )

    def recover_width_m(self, depth_m: float) -> float:
        """
        Recover the floor's width in metres from its four traced corners.

        Falls back to the raw pixel aspect ratio when the principal point is
        unknown, or when the geometry is too close to fronto-parallel for the
        focal length to be identified. That fallback is
        exact only when there is no perspective.
        """
        near1, near2, far2, far1 = self.floor_corners.astype(np.float64)

        # Take the averages of the lengths
        width_px = (np.linalg.norm(near2 - near1) + np.linalg.norm(far2 - far1)) / 2

        depth_px = (np.linalg.norm(far1 - near1) + np.linalg.norm(far2 - near2)) / 2

        affine_width_m = width_px / depth_px * depth_m

        if self.principal_point is None:
            return affine_width_m

        # Centre the image points on the principal point and work in
        # homogeneous coordinates.
        cx, cy = self.principal_point
        h1, h2, h3, h4 = (
            np.array([x - cx, y - cy, 1.0]) for x, y in (near1, near2, far2, far1)
        )

        # Vanishing points of the width edges and the depth edges
        vw = np.cross(np.cross(h1, h2), np.cross(h4, h3))
        vd = np.cross(np.cross(h1, h4), np.cross(h2, h3))

        # A vanishing point at infinity means that edge pair is parallel in the
        # image: no perspective in that direction and the focal length is not
        # identifiable, so fall back.
        near_parallel = abs(vw[2]) < 1e-9 * np.linalg.norm(vw[:2]) or abs(
            vd[2]
        ) < 1e-9 * np.linalg.norm(vd[:2])
        if near_parallel:
            return affine_width_m

        # The width and depth world directions are orthogonal, which fixes the
        # squared focal length. A non-positive value means the corners cannot
        # come from a rectangle under this camera model.
        f_squared = -(vw[0] * vd[0] + vw[1] * vd[1]) / (vw[2] * vd[2])
        if f_squared <= 0:
            return affine_width_m

        f = np.sqrt(f_squared)
        k_inv = np.diag([1.0 / f, 1.0 / f, 1.0])

        # Back-project the corners onto the floor plane (fixed at n . X = 1;
        # the plane's normal is orthogonal to both in-plane directions). The
        # ratio of world side lengths is independent of that fixed scale.
        normal = np.cross(k_inv @ vw, k_inv @ vd)

        def back_project(h):
            ray = k_inv @ h
            return ray / (normal @ ray)

        p1, p2, p3, p4 = (back_project(h) for h in (h1, h2, h3, h4))

        width_world = (np.linalg.norm(p2 - p1) + np.linalg.norm(p3 - p4)) / 2
        depth_world = (np.linalg.norm(p4 - p1) + np.linalg.norm(p3 - p2)) / 2

        return width_world / depth_world * depth_m

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
        """
        self.mapped_left_ankle_keypoints = self.map_joint_to_true_scale(
            self.left_ankle_keypoints
        )
        self.mapped_right_ankle_keypoints = self.map_joint_to_true_scale(
            self.right_ankle_keypoints
        )

    def project_true_scale(self):
        """
        Run the full true-scale projection pipeline.

        Builds the floor corners, the homography and the transformed ankle
        keypoints, defaulting to the unit square when edge lengths are
        unknown.
        """
        self.get_selected_corners()
        self.get_true_scale_homography()
        self.transform_keypoints_to_true_scale()
