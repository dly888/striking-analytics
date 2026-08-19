import numpy as np

from strike_analysis import Detections, PersonState, get_joint_speed, StrikeConfig, KEYPOINT_NAMES, STRIKE_TYPES, \
    count_segments

strike_config = StrikeConfig()


class StrikingStats:
    def __init__(self, person_state: PersonState, detections: Detections):
        self.person_state = person_state
        self.detections = detections
        self.fps = person_state.fps

        self.strike_records = detections.to_records(fps=self.fps)
        self.strike_frame_idx = self.get_strike_frame_idx()
        self.joint_speeds = self.get_joint_speeds()

        self.strike_counts = self.get_strike_count()
        self.max_speeds = self.get_max_speeds()
        self.combo_count = self.get_combo_count()

    # ============================================================
    # Data
    # ============================================================

    def get_strike_frame_idx(self) -> np.ndarray:
        """
        Get the frame indexes where strikes occur.

        Returns:
            Numpy array containing the frame indexes of each strike.
        """
        return np.array(
            [record["frame"] for record in self.strike_records],
            dtype=int
        )

    def get_joint_speeds(self) -> dict[str, np.ndarray]:
        """
        Get the speed of each joint for each frame.

        Returns:
            Dictionary containing the speed of each joint.
        """
        return {
            f"{side}_{joint}": get_joint_speed(
                state=self.person_state,
                joint_name=f"{side}_{joint}"
            )
            for side in ("left", "right")
            for joint in KEYPOINT_NAMES
        }

    # ============================================================
    # Strike statistics
    # ============================================================

    def get_strike_count(self) -> dict[str, int]:
        """
        Get the number of strikes of each type.

        Returns:
            Dictionary containing the count of each strike type.
        """
        strike_counts = {
            f"{side}_{strike_type}": 0
            for side in ("left", "right")
            for strike_type in STRIKE_TYPES
        }

        for record in self.strike_records:
            side = record["side"]
            strike_type = record["strike_type"]

            strike_name = f"{side}_{strike_type}"
            strike_counts[strike_name] += 1

        return strike_counts

    def get_max_speeds(self) -> dict[str, float]:
        """
        Get the maximum speed recorded for each strike type.

        Returns:
            Dictionary containing the maximum speed of each strike type.
        """
        max_speeds = {
            f"{side}_{strike_type}": 0
            for side in ("left", "right")
            for strike_type in STRIKE_TYPES
        }

        window = self.fps // 3

        for record in self.strike_records:
            side = record["side"]
            strike_type = record["strike_type"]
            strike_name = f"{side}_{strike_type}"

            start_frame = record["frame"]
            joint = f"{side}_wrist"

            current_max_speed = np.max(
                self.joint_speeds[joint][start_frame:start_frame + window]
            )

            max_speeds[strike_name] = max(
                current_max_speed,
                max_speeds[strike_name]
            )

        return max_speeds

    # ============================================================
    # Combo statistics
    # ============================================================

    def get_combo_mask(self) -> np.ndarray:
        """
        Get the mask indicating which consecutive strikes are part of a combo.

        Returns:
            Numpy boolean array indicating whether consecutive strikes
            occur close enough together to be considered part of a combo.
        """
        if len(self.strike_frame_idx) < 2:
            return np.array([], dtype=bool)

        return (
            self.strike_frame_idx[1:] - self.strike_frame_idx[:-1]
            <= self.fps // 3
        )

    def get_combo_count(self) -> int:
        """
        Get the number of combos detected.

        Returns:
            The number of combos detected.
        """
        mask = self.get_combo_mask()

        return count_segments(
            mask,
            min_length=2
        )

    def get_combo_frame_idx(self) -> np.ndarray:
        """
        Get the frame indexes of strikes that are part of a combo.

        Returns:
            Numpy array containing the frame indexes where strikes
            that are part of a combo occur.
        """
        mask = self.get_combo_mask()

        if len(mask) == 0:
            return np.array([], dtype=int)

        # Make new mask to be the same size as strike_frame_idx
        combo_mask = np.full(
            len(self.strike_frame_idx),
            False,
            dtype=bool
        )

        combo_mask[:-1] |= mask
        combo_mask[1:] |= mask

        return self.strike_frame_idx[combo_mask]