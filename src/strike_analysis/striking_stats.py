from dataclasses import dataclass

import numpy as np

from .config import StrikeConfig
from .constants import JOINT_NAMES, SIDES, STRIKE_TYPE_TO_JOINT, STRIKE_TYPES
from .detections import Detections
from .features import get_joint_speed, get_pixel_to_meter_ratio
from .tracking import PersonState

strike_config = StrikeConfig()

@dataclass(frozen=True)
class StrikingStats:
    total_strikes: int
    strike_counts: dict[str, float]
    strike_rates: dict[str, float]
    max_speeds_mps: dict[str, float]
    combo_count: int
    rhythm_cv: float
    pacing_bins: dict[float, int]
    strike_speeds: dict[str, np.ndarray]
    strike_times_s: dict[str, np.ndarray]
    

class StrikingStatsCalculator:
    def __init__(self, person_state: PersonState, detections: Detections):
        self.person_state = person_state
        self.detections = detections
        self.fps = person_state.fps
        self.n_frames = detections.n_frames

        self.strike_records = detections.to_records(fps=self.fps)
        self.strike_frames = self.get_strike_frames()
        self.joint_speeds = self.get_joint_speeds()

        self.strike_counts = self.get_strike_count()
        self.m_per_pixel = self.get_metres_per_pixel()
        self.strike_speeds = self.get_strike_speeds()
        self.max_speeds = self.get_max_speeds()
        self.combo_count = self.get_combo_count()


    # ============================================================
    # Return Stats
    # ============================================================

    def calculate_striking_stats(self) -> StrikingStats:
        """
        Get every striking statistic for the video in one object.

        Gathers the statistics already built during setup together with
        the ones calculated on demand.

        Returns:
            StrikingStats object holding the strike totals, counts and
            rates, the maximum speeds, the combo count and the rhythm
            score for the video.
        """
        return StrikingStats(
            total_strikes=self.get_total_strikes(),
            strike_counts=self.strike_counts,
            strike_rates=self.get_strike_rate(),
            max_speeds_mps=self.max_speeds,
            combo_count=self.combo_count,
            rhythm_cv=self.get_striking_rhythm(),
            pacing_bins=self.get_pacing_bins(),
            strike_speeds=self.strike_speeds,
            strike_times_s=self.get_strike_times(),
        )


    # ============================================================
    # Data
    # ============================================================

    def get_strike_frames(self) -> np.ndarray:
        """
        Get the frame indexes where strikes occur.

        Returns:
            Numpy array containing the frame indexes of each strike.
        """
        return np.array(
            [record["frame"] for record in self.strike_records]
        )

    def get_strike_times(self) -> dict[str, np.ndarray]:
        """
        Get the times at which strikes of each type occur.

        Returns:
            Dictionary containing the time of each strike in seconds,
            for each strike type.
        """
        strike_times = {
            f"{side}_{strike_type}": []
            for side in ("left", "right")
            for strike_type in STRIKE_TYPES
        }

        for record in self.strike_records:
            side = record["side"]
            strike_type = record["strike_type"]

            strike_name = f"{side}_{strike_type}"
            strike_times[strike_name].append(record["time_s"])

        return {
            strike_name: np.array(times, dtype=float)
            for strike_name, times in strike_times.items()
        }

    def get_joint_speeds(self) -> dict[str, np.ndarray]:
        """
        Get the speed of each joint for each frame.

        Returns:
            Dictionary containing the speed of each joint.
        """
        return {
            f"{side}_{joint}": get_joint_speed(
                state=self.person_state,
                joint_name=f"{side}_{joint}",
                strike_config=strike_config
            )
            for side in SIDES
            for joint in JOINT_NAMES
        }

    def get_metres_per_pixel(self) -> float:
        """
        Get the conversion ratio from pixels to metres for the video.

        The ratio is estimated by the median fighter's torso length
        from all frames.

        Returns:
            The number of metres a single pixel represents.
        """
        ratios = get_pixel_to_meter_ratio(self.person_state)

        return float(np.nanmedian(ratios))

    # ============================================================
    # Strike statistics
    # ============================================================

    def get_strike_count(self) -> dict[str, float]:
        """
        Get the number of strikes of each type.

        Returns:
            Dictionary containing the count of each strike type.
        """
        strike_counts = {
            f"{side}_{strike_type}": 0.0
            for side in ("left", "right")
            for strike_type in STRIKE_TYPES
        }

        for record in self.strike_records:
            side = record["side"]
            strike_type = record["strike_type"]

            strike_name = f"{side}_{strike_type}"
            strike_counts[strike_name] += 1

        return strike_counts

    def get_strike_rate(self) -> dict[str, float]:
        """
        Get the rate at which strikes occur.

        Returns:
            Dictionary containing strike rate for each strike type
        """
        total_time = self.n_frames / self.fps
        strike_rate = {
            f"{side}_{strike_type}": 0
            for side in ("left", "right")
            for strike_type in STRIKE_TYPES
        }

        for strike_type, count in self.strike_counts.items():
            strike_rate[strike_type] = count / total_time

        return strike_rate

    def get_total_strikes(self) -> int:
        """
        Get the total number of strikes of all strike types.

        Returns:
            The total number of strikes of all strike types.
        """
        total = 0
        for strike_type, count in self.strike_counts.items():
            total += count

        return int(total)

    def get_strike_speeds(self) -> dict[str, np.ndarray]:
        """
        Get the peak speed of every strike type.

        Strikes that could not be detected are NaN.

        Returns:
            Dictionary containing the peak speed of each strike in
            metres per second, for each strike type.
        """
        window = int(self.fps // 3)
        strike_speeds = {
            f"{side}_{strike_type}": []
            for side in ("left", "right")
            for strike_type in STRIKE_TYPES
        }

        for record in self.strike_records:
            side = record["side"]
            strike_type = record["strike_type"]

            strike_name = f"{side}_{strike_type}"
            start_frame = record["frame"]
            joint = f"{side}_{STRIKE_TYPE_TO_JOINT[strike_type]}"

            speeds = self.joint_speeds[joint][start_frame:start_frame + window]

            # Skip NaN values
            if not np.isfinite(speeds).any():
                strike_speeds[strike_name].append(np.nan)
                continue

            strike_speeds[strike_name].append(
                float(np.nanmax(speeds) * self.m_per_pixel)
            )

        return {
            strike_name: np.array(speeds, dtype=float)
            for strike_name, speeds in strike_speeds.items()
        }

    def get_max_speeds(self) -> dict[str, float]:
        """
        Get the maximum speed recorded for each strike type.

        Returns:
            Dictionary containing the maximum speed of each strike type
            in metres per second.
        """
        max_speeds = {}

        for strike_name, speeds in self.strike_speeds.items():
            # Strike types that were never detected
            if not np.isfinite(speeds).any():
                max_speeds[strike_name] = 0.0
                continue

            max_speeds[strike_name] = float(np.nanmax(speeds))

        return max_speeds

    def get_striking_rhythm(self) -> float:
        """
        Gets a score to measure striking rhythm.

        Finds time interval between strikes then calculates the coefficient
        of variation of these intervals.

        Returns:
            Coefficient of variation of the time intervals between strikes
        """
        frame_diff = np.diff(self.strike_frames)
        time_intervals_between_strikes_s = frame_diff / self.fps
        time_intervals_between_strikes_s = time_intervals_between_strikes_s[time_intervals_between_strikes_s != 0]

        if len(time_intervals_between_strikes_s) < 2:
            return 0.0

        std = np.std(time_intervals_between_strikes_s)
        mean = np.mean(time_intervals_between_strikes_s)
        coefficient_of_variation = std / mean

        return float(coefficient_of_variation)

    def get_pacing_bins(self, bin_size_s: float = 5.0) -> dict[float, int]:
        """
        Get the number of strikes thrown in each block of time.

        Splits the video into fixed length bins and counts the strikes
        during each one

        Args:
            bin_size_s: Length of each bin in seconds.

        Returns:
            Dictionary mapping the start time of each bin in seconds to
            the number of strikes thrown during it.
        """
        duration_s = self.n_frames / self.fps

        if duration_s <= 0:
            return {}

        edges = np.arange(0.0, duration_s + bin_size_s, bin_size_s)

        counts, _ = np.histogram(
            self.strike_frames / self.fps,
            bins=edges
        )

        return {
            float(start): int(count)
            for start, count in zip(edges[:-1], counts)
        }

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
        return self.detections.combo_mask(self.fps)

    def get_combo_count(self) -> int:
        """
        Get the number of combos detected.

        Returns:
            The number of combos detected.
        """
        return self.detections.combo_count(self.fps)

    def get_combo_frame_idx(self) -> np.ndarray:
        """
        Get the frame indexes of strikes that are part of a combo.

        Returns:
            Numpy array containing the frame indexes where strikes
            that are part of a combo occur.
        """
        return self.detections.combo_frames(self.fps)
