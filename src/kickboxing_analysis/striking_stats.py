from dataclasses import dataclass

import numpy as np

from .config import StrikeConfig
from .constants import JOINT_NAMES, SIDES, STRIKE_TYPE_TO_JOINT, STRIKE_TYPES
from .features import get_joint_speed, get_pixel_to_meter_ratio
from .strike_detections import StrikeDetections
from .tracking import PersonState

# Strikes thrown with the hand versus the leg, used to split the mix.
PUNCH_TYPES: tuple[str, ...] = ("straight", "hook", "uppercut")
KICK_TYPES: tuple[str, ...] = ("kick",)

BUSIEST_WINDOW_S = 3.0


@dataclass(frozen=True)
class StrikingStats:
    total_strikes: int
    strike_counts: dict[str, float]
    strike_rates: dict[str, float]
    max_speeds_mps: dict[str, float]
    avg_speeds_mps: dict[str, float]
    combo_count: int
    rhythm_cv: float
    pacing_bins: dict[float, int]
    strike_speeds: dict[str, np.ndarray]
    strike_times_s: dict[str, np.ndarray]

    # Balance
    side_counts: dict[str, int]
    lead_rear_counts: dict[str, int]

    # Mix
    strike_mix: dict[str, int]
    punch_count: int
    kick_count: int

    # Combos
    avg_combo_length: float
    longest_combo: int

    # Work rate
    mean_interval_s: float
    longest_rest_s: float
    busiest_window_size_s: float


class StrikingStatsCalculator:
    def __init__(
        self,
        person_state: PersonState,
        detections: StrikeDetections,
        strike_config: StrikeConfig,
    ):
        self.person_state = person_state
        self.detections = detections
        self.strike_config = strike_config
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
        side_counts = self.get_side_counts()
        strike_mix = self.get_strike_mix()

        return StrikingStats(
            total_strikes=self.get_total_strikes(),
            strike_counts=self.strike_counts,
            strike_rates=self.get_strike_rate(),
            max_speeds_mps=self.max_speeds,
            avg_speeds_mps=self.get_avg_speeds(),
            combo_count=self.combo_count,
            rhythm_cv=self.get_striking_rhythm(),
            pacing_bins=self.get_pacing_bins(),
            strike_speeds=self.strike_speeds,
            strike_times_s=self.get_strike_times(),
            side_counts=side_counts,
            lead_rear_counts=self.get_lead_rear_counts(side_counts),
            strike_mix=strike_mix,
            punch_count=sum(strike_mix[t] for t in PUNCH_TYPES),
            kick_count=sum(strike_mix[t] for t in KICK_TYPES),
            avg_combo_length=self.get_avg_combo_length(),
            longest_combo=self.get_longest_combo(),
            mean_interval_s=self.get_mean_strike_gap(),
            longest_rest_s=self.get_longest_rest(),
            busiest_window_size_s=BUSIEST_WINDOW_S,
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
        return np.array([record["frame"] for record in self.strike_records])

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
                strike_config=self.strike_config,
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
        for count in self.strike_counts.values():
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

            speeds = self.joint_speeds[joint][start_frame : start_frame + window]

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

    def get_avg_speeds(self) -> dict[str, float]:
        """
        Get the average peak speed recorded for each strike type.

        Returns:
            Dictionary containing the mean peak speed of each strike type
            in metres per second. Types never detected are 0.
        """
        avg_speeds = {}

        for strike_name, speeds in self.strike_speeds.items():
            if not np.isfinite(speeds).any():
                avg_speeds[strike_name] = 0.0
                continue

            avg_speeds[strike_name] = float(np.nanmean(speeds))

        return avg_speeds

    # ============================================================
    # Balance statistics
    # ============================================================

    def get_side_counts(self) -> dict[str, int]:
        """
        Get how many strikes were thrown with each side.

        Returns:
            Dictionary mapping "left" and "right" to the number of
            strikes thrown with that side.
        """
        counts = {side: 0 for side in SIDES}

        for record in self.strike_records:
            counts[record["side"]] += 1

        return counts

    def get_lead_rear_counts(self, side_counts: dict[str, int]) -> dict[str, int]:
        """
        Split the strike count into lead side and rear side.

        Which hand leads depends on the stance: an orthodox fighter
        leads with the left, a southpaw with the right.

        Args:
            side_counts: Strike counts keyed by side, from get_side_counts.

        Returns:
            Dictionary mapping "lead" and "rear" to the number of strikes
            thrown with that side.
        """
        lead_side = "left" if self.person_state.person.stance == "orthodox" else "right"
        rear_side = "right" if lead_side == "left" else "left"

        return {
            "lead": side_counts[lead_side],
            "rear": side_counts[rear_side],
        }

    # ============================================================
    # Mix statistics
    # ============================================================

    def get_strike_mix(self) -> dict[str, int]:
        """
        Get how many strikes of each type were thrown, summed over sides.

        Returns:
            Dictionary mapping each strike type to its total count across
            both sides.
        """
        mix = {strike_type: 0 for strike_type in STRIKE_TYPES}

        for record in self.strike_records:
            mix[record["strike_type"]] += 1

        return mix

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
        time_intervals_between_strikes_s = time_intervals_between_strikes_s[
            time_intervals_between_strikes_s != 0
        ]

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

        counts, _ = np.histogram(self.strike_frames / self.fps, bins=edges)

        return {float(start): int(count) for start, count in zip(edges[:-1], counts)}

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

    def get_combos(self) -> list[list[dict]]:
        """
        Group the strike records into the combos they belong to.

        Returns:
            List of combos, each a list of the strike records it holds,
            in the order they were thrown.
        """
        mask = self.get_combo_mask()

        if len(mask) == 0:
            return []

        combos = []
        current = [self.strike_records[0]]

        for i, linked in enumerate(mask):
            if linked:
                current.append(self.strike_records[i + 1])
            else:
                if len(current) >= 2:
                    combos.append(current)
                current = [self.strike_records[i + 1]]

        if len(current) >= 2:
            combos.append(current)

        return combos

    def get_avg_combo_length(self) -> float:
        """
        Get the average number of strikes in a combo.

        Returns:
            Mean strikes per combo, or 0 if no combos were thrown.
        """
        combos = self.get_combos()

        if not combos:
            return 0.0

        return float(np.mean([len(combo) for combo in combos]))

    def get_longest_combo(self) -> int:
        """
        Get the number of strikes in the longest combo.

        Returns:
            The strike count of the longest combo, or 0 if none was thrown.
        """
        combos = self.get_combos()

        if not combos:
            return 0

        return max(len(combo) for combo in combos)

    # ============================================================
    # Work rate statistics
    # ============================================================

    def get_strike_times_s(self) -> np.ndarray:
        """
        Get the time of every strike in seconds, in order.

        Returns:
            Sorted numpy array of strike times in seconds.
        """
        return np.sort(self.strike_frames) / self.fps

    def get_mean_strike_gap(self) -> float:
        """
        Get the average gap between one strike and the next.

        Returns:
            Mean interval in seconds, or 0 with fewer than two strikes.
        """
        times = self.get_strike_times_s()

        if times.size < 2:
            return 0.0

        return float(np.mean(np.diff(times)))

    def get_longest_rest(self) -> float:
        """
        Get the longest quiet gap between two strikes.

        Returns:
            Longest interval in seconds, or 0 with fewer than two strikes.
        """
        times = self.get_strike_times_s()

        if times.size < 2:
            return 0.0

        return float(np.max(np.diff(times)))

    # ============================================================
    # Fatigue statistics
    # ============================================================

    def get_speed_decline(self) -> float:
        """
        Implement linear regression later
        """
