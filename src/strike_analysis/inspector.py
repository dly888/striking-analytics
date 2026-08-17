from __future__ import annotations

import numpy as np

from .constants import KEYPOINT_INDEX
from .geometry import longest_nan_run
from .tracking import PersonState


class VelocityInspector:
    """
    Inspector tool to get stats on speed in the video for debugging.
    """

    def __init__(self, track: PersonState | None = None):
        self.track = track

    def get_stats(
        self,
        speed: np.ndarray,
        start: int | None = None,
        end: int | None = None,
    ) -> dict[str, float]:
        """
        Gets stats about a given time window in the video.

        Args:
            speed: Speed values for each frame.
            start: First frame in the window.
            end: Last frame in the window.

        Returns:
            Dictionary containing stats.
        """
        window = speed[start:end]
        missing = np.isnan(window)

        if missing.all():
            return {
                "start_frame": start or 0,
                "end_frame": end if end is not None else len(speed),
                "num_frames": len(window),
                "nan_rate": 1.0,
                "longest_nan_run": len(window),
            }

        return {
            "start_frame": start or 0,
            "end_frame": end if end is not None else len(speed),
            "num_frames": len(window),
            "nan_rate": float(missing.mean()),
            "median": float(np.nanmedian(window)),
            "90th": float(np.nanpercentile(window, 90)),
            "max": float(np.nanmax(window)),
            "longest_nan_run": longest_nan_run(missing),
        }

    def find_velocity_outliers(
        self,
        speed: np.ndarray,
        threshold: float = 3000.0,
    ) -> np.ndarray:
        return np.flatnonzero(speed > threshold)

    def get_maximum_velocity(self, speed: np.ndarray) -> tuple[int, float]:
        idx = int(np.nanargmax(speed))
        return idx, float(speed[idx])

    def velocity_window(
        self,
        speed: np.ndarray,
        frame: int,
        radius: int = 5,
    ) -> str:
        start = max(0, frame - radius)
        end = min(len(speed), frame + radius + 1)

        lines = [f"Frames {start}-{end - 1}"]
        for i in range(start, end):
            marker = " <--" if i == frame else ""
            lines.append(f"{i:5d}: {speed[i]:8.1f}{marker}")

        return "\n".join(lines)

    def inspect_frame_pair(self, frame: int, *names: str) -> str:
        """Inspect keypoint values for a frame and its previous frame."""

        if self.track is None:
            raise ValueError("No track found.")

        lines = []
        for name in names or (
            "right_shoulder",
            "right_elbow",
            "right_wrist",
        ):
            values = self.track.keypoints[
                max(0, frame - 1) : frame + 1,
                KEYPOINT_INDEX[name],
            ]
            lines.append(
                f"{name:16s} {np.array2string(values, precision=1)}"
            )

        return "\n".join(lines)

    def print_wrist_speeds(self, left_speed: np.ndarray | None, right_speed: np.ndarray | None):
        for side, speed in (
            ("left", left_speed),
            ("right", right_speed),
        ):
            if speed is None:
                continue

            print(f"  {side} wrist stats: {self.get_stats(speed)}")