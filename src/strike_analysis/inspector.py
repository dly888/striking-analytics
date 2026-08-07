from __future__ import annotations

import numpy as np

from .constants import KEYPOINT_INDEX
from .geometry import longest_nan_run
from .tracking import PersonTrack


class VelocityInspector:
    """
    Inspector tool to get stats on speed in the video for debugging.
    """

    def __init__(self, speed: np.ndarray, track: PersonTrack | None = None):
        self.speed = speed
        self.track = track

    def get_stats(
            self, start: int | None = None, end: int | None = None
    ) -> dict[str, float]:
        """
        Gets stats about a given time window in the video.

        Args:
            start: First frame in the window frame.
            end: Last frame in the window frame.

        Returns:
            Dictionary containing stats.
        """
        window = self.speed[start:end]
        missing = np.isnan(window)

        if missing.all():
            return {
                "start_frame": start or 0,
                "end_frame": end if end is not None else len(self.speed),
                "num_frames": len(window),
                "nan_rate": 1.0,
                "longest_nan_run": len(window),
            }

        return {
            "start_frame": start or 0,
            "end_frame": end if end is not None else len(self.speed),
            "num_frames": len(window),
            "nan_rate": float(missing.mean()),
            "median": float(np.nanmedian(window)),
            "90th": float(np.nanpercentile(window, 90)),
            "max": float(np.nanmax(window)),
            "longest_nan_run": longest_nan_run(missing),
        }

    def find_velocity_outliers(self, threshold: float = 3000.0) -> np.ndarray:
        return np.flatnonzero(self.speed > threshold)

    def get_maximum_velocity(self) -> tuple[int, float]:
        idx = int(np.nanargmax(self.speed))
        return idx, float(self.speed[idx])

    def velocity_window(self, frame: int, radius: int = 5) -> str:
        start = max(0, frame - radius)
        end = min(len(self.speed), frame + radius + 1)

        lines = [f"Frames {start}-{end - 1}"]
        for i in range(start, end):
            marker = " <--" if i == frame else ""
            lines.append(f"{i:5d}: {self.speed[i]:8.1f}{marker}")

        return "\n".join(lines)

    def inspect_frame_pair(self, frame: int, *names: str) -> str:
        """Inspect keypoint values for a frame and its previous frame.

        Gets keypoint coordinates and confidence values for specified
        joints at the given frame and the previous frame.

        If no keypoints provided, right shoulder, right elbow, and right
        wrist inspected by default.

        Args:
            frame: Frame index to inspect.
            *names: Names of the keypoints to inspect.

        Returns:
            String containing keypoint values for selected
            joints over the inspected frames.

        Raises:
            ValueError: If no PersonTrack is associated with the inspector.
        """

        if self.track is None:
            raise ValueError("No track found.")

        lines = []
        for name in names or ("right_shoulder", "right_elbow", "right_wrist"):
            values = self.track.keypoints[
                max(0, frame - 1): frame + 1, KEYPOINT_INDEX[name]
            ]
            lines.append(f"{name:16s} {np.array2string(values, precision=1)}")

        return "\n".join(lines)