from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .constants import Side
from .geometry import count_segments, segment_bounds


@dataclass(frozen=True)
class Strike:
    """
    Stores data for each type of strike i.e. left straight, right hook, left roundhouse
    """

    strike_type: str
    side: Side

    @property
    def label(self) -> str:
        return f"{self.side} {self.strike_type}".upper()


@dataclass(frozen=True)
class Detections:
    """
    Used to track when a strike occurs.

    Uses a boolean mask to identify when a strike occurs, where each index is a frame in teh video.
    Contains a boolean mask for each strike object.
    """

    strikes: tuple[Strike, ...]
    mask: np.ndarray

    @property
    def n_frames(self) -> int:
        return self.mask.shape[1]

    def __getitem__(self, strike: Strike) -> np.ndarray:
        """
        Get the corresponding boolean mask of strike object using the strike object in the index.
        """
        return self.mask[self.strikes.index(strike)]

    def active_at(self, frame_idx: int) -> list[Strike]:
        """
        Checks which strikes occur at a given frame.

        Args:
            frame_idx: Index of frame currently being checked

        Returns:
            List of strikes active at the given frame.
        """

        return [
            strike
            for strike, mask_row in zip(self.strikes, self.mask)
            if mask_row[frame_idx]
        ]

    def counts(self, min_frames: int = 1) -> dict[Strike, int]:
        """
        Counts the strike occurrence for each strike type.

        Args:
            min_frames: Minimum number of frames a strike needs to be detected for to be counted for.

        Returns:
            Dictionary containing each strike and its count.
        """

        return {
            strike: count_segments(row, min_frames)
            for strike, row in zip(self.strikes, self.mask)
        }

    def start_frames(self, strike: Strike) -> np.ndarray:
        """
        Gets the first frame index for when each strike occurs.

        Args:
            strike: Strike object

        Returns:
            List of indexes where a strike detection occurs in the mask.
        """
        starts, _ = segment_bounds(self[strike])
        return starts

    def expanded(self, before: int = 0, after: int = 30) -> Detections:
        """Create a window around each detection so the annotation remains
        visible for multiple frames.

        Args:
            before: The length of the window before the frame.
            after: The length of the window after the frame.

        Returns:
            Detection object which uses the new expanded boolean mask instead of the old boolean mask.
        """
        expanded = self.mask.copy()

        for row_out, row in zip(expanded, self.mask):
            for frame in np.flatnonzero(row):
                start = max(0, frame - before)
                end = min(self.n_frames, frame + after + 1)
                row_out[start:end] = True

        return Detections(self.strikes, expanded)
