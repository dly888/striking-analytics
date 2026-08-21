from dataclasses import dataclass

import numpy as np

from .geometry import segment_bounds


@dataclass(frozen=True)
class GuardDetections:
    mask: np.ndarray

    @property
    def n_frames(self) -> int:
        return self.mask.shape[0]

    def start_frames(self) -> np.ndarray:
        """
        Gets the first frame index for when each guard drop occurs.

        Returns:
            List of indexes where a  guard drop occurs in the mask.
        """
        starts, _ = segment_bounds(self.mask)
        return starts

    def expanded(self, before: int = 0, after: int = 30) -> GuardDetections:
        """
        Create a window around each detection so the annotation remains
        visible for multiple frames.

        Args:
            before: The length of the window before the frame.
            after: The length of the window after the frame.

        Returns:
            Detection object which uses the new expanded boolean mask instead of the old boolean mask.
        """
        expanded = self.mask.copy()

        for frame_idx in np.flatnonzero(self.mask):
            start = max(0, frame_idx - before)
            end = min(self.n_frames, frame_idx + after + 1)
            expanded[start:end] = True

        return GuardDetections(expanded)