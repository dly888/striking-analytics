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

        for row_out, row in zip(expanded, self.mask):
            for frame in np.flatnonzero(row):
                start = max(0, frame - before)
                end = min(self.n_frames, frame + after + 1)
                row_out[start:end] = True

        return Detections(self.strikes, expanded)

    def strike_frames(self, fps: float, min_frames: int = 1) -> np.ndarray:
        """
        Gets the frame index of every strike in order.

        Args:
            fps: Frames per second of the video.
            min_frames: Minimum number of frames a strike needs to be
                        detected for to be counted for.

        Returns:
            Numpy array containing the frame index of each strike.
        """
        return np.array(
            [
                record["frame"]
                for record in self.to_records(fps, min_frames)
            ],
            dtype=int,
        )

    def combo_mask(self, fps: float, min_frames: int = 1) -> np.ndarray:
        """
        Gets the mask indicating which consecutive strikes are part of a combo.

        Each entry is the gap between one strike and the next, so a run
        of n True entries is a combo of n + 1 strikes.

        Args:
            fps: Frames per second of the video.
            min_frames: Minimum number of frames a strike needs to be
                        detected for to be counted for.

        Returns:
            Numpy boolean array indicating whether consecutive strikes
            occur close enough together to be considered part of a combo.
        """
        strike_frames = self.strike_frames(fps, min_frames)

        if len(strike_frames) < 2:
            return np.array([], dtype=bool)

        return strike_frames[1:] - strike_frames[:-1] <= fps // 1

    def combo_count(self, fps: float, min_frames: int = 1) -> int:
        """
        Counts the number of combos thrown.

        Args:
            fps: Frames per second of the video.
            min_frames: Minimum number of frames a strike needs to be
                        detected for to be counted for.

        Returns:
            The number of combos detected.
        """
        return count_segments(
            self.combo_mask(fps, min_frames),
            min_length=2,
        )

    def combo_frames(self, fps: float, min_frames: int = 1) -> np.ndarray:
        """
        Gets the frame index of every strike that is part of a combo.

        Args:
            fps: Frames per second of the video.
            min_frames: Minimum number of frames a strike needs to be
                        detected for to be counted for.

        Returns:
            Numpy array containing the frame indexes where strikes that
            are part of a combo occur.
        """
        mask = self.combo_mask(fps, min_frames)

        if len(mask) == 0:
            return np.array([], dtype=int)

        strike_frames = self.strike_frames(fps, min_frames)

        # Make new mask to be the same size as strike_frames
        combo_mask = np.full(len(strike_frames), False, dtype=bool)

        combo_mask[:-1] |= mask
        combo_mask[1:] |= mask

        return strike_frames[combo_mask]

    def combo_frame_mask(
        self,
        fps: float,
        before: int = 0,
        after: int = 30,
        min_frames: int = 1,
    ) -> np.ndarray:
        """
        Gets a per frame mask marking the frames a combo is thrown on.

        Create a window around a combo detection so that it stays
        seen during the video annotation.

        Args:
            fps: Frames per second of the video.
            before: The length of the window before the frame.
            after: The length of the window after the frame.
            min_frames: Minimum number of frames a strike needs to be
                        detected for to be counted for.

        Returns:
            Numpy boolean array with one entry per frame in the video,
            True while a combo is being thrown.
        """
        frame_mask = np.full(self.n_frames, False, dtype=bool)

        for frame in self.combo_frames(fps, min_frames):
            start = max(0, frame - before)
            end = min(self.n_frames, frame + after + 1)
            frame_mask[start:end] = True

        return frame_mask

    def to_records(self, fps: float, min_frames: int = 1) -> list[dict]:
        """
        Flattens the detections into one record per strike thrown.

        A strike can be detected over several frames so each detection is
        recorded once at the starting frame. Strikes thrown at the same
        time from different limbs each have their own record.

        Args:
            fps: Frames per second of the video, used to time each strike.
            min_frames: Minimum number of frames a strike needs to be
                        detected for to be counted for.

        Returns:
            List of strikes in the order they were thrown, each holding the
            frame, the time in seconds, the strike type and the side.
        """
        records = []

        for strike in self.strikes:
            starts, ends = segment_bounds(self[strike])

            for start, end in zip(starts, ends):
                if end - start < min_frames:
                    continue

                records.append({
                    "frame": int(start),
                    "time_s": float(start / fps),
                    "strike_type": strike.strike_type,
                    "side": strike.side,
                })

        return sorted(records, key=lambda record: record["frame"])

