from __future__ import annotations

import numpy as np


def calculate_angles(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Calculates the angle between three vectors in fixed order, usually the keypoint positions.

    Args:
        a: Keypoint 1
        b: Keypoint 2, the middle point
        c: Keypoint 3

    Returns:
        Angle of abc in degrees.
    """

    ba = a - b
    bc = c - b

    dot = np.sum(ba * bc, axis=1)
    norms = np.linalg.norm(ba, axis=1) * np.linalg.norm(bc, axis=1)

    # Ignore warnings
    with np.errstate(divide="ignore", invalid="ignore"):
        cosine = np.clip(dot / norms, -1.0, 1.0)

    return np.degrees(np.arccos(cosine))


def segment_bounds(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Finds start and end indices of True segments in boolean mask.

    Mask padded with False values ends so changes between True and
    False segments are detected.

    Args:
        mask: Boolean array

    Returns:
        Tuple containing two arrays: start and end indices,  for each segment.
        .
    """
    edges = np.diff(np.pad(mask.astype(np.int8), 1))
    return np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)


def count_segments(mask: np.ndarray, min_length: int = 1) -> int:
    """
    Counts the number of segments of 1.

    Args:
        mask: Bitmask array
        min_length: Minimum length of a segment to be counted.

    Returns:
        The count of the number of segments in input mask.
    """

    starts, ends = segment_bounds(mask)
    return int(np.count_nonzero(ends - starts >= min_length))


def longest_nan_run(mask: np.ndarray) -> int:
    """Finds the longest sequence of consecutive nans in mask.

    Args:
        mask: Bitmask array.

    Returns:
        Longest sequence of consecutive nans in mask.
    """

    starts, ends = segment_bounds(mask)
    return int((ends - starts).max()) if starts.size else 0