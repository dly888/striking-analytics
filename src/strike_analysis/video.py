from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import cv2


@contextmanager
def open_video(path: Path) -> Iterator[cv2.VideoCapture]:
    """Context manager to open video file.

    Args:
        path: Path of the video to be opened.
    """

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise OSError(f"Could not open video: {path}")
    try:
        yield cap
    finally:
        cap.release()


def get_fps(path: Path) -> float:
    """
    Gets fps of the video file.

    Args:
        path: Path of the video file.

    Returns:
        Fps of the video file.
    """

    with open_video(path) as capture:
        fps = capture.get(cv2.CAP_PROP_FPS)

    if not fps > 0:
        raise ValueError(f"Unusable frame rate ({fps}): {path}")

    return float(fps)