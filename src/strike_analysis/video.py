from __future__ import annotations

import shutil
import subprocess
import warnings
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from functools import cache
from pathlib import Path

import cv2
import numpy as np


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


# Containers the H.264 muxer accepts. A path with any other suffix cannot hold H.264.
H264_SUFFIXES = frozenset({".mp4", ".m4v", ".mov", ".mkv", ".ts"})


@cache
def find_h264_ffmpeg() -> str | None:
    """
    Finds an ffmpeg binary that can actually encode H.264.

    An ffmpeg on PATH is not enough: LGPL and minimal builds ship without the libx264
    encoder, so the binary is probed rather than assumed. The result is cached, as it
    cannot change within a run.

    Returns:
        Path to a suitable ffmpeg binary, or None if there is not one.
    """

    ffmpeg = shutil.which("ffmpeg")

    if ffmpeg is None:
        return None

    probe = subprocess.run(
        [ffmpeg, "-hide_banner", "-h", "encoder=libx264"],
        capture_output=True,
        text=True,
        check=False,
    )

    return ffmpeg if probe.stdout.startswith("Encoder libx264") else None


@contextmanager
def open_video_writer(path: Path, fps: float, size: tuple[int, int]) -> Iterator[Callable[[np.ndarray], None]]:
    """
    Context manager yielding a function that writes BGR frames to a video file.

    Prefers browser-playable H.264, piping frames to ffmpeg: OpenCV's bundled FFmpeg
    cannot encode H.264 without the optional openh264 library. If no ffmpeg that can
    encode H.264 is available, this warns and falls back to the mp4v codec, which
    writes successfully but which most browsers cannot play.

    Args:
        path: Path of the video file to write. To get H.264 the suffix must be a
            container that can hold it, one of H264_SUFFIXES.
        fps: Frame rate of the written video.
        size: (width, height) of the written frames.
    """

    width, height = size
    ffmpeg = find_h264_ffmpeg()

    if ffmpeg is None:
        warnings.warn(
            "No ffmpeg that can encode H.264 was found on PATH; falling back to mp4v. "
            "The video will not play in most browsers.",
            stacklevel=2,
        )
    elif path.suffix.lower() not in H264_SUFFIXES:
        ffmpeg = None
        warnings.warn(
            f"{path.suffix} cannot hold H.264; falling back to mp4v. "
            f"Use one of {sorted(H264_SUFFIXES)} for a browser-playable video.",
            stacklevel=2,
        )

    if ffmpeg is None:
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
        if not writer.isOpened():
            raise OSError(f"Could not open video writer: {path}")
        try:
            yield writer.write
        finally:
            writer.release()
        return

    process = subprocess.Popen(
        [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-s",
            f"{width}x{height}",
            "-r",
            f"{fps}",
            "-i",
            "-",
            "-an",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            # Put the metadata first so browser can start playing before the full download.
            "-movflags",
            "+faststart",
            str(path),
        ],
        stdin=subprocess.PIPE,
    )

    def write(frame: np.ndarray) -> None:
        try:
            process.stdin.write(frame.tobytes())
        except BrokenPipeError as error:
            # ffmpeg died early; report that rather than surfacing a broken pipe.
            raise OSError(f"ffmpeg stopped while writing video: {path}") from error

    try:
        yield write
    finally:
        if not process.stdin.closed:
            process.stdin.close()
        if process.wait() != 0:
            raise OSError(f"ffmpeg failed to write video: {path}")
