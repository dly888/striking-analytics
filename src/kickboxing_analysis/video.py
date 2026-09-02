from __future__ import annotations

import shutil
import subprocess
import warnings
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from functools import cache
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True)
class VideoLimit:
    """Maximum video characteristics that will be accepted."""

    max_duration_s: float = 300.0
    max_fps: float = 30.0
    max_long_edge_px: int = 1920
    max_pixels: int = 1920 * 1080
    duration_tolerance_s: float = 1.0
    fps_tolerance: float = 0.1


@dataclass(frozen=True)
class VideoMetadata:
    """Metadata from a video."""

    fps: float
    frame_count: float
    width: float
    height: float
    duration_s: float


class VideoValidationError(Enum):
    """Reasons a video cannot be processed within the configured limits."""

    UNREADABLE = "The video could not be opened."
    INVALID_FPS = "The video has an invalid frame rate."
    INVALID_FRAME_COUNT = "The video has an invalid frame count."
    INVALID_WIDTH = "The video has an invalid width."
    INVALID_HEIGHT = "The video has an invalid height."
    INVALID_DURATION = "The video has an invalid duration."
    DURATION_TOO_LONG = "The video is longer than the allowed duration."
    FPS_TOO_HIGH = "The video frame rate is too high."
    LONG_EDGE_TOO_LARGE = "The video resolution is too large."
    TOO_MANY_PIXELS = "The video has too many pixels per frame."


def get_video_metadata(video_path: Path) -> VideoMetadata:
    """Return the metadata needed to validate a video.

    Raises:
        OSError: If OpenCV cannot open the video.
    """

    with open_video(video_path) as capture:
        fps = capture.get(cv2.CAP_PROP_FPS)
        frame_count = capture.get(cv2.CAP_PROP_FRAME_COUNT)
        width = capture.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = capture.get(cv2.CAP_PROP_FRAME_HEIGHT)

    duration_s = frame_count / fps if fps > 0 else float("nan")
    return VideoMetadata(
        fps=fps,
        frame_count=frame_count,
        width=width,
        height=height,
        duration_s=duration_s,
    )


def validate_video(video_path: Path) -> tuple[bool, VideoValidationError | None]:
    """Return whether a video is valid and its first validation error, if any."""

    try:
        metadata = get_video_metadata(video_path)
    except OSError:
        return False, VideoValidationError.UNREADABLE

    if not np.isfinite(metadata.fps) or metadata.fps <= 0:
        return False, VideoValidationError.INVALID_FPS
    if not np.isfinite(metadata.frame_count) or metadata.frame_count <= 0:
        return False, VideoValidationError.INVALID_FRAME_COUNT
    if not np.isfinite(metadata.width) or metadata.width <= 0:
        return False, VideoValidationError.INVALID_WIDTH
    if not np.isfinite(metadata.height) or metadata.height <= 0:
        return False, VideoValidationError.INVALID_HEIGHT
    if not np.isfinite(metadata.duration_s) or metadata.duration_s <= 0:
        return False, VideoValidationError.INVALID_DURATION

    limit = VideoLimit()

    if metadata.duration_s > limit.max_duration_s + limit.duration_tolerance_s:
        return False, VideoValidationError.DURATION_TOO_LONG
    if metadata.fps > limit.max_fps + limit.fps_tolerance:
        return False, VideoValidationError.FPS_TOO_HIGH
    if max(metadata.width, metadata.height) > limit.max_long_edge_px:
        return False, VideoValidationError.LONG_EDGE_TOO_LARGE
    if metadata.width * metadata.height > limit.max_pixels:
        return False, VideoValidationError.TOO_MANY_PIXELS

    return True, None


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
def open_video_writer(
    path: Path, fps: float, size: tuple[int, int]
) -> Iterator[Callable[[np.ndarray], None]]:
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
