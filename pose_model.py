from __future__ import annotations

from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Literal

import cv2
import numpy as np
from ultralytics import YOLO

Side = Literal["left", "right"]

KEYPOINT_NAMES: tuple[str, ...] = (
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
)
KEYPOINT_INDEX: dict[str, int] = {name: i for i, name in enumerate(KEYPOINT_NAMES)}
N_KEYPOINTS = len(KEYPOINT_NAMES)


@dataclass(frozen=True)
class Config:
    keypoint_conf: float = 0.4
    extension_angle_threshold: float = 150.0
    velocity_percentile: float = 90.0
    max_hold: int = 6
    min_punch_frames: int = 1


# ========================================================================= #
# GEOMETRY AND HELPERS
# ========================================================================= #


def calculate_angles(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    ba = a - b
    bc = c - b

    dot = np.sum(ba * bc, axis=1)
    norms = np.linalg.norm(ba, axis=1) * np.linalg.norm(bc, axis=1)

    # Ignore warnings
    with np.errstate(divide="ignore", invalid="ignore"):
        cosine = np.clip(dot / norms, -1.0, 1.0)

    return np.degrees(np.arccos(cosine))


def segment_bounds(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    edges = np.diff(np.pad(mask.astype(np.int8), 1))
    return np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)


def count_segments(mask: np.ndarray, min_length: int = 1) -> int:
    starts, ends = segment_bounds(mask)
    return int(np.count_nonzero(ends - starts >= min_length))


def longest_nan_run(mask: np.ndarray) -> int:
    starts, ends = segment_bounds(mask)
    return int((ends - starts).max()) if starts.size else 0


# ========================================================================= #
# VIDEO
# ========================================================================= #


@contextmanager
def open_video(path: Path) -> Iterator[cv2.VideoCapture]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise OSError(f"Could not open video: {path}")
    try:
        yield cap
    finally:
        cap.release()


def get_fps(path: Path) -> float:
    with open_video(path) as capture:
        fps = capture.get(cv2.CAP_PROP_FPS)

    if not fps > 0:
        raise ValueError(f"Unusable frame rate ({fps}): {path}")

    return float(fps)


# ========================================================================= #
# TRACKING
# ========================================================================= #


@dataclass(frozen=True)
class PersonTrack:
    track_id: int
    keypoints: np.ndarray  # (frames_processed, N_KEYPOINTS, 3) of x, y, confidence
    boxes: np.ndarray  # (frames_processed, 4) of xyxy
    box_conf: np.ndarray  # (frames_processed,)
    fps: float

    @property
    def frames_processed(self) -> int:
        return len(self.keypoints)

    @property
    def detected(self) -> np.ndarray:
        return ~np.isnan(self.boxes[:, 0])

    def positions(self, name: str) -> np.ndarray:
        return self.keypoints[:, KEYPOINT_INDEX[name], :2]

    def confidence(self, name: str) -> np.ndarray:
        return self.keypoints[:, KEYPOINT_INDEX[name], 2]


class PoseTracker:
    def __init__(self, model: str = "yolo26n-pose.pt", config: Config = Config()):
        self.model = YOLO(model)
        self.config = config

    def get_person_tracker(self, video_path: Path) -> dict[int, PersonTrack]:
        fps = get_fps(video_path)
        detections: dict[int, dict[int, tuple]] = defaultdict(dict)
        frames_processed = 0

        results = self.model.track(
            source=str(video_path),
            persist=True,
            tracker="bytetrack.yaml",
            stream=True,
            classes=[0],
        )

        for frame_idx, result in enumerate(results):
            frames_processed = frame_idx + 1

            if result.boxes.id is None or result.keypoints.conf is None:
                continue

            track_ids = result.boxes.id.int().tolist()
            xy = result.keypoints.xy.cpu().numpy()
            keypoint_conf = result.keypoints.conf.cpu().numpy()
            boxes = result.boxes.xyxy.cpu().numpy()
            box_conf = result.boxes.conf.cpu().numpy()

            for det_idx, track_id in enumerate(track_ids):
                detections[track_id][frame_idx] = (
                    np.hstack([xy[det_idx], keypoint_conf[det_idx, :, None]]),
                    boxes[det_idx],
                    box_conf[det_idx],
                )

        return {
            track_id: self._densify(track_id, frames, frames_processed, fps)
            for track_id, frames in detections.items()
        }

    def _densify(
            self,
            track_id: int,
            frames: dict[int, tuple],
            frames_processed: int,
            fps: float,
    ) -> PersonTrack:

        keypoints = np.full((frames_processed, N_KEYPOINTS, 3), np.nan, dtype=np.float32)
        boxes = np.full((frames_processed, 4), np.nan, dtype=np.float32)
        box_conf = np.full(frames_processed, np.nan, dtype=np.float32)

        for frame_idx, (frame_keypoints, box, conf) in frames.items():
            keypoints[frame_idx] = frame_keypoints
            boxes[frame_idx] = box
            box_conf[frame_idx] = conf

        keypoints[keypoints[:, :, 2] < self.config.keypoint_conf] = np.nan

        return PersonTrack(track_id, keypoints, boxes, box_conf, fps)

    @staticmethod
    def get_top_n_ids(person_tracker: dict[int, PersonTrack], n: int = 2) -> list[int]:
        return sorted(
            person_tracker,
            key=lambda track_id: int(person_tracker[track_id].detected.sum()),
            reverse=True,
        )[:n]

# ========================================================================= #
# ANNOTATERS
# ========================================================================= #


class VideoAnnotater:
    def __init__(self, config: Config = Config()):
        self.person_tracks: list[PersonTrack] = []
        self.config = config

    def add_tracker(self, tracker: PersonTrack):
        self.person_tracks.append(tracker)

    def expand_punch_detections(
            self,
            punch_detector: np.ndarray,
            before: int = 0,
            after: int = 30,
    ) -> np.ndarray:
        "Create a window of punch detections around where a punch was actually detected so the punch"
        "annotation can actually be seen for more than one frame"

        expanded = punch_detector.copy()

        punch_frames = np.flatnonzero(punch_detector)

        for frame in punch_frames:
            start = max(0, frame - before)
            end = min(len(punch_detector), frame + after + 1)
            expanded[start:end] = 1

        return expanded

    def annotate_frame(self, person_track: PersonTrack, frame, frame_idx: int, punch_detected: bool):
        box = person_track.boxes[frame_idx]

        if np.isnan(box).any():
            return

        x1, y1, x2, y2 = map(int, box)

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        cv2.putText(
            frame,
            f"ID {person_track.track_id}",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )

        if punch_detected:
            cv2.putText(
                frame,
                "PUNCH",
                (x2, y1 - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2,
            )

    def annotate_video(
            self,
            video_path: Path,
            new_file_path: str,
            side: Side = "left",
    ) -> None:

        punch_detectors = {
            track.track_id: self.expand_punch_detections(PunchAnalyser(track, self.config).get_punch_detector(side))
            for track in self.person_tracks
        }

        with open_video(video_path) as cap:
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            writer = cv2.VideoWriter(
                str(new_file_path),
                cv2.VideoWriter_fourcc(*"mp4v"),
                get_fps(video_path),
                (width, height),
            )

            frame_idx = 0

            while True:
                success, frame = cap.read()
                if not success:
                    break

                for track in self.person_tracks:
                    if frame_idx >= track.frames_processed:
                        continue

                    self.annotate_frame(
                        person_track=track,
                        frame=frame,
                        frame_idx=frame_idx,
                        punch_detected=punch_detectors[track.track_id][frame_idx],
                    )

                writer.write(frame)
                frame_idx += 1

            writer.release()


# ========================================================================= #
# ANALYSIS
# ========================================================================= #


class PunchAnalyser:
    def __init__(self, track: PersonTrack, config: Config = Config()):
        self.track = track
        self.config = config

    def get_wrist_velocities(self, side: Side) -> np.ndarray:
        xy = self.track.positions(f"{side}_wrist")
        visible = ~np.isnan(xy[:, 0])
        frames = np.arange(len(xy))

        seen_at_or_before = np.maximum.accumulate(np.where(visible, frames, -1))
        previous = np.roll(seen_at_or_before, 1)
        previous[0] = -1

        gap = frames - previous
        measurable = visible & (previous >= 0) & (gap <= self.config.max_hold + 1)

        speed = np.full(len(xy), np.nan)
        distance = np.linalg.norm(xy[measurable] - xy[previous[measurable]], axis=1)
        speed[measurable] = distance * self.track.fps / gap[measurable]

        return speed

    def get_is_arm_extended(self, side: Side) -> np.ndarray:
        angles = calculate_angles(
            self.track.positions(f"{side}_shoulder"),
            self.track.positions(f"{side}_elbow"),
            self.track.positions(f"{side}_wrist"),
        )
        return angles > self.config.extension_angle_threshold

    def get_velocity_threshold(self, speed: np.ndarray) -> float:
        if not np.any(~np.isnan(speed)):
            return np.inf

        return float(np.nanpercentile(speed, self.config.velocity_percentile))

    def get_punch_detector(self, side: Side) -> np.ndarray:
        speed = self.get_wrist_velocities(side)
        threshold = self.get_velocity_threshold(speed)

        return self.get_is_arm_extended(side) & (speed > threshold)

    def get_punch_count(self, side: Side) -> int:
        return count_segments(
            self.get_punch_detector(side), self.config.min_punch_frames
        )


class VelocityInspector:
    def __init__(self, speed: np.ndarray, track: PersonTrack | None = None):
        self.speed = speed
        self.track = track

    def get_stats(
            self, start: int | None = None, end: int | None = None
    ) -> dict[str, float]:
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
        if self.track is None:
            raise ValueError("No track found.")

        lines = []
        for name in names or ("right_shoulder", "right_elbow", "right_wrist"):
            values = self.track.keypoints[
                max(0, frame - 1) : frame + 1, KEYPOINT_INDEX[name]
            ]
            lines.append(f"{name:16s} {np.array2string(values, precision=1)}")

        return "\n".join(lines)


# ========================================================================= #
# ENTRY POINT
# ========================================================================= #


def main(
        video_path: Path,
        side: Side = "right",
        model: str = "yolo26n-pose.pt",
        fighters: int = 1,
        config: Config = Config(),
) -> None:
    tracker = PoseTracker(model=model, config=config)
    person_tracker = tracker.get_person_tracker(video_path)
    video_annotater = VideoAnnotater(config=config)

    if not person_tracker:
        print("No people were tracked.")
        return

    for track_id in tracker.get_top_n_ids(person_tracker, n=fighters):
        track = person_tracker[track_id]
        analyser = PunchAnalyser(track, config=config)
        punch_detector = analyser.get_punch_detector(side)
        video_annotater.add_tracker(track)

        starts, _ = segment_bounds(punch_detector)
        print(
            f"\nTrack {track_id}: seen in "
            f"{int(track.detected.sum())}/{track.frames_processed} frames"
        )
        print(f"  {side} punches: {count_segments(punch_detector, config.min_punch_frames)}")
        print(f"  at frames: {[int(s) for s in starts]}")
        print(
            f"  stats: "
            f"{VelocityInspector(analyser.get_wrist_velocities(side)).get_stats()}"
        )

    video_annotater.annotate_video(video_path=video_path, new_file_path="annotate_test_0003.mp4", side=side)


if __name__ == "__main__":
    FRAME_PATH = Path("assets") / "frames" / "Van Vs Royval" / "frame_00400.jpg"
    VIDEO_PATH = (
            Path("assets")
            / "clips"
            / "Joshua Van vs Brandon Royval ｜ FULL FIGHT ｜ UFC 328 [nwO2UPz7p28].webm"
    )

    main(video_path=VIDEO_PATH, side="right")
