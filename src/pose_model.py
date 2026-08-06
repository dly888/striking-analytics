from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

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
    """Configuration parameters controlling detection thresholds and tracking behaviour."""

    keypoint_conf: float = 0.4
    arm_extension_angle_threshold: float = 150.0
    arm_body_angle_threshold: float = 35.0
    velocity_percentile: float = 95.0
    max_hold: int = 6
    min_punch_frames: int = 1


# ========================================================================= #
# GEOMETRY AND HELPERS
# ========================================================================= #


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


# ========================================================================= #
# VIDEO
# ========================================================================= #


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


# ========================================================================= #
# TRACKING
# ========================================================================= #


@dataclass(frozen=True)
class Person:
    """
    Stores data of the person.
    """

    name: str
    weight: float
    height_m: float
    wingspan_m: float
    stance: Side


@dataclass(frozen=True)
class PersonTrack:
    """
    Stores data on the tracking of the person.
    """

    track_id: int
    keypoints: np.ndarray  # (frames_processed, N_KEYPOINTS, 3) of x, y, confidence
    boxes: np.ndarray  # (frames_processed, 4) of xyxy
    box_conf: np.ndarray  # (frames_processed,)
    fps: float
    person: Person

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


# ========================================================================= #
# TRACKERS
# ========================================================================= #


class PoseTracker:
    """
    Tracks a person's pose keypoints and bounding boxes using YOLO pose model.
    """

    def __init__(
        self, person: Person, model: str = "yolo26n-pose.pt", config: Config = Config()
    ):
        self.model = YOLO(model)
        self.config = config
        self.person = person

    def get_person_tracker(self, video_path: Path) -> dict[int, PersonTrack]:
        """
        Runs model on video and gets data of the person in the video.

        Args:
            video_path: Path of the video file.

        Returns:
            Dictionary containing the id and PersonTrack object.
        """

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
        """
        Converts data tracked from model into PersonTrack object.

        Args:
            track_id: ID that the model uses to track the person.
            frames: Dictionary containing the frame index and the actual OpenCV frame read.
            frames_proccessed: Number of frames processed by the model.
            fps: FPS of the video.

        Returns:
            PersonTrack object.
        """

        keypoints = np.full(
            (frames_processed, N_KEYPOINTS, 3), np.nan, dtype=np.float32
        )
        boxes = np.full((frames_processed, 4), np.nan, dtype=np.float32)
        box_conf = np.full(frames_processed, np.nan, dtype=np.float32)

        for frame_idx, (frame_keypoints, box, conf) in frames.items():
            keypoints[frame_idx] = frame_keypoints
            boxes[frame_idx] = box
            box_conf[frame_idx] = conf

        keypoints[keypoints[:, :, 2] < self.config.keypoint_conf] = np.nan

        return PersonTrack(track_id, keypoints, boxes, box_conf, fps, self.person)

    @staticmethod
    def get_top_n_ids(person_tracker: dict[int, PersonTrack], n: int = 2) -> list[int]:
        """
        Gets the ids of persons with the top n longest appearances in the video by the model.

        Args:
            person_tracker: Dictionary containing model id and PersonTrack.
            n: The number of persons to be returned.

        Returns:
            List of the top n most appeared persons.
        """
        return sorted(
            person_tracker,
            key=lambda track_id: int(person_tracker[track_id].detected.sum()),
            reverse=True,
        )[:n]


# ========================================================================= #
# ANNOTATORS
# ========================================================================= #


class VideoAnnotator:
    """
    Annotates a video using data tracked by the model.
    """

    def __init__(self, config: Config = Config()):
        self.person_tracks: list[tuple[PersonTrack, Detections]] = []
        self.config = config

    def add_tracker(self, tracker: PersonTrack, detections: Detections):
        """
        Adds a PersonTrack object for VideoAnnotater to include in the video annotations.

        Args:
            tracker: PersonTrack object
            detections: Detections object
        """

        self.person_tracks.append((tracker, detections.expanded()))

    def annotate_frame(
        self, person_track: PersonTrack, detections: Detections, frame, frame_idx: int
    ):
        """
        Annotates a single frame.

        Args:
            person_track: PersonTrack object
            detections: Detections object
            frame: OpenCV frame to be annotated on
            frame_idx: The index in which the frame appears in the video
        """

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

        # for row, strike in enumerate(detections.active_at(frame_idx)):
        #     cv2.putText(
        #         frame,
        #         strike.label,
        #         (x2, y1 - 20 + row * 35),
        #         cv2.FONT_HERSHEY_SIMPLEX,
        #         1,
        #         (0, 0, 255),
        #         2,
        #     )

        for row, strike in enumerate(detections.active_at(frame_idx)):
            if strike.strike_type == "straight" and strike.side == "right":
                cv2.putText(
                    frame,
                    strike.label,
                    (x2, y1 - 20 + row * 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    2,
                )

    def annotate_video(
        self,
        video_path: Path,
        new_file_path: str,
    ) -> None:
        """
        Annotates the entire video with: a box around persons tracked, ID label,
         and strike label when a strike occurs.

        Args:
            video_path: Path of the video file.
            new_path_file: The path of the new annotated video file.
        """

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

                for track, detections in self.person_tracks:
                    if frame_idx >= track.frames_processed:
                        continue

                    self.annotate_frame(
                        person_track=track,
                        detections=detections,
                        frame=frame,
                        frame_idx=frame_idx,
                    )

                writer.write(frame)
                frame_idx += 1

            writer.release()


# ========================================================================= #
# DETECTIONS
# ========================================================================= #


@dataclass(frozen=True)
class Strike:
    """
    Stores data for each type of strike i.e left straight, right hook, left roundhouse
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
        Counts the strike occurance for each strike type.

        Args:
            min_frames: Minumum number of frames a strike needs to be detected for to be counted for.

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


# ========================================================================= #
# FEATURES
# ========================================================================= #


def get_joint_speed(track: PersonTrack, name: str, config: Config) -> np.ndarray:
    """Calculate the speed of a joint in pixels per second.

    Missing keypoint detections handled by using the most recent valid
    position, as long as the number of skipped frames stays under the
    maximum hold distance. Frames where speed  can't be calculated
    accurately return NaN.

    Args:
        track: PersonTrack object
        name: Name of the joint/keypoint
        config: Config object

    Returns:
        Numpy array containing the joint speed for each frame.
        Frames where the speed can't be calculated have NaN values.
    """

    xy = track.positions(name)
    visible = ~np.isnan(xy[:, 0])
    frames = np.arange(len(xy))

    seen_at_or_before = np.maximum.accumulate(np.where(visible, frames, -1))
    previous = np.roll(seen_at_or_before, 1)
    previous[0] = -1

    gap = frames - previous
    measurable = visible & (previous >= 0) & (gap <= config.max_hold + 1)

    speed = np.full(len(xy), np.nan)
    distance = np.linalg.norm(xy[measurable] - xy[previous[measurable]], axis=1)
    speed[measurable] = distance * track.fps / gap[measurable]

    return speed


def get_joint_angle(track: PersonTrack, a: str, b: str, c: str) -> np.ndarray:
    """
    Gets the angle between three joints.

    Args:
        track: PersonTrack object.
        a: Position vector of the first keypoint/joint
        b: Position vector of the second keypoint/joint, the joint where the angle is actually at.
        c: Position vector of the third keypoint/joint

    Returns:
        Angle ABC in degrees.
    """
    return calculate_angles(
        track.positions(a),
        track.positions(b),
        track.positions(c),
    )


def get_relative_speed_threshold(speed: np.ndarray, config: Config) -> float:
    """
    Calculates the speed threshold using a top n percentile based on the entire video, set at config.

    Args:
        speed: Numpy array containing speeds for each frame.
        config: Config object

    Returns:
        The calculated speed threshold. Returns infinity if no valid speed
        values are available.
    """
    if not np.any(~np.isnan(speed)):
        return np.inf
    return float(np.nanpercentile(speed, config.velocity_percentile))


def get_pixel_to_meter_ratio(
    track: PersonTrack,
) -> np.ndarray:
    """
    Calculates pixel-to-meter conversion ratio for each frame.

    Conversion estimated using the user's wingspan to approximate shoulder
    width in real-world units.

    Args:
        track: PersonTrack object

    Returns:
        Numpy array containing the pixel to meter ratio for each frame.
    """
    left_shoulder = track.positions("left_shoulder")
    right_shoulder = track.positions("right_shoulder")

    shoulder_pixels = np.linalg.norm(
        right_shoulder - left_shoulder,
        axis=1,
    )

    wingspan_m = track.person.wingspan_m
    shoulder_width_m = wingspan_m * 0.25
    ratio = shoulder_width_m / shoulder_pixels

    return ratio


def get_punch_speed_threshold(track: PersonTrack):
    """
    Calculates punch speed threshold based on fixed real life value.

    Args:
        track: PersonTrack object

    Returns:
        Numpy array containing the threshold at each frame in pixels per second.
    """
    pixel_to_m_ratio = get_pixel_to_meter_ratio(track)
    thresholds = 5 / pixel_to_m_ratio
    return thresholds


# ========================================================================= #
# DETECTORS
# ========================================================================= #


def detect_straight(track: PersonTrack, side: Side, config: Config) -> np.ndarray:
    """
    Detects when a straight punch occurs.

    Args:
        track: PersonTrack object
        side: Side which the strike is, left or right
        config: Config object

    Returns:
        Numpy array which is a boolean mask where which detects if a punch occured for
        each frame.
    """

    speed = get_joint_speed(track, f"{side}_wrist", config)
    thresholds = get_relative_speed_threshold(speed, config)

    arm_lifted = (
        get_joint_angle(track, f"{side}_hip", f"{side}_shoulder", f"{side}_elbow")
        > config.arm_body_angle_threshold
    )

    arm_extended = (
        get_joint_angle(track, f"{side}_shoulder", f"{side}_elbow", f"{side}_wrist")
        > config.arm_extension_angle_threshold
    )
    return arm_extended & arm_lifted & (speed > thresholds)


DETECTORS = {
    "straight": detect_straight,
    # Add other strikes later
}


class MoveAnalyser:
    def __init__(self, track: PersonTrack, config: Config = Config()):
        self.track = track
        self.config = config
        self.thresholds = get_punch_speed_threshold(track)

    def get_detections(self) -> Detections:
        """
        Detects strikes for each strike type.

        Args:
            track: PersonTrack object
            config: Config object

        Returns:
            Detection object containing information on when a strike occurs.
        """
        strikes = tuple(
            Strike(technique, side)
            for technique in DETECTORS
            for side in ("left", "right")
        )
        mask = np.stack(
            [
                DETECTORS[strike.strike_type](self.track, strike.side, self.config)
                for strike in strikes
            ]
        ).astype(bool)
        return Detections(strikes, mask)


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
                max(0, frame - 1) : frame + 1, KEYPOINT_INDEX[name]
            ]
            lines.append(f"{name:16s} {np.array2string(values, precision=1)}")

        return "\n".join(lines)


# ========================================================================= #
# ENTRY POINT
# ========================================================================= #


def main(
    video_path: Path,
    person: Person,
    model: str = "yolo26n-pose.pt",
    fighters: int = 1,
    config: Config = Config(),
) -> None:
    tracker = PoseTracker(model=model, config=config, person=person)
    person_tracker = tracker.get_person_tracker(video_path)
    video_annotater = VideoAnnotator(config=config)

    if not person_tracker:
        print("No people were tracked.")
        return

    for track_id in tracker.get_top_n_ids(person_tracker, n=fighters):
        track = person_tracker[track_id]
        detections = MoveAnalyser(track, config=config).get_detections()
        video_annotater.add_tracker(track, detections)

        print(
            f"\nTrack {track_id}: seen in "
            f"{int(track.detected.sum())}/{track.frames_processed} frames"
        )

        for strike, count in detections.counts(config.min_punch_frames).items():
            starts = detections.start_frames(strike)
            print(f"  {strike.label}: {count}")
            print(f"  at frames: {[int(s) for s in starts]}")

        for side in ("left", "right"):
            print(
                f"  {side} wrist stats: "
                f"{VelocityInspector(get_joint_speed(track, f'{side}_wrist', config)).get_stats()}"
            )

    video_annotater.annotate_video(
        video_path=video_path, new_file_path="../outputs/annotate_test_0003.mp4"
    )


if __name__ == "__main__":
    FRAME_PATH = Path("../assets") / "frames" / "Van Vs Royval" / "frame_00400.jpg"
    VIDEO_PATH = (
        Path("../assets")
        / "clips"
        / "Joshua Van vs Brandon Royval ｜ FULL FIGHT ｜ UFC 328 [nwO2UPz7p28].webm"
    )

    royval = Person(
        name="Brandon Royval", height_m=1.75, weight=57, wingspan_m=1.73, stance="left"
    )

    main(video_path=VIDEO_PATH, person=royval)
