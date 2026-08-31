from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from .config import Config
from .constants import KEYPOINT_INDEX, N_KEYPOINTS, Stance
from .video import get_fps


def apply_kalman_filter(
        keypoints: np.ndarray,
        fps: float,
) -> np.ndarray:
    """
    Applies a Kalman filter to the xy position of every keypoint.

    Args:
        keypoints: Array of shape (frames, N_KEYPOINTS, 3)
                   containing x, y, confidence.
        fps: Video FPS.

    Returns:
        Kalman-filtered keypoints of the same shape.
    """

    dt = 1 / fps

    filtered = keypoints.copy()

    for keypoint_idx in range(N_KEYPOINTS):

        positions = keypoints[:, keypoint_idx, :2]

        # Find first valid observation
        valid = ~np.isnan(positions).any(axis=1)

        if not valid.any():
            continue

        first_idx = np.flatnonzero(valid)[0]
        x0, y0 = positions[first_idx]

        kf = cv2.KalmanFilter(4, 2)

        kf.statePost = np.array(
            [[x0], [y0], [0], [0]],
            dtype=np.float32,
        )

        kf.transitionMatrix = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1,  0],
            [0, 0, 0,  1],
        ], dtype=np.float32)

        kf.measurementMatrix = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
        ], dtype=np.float32)

        sigma_a = 100.0

        kf.processNoiseCov = sigma_a ** 2 * np.array([
            [dt ** 4 / 4, 0,          dt ** 3 / 2, 0],
            [0,            dt ** 4 / 4, 0,          dt ** 3 / 2],
            [dt ** 3 / 2, 0,          dt ** 2,     0],
            [0,            dt ** 3 / 2, 0,          dt ** 2],
        ], dtype=np.float32)

        kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 10

        kf.errorCovPost = np.diag([
            10,
            10,
            1000,
            1000,
        ]).astype(np.float32)

        for frame_idx in range(first_idx, len(keypoints)):

            position = positions[frame_idx]

            prediction = kf.predict()

            if np.isnan(position).any():
                continue

            measurement = position.reshape(2, 1).astype(np.float32)

            estimate = kf.correct(measurement)

            filtered[frame_idx, keypoint_idx, :2] = estimate[:2, 0]

    return filtered


@dataclass(frozen=True)
class Person:
    """
    Stores data of the person.
    """

    name: str
    weight: float
    height_m: float
    wingspan_m: float
    stance: Stance


@dataclass(frozen=True)
class PersonState:
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


class PoseTracker:
    """
    Tracks a person's pose keypoints and bounding boxes using YOLO pose model.
    """

    def __init__(
            self, person: Person, config: Config, model_name: str = "yolo26s-pose.pt"
    ):
        self.person_states: dict[int, PersonState] = {}
        self.model = YOLO(model_name)
        self.model_name = model_name
        self.config = config
        self.person = person

    def get_person_state(self, person: Person) -> PersonState:
        """
        PersonState object of Person inputted

        Args:
            person: The person who's person_track will be returned

        Returns:
            PersonState object of the Person inputted.
        """
        if self.person_states is None:
            raise ValueError(
                "No persons has been tracked yet. Please run get_person_tracker to track."
            )

        for person_state in self.person_states.values():
            if person_state.person == person:
                return person_state
        raise ValueError("Person not tracked.")

    def track(
            self,
            video_path: Path,
            track_progress: Callable[[int, int], None] | None = None,
    ) -> None:
        """
        Runs model on video and tracks the state(boxes, keypoints) of the person in the video.

        Args:
            video_path: Path of the video file.
            track_progress: Optional callback invoked after each frame with
                            the number of frames processed so far and the
                            total, to report tracking progress.
        """
        fps = get_fps(video_path)
        detections: dict[int, dict[int, tuple]] = defaultdict(dict)
        frames_processed = 0
        n_frames = self.get_n_frames(video_path)

        results = self.model.track(
            source=str(video_path),
            persist=True,
            tracker="bytetrack.yaml",
            stream=True,
            classes=[0],
        )

        for frame_idx, result in enumerate(results):
            frames_processed = frame_idx + 1

            if track_progress is not None:
                track_progress(frames_processed, n_frames)

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

        self.person_states = {
            track_id: self.densify(track_id, frames, frames_processed, fps)
            for track_id, frames in detections.items()
        }

    def densify(
            self,
            track_id: int,
            frames: dict[int, tuple],
            frames_processed: int,
            fps: float,
    ) -> PersonState:
        """
        Converts data tracked from model into PersonTrack object.

        Args:
            track_id: ID that the model uses to track the person.
            frames: Dictionary containing the frame index and the actual OpenCV frame read.
            frames_processed: Number of frames processed by the model.
            fps: FPS of the video.

        Returns:
            PersonState object.
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

        keypoints = apply_kalman_filter(
            keypoints=keypoints,
            fps=fps,
        )

        # Currently assigns the same Person object to every tracked fighter, change this later
        return PersonState(track_id, keypoints, boxes, box_conf, fps, self.person)

    def get_top_n_ids(self, n: int = 2) -> list[int]:
        """
        Gets the ids of persons with the top n longest appearances in the video by the model.

        Args:
            n: The number of persons to be returned.

        Returns:
            List of the top n most appeared persons.
        """
        return sorted(
            self.person_states,
            key=lambda track_id: int(self.person_states[track_id].detected.sum()),
            reverse=True,
        )[:n]

    def get_n_frames(self, video_path: Path) -> int:
        """Returns the number of frames in a video."""
        cap = cv2.VideoCapture(str(video_path))
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        return n_frames
