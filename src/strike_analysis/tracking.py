from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from ultralytics import YOLO

from .config import Config
from .constants import KEYPOINT_INDEX, N_KEYPOINTS, Side
from .video import get_fps

def smooth_keypoints(
    keypoints: np.ndarray,
    window: int,
) -> np.ndarray:
    """
    Smooth keypoints positions over time

    Args:
        keypoints: Keypoints to be smoothed
        window: Size of the window where keypoints will be smoothed

    Returns:
        Numpy array of the smoothed keypoints
    """
    if window <= 1:
        return keypoints

    if window % 2 == 0:
        raise ValueError("Smoothing window must be odd.")

    smoothed = keypoints.copy()
    kernel = np.ones(window, dtype=float)

    for keypoint_idx in range(keypoints.shape[1]):
        for axis in range(2):  # x and y only
            values = keypoints[:, keypoint_idx, axis]
            visible = np.isfinite(values)

            # Replace NaNs with zero
            values_filled = np.where(visible, values, 0.0)

            # Sum of valid values in each window
            num = np.convolve(
                values_filled,
                kernel,
                mode="same",
            )

            # Number of valid values in each window
            den = np.convolve(
                visible.astype(float),
                kernel,
                mode="same",
            )

            result = np.full_like(values, np.nan, dtype=float)

            valid = den > 0
            result[valid] = num[valid] / den[valid]

            # Dont fill originally missing observations.
            result[~visible] = np.nan

            smoothed[:, keypoint_idx, axis] = result

    return smoothed


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
        self,
        person: Person,
        model_name: str = "yolo26s-pose.pt",
        config: Config = Config(),
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

        for _, person_state in self.person_states.items():
            if person_state.person == person:
                return person_state
        raise ValueError("Person not tracked.")

    def track(self, video_path: Path) -> None:
        """
        Runs model on video and tracks the state(boxes, keypoints) of the person in the video.

        Args:
            video_path: Path of the video file.
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

        self.person_states = {
            track_id: self._densify(track_id, frames, frames_processed, fps)
            for track_id, frames in detections.items()
        }

    def _densify(
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

        keypoints = smooth_keypoints(
            keypoints,
            window=self.config.keypoint_smoothing_window,
        )

        boxes = np.full((frames_processed, 4), np.nan, dtype=np.float32)
        box_conf = np.full(frames_processed, np.nan, dtype=np.float32)

        for frame_idx, (frame_keypoints, box, conf) in frames.items():
            keypoints[frame_idx] = frame_keypoints
            boxes[frame_idx] = box
            box_conf[frame_idx] = conf

        keypoints[keypoints[:, :, 2] < self.config.keypoint_conf] = np.nan

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
