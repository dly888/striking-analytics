from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from ultralytics import YOLO

from .config import Config
from .constants import KEYPOINT_INDEX, N_KEYPOINTS, Side
from .video import get_fps


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
            frames_processed: Number of frames processed by the model.
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