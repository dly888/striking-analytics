from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .config import Config
from .detections import Detections
from .tracking import PersonState
from .video import get_fps, open_video


class VideoAnnotator:
    """
    Annotates a video using data tracked by the model.
    """

    def __init__(self, config: Config = Config()):
        self.person_detections: list[tuple[PersonState, Detections]] = []
        self.config = config

    def add_tracker(self, states: PersonState, detections: Detections):
        """
        Adds a PersonTrack object for VideoAnnotater to include in the video annotations.

        Args:
            states: PersonTrack object
            detections: Detections object
        """

        self.person_detections.append((states, detections.expanded()))

    @staticmethod
    def annotate_frame(
        person_state: PersonState, detections: Detections, frame, frame_idx: int
    ):
        """
        Annotates a single frame.

        Args:
            person_state: PersonTrack object
            detections: Detections object
            frame: OpenCV frame to be annotated on
            frame_idx: The index in which the frame appears in the video
        """

        box = person_state.boxes[frame_idx]

        if np.isnan(box).any():
            return

        x1, y1, x2, y2 = map(int, box)

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        cv2.putText(
            frame,
            f"ID {person_state.track_id}",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )

        for row, strike in enumerate(detections.active_at(frame_idx)):
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
        new_file_path: Path,
    ) -> None:
        """
        Annotates the entire video with: a box around persons tracked, ID label,
         and strike label when a strike occurs.

        Args:
            video_path: Path of the video file.
            new_file_path: The path of the new annotated video file.
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

                for track, detections in self.person_detections:
                    if frame_idx >= track.frames_processed:
                        continue

                    self.annotate_frame(
                        person_state=track,
                        detections=detections,
                        frame=frame,
                        frame_idx=frame_idx,
                    )

                writer.write(frame)
                frame_idx += 1

            writer.release()
