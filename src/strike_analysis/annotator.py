from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .config import Config
from .constants import SKELETON_EDGES
from .defense_detections import GuardDetections
from .strike_detections import StrikeDetections
from .tracking import PersonState
from .video import get_fps, open_video


class VideoAnnotator:
    """
    Annotates a video using data tracked by the model.
    """

    def __init__(self, config: Config):
        self.person_detections: list[tuple[PersonState, StrikeDetections, np.ndarray, GuardDetections]] = []
        self.config = config

    def add_tracker(self, states: PersonState, strike_detections: StrikeDetections, guard_detections: GuardDetections):
        """
        Adds a PersonTrack object for VideoAnnotater to include in the video annotations.

        Args:
            states: PersonTrack object
            strike_detections: StrikeDetections object
            guard_detections: GuardDetections object
        """

        self.person_detections.append(
            (
                states,
                strike_detections.expanded(),
                strike_detections.combo_frame_mask(states.fps),
                guard_detections.expanded()
            )
        )

    @staticmethod
    def annotate_frame(
        person_state: PersonState,
        strike_detections: StrikeDetections,
        guard_detections: GuardDetections,
        frame,
        frame_idx: int,
        combo_frame_mask: np.ndarray | None = None,
    ):
        """
        Annotates a single frame.

        Draws the ID box, strike detection, current frame index, combo
        label, and skeleton.

        Args:
            person_state: PersonTrack object
            strike_detections: StrikeDetections object
            guard_detections: GuardDetections object
            frame: OpenCV frame to be annotated on
            frame_idx: The index in which the frame appears in the video
            combo_frame_mask: Boolean array with one entry per frame,
                True on the frames a combo is thrown on.
        """

        box = person_state.boxes[frame_idx]

        if np.isnan(box).any():
            return

        x1, y1, x2, y2 = map(int, box)

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Output person id
        cv2.putText(
            frame,
            f"ID {person_state.track_id}",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )

        # Output the current frame
        cv2.putText(
            frame,
            f"Frame: {frame_idx}",
            (50, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )

        # Output name of strike detected
        for row, strike in enumerate(strike_detections.active_at(frame_idx)):
            cv2.putText(
                frame,
                strike.label,
                (x2, y1 - 20 + row * 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2,
            )

        # Output combo label
        if combo_frame_mask is not None and combo_frame_mask[frame_idx]:
            cv2.putText(
                frame,
                "COMBO",
                (x1, y2 + 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                (0, 255, 255),
                3,
            )

        if guard_detections is not None and guard_detections.mask[frame_idx]:
            cv2.putText(
                frame,
                "Guard dropped",
                (50, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
            )

        VideoAnnotator.draw_skeleton(person_state, frame, frame_idx)

    @staticmethod
    def draw_skeleton(person_state: PersonState, frame, frame_idx: int):
        """
        Draws the keypoint skeleton on a single frame.

        Args:
            person_state: PersonTrack object
            frame: OpenCV frame to be annotated on
            frame_idx: The index in which the frame appears in the video
        """

        for a, b in SKELETON_EDGES:
            point_a = person_state.positions(a)[frame_idx]
            point_b = person_state.positions(b)[frame_idx]

            if np.isnan(point_a).any() or np.isnan(point_b).any():
                continue

            cv2.line(
                frame,
                tuple(point_a.astype(int)),
                tuple(point_b.astype(int)),
                (0, 200, 255),
                2,
            )

        for name in ("left_wrist", "right_wrist", "left_ankle", "right_ankle"):
            point = person_state.positions(name)[frame_idx]

            if np.isnan(point).any():
                continue

            cv2.circle(frame, tuple(point.astype(int)), 4, (0, 200, 255), -1)

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
                # H.264, browsers can't play the mp4v codec
                cv2.VideoWriter_fourcc(*"avc1"),
                get_fps(video_path),
                (width, height),
            )

            frame_idx = 0

            while True:
                success, frame = cap.read()
                if not success:
                    break

                for track, strike_detections, combo_frame_mask, guard_detections in self.person_detections:
                    if frame_idx >= track.frames_processed:
                        continue

                    self.annotate_frame(
                        person_state=track,
                        strike_detections=strike_detections,
                        guard_detections=guard_detections,
                        frame=frame,
                        frame_idx=frame_idx,
                        combo_frame_mask=combo_frame_mask,
                    )

                writer.write(frame)
                frame_idx += 1

            writer.release()
