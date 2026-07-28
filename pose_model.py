from typing import Any
import cv2
import math
from pathlib import Path
import numpy as np
import numpy.typing as npt
from ultralytics import YOLO
from pathlib import Path


model = YOLO("yolo26n-pose.pt")
frame_path = Path("assets") / "frames" / "Van Vs Royval" / "frame_00400.jpg"

TEST_WRIST_TRACK = [
    (100, 200, 0.95),
    (105, 205, 0.90),
    (110, 210, 0.70),
    (115, 213, 0.80),
    (120, 211, 0.84),
    (125, 210, 0.84),
    (125, 210, 0.84),
    (125, 210, 0.84),
    (130, 210, 0.2),
    (130, 210, 0.2),
    (130, 210, 0.2),
    (130, 210, 0.2),
    (130, 210, 0.2),
    (130, 210, 0.2),
    (130, 210, 0.2),
    (130, 210, 0.2),
    (145, 210, 0.90),
    (345, 410, 0.90),
    (345, 410, 0.10),
    (345, 410, 0.10),
    (350, 411, 0.96)
]


def get_keypoints_on_single_frame(model: YOLO, frame_path: Path) -> Any:
    results = model(source=frame_path)
    keypoints = results[0].keypoints

    return keypoints

def get_person_keypoints(model: YOLO, frame_path: Path, person_id: int) -> dict[str, tuple[float, float, float]]:
    results = model(source=str(frame_path))
    person_keypoints = results[0].keypoints[person_id]

    KEYPOINT_NAMES = {
        0: "nose",
        1: "left_eye",
        2: "right_eye",
        3: "left_ear",
        4: "right_ear",
        5: "left_shoulder",
        6: "right_shoulder",
        7: "left_elbow",
        8: "right_elbow",
        9: "left_wrist",
        10: "right_wrist",
        11: "left_hip",
        12: "right_hip",
        13: "left_knee",
        14: "right_knee",
        15: "left_ankle",
        16: "right_ankle",
    }
    keypoint_tracker = {}

    for idx, ((x, y), conf) in enumerate(zip(person_keypoints.xy[0], person_keypoints.conf[0])):
        keypoint_tracker[KEYPOINT_NAMES[idx]] = (
            round(x.item(), 2),
            round(y.item(), 2),
            round(conf.item(), 2)
        )

    return keypoint_tracker


def wrist_velocity(wrist_track, fps, conf_threshold=0.5, max_hold=6) -> list[float]:
    velocities = []
    anchor_pos = None        # last OBSERVED (x, y)
    anchor_frame = None      # frame index it was observed at
    hold_count = 0

    for i, (x, y, conf) in enumerate(wrist_track):
        if conf >= conf_threshold:
            if anchor_pos is None:
                velocities.append(float("nan"))
                anchor_pos = (x, y)
                anchor_frame = i
            else:
                distance = math.dist(anchor_pos, (x, y))
                elapsed_frames = i - anchor_frame
                dt = elapsed_frames / fps
                velocities.append(distance / dt)

            hold_count = 0
        else:
            hold_count += 1
            if hold_count > max_hold:
                velocities.append(float('nan'))
                hold_count = 0
                anchor_frame = None
                anchor_pos = None
            else:
                velocities.append(float('nan'))

    return velocities

def annotate_single_frame(model: YOLO, frame_path: Path) -> None:
    frame = cv2.imread(str(frame_path))
    results = model(frame)
    annotated = results[0].plot()

    cv2.imshow("Annotated", annotated)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# print(get_keypoints_on_single_frame(model=model, frame_path=frame_path))
# print(get_person_keypoints(model=model, frame_path=frame_path, person_id=0))
# annotate_single_frame(model=model, frame_path=frame_path)
for i, v in enumerate(wrist_velocity(TEST_WRIST_TRACK, fps=29.97)):
    print(i, v)
