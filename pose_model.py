import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from numpy import dtype, ndarray
from ultralytics import YOLO

model = YOLO("yolo26n-pose.pt")
frame_path = Path("assets") / "frames" / "Van Vs Royval" / "frame_00400.jpg"
video_path = (
    Path("assets")
    / "clips"
    / "Joshua Van vs Brandon Royval ｜ FULL FIGHT ｜ UFC 328 [nwO2UPz7p28].webm"
)

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
    (350, 411, 0.96),
]


def calculate_angles(
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
) -> np.ndarray:

    ba = a - b
    bc = c - b

    dot = np.sum(ba * bc, axis=1)

    magnitude_ba = np.linalg.norm(ba, axis=1)
    magnitude_bc = np.linalg.norm(bc, axis=1)

    with np.errstate(divide="ignore", invalid="ignore"):
        cosine_angle = dot / (magnitude_ba * magnitude_bc)

    cosine_angle = np.clip(cosine_angle, -1, 1)

    return np.degrees(np.arccos(cosine_angle))


def get_fps(video_path: Path) -> float:
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    return fps


def get_id_tracker(model: YOLO, video_path: Path) -> dict:
    id_tracker = {}

    results = model.track(
        source=video_path,
        persist=True,
        stream=True,
        tracker="bytetrack.yaml",
        classes=[0],
    )

    for frame_idx, result in enumerate(results):
        for detection_idx, (track_id, box, conf) in enumerate(
            zip(
                result.boxes.id,
                result.boxes.xyxy,
                result.boxes.conf,
            ),
            start=1,
        ):
            track_id = int(track_id)

            if track_id not in id_tracker:
                id_tracker[track_id] = []

            id_tracker[track_id].append(
                [frame_idx, box.tolist(), round(float(conf), 2)]
            )

    return id_tracker


def get_top_n_ids(id_tracker: dict, n=2):
    top_n_ids = sorted(id_tracker.items(), key=lambda item: len(item[1]), reverse=True)[
        :n
    ]

    return top_n_ids


def get_keypoints_on_single_frame(model: YOLO, frame_path: Path) -> Any:
    results = model(source=frame_path)
    keypoints = results[0].keypoints

    return keypoints


def get_frame_person_keypoints(
    model: YOLO, frame_path: Path, person_id: int
) -> dict[str, tuple[float, float, float] | None]:
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

    for idx, ((x, y), conf) in enumerate(
        zip(person_keypoints.xy[0], person_keypoints.conf[0])
    ):
        keypoint_tracker[KEYPOINT_NAMES[idx]] = (
            round(x.item(), 2),
            round(y.item(), 2),
            round(conf.item(), 2),
        )

    return keypoint_tracker


def get_video_person_keypoints(
    model: YOLO, video_path: Path, person_id: int, conf_threshold=0.5
) -> tuple[dict[Any, Any], int]:
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
    frames_processed = 0

    results = model.track(
        source=video_path,
        persist=True,
        stream=True,
        tracker="bytetrack.yaml",
        classes=[0],
    )

    for frame_idx, result in enumerate(results):
        if result.boxes.id is None:
            frames_processed += 1
            continue

        ids = result.boxes.id.int().tolist()

        if person_id not in ids:
            frames_processed += 1
            continue

        i = ids.index(person_id)
        xy = result.keypoints.xy[i]
        conf = result.keypoints.conf[i]

        current_keypoints = {}

        for k, name in KEYPOINT_NAMES.items():
            c = round(conf[k].item(), 2)
            current_keypoints[name] = (
                (None, None, None)
                if c < conf_threshold
                else (
                    round(xy[k, 0].item(), 2),
                    round(xy[k, 1].item(), 2),
                    c,
                )
            )
        frames_processed += 1
        keypoint_tracker[frame_idx] = current_keypoints

    return keypoint_tracker, frames_processed


def get_wrist_tracker(
    keypoint_tracker: dict[int, dict[str, tuple[float, float, float] | None]],
    frames_processed: int,
    hand="left",
) -> list[float | None]:

    wrist_tracker = [(None, None, None)] * frames_processed
    hand = hand.lower()

    if hand != "left" and hand != "right":
        raise ValueError("Invalid input for hand.")

    for frame_idx, keypoints in keypoint_tracker.items():
        if hand == "left":
            wrist_tracker[frame_idx] = keypoints["left_wrist"]
        else:
            wrist_tracker[frame_idx] = keypoints["right_wrist"]

    return wrist_tracker


def get_wrist_velocities(
    wrist_tracker: list[tuple[float | None, float | None, float | None]],
    fps: float,
    conf_threshold=0.5,
    max_hold=6,
) -> ndarray:

    "Check if vectorisation is possible here"

    velocities = []
    prev_pos = None
    hold_count = 0

    dt = 1 / fps

    for x, y, conf in wrist_tracker:
        if conf is not None and conf >= conf_threshold:
            if prev_pos is None:
                velocities.append(float("nan"))
            else:
                distance = math.dist(prev_pos, (x, y))
                velocities.append(distance / (dt * (hold_count + 1)))

            prev_pos = (x, y)
            hold_count = 0
        else:
            hold_count += 1
            velocities.append(float("nan"))

            if hold_count > max_hold:
                prev_pos = None
                hold_count = 0

    return np.array(velocities)


def get_keypoint_trackers(
    keypoint_tracker: dict[int, dict[str, tuple[float, float, float] | None]],
    frames_processed: int,
    side="left",
):

    shoulder_tracker = [(None, None, None)] * frames_processed
    wrist_tracker = [(None, None, None)] * frames_processed
    elbow_tracker = [(None, None, None)] * frames_processed

    side = side.lower()

    if side != "left" and side != "right":
        raise ValueError("Invalid input for side.")

    for frame_idx, keypoints in keypoint_tracker.items():
        if side == "left":
            wrist_tracker[frame_idx] = keypoints["left_wrist"]
            shoulder_tracker[frame_idx] = keypoints["left_shoulder"]
            elbow_tracker[frame_idx] = keypoints["left_elbow"]
        else:
            wrist_tracker[frame_idx] = keypoints["right_wrist"]
            shoulder_tracker[frame_idx] = keypoints["right_shoulder"]
            elbow_tracker[frame_idx] = keypoints["right_elbow"]

    return shoulder_tracker, elbow_tracker, wrist_tracker

def get_velocity_percentiles(velocities: np.ndarray) -> dict[Any, Any]:
    percentiles = [90, 95, 99]

    values = np.nanpercentile(
        velocities,
        percentiles
    )

    return dict(zip(percentiles, values))

def get_is_arm_extended(
    wrist_tracker: list[tuple[float | None, float | None, float | None]],
    shoulder_tracker: list[tuple[float | None, float | None, float | None]],
    elbow_tracker: list[tuple[float | None, float | None, float | None]],
    threshold=150,
):

    wrist = np.array(
        [
            (x, y) if x is not None and y is not None else (np.nan, np.nan)
            for x, y, _ in wrist_tracker
        ]
    )

    shoulder = np.array(
        [
            (x, y) if x is not None and y is not None else (np.nan, np.nan)
            for x, y, _ in shoulder_tracker
        ]
    )

    elbow = np.array(
        [
            (x, y) if x is not None and y is not None else (np.nan, np.nan)
            for x, y, _ in elbow_tracker
        ]
    )

    angles = calculate_angles(
        shoulder,
        elbow,
        wrist,
    )

    return ((angles > threshold) & ~np.isnan(angles)).astype(int).tolist()


def get_punch_detector(wrist_velocities: np.ndarray, is_arm_extended: np.ndarray):
    percentile_values = get_velocity_percentiles(wrist_velocities)
    threshold = percentile_values[90]
    punch_detector = is_arm_extended & (wrist_velocities > threshold)  # Use 90th percentile

    return punch_detector

def get_punch_count(punch_detector: np.ndarray) -> int:
    padded = np.pad(punch_detector, (1, 1), constant_values=0)
    starts = (padded[:-1] == 0) & (padded[1:] == 1)

    return int(np.sum(starts))

def annotate_single_frame(model: YOLO, frame_path: Path) -> None:
    frame = cv2.imread(str(frame_path))
    results = model(frame)
    annotated = results[0].plot()

    cv2.imshow("Annotated", annotated)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def find_velocity_outliers(velocities: np.ndarray, threshold=3000) -> np.ndarray:
    outliers = np.flatnonzero(velocities > threshold)
    return outliers


def longest_nan_run(arr: np.ndarray) -> int:
    nan_mask = np.isnan(arr)

    max_run = 0
    current_run = 0

    for is_nan in nan_mask:
        if is_nan:
            current_run += 1
            max_run = max(max_run, current_run)
        else:
            current_run = 0

    return max_run


def get_maximum_velocity(velocities: np.ndarray) -> tuple[Any, Any]:
    idx = np.nanargmax(velocities)
    return idx, velocities[idx]


def print_velocity_window(velocities: np.ndarray, frame: int, radius: int) -> None:
    start = max(0, frame - radius)
    end = min(len(velocities), frame + radius + 1)

    print(f"\nFrames {start}–{end - 1}")

    for i in range(start, end):
        marker = "<--" if i == frame else ""
        print(f"{i:5d}: {velocities[i]:8.1f} {marker}")


def inspect_frame_pair(
    keypoint_tracker: dict,
    frame: int,
):
    print(f"\nFrame {frame - 1}")

    prev = keypoint_tracker[frame - 1]
    curr = keypoint_tracker[frame]

    for name in (
        "right_wrist",
        "right_shoulder",
        "right_hip",
    ):
        print(
            name,
            "prev =",
            prev[name],
            "curr =",
            curr[name],
        )


def get_stats(
    array: np.ndarray, start_frame: int | None = None, end_frame: int | None = None
) -> dict[str, float]:
    a = array[start_frame:end_frame]

    return {
        "start_frame": 0 if start_frame is None else start_frame,
        "end_frame": len(array) if end_frame is None else end_frame,
        "num_frames": len(a),
        "nan_rate": np.isnan(a).mean(),
        "median": np.nanmedian(a),
        "90th": np.nanpercentile(a, 90),
        "max": np.nanmax(a),
        "longest_nan_run": longest_nan_run(a),
    }


# print(get_keypoints_on_single_frame(model=model, frame_path=frame_path))
# print(get_person_keypoints(model=model, frame_path=frame_path, person_id=0))
# annotate_single_frame(model=model, frame_path=frame_path)

# for i, v in enumerate(wrist_velocity(TEST_WRIST_TRACK, fps=29.97)):
#     print(i, v)

id_tracker = get_id_tracker(model=model, video_path=video_path)
top_2_ids = get_top_n_ids(id_tracker=id_tracker, n=2)

keypoint_tracker, frames_processed = get_video_person_keypoints(
    model=model, video_path=video_path, person_id=top_2_ids[0][0]
)
#
# print(keypoint_tracker)
# print(len(keypoint_tracker))
#
fps = get_fps(video_path=video_path)
#
# wrist_tracker = get_wrist_tracker(
#     keypoint_tracker=keypoint_tracker, frames_processed=frames_processed, hand="right"
# )
# wrist_velocities = np.array(get_wrist_velocities(wrist_tracker=wrist_tracker, fps=fps))
# print(wrist_velocities)
# print("Max velocity: ", np.nanmax(wrist_velocities))
# print("Min velocity: ", np.nanmin(wrist_velocities))
# print("Median velocity: ", np.nanmedian(wrist_velocities))
# print("Wrist_velocity length:", len(wrist_velocities))
# print("Wrist tracker length:", len(wrist_tracker))
# print("First person: ", get_stats(wrist_velocities, end_frame=340))
# print("Second person: ", get_stats(wrist_velocities, start_frame=341))
# print("Frames processed: ", frames_processed)
#
# outliers = find_velocity_outliers(velocities=wrist_velocities, threshold=3000)
# print("Outliers: ", outliers)
#
# frame_idx, max_velocity = get_maximum_velocity(velocities=wrist_velocities)
# print_velocity_window(velocities=wrist_velocities, frame=frame_idx, radius=10)
#
# inspect_frame_pair(keypoint_tracker=keypoint_tracker, frame=frame_idx)
# inspect_frame_pair(keypoint_tracker=keypoint_tracker, frame=frame_idx - 1)

shoulder_tracker, elbow_tracker, wrist_tracker = get_keypoint_trackers(
    keypoint_tracker=keypoint_tracker,
    frames_processed=frames_processed,
    side="left"
)

is_arm_extended = get_is_arm_extended(
    shoulder_tracker=shoulder_tracker,
    elbow_tracker=elbow_tracker,
    wrist_tracker=wrist_tracker
)

wrist_velocities = get_wrist_velocities(wrist_tracker=wrist_tracker, fps=fps)

punch_detector = get_punch_detector(wrist_velocities=wrist_velocities, is_arm_extended=is_arm_extended)
punch_count = get_punch_count(punch_detector=punch_detector)

print("ID tracked: ", top_2_ids[0][0])
print("Punch detector: ", punch_detector)
print("Punch count: ", punch_count)