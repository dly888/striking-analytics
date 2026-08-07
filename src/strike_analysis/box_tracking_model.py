from pathlib import Path
from typing import Any

import cv2
from ultralytics import YOLO

model = YOLO("../../models/yolo26n.pt")
frame_path = Path("assest") / "frames" / "Van Vs Royval/frame_00400.jpg"


def detect_single_frame(model: YOLO, frame_path: Path) -> tuple[list[Any], list[Any]]:
    results = model(frame_path, classes=[0])

    xyxys = [box.xyxy for box in results[0].boxes]
    confs = [box.conf for box in results[0].boxes]

    return xyxys, confs


def track_fighters(clip_path: Path, model: YOLO, max_frame: int) -> dict:
    cap = cv2.VideoCapture(clip_path)
    id_tracker = {}
    current_frame_idx = 0

    while cap.isOpened():
        success, frame = cap.read()

        if not success:
            break

        result = model.track(
            source=frame, persist=True, classes=[0], tracker="bytetrack.yaml"
        )

        current_result = result[0]

        if current_result.boxes.id is None:
            current_frame_idx += 1
            continue

        for detection_idx, (track_id, box, conf) in enumerate(
            zip(
                current_result.boxes.id,
                current_result.boxes.xyxy,
                current_result.boxes.conf,
            ),
            start=1,
        ):
            track_id = int(track_id)

            if track_id not in id_tracker:
                id_tracker[track_id] = []

            id_tracker[track_id].append(
                [current_frame_idx, box.tolist(), round(float(conf), 2)]
            )

        current_frame_idx += 1

        if current_frame_idx >= max_frame:
            break

    cap.release()

    return id_tracker


def print_track_fighters(id_tracker: dict[int, list[list[Any]]]) -> None:
    print("\n" + "=" * 70)
    print("TRACKED FIGHTERS")
    print("=" * 70)

    if not id_tracker:
        print("No tracked fighters.")
        return

    for track_id in sorted(id_tracker):
        detections = id_tracker[track_id]

        print(f"\nTrack ID {track_id}")
        print("-" * 70)
        print(f"Frames tracked: {len(detections)}")

        for frame_idx, box, conf in detections:
            box = [round(x, 1) for x in box]

            print(f"  Frame {frame_idx:4d} | Conf: {conf:.2f} | Box: {box}")


def print_frame_data(results: list[Any]) -> None:
    for frame_idx, result in enumerate(results):
        print(f"\n{'=' * 50}")
        print(f"FRAME {frame_idx}")
        print(f"{'=' * 50}")

        if result.boxes.id is None:
            print("No detections")
            continue

        for detection_idx, (track_id, box, conf) in enumerate(
            zip(
                result.boxes.id,
                result.boxes.xyxy,
                result.boxes.conf,
            ),
            start=1,
        ):
            print(f"\nDetection {detection_idx}")
            print(f"  ID:         {int(track_id)}")
            print(f"  Box:        {[round(x, 1) for x in box.tolist()]}")
            print(f"  Confidence: {float(conf):.2f}")


def annotate_video(video_path: Path, file_name: str, model: YOLO) -> None:
    cap = cv2.VideoCapture(clip_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    writer = cv2.VideoWriter(
        file_name, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )

    while cap.isOpened():
        success, frame = cap.read()

        if not success:
            break

        results = model.track(
            source=frame, persist=True, classes=[0], tracker="bytetrack.yaml"
        )

        annotated_frame = results[0].plot()

        writer.write(annotated_frame)

    cap.release()
    writer.release()


clip_path = (
    Path("../../assets")
    / "clips"
    / "Joshua Van vs Brandon Royval ｜ FULL FIGHT ｜ UFC 328 [nwO2UPz7p28].webm"
)

annotate_video(video_path=clip_path, file_name="../../outputs/test_0001.mp4", model=model)
