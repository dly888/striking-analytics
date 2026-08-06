from pathlib import Path

import cv2
import numpy as np

clip_path = (
    Path("../assets") / "clips" / "Joshua Van vs Brandon Royval ｜"
    " FULL FIGHT ｜ UFC 328 [nwO2UPz7p28].webm"
)

cap = cv2.VideoCapture(clip_path)


def get_clip_metadata(cap: cv2.VideoCapture) -> tuple[int | float]:
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    return fps, width, height, frame_count


def print_clip_meta_data(cap: cv2.VideoCapture) -> None:
    fps, width, height, frame_count = get_clip_metadata(cap)

    print(f"fps={fps}\nwidthxheight={width}x{height}\nframe_count={frame_count}")


def save_frame(cap: cv2.VideoCapture, frame_number: int, fight_title: str) -> None:
    _, _, _, frame_count = get_clip_metadata(cap)

    if frame_number < 0 or frame_number >= frame_count:
        raise ValueError("Invalid frame number.")

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, frame = cap.read()

    if not ret:
        raise ValueError(f"Could not read frame {frame_number}.")

    base_path = Path("../assets") / "frames" / fight_title
    base_path.mkdir(parents=True, exist_ok=True)
    file_path = base_path / f"frame_{frame_number:05d}.jpg"

    success = cv2.imwrite(str(file_path), frame)

    if not success:
        raise OSError(f"Failed to write frame to {file_path}")


def get_frame_metadata(frame_path: str) -> tuple[tuple[int], np.dtype]:
    img = cv2.imread(frame_path)

    return img.shape, img.dtype


def print_frame_metadata(frame_path: str) -> None:
    shape, dtype = get_frame_metadata(frame_path)
    print(f"Shape={shape}\ndtype={dtype}")


def count_frames(clip_path: Path) -> int:
    cap = cv2.VideoCapture(clip_path)
    count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        count += 1

    return count


def print_actual_frame_count(clip_path: Path) -> None:
    count = count_frames(clip_path)
    print(f"Actual frame count={count}")


print_clip_meta_data(cap=cap)
print_frame_metadata("../assets/frames/Van Vs Royval/frame_00400.jpg")
print_actual_frame_count(clip_path=clip_path)
# save_frame(cap=cap, frame_number=400, fight_title="Van Vs Royval")
