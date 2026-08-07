from __future__ import annotations

from pathlib import Path

from .annotator import VideoAnnotator
from .config import Config, StrikeConfig
from .detectors import MoveAnalyser
from .features import get_joint_speed
from .inspector import VelocityInspector
from .tracking import Person, PoseTracker


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
    strike_config = StrikeConfig()

    if not person_tracker:
        print("No people were tracked.")
        return

    for track_id in tracker.get_top_n_ids(person_tracker, n=fighters):
        track = person_tracker[track_id]
        detections = MoveAnalyser(track, config=strike_config).get_detections()
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
                f"{VelocityInspector(get_joint_speed(track, f'{side}_wrist', strike_config)).get_stats()}"
            )

    video_annotater.annotate_video(
        video_path=video_path, new_file_path="../outputs/annotate_test_0003.mp4"
    )

if __name__ == "__main__":
    PROJECT_ROOT = Path(__file__).resolve().parents[2]

    VIDEO_PATH = (
        PROJECT_ROOT
        / "assets"
        / "clips"
        / "Joshua Van vs Brandon Royval ｜ FULL FIGHT ｜ UFC 328 [nwO2UPz7p28].webm"
    )

    royval = Person(
        name="Brandon Royval", height_m=1.75, weight=57, wingspan_m=1.73, stance="left"
    )

    main(
        video_path=VIDEO_PATH,
        person=royval,
        model=str(PROJECT_ROOT / "models" / "yolo26n-pose.pt"),
    )