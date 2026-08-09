from __future__ import annotations

from pathlib import Path

from strike_analysis import TrackCache

from .annotator import VideoAnnotator
from .config import Config, StrikeConfig
from .detectors import MoveAnalyser
from .features import get_joint_speed
from .inspector import VelocityInspector
from .tracking import Person, PoseTracker


def main(
    video_path: Path,
    person: Person,
    root: Path,
    model: str = "yolo26n-pose.pt",
    config: Config = Config(),
) -> None:
    track_cache = TrackCache()
    output_path = root / "outputs" / "pose_tracker.npz"

    if output_path.exists():
        tracker = track_cache.load_pose_tracker(path=output_path)
    else:
        tracker = PoseTracker(person=person, model_name=model, config=Config())
        tracker.track(video_path=video_path)
        track_cache.save_pose_tracker(pose_tracker=tracker, new_path=output_path)

    person_state = tracker.get_person_state(person=person)

    video_annotater = VideoAnnotator(config=config)
    strike_config = StrikeConfig()

    if person_state is None:
        print("Fighter not tracked.")
        return

    track_id = tracker.get_top_n_ids(n=1)[0]
    person_state = tracker.person_states[track_id]

    detections = MoveAnalyser(
        person_state,
        strike_config=strike_config,
    ).get_detections()

    video_annotater.add_tracker(
        person_state,
        detections,
    )

    print(
        f"\nTrack {person_state.track_id}: seen in "
        f"{int(person_state.detected.sum())}/{person_state.frames_processed} frames"
    )

    for strike, count in detections.counts(config.min_punch_frames).items():
        starts = detections.start_frames(strike)
        print(f"  {strike.label}: {count}")
        print(f"  at frames: {[int(s) for s in starts]}")

    for side in ("left", "right"):
        print(
            f"  {side} wrist stats: "
            f"{
                VelocityInspector(
                    get_joint_speed(
                        person_state,
                        f'{side}_wrist',
                        strike_config,
                    )
                ).get_stats()
            }"
        )

    video_annotater.annotate_video(
        video_path=video_path,
        new_file_path=root / "outputs" / "annotate_test_0003.mp4",
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
        name="Brandon Royval",
        height_m=1.75,
        weight=57,
        wingspan_m=1.73,
        stance="left",
    )

    main(
        video_path=VIDEO_PATH,
        person=royval,
        root=PROJECT_ROOT,
        model=str(PROJECT_ROOT / "models" / "yolo26n-pose.pt"),
    )
