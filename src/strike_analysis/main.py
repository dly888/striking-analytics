from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

from .config import Config, StrikeConfig
from .features import get_joint_speed
from .footwork_analysis import FootworkAnalyser
from .inspector import VelocityInspector
from .pipeline import analyse_video, render_annotated_video
from .tracking import Person


def main(
    video_path: Path,
    person: Person,
    root: Path,
    cache_path: Path,
    floor_edge1: list[tuple[float, float]],
    floor_edge2: list[tuple[float, float]],
    config: Config,
    strike_config: StrikeConfig,
    model: str = "yolo26s-pose.pt",
    annotate: bool = False,
    debug: bool = False,
) -> None:
    """
    Analyses a video and reports the strikes and footwork it found.

    Args:
        video_path: Path of the video file.
        person: The person being tracked.
        root: Root of the project, annotated videos are written under it.
        cache_path: Path the tracked keypoints are cached at.
        floor_edge1: Two (x, y) points along one floor edge, near end first.
        floor_edge2: Two (x, y) points along the opposite edge, near end first.
        model: Name or path of the pose model.
        config: Config object.
        strike_config: StrikeConfig object.
        annotate: Whether to write an annotated copy of the video, which
                  reads the whole video a second time.
        debug: Whether to print wrist speed stats as well as the results.
    """
    result = analyse_video(
        video_path=video_path,
        person=person,
        cache_path=cache_path,
        model=model,
        config=config,
        strike_config=strike_config,
    )

    person_state = result.person_state

    print(
        f"\nTrack {person_state.track_id}: seen in "
        f"{int(person_state.detected.sum())}/{person_state.frames_processed} frames"
    )

    for strike, count in result.detections.counts(config.min_punch_frames).items():
        print(f"  {strike.label}: {count}")

    print(f"\n{len(result.strike_records)} strikes thrown:")

    for record in result.strike_records:
        print(f"  {record['time_s']:6.2f}s  {record['side']} {record['type']}")

    if debug:
        VelocityInspector().print_wrist_speeds(
            left_speed=get_joint_speed(person_state, "left_wrist", strike_config)
        )

    footwork_analyser = FootworkAnalyser(person_state)
    footwork_analyser.select_floor(edge1=floor_edge1, edge2=floor_edge2)
    footwork_analyser.get_plot_figure()
    plt.show()

    if annotate:
        annotated_path = root / "outputs" / f"{video_path.stem}_annotated.mp4"

        render_annotated_video(
            result=result,
            video_path=video_path,
            output_path=annotated_path,
            config=config,
        )
        print(f"\nAnnotated video written to {annotated_path}")


if __name__ == "__main__":
    PROJECT_ROOT = Path(__file__).resolve().parents[2]

    VIDEO_PATH = (
        PROJECT_ROOT
        / "assets"
        / "clips"
        / "Full MMA Shadow Boxing With Sabaki by Giga Chikadze [IEu0SAWLiXw].mp4"
    )

    CACHE_PATH = PROJECT_ROOT / "outputs" / "giga.npz"

    fighter = Person(
        name="Giga Chikadze",
        height_m=1.83,
        weight=66,
        wingspan_m=1.88,
        stance="orthodox",
    )

    main(
        video_path=VIDEO_PATH,
        person=fighter,
        root=PROJECT_ROOT,
        cache_path=CACHE_PATH,
        # The left and right edges of the mat, each traced near end first.
        # Estimated from the mat seams, adjust for other footage
        floor_edge1=[(408, 1070), (600, 640)],
        floor_edge2=[(1612, 1070), (1400, 640)],
        model=str(PROJECT_ROOT / "models" / "yolo26s-pose.pt"),
        annotate=True,
    )
