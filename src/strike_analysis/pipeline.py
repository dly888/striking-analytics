from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from .annotator import VideoAnnotator
from .cache import TrackCache
from .config import Config, StrikeConfig
from .defense_analysis import DefenseAnalyser
from .defense_detections import GuardDetections
from .strike_detections import StrikeDetections
from .strike_analysis import StrikeAnalyser
from .tracking import Person, PersonState, PoseTracker


@dataclass(frozen=True)
class AnalysisResult:
    """
    Everything the analysis knows about one video.

    Holds the tracked keypoints and the detections themselves, so you
    can annotate a video or analyse footwork, with a flat list of the
    strikes thrown for reporting.
    """

    person_state: PersonState
    strike_detections: StrikeDetections
    guard_detections: GuardDetections
    strike_records: list[dict] = field(default_factory=list)


def analyse_video(
    video_path: Path,
    person: Person,
    cache_path: Path,
    model: str,
    config: Config,
    strike_config: StrikeConfig,
    track_progress: Callable[[int, int], None] | None = None,
) -> AnalysisResult:
    """
    Tracks the fighter in a video and detects the strikes they throw.

    Tracked keypoints are cached and reused. Runs the model to track the fighter
    then analyses the striking.

    Args:
        video_path: Path of the video file.
        person: The person being tracked.
        cache_path: Path the tracked keypoints are cached at.
        model: Name or path of the pose model.
        config: Config object.
        strike_config: StrikeConfig object.
        track_progress: Optional callback called after each frame with the
                        number of frames processed so far and the total.

    Returns:
        AnalysisResult of the video.
    """
    if cache_path.exists():
        tracker = TrackCache.load_pose_tracker(path=cache_path)
    else:
        tracker = PoseTracker(person=person, model_name=model, config=config)
        tracker.track(video_path=video_path, track_progress=track_progress)
        TrackCache.save_pose_tracker(pose_tracker=tracker, new_path=cache_path)

    track_ids = tracker.get_top_n_ids(n=1)

    if not track_ids:
        raise ValueError(f"No person was tracked in {video_path.name}.")

    person_state = tracker.person_states[track_ids[0]]

    strike_detections = StrikeAnalyser(
        person_state,
        strike_config=strike_config,
    ).get_strike_detections()

    guard_detections = DefenseAnalyser(
        person_state,
    ).get_guard_dropped_detections()

    return AnalysisResult(
        person_state=person_state,
        strike_detections=strike_detections,
        guard_detections=guard_detections,
        strike_records=strike_detections.to_records(person_state.fps, config.min_punch_frames),
    )


def render_annotated_video(
    result: AnalysisResult,
    video_path: Path,
    output_path: Path,
    config: Config,
) -> None:
    """
    Creates video with the tracking and strikes drawn on it.

    Args:
        result: AnalysisResult of the video.
        video_path: Path of the video file.
        output_path: Path of the new annotated video file.
        config: Config object.
    """
    annotator = VideoAnnotator(config=config)
    annotator.add_tracker(result.person_state, result.strike_detections, guard_detections=result.guard_detections)
    annotator.annotate_video(video_path=video_path, new_file_path=output_path)
