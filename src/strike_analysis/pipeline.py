from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from .annotator import VideoAnnotator
from .cache import TrackCache
from .config import Config, StrikeConfig
from .detections import Detections
from .strike_analyser import StrikeAnalyser
from .tracking import Person, PersonState, PoseTracker


@dataclass(frozen=True)
class AnalysisResult:
    """
    Everything the analysis knows about one video.

    Holds the tracked keypoints and the detections themselves, so callers
    can annotate a video or analyse footwork, alongside a flat list of the
    strikes thrown for reporting.
    """

    person_state: PersonState
    detections: Detections
    strikes: list[dict] = field(default_factory=list)


def analyse(
    video_path: Path,
    person: Person,
    cache_path: Path,
    model: str,
    config: Config = Config(),
    strike_config: StrikeConfig = StrikeConfig(),
    track_progress=None,
) -> AnalysisResult:
    """
    Tracks the fighter in a video and detects the strikes they throw.

    Tracked keypoints are cached and reused.

    Args:
        video_path: Path of the video file.
        person: The person being tracked.
        cache_path: Path the tracked keypoints are cached at.
        model: Name or path of the pose model.
        config: Config object.
        strike_config: StrikeConfig object.
        track_progress: Optional callback invoked after each frame with the
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

    detections = StrikeAnalyser(
        person_state,
        strike_config=strike_config,
    ).get_detections()

    return AnalysisResult(
        person_state=person_state,
        detections=detections,
        strikes=detections.to_records(person_state.fps, config.min_punch_frames),
    )


def render_annotated_video(
    result: AnalysisResult,
    video_path: Path,
    new_file_path: Path,
    config: Config = Config(),
) -> Path:
    """
    Creates video with the tracking and strikes drawn on it.

    Args:
        result: AnalysisResult of the video.
        video_path: Path of the video file.
        new_file_path: Path of the new annotated video file.
        config: Config object.

    Returns:
        Path the annotated video was written to.
    """
    annotator = VideoAnnotator(config=config)
    annotator.add_tracker(result.person_state, result.detections)
    annotator.annotate_video(video_path=video_path, new_file_path=new_file_path)

    return new_file_path
