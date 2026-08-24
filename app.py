import tempfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from streamlit_image_coordinates import streamlit_image_coordinates

import strike_analysis as sa

CACHE_DIR = Path("cache")
MODEL = "yolo26s-pose.pt"
DISPLAY_WIDTH = 800
PACING_BIN_S = 5.0

CONFIG = sa.Config()
STRIKE_CONFIG = sa.StrikeConfig()


def build_strike_table(stats: sa.StrikingStats) -> pd.DataFrame:
    """
    Reshape the strike dictionaries into a table.

    The stats are keyed by side and strike type, which reads poorly
    as a list, so they are pivoted into one row per strike.

    Args:
        stats: StrikingStats object for the analysed video.

    Returns:
        DataFrame with one row per side and strike type.
    """
    rows = []

    for side in sa.SIDES:
        for strike_type in sa.STRIKE_TYPES:
            key = f"{side}_{strike_type}"

            rows.append(
                {
                    "Strike": f"{side.title()} {strike_type}",
                    "Count": int(stats.strike_counts[key]),
                    "Per sec": round(stats.strike_rates[key], 1),
                    "Max speed (m/s)": round(stats.max_speeds_mps[key], 1),
                }
            )

    return pd.DataFrame(rows)


def build_guard_drop_table(
    guard_detections: sa.GuardDetections,
    fps: float,
) -> pd.DataFrame:
    """
    Reshape the guard drop mask into one row per drop.

    The mask is a per frame flag, which says nothing about when each
    drop happened, so the True runs are turned into timed rows.

    Args:
        guard_detections: GuardDetections object for the analysed video.
        fps: Frames per second of the analysed video.

    Returns:
        DataFrame with one row per guard drop.
    """
    starts, ends = sa.segment_bounds(guard_detections.mask)

    return pd.DataFrame(
        {
            "Drop": np.arange(1, starts.size + 1),
            "Start (s)": np.round(starts / fps, 1),
            "End (s)": np.round(ends / fps, 1),
            "Duration (s)": np.round((ends - starts) / fps, 1),
        }
    )


def build_guard_timeline(
    guard_detections: sa.GuardDetections,
    fps: float,
    bin_size_s: float = PACING_BIN_S,
) -> pd.DataFrame:
    """
    Count how long the guard was down in each block of time.

    Mirrors the striking pacing chart so the two can be read against
    each other.

    Args:
        guard_detections: GuardDetections object for the analysed video.
        fps: Frames per second of the analysed video.
        bin_size_s: Length of each block in seconds.

    Returns:
        DataFrame holding the start time of each block and the seconds
        spent with the guard down during it.
    """
    duration_s = guard_detections.n_frames / fps

    if duration_s <= 0:
        return pd.DataFrame({"Time (s)": [], "Guard down (s)": []})

    edges = np.arange(0.0, duration_s + bin_size_s, bin_size_s)

    dropped_times_s = np.flatnonzero(guard_detections.mask) / fps

    counts, _ = np.histogram(dropped_times_s, bins=edges)

    return pd.DataFrame(
        {
            "Time (s)": edges[:-1],
            "Guard down (s)": counts / fps,
        }
    )


RHYTHM_HELP = """
| Score | What it means |
| --- | --- |
| Below 0.5 | Metronomic. Nearly every gap is the same length. |
| 0.5 to 1.0 | Fairly even. More regular than random, |
| Around 1.0 | As unpredictable as random timing. |
| Above 1.0 | Bursty. Tight combinations separated by longer resets. |

**NOTE.** A fighter repeating one short gap
and one long gap over and over is highly readable, but counts
as a higher score. Feints, level changes etc are not counted
here.
"""


st.title("Kickboxing Analysis")


# ============================================================
# Input
# ============================================================

video = st.file_uploader(
    "Upload a video",
    type=["mp4", "mov", "avi"],
)

name = st.text_input("Name")

height = st.number_input(
    "Height (m)",
    min_value=1.0,
    max_value=2.5,
    value=None,
)

wingspan = st.number_input(
    "Wingspan (m)",
    min_value=1.0,
    max_value=2.5,
    value=None,
)

weight = st.number_input(
    "Weight (kg)",
    min_value=30.0,
    max_value=200.0,
    value=None,
)

stance = st.selectbox(
    "Stance",
    ["Orthodox", "Southpaw"],
)


# ============================================================
# Create fighter
# ============================================================

person = sa.Person(
    name=name,
    weight=weight,
    height_m=height,
    wingspan_m=wingspan,
    stance=stance.lower(),
)


# ============================================================
# Analyse button
# ============================================================

if st.button("Analyse"):
    if video is None:
        st.error("Please upload a video.")
        st.stop()

    CACHE_DIR.mkdir(exist_ok=True)

    # --------------------------------------------------------
    # Save uploaded video to a temporary file
    # --------------------------------------------------------

    with tempfile.NamedTemporaryFile(
        suffix=Path(video.name).suffix,
        delete=False,
    ) as f:
        f.write(video.getbuffer())
        video_path = Path(f.name)

    cache_path = CACHE_DIR / f"{Path(video.name).stem}.npz"

    # --------------------------------------------------------
    # Progress bar
    # --------------------------------------------------------

    progress = st.progress(
        0.0,
        text="Tracking fighter",
    )

    def show_progress(
        frames_processed: int,
        n_frames: int,
    ) -> None:

        fraction = frames_processed / n_frames if n_frames else 0.0

        progress.progress(
            min(fraction, 1.0),
            text=(f"Tracking fighter: {frames_processed}/{n_frames} frames"),
        )

    # --------------------------------------------------------
    # Run analysis
    # --------------------------------------------------------

    result = sa.analyse_video(
        video_path=video_path,
        person=person,
        cache_path=cache_path,
        model=MODEL,
        config=CONFIG,
        strike_config=STRIKE_CONFIG,
        track_progress=show_progress,
    )

    progress.empty()
    TOTAL_SECONDS = result.strike_detections.n_frames / result.person_state.fps

    # --------------------------------------------------------
    # Calculate striking stats
    # --------------------------------------------------------

    striking_stats_calculator = sa.StrikingStatsCalculator(
        person_state=result.person_state,
        detections=result.strike_detections,
        strike_config=STRIKE_CONFIG,
    )

    striking_stats = striking_stats_calculator.calculate_striking_stats()

    defense_stats_calculator = sa.DefenseStatsCalculator(
        person_state=result.person_state,
        guard_detections=result.guard_detections
    )

    guard_stats = defense_stats_calculator.calculate_guard_stats()

    # --------------------------------------------------------
    # Render annotated video
    # --------------------------------------------------------

    annotated_video_path = video_path.parent / f"annotated_{video_path.name}"

    sa.render_annotated_video(
        result=result,
        video_path=video_path,
        output_path=annotated_video_path,
        config=CONFIG,
    )

    # --------------------------------------------------------
    # Extract a frame from the original video
    # --------------------------------------------------------

    cap = cv2.VideoCapture(str(video_path))

    frame_idx = 1

    cap.set(
        cv2.CAP_PROP_POS_FRAMES,
        frame_idx,
    )

    success, frame = cap.read()

    cap.release()

    if not success:
        st.error("Could not read frame from video.")
        st.stop()

    # Convert to RGB from BGR
    frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB,
    )

    # --------------------------------------------------------
    # Store things we need after the Streamlit rerun
    # --------------------------------------------------------

    st.session_state["result"] = result
    st.session_state["stats"] = striking_stats
    st.session_state["guard_stats"] = guard_stats
    st.session_state["video_path"] = video_path
    st.session_state["frame"] = frame
    st.session_state["annotated_video_path"] = annotated_video_path

    # Reset floor selection when a new analysis is performed
    st.session_state.pop("floor_points", None)
    st.session_state.pop("floor", None)
    st.session_state.pop("footwork_analyser", None)

# ============================================================
# Floor selection
# ============================================================

if "result" in st.session_state:
    st.subheader("Select floor edges")

    st.write(
        "Trace the left edge then the right edge, each near end first: "
        "near-left, far-left, near-right, far-right."
    )

    frame = st.session_state["frame"]

    # --------------------------------------------------------
    # Resize frame for display while maintaining aspect ratio
    # --------------------------------------------------------

    display_frame = Image.fromarray(frame)

    display_height = int(frame.shape[0] * DISPLAY_WIDTH / frame.shape[1])

    display_frame = display_frame.resize((DISPLAY_WIDTH, display_height))

    value = streamlit_image_coordinates(
        display_frame,
        key="floor",
    )

    if st.button("Clear points"):
        st.session_state.pop("floor_points", None)
        st.session_state.pop("floor", None)
        st.rerun()

    # --------------------------------------------------------
    # Get floor edges
    # --------------------------------------------------------

    if "floor_points" not in st.session_state:
        st.session_state.floor_points = []

    if value is not None:
        scale_x = frame.shape[1] / value["width"]
        scale_y = frame.shape[0] / value["height"]

        point = (value["x"] * scale_x, value["y"] * scale_y)

        if not st.session_state.floor_points or (
            point != st.session_state.floor_points[-1]
        ):
            st.session_state.floor_points.append(point)

    points = st.session_state.floor_points

    st.write(f"Selected {len(points)}/4 points")

    for i, point in enumerate(points, start=1):
        st.write(f"Point {i}: {point}")

    # ----------------------------------------------------
    # Create footwork analyser
    # ----------------------------------------------------

    if len(points) == 4:
        edge1 = (points[0], points[1])
        edge2 = (points[2], points[3])

        footwork_analyser = sa.FootworkAnalyser(st.session_state["result"].person_state)

        footwork_analyser.select_floor(
            edge1=edge1,
            edge2=edge2,
        )

        st.session_state["footwork_analyser"] = footwork_analyser

        st.success("Floor selected.")

    else:
        st.info(f"Select {4 - len(points)} more point(s).")
else:
    st.info("Draw two lines along the floor edges.")


# ============================================================
# Annotated video
# ============================================================

if "annotated_video_path" in st.session_state:
    annotated_video_path = st.session_state["annotated_video_path"]

    if annotated_video_path.exists():
        st.subheader("Analysis")

        st.video(str(annotated_video_path))

    else:
        st.error("Annotated video was not rendered.")

# ============================================================
# Striking Stats
# ============================================================

if "stats" in st.session_state:
    striking_stats = st.session_state["stats"]

    st.subheader("Striking Stats")

    # --------------------------------------------------------
    # Headline numbers
    # --------------------------------------------------------

    strikes_per_sec = sum(striking_stats.strike_rates.values())

    total_col, rate_col, combo_col, rhythm_col = st.columns(4)

    total_col.metric("Total strikes", striking_stats.total_strikes)
    rate_col.metric("Strikes per sec", f"{strikes_per_sec:.1f}")
    combo_col.metric("Combo count", striking_stats.combo_count)
    rhythm_col.metric("Rhythm score (CV)", f"{striking_stats.rhythm_cv:.2f}")

    st.caption(
        "Rhythm score is the coefficient of variation of the gaps "
        "between strikes. Lower means more predictable timing."
    )

    with st.expander("How to read the rhythm score"):
        st.markdown(RHYTHM_HELP)

    # --------------------------------------------------------
    # Breakdown by strike type
    # --------------------------------------------------------

    st.dataframe(
        build_strike_table(striking_stats),
        hide_index=True,
    )

    # --------------------------------------------------------
    # Number of Strikes over time
    # --------------------------------------------------------

    pacing = pd.DataFrame(
        {
            "Time (s)": list(striking_stats.pacing_bins.keys()),
            "Strikes": list(striking_stats.pacing_bins.values()),
        }
    )

    st.caption(f"Strikes thrown per {PACING_BIN_S:.0f} second block")

    st.bar_chart(
        pacing,
        x="Time (s)",
        y="Strikes",
    )

    # --------------------------------------------------------
    # Fatigue over time
    # --------------------------------------------------------

    st.caption("Peak speed of every strike thrown.")

    thrown = [
        strike_name
        for strike_name, speeds in striking_stats.strike_speeds.items()
        if np.isfinite(speeds).any()
    ]

    if thrown:
        labels = [strike_name.replace("_", " ").capitalize() for strike_name in thrown]

        for tab, strike_name in zip(st.tabs(labels), thrown):
            fatigue = pd.DataFrame(
                {
                    "Time (s)": striking_stats.strike_times_s[strike_name],
                    "Speed (m/s)": striking_stats.strike_speeds[strike_name],
                }
            )

            tab.scatter_chart(
                fatigue,
                x="Time (s)",
                y="Speed (m/s)",
            )
    else:
        st.info("No strikes were measured, so there is nothing to plot.")

# ============================================================
# Guard Stats
# ============================================================

if "guard_stats" in st.session_state:
    guard_stats = st.session_state["guard_stats"]

    result = st.session_state["result"]
    guard_detections = result.guard_detections
    fps = result.person_state.fps

    st.subheader("Guard Stats")

    guard_drops = build_guard_drop_table(
        guard_detections=guard_detections,
        fps=fps,
    )

    # --------------------------------------------------------
    # Headline numbers
    # --------------------------------------------------------

    longest_drop_s = guard_drops["Duration (s)"].max() if not guard_drops.empty else 0.0

    up_time_col, up_pct_col, guard_drops_col, longest_col = st.columns(4)

    up_time_col.metric("Guard up time", f"{guard_stats.guard_up_time:.1f} s")
    up_pct_col.metric("Guard up", f"{guard_stats.guard_up_time_percentage:.0%}")
    guard_drops_col.metric("Guard drops", guard_stats.guard_drop_count)
    longest_col.metric("Longest drop", f"{longest_drop_s:.1f} s")


    # --------------------------------------------------------
    # Every drop
    # --------------------------------------------------------

    if guard_drops:
        st.caption("Every guard drop, in the order they happened.")

        st.dataframe(
            guard_drops,
            hide_index=True,
        )


# ============================================================
# Footwork Analysis
# ============================================================

if "footwork_analyser" in st.session_state:
    st.subheader("Footwork Analysis")

    fig = st.session_state["footwork_analyser"].get_plot_figure()

    st.pyplot(fig)
