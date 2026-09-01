import base64
import os
import tempfile
from io import BytesIO
from pathlib import Path
from uuid import uuid4

import cv2
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image

import kickboxing_analysis as kba

CACHE_DIR = Path("cache")
MODEL = "yolo26s-pose.pt"
DISPLAY_WIDTH = 800
PACING_BIN_S = 5.0
FLOOR_SELECTOR_DIR = Path(__file__).parent / "components" / "floor_selector"
CONFIG = kba.Config()
STRIKE_CONFIG = kba.StrikeConfig()

APP_CSS = """
<style>
.stApp,
.stApp p,
.stApp label,
.stApp button,
.stApp input,
.stApp textarea,
.stApp [role="tab"],
.stApp [data-testid="stMetricLabel"],
.stApp [data-testid="stMetricValue"],
.stApp [data-testid="stCaptionContainer"] {
    font-family: "Segoe UI", Arial, Helvetica, sans-serif !important;
}
.stApp { background: #121513; color: #e7e4dd; }
[data-testid="stHeader"] { background: rgba(18,21,19,.9); border-bottom: 1px solid #343a34; }
.block-container { max-width: 1420px; padding-top: 1.25rem; padding-bottom: 3rem; }
h1,h2,h3 { color: #f4f0e6 !important; font-weight: 600 !important; letter-spacing: -.02em; }
h1 { font-size: clamp(2.3rem, 5vw, 3.7rem) !important; }
[data-testid="stCaptionContainer"] { color: #9ca49b; }
[data-testid="stMetric"] { border-top: 1px solid #3b433b; border-radius: 0; padding: .75rem 0 .9rem; }
[data-testid="stMetricLabel"] { color: #9ca49b; }
[data-testid="stMetricValue"] { color: #f4f0e6; font-weight: 600; }
.stButton > button[kind="primary"], [data-testid="stFormSubmitButton"] > button { background: #1b619f; border: 1px solid #63b5ff; border-radius: 2px; color: #f4f0e6; font-weight: 500; }
.stButton > button[kind="primary"]:hover, [data-testid="stFormSubmitButton"] > button:hover { background: #2678bd; border-color: #a4d7ff; color: white; }
.stTabs [data-baseweb="tab-list"] { gap: 1.6rem; border-bottom: 1px solid #3b433b; }
.stTabs [data-baseweb="tab"] { height: 42px; padding: 0; color: #9ca49b; }
.stTabs [aria-selected="true"] { color: #8bc8ff !important; }
.stTabs [data-baseweb="tab-highlight"] { background: #63b5ff; }
[data-testid="stDataFrame"] { border: 1px solid #3b433b; }
[data-testid="stExpander"] { background: #171a18; border: 1px solid #3b433b; border-radius: 0; }
.film-wordmark { color: #f4f0e6; font-size: .92rem; letter-spacing: .13em; margin-bottom: .35rem; }
.film-wordmark span,.film-kicker,.workflow-number { color: #63b5ff; }
.film-kicker { font-size: .72rem; font-weight: 600; letter-spacing: .16em; text-transform: uppercase; }
.film-intro,.stage-copy { color: #9ca49b; margin: .55rem 0 2rem; }
.workflow { display: grid; grid-template-columns: repeat(3,minmax(0,1fr)); margin: 1.75rem 0 2.2rem; border-top: 1px solid #3b433b; border-bottom: 1px solid #3b433b; }
.workflow-step { min-height: 63px; padding: .75rem .9rem; border-left: 1px solid #3b433b; color: #778078; }
.workflow-step:first-child { border-left: 0; }
.workflow-step.active { background: #182633; box-shadow: inset 0 -2px 0 #63b5ff; color: #f4f0e6; }
.workflow-number { font-size: .68rem; font-weight: 600; letter-spacing: .13em; }
.workflow-name { display: block; margin-top: .25rem; font-size: .88rem; }
.stage-rule { border-top: 1px solid #3b433b; margin: 1.8rem 0 .7rem; }
.stage-heading { color: #f4f0e6; font-size: 1.7rem; margin: 0 0 .3rem; }
.stage-copy { margin: 0 0 1.25rem; }
.workspace-bar { align-items: baseline; border-bottom: 1px solid #3b433b; display: flex; justify-content: space-between; margin-bottom: 1.4rem; padding-bottom: .8rem; }
.workspace-name { color: #f4f0e6; font-size: 1.18rem; font-weight: 600; letter-spacing: .04em; margin: 0; }
.workspace-name span { color: #63b5ff; }
.workspace-tag { color: #879187; font-size: .72rem; letter-spacing: .12em; text-transform: uppercase; }
.panel-title { color: #f4f0e6; font-size: 1.35rem; font-weight: 600; margin: 0 0 .28rem; }
.panel-copy { color: #9ca49b; font-size: .88rem; line-height: 1.45; margin: 0 0 1.25rem; }
.panel-rule { border-top: 1px solid #3b433b; margin: 1.4rem 0 .8rem; }
.workspace-kicker { color: #63b5ff; font-size: .7rem; font-weight: 600; letter-spacing: .14em; margin: 0 0 .4rem; text-transform: uppercase; }
.workspace-title { color: #f4f0e6; font-size: 2rem; font-weight: 600; margin: 0 0 .35rem; }
.workspace-copy { color: #9ca49b; margin: 0 0 1.25rem; }
.empty-workspace { border-top: 1px solid #3b433b; color: #9ca49b; margin-top: .2rem; padding: 2rem 0; }
.empty-workspace strong { color: #f4f0e6; display: block; font-size: 1.7rem; font-weight: 600; margin-bottom: .5rem; }
.stButton > button[kind="secondary"] { background: transparent; border-color: #4b574b; color: #c7cec7; }
.stButton > button[kind="secondary"]:hover { background: #1b211d; border-color: #79bfff; color: #f4f0e6; }
@media (max-width: 700px) { .block-container { padding-left: 1rem; padding-right: 1rem; } .workflow-step { padding-left:.55rem; padding-right:.55rem; } .workflow-name { font-size:.76rem; } }
</style>
"""

st.set_page_config(page_title="Kickboxing Analysis", layout="wide", initial_sidebar_state="collapsed")
floor_selector = components.declare_component("floor_selector", path=str(FLOOR_SELECTOR_DIR))


def select_floor_corners(image: Image.Image, points: list[tuple[float, float]], key: str) -> list[list[float]] | None:
    buffer = BytesIO()
    image.save(buffer, format="JPEG", quality=90)
    encoded_image = base64.b64encode(buffer.getvalue()).decode("ascii")
    return floor_selector(
        image=f"data:image/jpeg;base64,{encoded_image}",
        image_width=image.width,
        image_height=image.height,
        points=[[float(x), float(y)] for x, y in points],
        key=key,
        default=None,
    )


def build_strike_table(stats: kba.StrikingStats) -> pd.DataFrame:
    rows = []
    for side in kba.SIDES:
        for strike_type in kba.STRIKE_TYPES:
            key = f"{side}_{strike_type}"
            rows.append({
                "Strike": f"{side.title()} {strike_type}",
                "Count": int(stats.strike_counts[key]),
                "Per sec": round(stats.strike_rates[key], 1),
                "Avg speed (m/s)": round(stats.avg_speeds_mps[key], 1),
                "Max speed (m/s)": round(stats.max_speeds_mps[key], 1),
            })
    return pd.DataFrame(rows)


def build_guard_drop_table(detections: kba.GuardDetections, fps: float) -> pd.DataFrame:
    starts, ends = kba.segment_bounds(detections.mask)
    return pd.DataFrame({
        "Drop": np.arange(1, starts.size + 1),
        "Start (s)": np.round(starts / fps, 1),
        "End (s)": np.round(ends / fps, 1),
        "Duration (s)": np.round((ends - starts) / fps, 1),
    })


def build_guard_timeline(detections: kba.GuardDetections, fps: float) -> pd.DataFrame:
    duration_s = detections.n_frames / fps
    if duration_s <= 0:
        return pd.DataFrame({"Time (s)": [], "Guard down (s)": []})
    edges = np.arange(0.0, duration_s + PACING_BIN_S, PACING_BIN_S)
    counts, _ = np.histogram(np.flatnonzero(detections.mask) / fps, bins=edges)
    return pd.DataFrame({"Time (s)": edges[:-1], "Guard down (s)": counts / fps})


def analyse_session(video, person: kba.Person) -> None:
    CACHE_DIR.mkdir(exist_ok=True)
    with tempfile.NamedTemporaryFile(suffix=Path(video.name).suffix, delete=False) as file:
        file.write(video.getbuffer())
        video_path = Path(file.name)
    progress = st.progress(0.0, text="Tracking fighter")

    def show_progress(frames_processed: int, n_frames: int) -> None:
        fraction = frames_processed / n_frames if n_frames else 0.0
        progress.progress(min(fraction, 1.0), text=f"Tracking fighter: {frames_processed}/{n_frames} frames")

    result = kba.analyse_video(
        video_path=video_path,
        person=person,
        cache_path=CACHE_DIR / f"{Path(video.name).stem}.npz",
        model=MODEL,
        config=CONFIG,
        strike_config=STRIKE_CONFIG,
        track_progress=show_progress,
    )
    progress.empty()
    striking_stats = kba.StrikingStatsCalculator(result.person_state, result.strike_detections, STRIKE_CONFIG).calculate_striking_stats()
    guard_stats = kba.DefenseStatsCalculator(result.person_state, result.guard_detections).calculate_guard_stats()
    annotated_path = video_path.parent / f"annotated_{video_path.stem}.mp4"
    kba.render_annotated_video(result, video_path, annotated_path, CONFIG)

    capture = cv2.VideoCapture(str(video_path))
    capture.set(cv2.CAP_PROP_POS_FRAMES, 1)
    success, frame = capture.read()
    capture.release()
    if not success:
        st.error("Could not read a frame from the uploaded video.")
        return

    st.session_state.update({
        "result": result,
        "stats": striking_stats,
        "guard_stats": guard_stats,
        "video_path": video_path,
        "frame": cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
        "annotated_video_path": annotated_path,
    })
    for key in ("floor_points", "floor", "floor_selector_revision", "footwork_analyser", "footwork_output"):
        st.session_state.pop(key, None)


def render_setup() -> None:
    st.markdown("<p class='panel-title'>New session</p>", unsafe_allow_html=True)
    st.markdown("<p class='panel-copy'>Upload one shadowboxing round to begin analysis.</p>", unsafe_allow_html=True)
    with st.form("session_setup"):
        video = st.file_uploader("Training clip", type=["mp4", "mov", "avi"], help="One fighter in frame; shadowboxing footage works best.")
        name = st.text_input("Fighter name")
        stance = st.selectbox("Stance", ["Orthodox", "Southpaw"])
        height = st.number_input("Height (m)", min_value=1.0, max_value=2.5, value=None)
        wingspan = st.number_input("Wingspan (m)", min_value=1.0, max_value=2.5, value=None)
        weight = st.number_input("Weight (kg)", min_value=30.0, max_value=200.0, value=None)
        submitted = st.form_submit_button("Analyse session", type="primary")

    if submitted:
        if video is None:
            st.error("Upload a video before starting analysis.")
        else:
            analyse_session(video, kba.Person(name=name, weight=weight, height_m=height, wingspan_m=wingspan, stance=stance.lower()))


def render_calibration(compact: bool = False) -> kba.FootworkAnalyser | None:
    if compact:
        st.caption("Adjust corners or floor dimensions. Changes replace the current footwork output.")
    else:
        st.markdown("<p class='workspace-kicker'>Floor calibration</p>", unsafe_allow_html=True)
        st.markdown("<p class='workspace-title'>Trace the fighting surface.</p>", unsafe_allow_html=True)
        st.markdown("<p class='workspace-copy'>Set the four floor corners, then drag a marker to refine the projection.</p>", unsafe_allow_html=True)
    frame = st.session_state["frame"]
    display_height = int(frame.shape[0] * DISPLAY_WIDTH / frame.shape[1])
    display_frame = Image.fromarray(frame).resize((DISPLAY_WIDTH, display_height), Image.Resampling.LANCZOS)
    st.session_state.setdefault("floor_points", [])
    revision = st.session_state.setdefault("floor_selector_revision", 0)
    display_points = [
        (point[0] * DISPLAY_WIDTH / frame.shape[1], point[1] * display_height / frame.shape[0])
        for point in st.session_state.floor_points
    ]
    value = select_floor_corners(display_frame, display_points, f"floor-selector-{revision}")
    if st.button("Clear floor corners", key="clear_floor_points"):
        for key in ("floor_points", "floor", "footwork_analyser", "footwork_output"):
            st.session_state.pop(key, None)
        st.session_state.floor_selector_revision = revision + 1
        st.rerun()

    if isinstance(value, list) and len(value) <= 4:
        selected_points = [
            (float(point[0]) * frame.shape[1] / DISPLAY_WIDTH, float(point[1]) * frame.shape[0] / display_height)
            for point in value if isinstance(point, list) and len(point) == 2
        ]
        if selected_points != st.session_state.floor_points:
            st.session_state.floor_points = selected_points
            st.session_state.pop("footwork_analyser", None)
            st.session_state.pop("footwork_output", None)

    points = st.session_state.floor_points
    if len(points) != 4:
        st.info(f"Set {4 - len(points)} more floor corner(s) to unlock the review.")
        return None

    analyser = kba.FootworkAnalyser(st.session_state["result"].person_state, principal_point=(frame.shape[1] / 2, frame.shape[0] / 2))
    analyser.select_floor(edge1=(points[0], points[1]), edge2=(points[2], points[3]))
    with st.expander("Scale the floor projection (optional)"):
        st.caption("Enter both side lengths for results in metres. Leave them at zero to use a unit-square projection.")
        left_col, right_col, width_col = st.columns(3)
        with left_col:
            left_length = st.number_input("Left side (m)", min_value=0.0, step=0.1, key="floor_length1")
        with right_col:
            right_length = st.number_input("Right side (m)", min_value=0.0, step=0.1, key="floor_length2")
        with width_col:
            width_m = st.number_input("Front edge / width (m)", min_value=0.0, step=0.1, key="floor_width")
        if left_length > 0 and right_length > 0:
            analyser.select_floor_edge_lengths_m(left_length, right_length)
        if width_m > 0:
            analyser.select_floor_width_m(width_m)
    st.session_state["footwork_analyser"] = analyser
    if not compact:
        st.success("Floor calibrated. Open Review from the workspace panel.")
    return analyser


def render_striking(stats: kba.StrikingStats) -> None:
    totals = st.columns(4)
    totals[0].metric("Total strikes", stats.total_strikes)
    totals[1].metric("Strikes per sec", f"{sum(stats.strike_rates.values()):.1f}")
    totals[2].metric("Combo count", stats.combo_count)
    totals[3].metric("Rhythm score", f"{stats.rhythm_cv:.2f}")

    st.caption("Strike mix")
    mix = st.columns(2)
    mix[0].metric("Punches", stats.punch_count)
    mix[1].metric("Kicks", stats.kick_count)

    st.caption("Combo and work rate")
    timing = st.columns(4)
    timing[0].metric("Average combo", f"{stats.avg_combo_length:.1f}")
    timing[1].metric("Longest combo", stats.longest_combo)
    timing[2].metric("Mean gap", f"{stats.mean_interval_s:.1f} s")
    timing[3].metric("Longest rest", f"{stats.longest_rest_s:.1f} s")

    st.caption("Strike balance")
    balance = st.columns(4)
    balance[0].metric("Lead", stats.lead_rear_counts["lead"])
    balance[1].metric("Rear", stats.lead_rear_counts["rear"])
    balance[2].metric("Left", stats.side_counts["left"])
    balance[3].metric("Right", stats.side_counts["right"])

    st.caption("Strike breakdown")
    st.dataframe(build_strike_table(stats), hide_index=True, use_container_width=True)
    pacing = pd.DataFrame({"Time (s)": list(stats.pacing_bins.keys()), "Strikes": list(stats.pacing_bins.values())})
    st.caption(f"Output per {PACING_BIN_S:.0f}-second block")
    st.bar_chart(pacing, x="Time (s)", y="Strikes", color="#63b5ff")
    thrown = [name for name, speeds in stats.strike_speeds.items() if np.isfinite(speeds).any()]
    if not thrown:
        st.info("No strikes were measured in this clip.")
        return
    st.caption("Peak speed by strike")
    labels = [name.replace("_", " ").capitalize() for name in thrown]
    for tab, name in zip(st.tabs(labels), thrown):
        chart = pd.DataFrame({"Time (s)": stats.strike_times_s[name], "Speed (m/s)": stats.strike_speeds[name]})
        tab.scatter_chart(chart, x="Time (s)", y="Speed (m/s)", color="#63b5ff")


def render_guard(guard_stats: kba.GuardStats, result) -> None:
    detections = result.guard_detections
    drops = build_guard_drop_table(detections, result.person_state.fps)
    longest = drops["Duration (s)"].max() if not drops.empty else 0.0
    totals = st.columns(4)
    totals[0].metric("Guard up time", f"{guard_stats.guard_up_time:.1f} s")
    totals[1].metric("Guard up", f"{guard_stats.guard_up_time_percentage:.0%}")
    totals[2].metric("Guard drops", guard_stats.guard_drop_count)
    totals[3].metric("Longest drop", f"{longest:.1f} s")
    timeline = build_guard_timeline(detections, result.person_state.fps)
    if timeline["Guard down (s)"].sum() > 0:
        st.caption(f"Guard-down time per {PACING_BIN_S:.0f}-second block")
        st.bar_chart(timeline, x="Time (s)", y="Guard down (s)", color="#63b5ff")
    if drops.empty:
        st.info("No guard drops were measured in this clip.")
    else:
        st.caption("Guard drops")
        st.dataframe(drops, hide_index=True, use_container_width=True)


def create_footwork_output(analyser: kba.FootworkAnalyser) -> None:
    with st.spinner("Rendering footwork visualisations..."):
        footwork_stats = analyser.get_footwork_stats()
        figure = analyser.get_plot_figure()
        source_path = st.session_state["video_path"]
        final_path = source_path.parent / f"footwork_{source_path.stem}.mp4"
        temporary_path = source_path.parent / f"footwork_{uuid4().hex}.mp4"
        kba.render_annotated_video(
            result=st.session_state["result"], video_path=source_path, output_path=temporary_path,
            config=CONFIG, distance_per_frame=footwork_stats.distance_travelled_cumsum,
        )
        os.replace(temporary_path, final_path)
        st.session_state["footwork_output"] = {
            "stats": footwork_stats,
            "width_unit": "m" if analyser.projector.floor_edge_lengths is not None else "sq",
            "figure": figure,
            "video": str(final_path),
        }


def render_footwork(analyser: kba.FootworkAnalyser) -> None:
    output = st.session_state.get("footwork_output")
    if output is None:
        st.caption("Generate the existing footwork video, 40 × 40-bin heatmap, and movement statistics.")
        if st.button("Render footwork visualisations", type="primary"):
            create_footwork_output(analyser)
            output = st.session_state["footwork_output"]
    if output is None:
        return
    stats = output["stats"]
    unit = output["width_unit"]
    distance = stats.distance_travelled_cumsum[-1] if stats.distance_travelled_cumsum.size else 0.0
    totals = st.columns(4)
    totals[0].metric("Floor coverage", f"{stats.floor_coverage:.0%}")
    totals[1].metric("Mean stance width", f"{stats.stance_width_mean:.2f} {unit}")
    totals[2].metric("Stance variation", f"{stats.stance_width_std_dev:.2f} {unit}")
    totals[3].metric("Distance travelled", f"{distance:.2f} {unit}")
    if Path(output["video"]).exists():
        st.caption("Footwork video")
        st.video(output["video"])
    st.pyplot(output["figure"], use_container_width=True)


def render_review(analyser: kba.FootworkAnalyser) -> None:
    stats = st.session_state["stats"]
    guard_stats = st.session_state["guard_stats"]
    output = st.session_state.get("footwork_output")
    distance = None
    if output is not None and output["stats"].distance_travelled_cumsum.size:
        distance = output["stats"].distance_travelled_cumsum[-1]
    video_col, metrics_col = st.columns((1.65, .8))
    with video_col:
        video_path = st.session_state["annotated_video_path"]
        if video_path.exists():
            st.video(str(video_path))
        else:
            st.error("Annotated video was not rendered.")
    with metrics_col:
        st.metric("Strikes", stats.total_strikes)
        st.metric("Combos", stats.combo_count)
        st.metric("Guard up", f"{guard_stats.guard_up_time_percentage:.0%}")
        st.metric("Distance", "—" if distance is None else f"{distance:.2f}")
        if distance is None:
            st.caption("Distance appears after footwork visualisations are rendered.")
    striking_tab, guard_tab, footwork_tab = st.tabs(("Striking", "Guard", "Footwork"))
    with striking_tab:
        render_striking(stats)
    with guard_tab:
        render_guard(guard_stats, st.session_state["result"])
    with footwork_tab:
        render_footwork(analyser)


def select_workspace_view(view: str) -> None:
    """Update the active workspace view before Streamlit renders controls."""
    st.session_state["workspace_view"] = view


def render_session_panel() -> None:
    """Render the persistent workspace controls for an analysed session."""
    active_view = st.session_state.get("workspace_view", "calibrate")

    st.markdown("<p class='panel-title'>Session</p>", unsafe_allow_html=True)

    st.button(
        "Floor calibration",
        type="primary" if active_view == "calibrate" else "secondary",
        use_container_width=True,
        key="show_calibration",
        on_click=select_workspace_view,
        args=("calibrate",),
    )

    calibrated = len(st.session_state.get("floor_points", [])) == 4
    st.button(
        "Review session",
        type="primary" if active_view == "review" else "secondary",
        use_container_width=True,
        disabled=not calibrated,
        key="show_review",
        on_click=select_workspace_view,
        args=("review",),
    )

    st.markdown("<div class='panel-rule'></div>", unsafe_allow_html=True)
    if st.button("New session", type="secondary", use_container_width=True, key="new_session"):
        for key in (
            "result",
            "stats",
            "guard_stats",
            "video_path",
            "frame",
            "annotated_video_path",
            "floor_points",
            "floor",
            "floor_selector_revision",
            "footwork_analyser",
            "footwork_output",
            "workspace_view",
            "floor_length1",
            "floor_length2",
            "floor_width",
        ):
            st.session_state.pop(key, None)
        st.rerun()


def render_empty_workspace() -> None:
    st.markdown(
        "<div class='empty-workspace'><strong>Load a training clip.</strong>"
        "Your annotated footage, measured stats, and movement visualisations will appear here.</div>",
        unsafe_allow_html=True,
    )


st.markdown(APP_CSS, unsafe_allow_html=True)
st.markdown(
    "<div class='workspace-bar'><p class='workspace-name'>Kickboxing Analysis</p>"
    "<span class='workspace-tag'>Session workspace</span></div>",
    unsafe_allow_html=True,
)

panel_col, main_col = st.columns((0.32, 0.68), gap="large")
with panel_col:
    if "result" in st.session_state:
        render_session_panel()
    else:
        render_setup()

with main_col:
    if "result" not in st.session_state:
        render_empty_workspace()
    elif st.session_state.get("workspace_view", "calibrate") == "review":
        render_review(st.session_state["footwork_analyser"])
    else:
        footwork_analyser = render_calibration()
        if footwork_analyser is not None:
            st.session_state["footwork_analyser"] = footwork_analyser
