from pathlib import Path
import tempfile

import cv2
import streamlit as st
from PIL import Image
from streamlit_image_coordinates import streamlit_image_coordinates

import strike_analysis as sa


CACHE_DIR = Path("cache")
MODEL = "yolo26s-pose.pt"
DISPLAY_WIDTH = 800


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

weight = st.number_input(
    "Weight (kg)",
    min_value=30.0,
    max_value=200.0,
    value=None,
)

wingspan = st.number_input(
    "Wingspan (m)",
    min_value=1.0,
    max_value=2.5,
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

        fraction = (
            frames_processed / n_frames
            if n_frames
            else 0.0
        )

        progress.progress(
            min(fraction, 1.0),
            text=(
                f"Tracking fighter: "
                f"{frames_processed}/{n_frames} frames"
            ),
        )

    # --------------------------------------------------------
    # Run analysis
    # --------------------------------------------------------

    result = sa.analyse(
        video_path=video_path,
        person=person,
        cache_path=cache_path,
        model=MODEL,
        config=sa.Config(),
        strike_config=sa.StrikeConfig(),
        track_progress=show_progress,
    )

    progress.empty()

    # --------------------------------------------------------
    # Render annotated video
    # --------------------------------------------------------

    annotated_video_path = (
        video_path.parent
        / f"annotated_{video_path.name}"
    )

    sa.render_annotated_video(
        result=result,
        video_path=video_path,
        output_path=annotated_video_path,
        config=sa.Config(),
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
    st.session_state["video_path"] = video_path
    st.session_state["frame"] = frame
    st.session_state["annotated_video_path"] = (
        annotated_video_path
    )

    # Reset floor selection when a new analysis is performed
    st.session_state.pop("floor_points", None)
    st.session_state.pop("floor", None)

# ============================================================
# Floor selection
# ============================================================

if "result" in st.session_state:

    st.subheader("Select floor edges")

    st.write(
        "Draw two lines along the edges of the floor."
    )

    frame = st.session_state["frame"]

    # --------------------------------------------------------
    # Resize frame for display while maintaining aspect ratio
    # --------------------------------------------------------

    display_frame = Image.fromarray(frame)

    display_height = int(
        frame.shape[0]
        * DISPLAY_WIDTH
        / frame.shape[1]
    )

    display_frame = display_frame.resize(
        (DISPLAY_WIDTH, display_height)
    )

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


        footwork_analyser = sa.FootworkAnalyser(
            st.session_state["result"].person_state
        )

        footwork_analyser.select_floor(
            edge1=edge1,
            edge2=edge2,
        )

        st.success("Floor selected.")

    else:
        st.info(
            f"Select {4 - len(points)} more point(s)."
        )
else:

    st.info(
        "Draw two lines along the floor edges."
    )


# ============================================================
# Annotated video
# ============================================================

if "annotated_video_path" in st.session_state:

    annotated_video_path = (
        st.session_state["annotated_video_path"]
    )

    if annotated_video_path.exists():

        st.subheader("Analysis")

        st.video(
            str(annotated_video_path)
        )

    else:

        st.error(
            "Annotated video was not rendered."
        )

# ============================================================
# Footwork Analysis
# ============================================================
st.subheader("Footwork Analysis")

fig = footwork_analyser.get_plot_figure()

st.pyplot(fig)