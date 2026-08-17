import streamlit as st


import strike_analysis as sa
from pathlib import Path
import tempfile


CACHE_DIR: Path = Path("cache")
MODEL : str = "yolo26s-pose.pt"

st.title("Kickboxing Analysis")

# Video upload
video = st.file_uploader(
    "Upload a video",
    type=["mp4", "mov", "avi"]
)

# Get fighter details
name = st.text_input("Name")

height = st.number_input(
    "Height (m)",
    min_value=1.0,
    max_value=2.5,
    value=None
)

weight = st.number_input(
    "Weight (kg)",
    min_value=30.0,
    max_value=200.0,
    value=None
)


wingspan = st.number_input(
    "Wingspan (m)",
    min_value=1.0,
    max_value=2.5,
    value=None
)

stance = st.selectbox(
    "Stance",
    ["Orthodox", "Southpaw"]
)

person = sa.Person(
    name=name,
    weight=weight,
    height_m=height,
    wingspan_m=wingspan,
    stance=stance.lower()
)

if st.button("Analyse"):
    if video:
        with tempfile.NamedTemporaryFile(
            suffix=Path(video.name).suffix,
            delete=False,
        ) as f:
            f.write(video.getbuffer())

        VIDEO_PATH = Path(f.name)

        # np.savez_compressed adds .npz, so without it here the file that
        # gets written is never the file that gets looked for
        CACHE_PATH = CACHE_DIR / f"{Path(video.name).stem}.npz"

        progress = st.progress(0.0, text="Tracking fighter")

        def show_progress(frames_processed: int, n_frames: int) -> None:
            fraction = frames_processed / n_frames if n_frames else 0.0
            progress.progress(
                min(fraction, 1.0),
                text=f"Tracking fighter: {frames_processed}/{n_frames} frames",
            )

        result = sa.analyse(
            video_path=VIDEO_PATH,
            person=person,
            cache_path=CACHE_PATH,
            model=MODEL,
            config=sa.Config(),
            strike_config=sa.StrikeConfig(),
            track_progress=show_progress,
        )

        progress.empty()

        # Written beside the upload, joining a relative path onto an
        # absolute one just gives back the absolute one
        ANNOTATED_VIDEO_PATH = VIDEO_PATH.with_name(f"annotated_{VIDEO_PATH.name}")

        sa.render_annotated_video(
            result=result,
            video_path=VIDEO_PATH,
            output_path=ANNOTATED_VIDEO_PATH,
            config=sa.Config()
        )

        if ANNOTATED_VIDEO_PATH.exists():
            st.video(ANNOTATED_VIDEO_PATH)
        else:
            raise ValueError("Video not rendered.")

    else:
        raise ValueError("File not uploaded.")