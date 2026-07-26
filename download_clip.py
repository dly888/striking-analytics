import subprocess

def download_clip():
    url = input("Enter YouTube URL: ")
    start = input("Start time (MM:SS): ")
    end = input("End time (MM:SS): ")

    command = [
        "yt-dlp",
        "--download-sections", f"*{start}-{end}",
        "--force-keyframes-at-cuts",
        "-f", "bv*+ba/b",
        "--merge-output-format", "mp4",
        url
    ]

    subprocess.run(command)


if __name__ == "__main__":
    download_clip()