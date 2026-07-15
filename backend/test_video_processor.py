from pathlib import Path

from backend.video_processor import VideoProcessor

BASE_DIR = Path(__file__).resolve().parent

INPUT_VIDEO = BASE_DIR / "test.mp4"

OUTPUT_VIDEO = BASE_DIR / "output.mp4"

processor = VideoProcessor()

processor.process_video(
    str(INPUT_VIDEO),
    str(OUTPUT_VIDEO)
)