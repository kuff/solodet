"""Video I/O and frame extraction utilities."""

from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


class VideoReader:
    """Simple video reader with iterator support.

    Args:
        path: Path to video file.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.cap = cv2.VideoCapture(str(self.path))
        if not self.cap.isOpened():
            raise IOError(f"Cannot open video: {self.path}")

    @property
    def fps(self) -> float:
        return self.cap.get(cv2.CAP_PROP_FPS) or 30.0

    @property
    def width(self) -> int:
        return int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    @property
    def height(self) -> int:
        return int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    @property
    def frame_count(self) -> int:
        return int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def __iter__(self):
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        return self

    def __next__(self) -> np.ndarray:
        ret, frame = self.cap.read()
        if not ret:
            raise StopIteration
        return frame

    def __len__(self) -> int:
        return self.frame_count

    def __del__(self):
        if hasattr(self, "cap"):
            self.cap.release()


class VideoWriter:
    """Simple video writer.

    Args:
        path: Output video path.
        fps: Frames per second.
        width: Frame width.
        height: Frame height.
        codec: FourCC codec string.
    """

    def __init__(
        self,
        path: str | Path,
        fps: float,
        width: int,
        height: int,
        codec: str = "mp4v",
    ):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*codec)
        self.writer = cv2.VideoWriter(str(self.path), fourcc, fps, (width, height))

    def write(self, frame: np.ndarray) -> None:
        self.writer.write(frame)

    def release(self) -> None:
        self.writer.release()

    def __del__(self):
        if hasattr(self, "writer"):
            self.writer.release()


def extract_frames(
    video_path: str | Path,
    output_dir: str | Path,
    every_n: int = 1,
    ext: str = ".jpg",
) -> list[Path]:
    """Extract frames from a video file.

    Args:
        video_path: Input video path.
        output_dir: Directory to save extracted frames.
        every_n: Extract every Nth frame (1 = all frames).
        ext: Output image extension.

    Returns:
        List of saved frame paths.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    reader = VideoReader(video_path)
    saved = []

    for i, frame in enumerate(tqdm(reader, total=len(reader), desc="Extracting frames")):
        if i % every_n != 0:
            continue
        frame_path = output_dir / f"{i:06d}{ext}"
        cv2.imwrite(str(frame_path), frame)
        saved.append(frame_path)

    return saved
