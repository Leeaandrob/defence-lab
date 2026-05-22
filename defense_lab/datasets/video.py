"""Video ingestion: frame extraction + a streaming frame source.

SAM2's video predictor consumes a directory of numbered JPEG frames, so
``extract_frames`` materializes that layout from any video file. ``VideoFrameSource``
streams frames as RGB arrays from either a video file or a frame directory --
the input contract for Phase-5 streaming inference.

On ARM64 we deliberately rely on OpenCV (present) rather than ``decord`` (no
ARM wheel), so video ingestion has no x86-only dependency.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterator, Optional

import numpy as np


def extract_frames(
    video_path: str | Path,
    out_dir: str | Path,
    *,
    stride: int = 1,
    max_frames: Optional[int] = None,
    quality: int = 95,
) -> list[Path]:
    """Decode a video to ``out_dir/00000.jpg ...`` (the SAM2 video layout)."""
    import cv2

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"cannot open video: {video_path}")
    written: list[Path] = []
    idx = 0
    kept = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if idx % stride == 0:
                p = out_dir / f"{kept:05d}.jpg"
                cv2.imwrite(str(p), frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
                written.append(p)
                kept += 1
                if max_frames is not None and kept >= max_frames:
                    break
            idx += 1
    finally:
        cap.release()
    return written


class VideoFrameSource:
    """Iterate frames (as HxWx3 RGB uint8) from a video file or frame directory."""

    def __init__(self, source: str | Path, *, stride: int = 1, max_frames: Optional[int] = None) -> None:
        self.source = Path(source)
        self.stride = stride
        self.max_frames = max_frames
        self.is_dir = self.source.is_dir()
        if self.is_dir:
            exts = {".jpg", ".jpeg", ".png"}
            self._frames = sorted(p for p in self.source.iterdir() if p.suffix.lower() in exts)

    @property
    def num_frames(self) -> Optional[int]:
        if self.is_dir:
            n = len(self._frames[:: self.stride])
            return min(n, self.max_frames) if self.max_frames else n
        return None

    def __iter__(self) -> Iterator[tuple[int, np.ndarray]]:
        import cv2

        if self.is_dir:
            for i, p in enumerate(self._frames[:: self.stride]):
                if self.max_frames and i >= self.max_frames:
                    break
                bgr = cv2.imread(str(p), cv2.IMREAD_COLOR)
                yield i, np.ascontiguousarray(bgr[:, :, ::-1])
        else:
            cap = cv2.VideoCapture(str(self.source))
            if not cap.isOpened():
                raise IOError(f"cannot open video: {self.source}")
            idx, kept = 0, 0
            try:
                while True:
                    ok, frame = cap.read()
                    if not ok:
                        break
                    if idx % self.stride == 0:
                        if self.max_frames and kept >= self.max_frames:
                            break
                        yield kept, np.ascontiguousarray(frame[:, :, ::-1])
                        kept += 1
                    idx += 1
            finally:
                cap.release()
