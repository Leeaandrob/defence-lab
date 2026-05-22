"""Dataset value types shared across ingestion + the data engine.

Backend-agnostic containers: an ``Instance`` is one segmented object (mask +
box + optional category/scores), a ``Sample`` is one image with its instances.
Category is *optional* on purpose -- the data engine produces class-agnostic
masks (SAM philosophy); categories are attached only when a labeled source
(e.g. COCO) provides them.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np


def mask_to_xyxy(mask: np.ndarray) -> tuple[float, float, float, float]:
    ys, xs = np.where(mask)
    if xs.size == 0:
        return (0.0, 0.0, 0.0, 0.0)
    return (float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1))


@dataclass
class Instance:
    box: tuple[float, float, float, float]  # xyxy
    mask: Optional[np.ndarray] = None       # bool HxW
    category_id: Optional[int] = None
    category_name: Optional[str] = None
    score: Optional[float] = None           # predicted IoU / confidence
    stability: Optional[float] = None       # SAM stability score
    obj_id: Optional[int] = None
    source: Optional[str] = None            # 'gt' | 'assisted' | 'auto' | 'pseudo' | 'refined'

    @property
    def area(self) -> int:
        return int(self.mask.sum()) if self.mask is not None else 0


@dataclass
class Sample:
    image_id: Any
    height: int
    width: int
    file_name: Optional[str] = None
    image_dir: Optional[str] = None
    instances: list[Instance] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)

    def image_path(self) -> Optional[Path]:
        if self.file_name is None:
            return None
        return Path(self.image_dir or ".") / self.file_name

    def load_image(self) -> np.ndarray:
        """Load HxWx3 RGB uint8 from disk (BGR->RGB)."""
        import cv2

        p = self.image_path()
        if p is None or not p.exists():
            raise FileNotFoundError(f"image not found for sample {self.image_id}: {p}")
        bgr = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if bgr is None:
            raise IOError(f"failed to read image: {p}")
        return np.ascontiguousarray(bgr[:, :, ::-1])
