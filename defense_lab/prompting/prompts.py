"""Typed visual-prompt abstractions (SAM's promptable-segmentation interface).

These dataclasses are the lab's *prompt contract*: a domain-agnostic way to
express "what to segment" as points / boxes / masks / regions, with no notion of
a fixed class vocabulary. The mapping onto a concrete backend (SAM2's image
predictor) lives in :mod:`defense_lab.segmentation`, so the same prompts can
later drive a different foundation model or a VLM-grounded prompt source.

Conventions
-----------
* Coordinates are absolute pixel ``(x, y)`` in the original image frame.
* Point labels: ``1`` = include (foreground), ``0`` = exclude (background) --
  the SAM convention that enables interactive add/subtract refinement.
* Boxes are ``xyxy`` (x0, y0, x1, y1).
* Mask prompts are *low-resolution logits* (256x256), i.e. the previous
  decode's output fed back in -- the mechanism behind iterative refinement.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

FOREGROUND = 1
BACKGROUND = 0
_MASK_RES = 256  # SAM/SAM2 low-res mask side length


@dataclass(frozen=True)
class Point:
    x: float
    y: float
    label: int = FOREGROUND  # 1 = include, 0 = exclude

    def __post_init__(self) -> None:
        if self.label not in (FOREGROUND, BACKGROUND):
            raise ValueError(f"Point.label must be 0 or 1, got {self.label}")


@dataclass
class PointPrompt:
    points: list[Point] = field(default_factory=list)

    def add(self, x: float, y: float, foreground: bool = True) -> "PointPrompt":
        self.points.append(Point(x, y, FOREGROUND if foreground else BACKGROUND))
        return self

    def to_arrays(self) -> tuple[np.ndarray, np.ndarray]:
        if not self.points:
            return np.empty((0, 2), np.float32), np.empty((0,), np.int32)
        coords = np.array([[p.x, p.y] for p in self.points], dtype=np.float32)
        labels = np.array([p.label for p in self.points], dtype=np.int32)
        return coords, labels


@dataclass
class BoxPrompt:
    x0: float
    y0: float
    x1: float
    y1: float

    def __post_init__(self) -> None:
        # normalize so x0<x1, y0<y1 regardless of click order
        self.x0, self.x1 = sorted((self.x0, self.x1))
        self.y0, self.y1 = sorted((self.y0, self.y1))

    def to_array(self) -> np.ndarray:
        return np.array([self.x0, self.y0, self.x1, self.y1], dtype=np.float32)


@dataclass
class MaskPrompt:
    """Low-res mask logits (256x256) fed back for refinement."""

    logits: np.ndarray

    def to_array(self) -> np.ndarray:
        arr = np.asarray(self.logits, dtype=np.float32)
        if arr.ndim == 2:
            arr = arr[None]  # -> (1, H, W)
        if arr.shape[-2:] != (_MASK_RES, _MASK_RES):
            raise ValueError(
                f"MaskPrompt expects {_MASK_RES}x{_MASK_RES} low-res logits, got {arr.shape}"
            )
        return arr


@dataclass
class PromptSet:
    """A single promptable query: any combination of point(s), box, mask.

    This is exactly what one ``predict`` call consumes. Combining a box with
    refining points is the canonical SAM interaction.
    """

    points: Optional[PointPrompt] = None
    box: Optional[BoxPrompt] = None
    mask: Optional[MaskPrompt] = None

    # -- ergonomic constructors ------------------------------------------- #
    @classmethod
    def from_point(cls, x: float, y: float, foreground: bool = True) -> "PromptSet":
        return cls(points=PointPrompt().add(x, y, foreground))

    @classmethod
    def from_box(cls, x0: float, y0: float, x1: float, y1: float) -> "PromptSet":
        return cls(box=BoxPrompt(x0, y0, x1, y1))

    def is_empty(self) -> bool:
        return not (self.points and self.points.points) and self.box is None and self.mask is None

    def to_predictor_kwargs(self) -> dict[str, Any]:
        """Render to SAM2ImagePredictor.predict kwargs (numpy)."""
        kw: dict[str, Any] = {}
        if self.points is not None and self.points.points:
            coords, labels = self.points.to_arrays()
            kw["point_coords"] = coords
            kw["point_labels"] = labels
        if self.box is not None:
            kw["box"] = self.box.to_array()
        if self.mask is not None:
            kw["mask_input"] = self.mask.to_array()
        return kw


# Region prompt is semantically a coarse box/mask cue; we expose it as an alias
# constructor to keep intent explicit at call sites (operational ROI selection).
def region_prompt(x0: float, y0: float, x1: float, y1: float) -> PromptSet:
    return PromptSet.from_box(x0, y0, x1, y1)


@dataclass
class TemporalPrompt:
    """A prompt anchored to a specific (frame, object) -- the video interface.

    Consumed by the temporal/video engine (Phase 5): prompt object ``obj_id`` on
    ``frame_idx``, then propagate. Defined here so the prompt contract is one
    coherent vocabulary across image and video.
    """

    frame_idx: int
    obj_id: int
    prompt: PromptSet
