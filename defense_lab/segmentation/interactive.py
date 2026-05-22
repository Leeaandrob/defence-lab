"""Interactive segmentation session -- SAM's iterative click-refinement loop.

Holds a single encoded image and accumulates prompts across user interactions.
Follows the SAM interaction protocol:

  * first prompt → ``multimask_output=True``, keep the highest-IoU hypothesis;
  * subsequent clicks → feed the previous decode's low-res logits back as a
    mask-prompt with ``multimask_output=False`` so the mask is *refined* rather
    than re-guessed.

Each interaction is a cheap decode (~5 ms on a GH200) against the cached
embedding, which is exactly why SAM's encode/decode split matters for
operational human-in-the-loop annotation.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from defense_lab.prompting.prompts import BoxPrompt, PointPrompt, PromptSet
from defense_lab.segmentation.predictor import (
    PromptableSegmenter,
    SegmentationResult,
)


class InteractiveSession:
    def __init__(self, segmenter: PromptableSegmenter, image: Optional[np.ndarray] = None) -> None:
        self.seg = segmenter
        if image is not None:
            self.seg.set_image(image)
        self._points = PointPrompt()
        self._box: Optional[BoxPrompt] = None
        self._last: Optional[SegmentationResult] = None
        self._history: list[SegmentationResult] = []

    # -- interactions ------------------------------------------------------- #
    def add_point(self, x: float, y: float, foreground: bool = True) -> SegmentationResult:
        # the very first interaction establishes the mask; later clicks refine it
        primary = self._last is None
        self._points.add(x, y, foreground)
        return self._decode(primary=primary)

    def set_box(self, x0: float, y0: float, x1: float, y1: float) -> SegmentationResult:
        # a box is always a primary prompt (re-establishes the mask)
        self._box = BoxPrompt(x0, y0, x1, y1)
        return self._decode(primary=True)

    def undo(self) -> Optional[SegmentationResult]:
        if self._points.points:
            self._points.points.pop()
        if not self._points.points and self._box is None:
            self._last = None
            self._history.clear()
            return None
        # rebuild from scratch (no stale mask feedback) after an undo
        return self._decode(primary=True)

    def reset(self) -> None:
        self._points = PointPrompt()
        self._box = None
        self._last = None
        self._history.clear()

    # -- state -------------------------------------------------------------- #
    def current_mask(self) -> Optional[np.ndarray]:
        return None if self._last is None else self._last.best()[0]

    def current_score(self) -> Optional[float]:
        return None if self._last is None else self._last.best()[1]

    @property
    def num_clicks(self) -> int:
        return len(self._points.points)

    # -- core --------------------------------------------------------------- #
    def _decode(self, primary: bool) -> SegmentationResult:
        if primary or self._last is None:
            # establish the mask from the initial prompt (box and/or points);
            # multimask resolves prompt ambiguity, we keep the best hypothesis.
            prompt = PromptSet(points=self._points if self._points.points else None, box=self._box)
            result = self.seg.predict(prompt, multimask_output=True)
        else:
            # refine: accumulated points + previous low-res logits as feedback,
            # WITHOUT re-passing the box -- re-passing it alongside the mask
            # feedback over-constrains the decoder and degrades the mask.
            prompt = PromptSet(
                points=self._points if self._points.points else None,
                mask=self._last.best_mask_prompt(),
            )
            result = self.seg.predict(prompt, multimask_output=False)
        self._last = result
        self._history.append(result)
        return result
