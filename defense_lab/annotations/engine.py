"""SAM-assisted labeling (data-engine stages 1-2: assisted & semi-automatic).

Given weak supervision -- a detector's boxes, GT boxes to be upgraded to masks,
or a few clicks -- the assisted labeler turns them into high-quality masks using
the Phase-2 :class:`PromptableSegmenter`. The image is encoded once and every
prompt is a cheap decode, which is what makes model-assisted annotation scale.
"""
from __future__ import annotations

from typing import Iterable, Optional, Sequence

import numpy as np

from defense_lab.datasets.types import Instance, mask_to_xyxy
from defense_lab.prompting.prompts import PromptSet
from defense_lab.segmentation.predictor import PromptableSegmenter


class AssistedLabeler:
    def __init__(self, segmenter: PromptableSegmenter) -> None:
        self.seg = segmenter

    def label_from_boxes(
        self,
        image: np.ndarray,
        boxes: Sequence[Sequence[float]],
        category_ids: Optional[Sequence[int]] = None,
    ) -> list[Instance]:
        """Upgrade boxes (xyxy) to masks. One encode, N cheap decodes."""
        self.seg.set_image(image)
        out: list[Instance] = []
        for i, b in enumerate(boxes):
            res = self.seg.predict(PromptSet.from_box(float(b[0]), float(b[1]), float(b[2]), float(b[3])))
            mask, score, _ = res.best()
            out.append(
                Instance(
                    box=mask_to_xyxy(mask),
                    mask=mask,
                    score=float(score),
                    category_id=(category_ids[i] if category_ids is not None else None),
                    source="assisted",
                )
            )
        return out

    def label_from_points(
        self,
        image: np.ndarray,
        points: Iterable[Sequence[float]],
    ) -> list[Instance]:
        """One positive click per object -> mask."""
        self.seg.set_image(image)
        out: list[Instance] = []
        for (x, y) in points:
            res = self.seg.predict(PromptSet.from_point(float(x), float(y), True))
            mask, score, _ = res.best()
            out.append(Instance(box=mask_to_xyxy(mask), mask=mask, score=float(score), source="assisted"))
        return out
