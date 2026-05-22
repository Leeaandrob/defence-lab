"""Human-in-the-loop mask refinement (closing the data-engine loop).

When a proposed mask is sent to review, a human fixes it with a few corrective
clicks. ``refine`` replays that: seed from a (possibly coarse) box, then apply
positive/negative correction points through the SAM interactive protocol, which
feeds the previous mask back as a prompt so each click *edits* rather than
restarts the mask.
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from defense_lab.datasets.types import Instance, mask_to_xyxy
from defense_lab.segmentation.interactive import InteractiveSession
from defense_lab.segmentation.predictor import PromptableSegmenter

# a correction is (x, y, is_foreground)
Correction = tuple[float, float, bool]


def refine(
    segmenter: PromptableSegmenter,
    image: np.ndarray,
    *,
    box: Optional[Sequence[float]] = None,
    seed_point: Optional[Sequence[float]] = None,
    corrections: Sequence[Correction] = (),
) -> tuple[Instance, list[float]]:
    """Refine a mask interactively. Returns (final Instance, per-step scores)."""
    if box is None and seed_point is None:
        raise ValueError("refine needs a box or a seed_point to start from")
    sess = InteractiveSession(segmenter, image)
    scores: list[float] = []
    if box is not None:
        r = sess.set_box(float(box[0]), float(box[1]), float(box[2]), float(box[3]))
        scores.append(r.best()[1])
    if seed_point is not None:
        r = sess.add_point(float(seed_point[0]), float(seed_point[1]), True)
        scores.append(r.best()[1])
    for (x, y, fg) in corrections:
        r = sess.add_point(float(x), float(y), bool(fg))
        scores.append(r.best()[1])
    mask = sess.current_mask()
    inst = Instance(box=mask_to_xyxy(mask), mask=mask, score=sess.current_score(), source="refined")
    return inst, scores
