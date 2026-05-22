"""Segmentation metrics. Phase 2 uses IoU/Dice; Phase 6 extends with boundary
and temporal-consistency metrics. Kept dependency-light (numpy only) and
vectorized so it scales to full-frame operational masks.
"""
from __future__ import annotations

import numpy as np


def _b(x: np.ndarray) -> np.ndarray:
    return np.asarray(x, dtype=bool)


def mask_iou(pred: np.ndarray, gt: np.ndarray) -> float:
    """Intersection-over-Union of two binary masks. Empty/empty -> 1.0."""
    pred, gt = _b(pred), _b(gt)
    inter = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    return float(inter / union) if union > 0 else 1.0


def dice_coefficient(pred: np.ndarray, gt: np.ndarray) -> float:
    pred, gt = _b(pred), _b(gt)
    denom = pred.sum() + gt.sum()
    return float(2 * np.logical_and(pred, gt).sum() / denom) if denom > 0 else 1.0


def boundary_iou(pred: np.ndarray, gt: np.ndarray, dilation: int = 2) -> float:
    """Boundary IoU (Cheng et al., 2021): IoU restricted to a band around the
    contours -- sensitive to boundary quality where region IoU saturates."""
    import cv2

    pred, gt = _b(pred), _b(gt)
    k = np.ones((2 * dilation + 1, 2 * dilation + 1), np.uint8)

    def band(m: np.ndarray) -> np.ndarray:
        m8 = m.astype(np.uint8)
        eroded = cv2.erode(m8, k, iterations=1)
        return (m8 - eroded).astype(bool)

    pb, gb = band(pred), band(gt)
    inter = np.logical_and(pb, gb).sum()
    union = np.logical_or(pb, gb).sum()
    return float(inter / union) if union > 0 else 1.0


# --------------------------------------------------------------------------- #
# temporal metrics (Phase 5/6) -- operate on a per-frame sequence of masks
# --------------------------------------------------------------------------- #
def sequence_iou(preds: list[np.ndarray], gts: list[np.ndarray]) -> float:
    """Mean per-frame IoU of a predicted mask track against GT."""
    pairs = list(zip(preds, gts))
    return float(np.mean([mask_iou(p, g) for p, g in pairs])) if pairs else 1.0


def temporal_consistency(masks: list[np.ndarray]) -> float:
    """Mean IoU between consecutive frames -- high = stable/low-flicker track.

    Note: this measures temporal stability of the prediction itself, independent
    of GT, so it penalizes flicker even when per-frame GT is unavailable.
    """
    if len(masks) < 2:
        return 1.0
    return float(np.mean([mask_iou(masks[i - 1], masks[i]) for i in range(1, len(masks))]))
