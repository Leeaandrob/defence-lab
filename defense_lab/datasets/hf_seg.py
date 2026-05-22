"""Adapter: build EvalSamples from a Hugging Face image+mask segmentation dataset.

Generic over any HF dataset exposing an image column and a mask/label column
(binary or class-index). For each sample we take the largest connected component
of the foreground as the target structure, deriving a box + centroid-point
prompt and the component mask as ground truth -- i.e. a promptable-segmentation
evaluation on real overhead imagery. Neutral land-cover / building / road
structures only.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from defense_lab.evaluation.evaluator import EvalSample


def _largest_component(mask_bool: np.ndarray):
    import cv2

    n, labels, stats, cents = cv2.connectedComponentsWithStats(
        mask_bool.astype(np.uint8), connectivity=8
    )
    if n <= 1:
        return None
    areas = stats[1:, cv2.CC_STAT_AREA]
    idx = 1 + int(np.argmax(areas))
    comp = labels == idx
    x, y = stats[idx, cv2.CC_STAT_LEFT], stats[idx, cv2.CC_STAT_TOP]
    w, h = stats[idx, cv2.CC_STAT_WIDTH], stats[idx, cv2.CC_STAT_HEIGHT]
    cx, cy = cents[idx]
    return comp, (float(x), float(y), float(x + w), float(y + h)), (float(cx), float(cy))


def load_hf_eval_samples(
    repo: str,
    split: str = "train",
    n: int = 40,
    image_col: str = "image",
    label_col: str = "label",
    foreground_threshold: int = 128,
    min_area: int = 400,
    seed: int = 1234,
) -> list[EvalSample]:
    """Sample up to ``n`` EvalSamples whose target is the largest foreground
    component (box + centroid prompt). Skips empty / tiny masks."""
    from datasets import load_dataset

    ds = load_dataset(repo, split=split)
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(ds))
    out: list[EvalSample] = []
    for i in order:
        s = ds[int(i)]
        img = np.ascontiguousarray(np.array(s[image_col].convert("RGB")))
        lab = np.array(s[label_col].convert("L"))
        fg = lab >= foreground_threshold
        comp = _largest_component(fg)
        if comp is None:
            continue
        mask, box, point = comp
        if int(mask.sum()) < min_area:
            continue
        out.append(EvalSample(image=img, gt=mask, box=box, point=point, sample_id=int(i)))
        if len(out) >= n:
            break
    return out
