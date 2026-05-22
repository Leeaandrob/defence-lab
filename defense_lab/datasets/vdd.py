"""VDD (Varied Drone Dataset) ingester -> EvalSamples for prompted segmentation.

VDD layout: ``VDD/{split}/src/*.JPG`` (drone RGB) + ``VDD/{split}/gt/*.png``
(single-channel class-index masks, 7 land-cover classes). For each image we take
the largest connected component of each present class as a target structure,
deriving a box + centroid-point prompt and the component mask as ground truth.
Neutral land-cover classes only (no person/identity notion).
"""
from __future__ import annotations

import pathlib
from typing import Optional

import numpy as np

from defense_lab.datasets.hf_seg import _largest_component
from defense_lab.evaluation.evaluator import EvalSample

# Class order per the VDD paper (Cai et al., 2023). Index 0 is background/clutter.
VDD_CLASSES = {0: "other", 1: "wall", 2: "road", 3: "vegetation", 4: "vehicle", 5: "roof", 6: "water"}


def _resize(img: np.ndarray, mask: np.ndarray, long_side: int):
    import cv2

    h, w = mask.shape
    s = long_side / max(h, w)
    if s >= 1.0:
        return img, mask
    nw, nh = int(round(w * s)), int(round(h * s))
    img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    mask = cv2.resize(mask, (nw, nh), interpolation=cv2.INTER_NEAREST)
    return img, mask


def load_vdd_eval_samples(
    root: str,
    split: str = "val",
    n_images: int = 20,
    ignore_index: int = 0,
    min_area: int = 1500,
    max_per_image: int = 4,
    long_side: int = 1024,
    seed: int = 1234,
) -> list[EvalSample]:
    from PIL import Image

    root = pathlib.Path(root)
    src_dir, gt_dir = root / split / "src", root / split / "gt"
    stems = sorted(p.stem for p in gt_dir.glob("*.png"))
    rng = np.random.default_rng(seed)
    rng.shuffle(stems)

    out: list[EvalSample] = []
    for stem in stems[:n_images]:
        gp = gt_dir / f"{stem}.png"
        srcs = list(src_dir.glob(stem + ".*"))
        if not srcs:
            continue
        m = np.array(Image.open(gp))
        im = np.array(Image.open(srcs[0]).convert("RGB"))
        if m.ndim == 3:
            m = m[..., 0]
        im, m = _resize(im, m, long_side)
        added = 0
        for c in (int(v) for v in np.unique(m) if int(v) != ignore_index):
            comp = _largest_component(m == c)
            if comp is None:
                continue
            mask, box, point = comp
            if int(mask.sum()) < min_area:
                continue
            out.append(EvalSample(image=np.ascontiguousarray(im), gt=mask, box=box, point=point,
                                  sample_id=(stem, c)))
            added += 1
            if added >= max_per_image:
                break
    return out
