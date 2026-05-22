#!/usr/bin/env python3
"""Prompt-geometry sweep for thin airstrips: box vs box+point vs K centerline points.

Thin diagonal airstrips are poorly covered by a loose bbox prompt. We sample K
positive points along the mask's principal axis (PCA centerline) and compare
prompt strategies on the airstrip test positives. Eval-only (zero-shot SAM2).

Usage:
    python scripts/airstrip_prompt_eval.py --root /home/ubuntu/.cache/airstrip_s1aad
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from defense_lab.datasets.types import mask_to_xyxy  # noqa: E402
from defense_lab.evaluation.metrics import dice_coefficient, mask_iou  # noqa: E402
from defense_lab.prompting.prompts import BoxPrompt, PointPrompt, PromptSet  # noqa: E402
from defense_lab.repro.experiment import Experiment  # noqa: E402
from defense_lab.segmentation.predictor import PromptableSegmenter, SegmenterConfig  # noqa: E402


def centerline_points(mask: np.ndarray, k: int) -> np.ndarray:
    pts = np.argwhere(mask)  # (N,2) [y,x]
    if len(pts) < k:
        k = max(1, len(pts))
    c = pts.mean(0)
    _, _, vt = np.linalg.svd(pts - c, full_matrices=False)
    proj = (pts - c) @ vt[0]
    order = np.argsort(proj)
    idx = np.linspace(0, len(order) - 1, k).astype(int)
    sel = pts[order[idx]]              # (k,2) [y,x]
    return sel[:, ::-1].astype(float)  # -> [x,y]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/home/ubuntu/.cache/airstrip_s1aad")
    ap.add_argument("--name", default="airstrip_prompt_eval")
    args = ap.parse_args()
    root = Path(args.root)
    import cv2

    man = json.load(open(root / "manifest.json"))
    test = [m for m in man if m["split"] == "test" and m["positive"]]

    with Experiment(args.name, config={"root": str(root), "n_test": len(test)}, seed=1234) as exp:
        seg = PromptableSegmenter(SegmenterConfig())
        strategies = ["box", "box+pt", "cl3", "cl5", "cl9"]
        acc = {s: [] for s in strategies}
        for m in test:
            img = np.ascontiguousarray(cv2.imread(str(root / "test" / "images" / m["file"]))[:, :, ::-1])
            msk = cv2.imread(str(root / "test" / "masks" / m["file"]), 0) >= 128
            seg.set_image(img)
            box = mask_to_xyxy(msk)
            cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
            prompts = {
                "box": PromptSet.from_box(*box),
                "box+pt": PromptSet(points=PointPrompt().add(cx, cy, True), box=BoxPrompt(*box)),
            }
            for k in (3, 5, 9):
                pp = PointPrompt()
                for x, y in centerline_points(msk, k):
                    pp.add(float(x), float(y), True)
                prompts[f"cl{k}"] = PromptSet(points=pp)
            for s, ps in prompts.items():
                mask = seg.predict(ps).best()[0]
                acc[s].append(mask_iou(mask, msk))

        rows = [{"strategy": s, "mean_iou": round(float(np.mean(v)), 4), "n": len(v)} for s, v in acc.items()]
        rows.sort(key=lambda r: -r["mean_iou"])
        exp.save_json("prompt_eval.json", {"rows": rows})
        exp.save_summary({"rows": rows})

        print("\n=== Airstrip prompt-geometry sweep (test positives, zero-shot) ===")
        print(f"n_test_pos={len(test)}")
        for r in rows:
            print(f"  {r['strategy']:8s} IoU={r['mean_iou']:.3f}")
        print(f"best: {rows[0]['strategy']} ({rows[0]['mean_iou']:.3f})  vs box ({[r['mean_iou'] for r in rows if r['strategy']=='box'][0]:.3f})")
        print(f"artifacts: {exp.artifacts}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
