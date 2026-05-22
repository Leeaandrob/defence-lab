#!/usr/bin/env python3
"""Demo: SAM2 segments (class-agnostic) -> Claude vision labels each region.

Runs AMG on one frame, sends the top-K numbered regions to Claude (Haiku) in a
single call, prints region->class + cost, and saves a labeled overlay. Proves
the VLM-in-the-loop labeling that turns class-agnostic masks into class-tagged
pseudo-labels.

Usage:
    python scripts/vlm_label_demo.py --frame <path.jpg> --top-k 8 --model haiku --backend cli
"""
from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from defense_lab.annotations.auto_mask import AutoMaskConfig, AutoMaskLabeler  # noqa: E402
from defense_lab.annotations.vlm_labeler import ClaudeVLMLabeler, VLMLabelerConfig  # noqa: E402
from defense_lab.repro.experiment import Experiment  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frame", default=None)
    ap.add_argument("--top-k", type=int, default=8)
    ap.add_argument("--model", default="haiku")
    ap.add_argument("--backend", default="cli", choices=["cli", "api"])
    ap.add_argument("--name", default="vlm_label_demo")
    args = ap.parse_args()

    frame = args.frame or next(iter(sorted(glob.glob("experiments/youtube_farms/*/artifacts/frames/*.jpg"))), None)
    if not frame:
        print("no frame found; pass --frame")
        return 1

    import cv2

    with Experiment(args.name, config={"frame": frame, "top_k": args.top_k, "model": args.model,
                                       "backend": args.backend}, seed=1234) as exp:
        img = np.ascontiguousarray(cv2.imread(frame)[:, :, ::-1])
        exp.logger.info("AMG on %s", frame)
        insts = AutoMaskLabeler(AutoMaskConfig(points_per_side=16)).generate(img)
        exp.logger.info("%d masks; labeling top-%d with %s/%s", len(insts), args.top_k, args.model, args.backend)

        labeler = ClaudeVLMLabeler(VLMLabelerConfig(model=args.model, backend=args.backend, top_k=args.top_k))
        labeled, cost = labeler.label(img, insts)

        # persistent labeled overlay (id:class)
        vis = np.ascontiguousarray(img[:, :, ::-1]).copy()
        rows = []
        for k, ins in enumerate(labeled):
            m = ins.mask.astype(np.uint8)
            cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(vis, cnts, -1, (0, 0, 255), 2)
            ys, xs = np.where(ins.mask)
            cx, cy = int(xs.mean()), int(ys.mean())
            cv2.putText(vis, f"{k}:{ins.category_name}", (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            rows.append({"id": k, "class": ins.category_name, "area": ins.area, "score": round(ins.score or 0, 3)})
        out_png = exp.artifact_path("vlm_labeled.png")
        cv2.imwrite(str(out_png), vis)

        exp.save_json("vlm_labels.json", {"frame": frame, "cost_usd": cost, "labels": rows})
        exp.save_summary({"n_masks": len(insts), "n_labeled": len(labeled), "cost_usd": cost})

        print(f"\n=== VLM labeling ({args.model}/{args.backend}) ===")
        print(f"masks={len(insts)}  labeled top-{len(labeled)}  cost=${cost:.4f}")
        for r in rows:
            print(f"  {r['id']:2d}: {r['class']:12s} area={r['area']}")
        print(f"overlay: {out_png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
