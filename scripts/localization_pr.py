#!/usr/bin/env python3
"""Localization operating curve: trade false-positives for recall via score filter.

Runs AMG once per image (permissive) keeping each proposal's predicted-IoU, then
sweeps a confidence threshold and recomputes recall / precision / FP-per-image.
Produces the FP-vs-recall operating curve -- the figure that shows the
low-false-positive localization tradeoff (the project's core bottleneck).

Usage:
    python scripts/localization_pr.py --n 15
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE.parent))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import no_oracle_eval as noe  # noqa: E402
from defense_lab.annotations.auto_mask import AutoMaskConfig, AutoMaskLabeler  # noqa: E402
from defense_lab.repro.experiment import Experiment  # noqa: E402


def collect(gen, amg):
    cache = []
    for img, gts in gen:
        if not gts:
            continue
        props = amg.generate(img)
        cache.append((gts, [(p.mask, float(p.score or 0)) for p in props]))
    return cache


def sweep(cache, thrs):
    rows = []
    for t in thrs:
        TP = FP = FN = 0; IOU = []
        for gts, props in cache:
            masks = [m for m, s in props if s >= t]
            tp, fp, fn, ious = noe.match(masks, gts)
            TP += tp; FP += fp; FN += fn; IOU += ious
        rec = TP / (TP + FN) if TP + FN else 0.0
        prec = TP / (TP + FP) if TP + FP else 0.0
        rows.append({"t": round(t, 2), "recall": round(rec, 3), "precision": round(prec, 3),
                     "fp_img": round(FP / max(len(cache), 1), 1),
                     "iou": round(float(np.mean(IOU)), 3) if IOU else 0.0})
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=15)
    ap.add_argument("--name", default="localization_pr")
    args = ap.parse_args()
    thrs = [0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.93, 0.96, 0.98]

    with Experiment(args.name, config={"n": args.n, "thresholds": thrs}, seed=1234) as exp:
        amg = AutoMaskLabeler(AutoMaskConfig(points_per_side=16, pred_iou_thresh=0.5,
                                             stability_score_thresh=0.7, min_mask_region_area=64))
        specs = [
            ("VDD", noe.vdd_images(args.n)),
            ("satellite", noe.hf_images("saidines12/satellite-imagery-segmentation", args.n)),
            ("morocco", noe.hf_images("tferhan/morocco_satellite_buildings_semantic_segmentation_512_v2",
                                      args.n, image_col="pixel_values", binary_thresh=1)),
        ]
        results = {}
        fig, ax = plt.subplots(figsize=(6.5, 4.6))
        for nm, gen in specs:
            try:
                cache = collect(gen, amg)
                rows = sweep(cache, thrs)
                results[nm] = rows
                ax.plot([r["recall"] for r in rows], [r["fp_img"] for r in rows], marker="o", label=nm)
                exp.logger.info("%s done (%d imgs)", nm, len(cache))
            except Exception as e:
                exp.logger.error("%s: %s", nm, e); results[nm] = {"error": str(e)}
        ax.set_xlabel("recall@0.5"); ax.set_ylabel("false-positives / image")
        ax.set_title("Localization operating curve (AMG score filter)")
        ax.legend(); ax.grid(True, alpha=0.3)
        fig.tight_layout(); fig.savefig(exp.artifact_path("fp_recall_curve.png"), dpi=140); plt.close(fig)

        exp.save_json("localization_pr.json", results)
        exp.save_summary(results)

        print("\n=== Localization operating curve (FP vs recall) ===")
        for nm, rows in results.items():
            if isinstance(rows, dict):
                print(f"  {nm}: ERROR"); continue
            print(f"\n{nm}:")
            print(f"  {'thr':>5} {'recall':>7} {'prec':>6} {'FP/img':>7} {'IoU':>6}")
            for r in rows:
                print(f"  {r['t']:5.2f} {r['recall']:7.3f} {r['precision']:6.3f} {r['fp_img']:7.1f} {r['iou']:6.3f}")
        print(f"\nartifacts: {exp.artifacts}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
