#!/usr/bin/env python3
"""Benchmark SAM2 prompted segmentation on VDD (Varied Drone Dataset), real drone
imagery, neutral land-cover classes. Overall method comparison + per-class IoU.

Usage:
    python scripts/benchmark_vdd.py --root /home/ubuntu/.cache/vdd/VDD --n 20
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from defense_lab.datasets.vdd import VDD_CLASSES, load_vdd_eval_samples  # noqa: E402
from defense_lab.evaluation.evaluator import compare_methods, report_rows  # noqa: E402
from defense_lab.evaluation.methods import (  # noqa: E402
    ClassicalBoxMethod, Sam2BoxMethod, Sam2BoxPointMethod, Sam2PointMethod,
)
from defense_lab.evaluation.metrics import mask_iou  # noqa: E402
from defense_lab.repro.experiment import Experiment  # noqa: E402
from defense_lab.segmentation.predictor import PromptableSegmenter, SegmenterConfig  # noqa: E402
from defense_lab.visualization.overlay import render_panels  # noqa: E402
from defense_lab.visualization.plots import bar_metric  # noqa: E402


def md_table(rows):
    cols = list(rows[0].keys())
    out = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    return "\n".join(out) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/home/ubuntu/.cache/vdd/VDD")
    ap.add_argument("--split", default="val")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--name", default="vdd_benchmark")
    args = ap.parse_args()

    with Experiment(args.name, config={"root": args.root, "split": args.split, "n": args.n}, seed=1234) as exp:
        dataset = load_vdd_eval_samples(args.root, split=args.split, n_images=args.n)
        exp.logger.info("loaded %d class-region samples from %d images", len(dataset), args.n)
        if not dataset:
            exp.logger.error("no samples"); return 1

        seg = PromptableSegmenter(SegmenterConfig())
        methods = [Sam2BoxMethod(seg), Sam2BoxPointMethod(seg), Sam2PointMethod(seg), ClassicalBoxMethod()]
        reports = compare_methods(methods, dataset)
        rows = report_rows(reports)
        rep_list = list(reports.values())
        best = max(rep_list, key=lambda r: r.mean_iou)
        bm = next(m for m in methods if m.name == best.name)

        # per-class IoU for the best method
        by_cls = defaultdict(list)
        for s in dataset:
            by_cls[s.sample_id[1]].append(s)
        per_class = []
        for cid in sorted(by_cls):
            ious = [mask_iou(bm.predict(s), s.gt) for s in by_cls[cid]]
            per_class.append({"class": f"{cid}:{VDD_CLASSES.get(cid, '?')}",
                              "n": len(ious), "mean_iou": round(float(np.mean(ious)), 4)})

        exp.artifact_path("methods_table.md").write_text(md_table(rows))
        exp.artifact_path("per_class.md").write_text(md_table(per_class))
        with open(exp.artifact_path("methods_table.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
        bar_metric(rep_list, "mean_iou", "VDD — IoU by method", "mean IoU", exp.artifact_path("iou_by_method.png"))
        for i in range(min(3, len(dataset))):
            s = dataset[i]
            render_panels(s.image, [s.gt, bm.predict(s)],
                          ["ground truth", f"{best.name} ({VDD_CLASSES.get(s.sample_id[1],'?')})"],
                          exp.artifact_path(f"sample_{i}.png"))
        exp.save_json("results.json", {"methods": rows, "per_class": per_class, "n": len(dataset)})
        exp.save_summary({"methods": rows, "per_class": per_class, "best": best.name})

        print(f"\n=== VDD prompted segmentation ({len(dataset)} class-regions / {args.n} drone images) ===")
        print(md_table(rows))
        print("per-class IoU (best = %s):" % best.name)
        print(md_table(per_class))
        print(f"artifacts: {exp.artifacts}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
