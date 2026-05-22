#!/usr/bin/env python3
"""Benchmark SAM2 prompted segmentation on a real Hugging Face overhead-imagery
segmentation dataset (neutral land-cover / building / road structures).

For each tile we take the largest foreground component as the target, prompt
SAM2 (box, point, box+point) and a classical GrabCut baseline, and report
IoU / Dice / boundary-IoU / FPS / memory on real imagery. Honest numbers --
prompted-segmentation quality, not full semantic accuracy.

Usage
-----
    python scripts/benchmark_hf_dataset.py
    python scripts/benchmark_hf_dataset.py --repo saidines12/satellite-imagery-segmentation --n 40 --split train
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from defense_lab.config.base import to_dict  # noqa: E402
from defense_lab.datasets.hf_seg import load_hf_eval_samples  # noqa: E402
from defense_lab.evaluation.evaluator import compare_methods, report_rows  # noqa: E402
from defense_lab.evaluation.methods import (  # noqa: E402
    ClassicalBoxMethod,
    Sam2BoxMethod,
    Sam2BoxPointMethod,
    Sam2PointMethod,
)
from defense_lab.repro.experiment import Experiment  # noqa: E402
from defense_lab.segmentation.predictor import PromptableSegmenter, SegmenterConfig  # noqa: E402
from defense_lab.visualization.overlay import render_panels  # noqa: E402
from defense_lab.visualization.plots import bar_metric  # noqa: E402


def md_table(rows: list[dict]) -> str:
    if not rows:
        return "_(no rows)_\n"
    cols = list(rows[0].keys())
    out = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    return "\n".join(out) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description="Benchmark SAM2 on a real HF segmentation dataset")
    ap.add_argument("--repo", default="saidines12/satellite-imagery-segmentation")
    ap.add_argument("--split", default="train")
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--name", default="hf_satellite_benchmark")
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--model-cfg", default=None)
    args = ap.parse_args()

    seg_cfg = SegmenterConfig()
    if args.checkpoint:
        seg_cfg.checkpoint = args.checkpoint
    if args.model_cfg:
        seg_cfg.model_cfg = args.model_cfg

    with Experiment(args.name, config={"repo": args.repo, "split": args.split, "n": args.n,
                                       "seg": to_dict(seg_cfg)}, seed=1234) as exp:
        exp.logger.info("loading up to %d samples from %s [%s] ...", args.n, args.repo, args.split)
        dataset = load_hf_eval_samples(args.repo, split=args.split, n=args.n)
        if not dataset:
            exp.logger.error("no usable samples (no foreground components found)")
            return 1
        exp.logger.info("loaded %d real samples", len(dataset))

        segmenter = PromptableSegmenter(seg_cfg)
        methods = [
            Sam2BoxMethod(segmenter),
            Sam2BoxPointMethod(segmenter),
            Sam2PointMethod(segmenter),
            ClassicalBoxMethod(),
        ]
        reports = compare_methods(methods, dataset)
        rows = report_rows(reports)

        exp.artifact_path("benchmark_table.md").write_text(md_table(rows))
        with open(exp.artifact_path("benchmark_table.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)
        rep_list = list(reports.values())
        bar_metric(rep_list, "mean_iou", f"IoU on {args.repo}", "mean IoU", exp.artifact_path("iou_by_method.png"))
        bar_metric(rep_list, "fps", "Throughput", "FPS", exp.artifact_path("fps_by_method.png"))

        # qualitative panels for first 3 samples: image | GT | best SAM2 prediction
        best_method = max(rep_list, key=lambda r: r.mean_iou)
        bm = next(m for m in methods if m.name == best_method.name)
        for i in range(min(3, len(dataset))):
            s = dataset[i]
            pred = bm.predict(s)
            render_panels(s.image, [s.gt, pred], ["ground truth", f"{best_method.name}"],
                          exp.artifact_path(f"sample_{i}.png"))

        exp.save_json("results.json", {"repo": args.repo, "split": args.split, "n": len(dataset), "rows": rows})
        exp.save_summary({"repo": args.repo, "n": len(dataset), "rows": rows, "best": best_method.name})

        print(f"\n=== SAM2 prompted segmentation on {args.repo} ({len(dataset)} real tiles) ===")
        print(md_table(rows))
        print(f"best IoU: {best_method.name} ({best_method.mean_iou})")
        print(f"artifacts: {exp.artifacts}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
