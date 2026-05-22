#!/usr/bin/env python3
"""Headline land-use segmentation experiment (SIGE deliverable).

Consolidates, on the SAME VDD val protocol: SAM2 zero-shot (box) vs Classical
(GrabCut) baseline vs SAM2+LoRA (3-seed mean +/- std). Emits a paper-ready table
+ figures (headline bars with error bar; per-class before/after). LoRA seeds are
read from prior runs; zero-shot + classical are evaluated fresh here.

Usage:
    python scripts/landuse_experiment.py
"""
from __future__ import annotations

import glob
import json
import statistics as st
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from defense_lab.datasets.vdd import load_vdd_eval_samples  # noqa: E402
from defense_lab.evaluation.evaluator import compare_methods, report_rows  # noqa: E402
from defense_lab.evaluation.methods import ClassicalBoxMethod, Sam2BoxMethod  # noqa: E402
from defense_lab.repro.experiment import Experiment  # noqa: E402
from defense_lab.segmentation.predictor import PromptableSegmenter, SegmenterConfig  # noqa: E402


def _lora_seeds():
    afters, before = [], None
    for pat in ("lora_vdd_full", "lora_vdd_full_seed11", "lora_vdd_full_seed22"):
        for f in glob.glob(f"experiments/{pat}/*/metrics_summary.json"):
            s = json.load(open(f))
            afters.append(s["val_iou_after"]); before = s["val_iou_before"]
    return before, afters


def _per_class():
    for f in glob.glob("experiments/lora_vdd_full/*/artifacts/lora_vdd_results.json"):
        r = json.load(open(f))
        return r.get("per_class_before", {}), r.get("per_class_after", {})
    return {}, {}


def main() -> int:
    with Experiment("landuse_experiment", config={"dataset": "VDD val", "task": "land-use parcel seg"}, seed=1234) as exp:
        val = load_vdd_eval_samples("/home/ubuntu/.cache/vdd/VDD", "val", n_images=20, seed=1234)
        exp.logger.info("val regions=%d", len(val))
        seg = PromptableSegmenter(SegmenterConfig())
        reports = compare_methods([Sam2BoxMethod(seg), ClassicalBoxMethod()], val)
        zs = reports["SAM2 (box prompt)"]; cl = reports["Classical (GrabCut+box)"]

        b, afters = _lora_seeds()
        lora_mean, lora_std = st.mean(afters), st.pstdev(afters)
        pc_b, pc_a = _per_class()

        table = [
            {"method": "Classical (GrabCut+box)", "IoU": cl.mean_iou, "FPS": cl.fps, "trainable_params": 0},
            {"method": "SAM2 zero-shot (box)", "IoU": zs.mean_iou, "FPS": zs.fps, "trainable_params": 0},
            {"method": f"SAM2 + LoRA (box, 3-seed)", "IoU": round(lora_mean, 4),
             "IoU_std": round(lora_std, 4), "FPS": zs.fps, "trainable_params": 167936},
        ]

        # headline figure
        labels = ["Classical\nGrabCut", "SAM2\nzero-shot", "SAM2+LoRA\n(3-seed)"]
        vals = [cl.mean_iou, zs.mean_iou, lora_mean]
        errs = [0, 0, lora_std]
        fig, ax = plt.subplots(figsize=(6, 4.5))
        bars = ax.bar(labels, vals, yerr=errs, capsize=6, color=["#9e9e9e", "#42a5f5", "#1565c0"])
        ax.set_ylabel("mean IoU (VDD val)"); ax.set_title("Land-use segmentation: SAM2+LoRA vs baselines")
        ax.set_ylim(0, max(vals) * 1.25)
        for bbar, v in zip(bars, vals):
            ax.text(bbar.get_x() + bbar.get_width() / 2, v, f"{v:.3f}", ha="center", va="bottom", fontsize=10)
        fig.tight_layout(); fig.savefig(exp.artifact_path("headline_iou.png"), dpi=140); plt.close(fig)

        # per-class before/after
        if pc_b and pc_a:
            classes = [c for c in pc_a if c in pc_b]
            x = np.arange(len(classes)); w = 0.38
            fig, ax = plt.subplots(figsize=(7, 4.2))
            ax.bar(x - w / 2, [pc_b[c] for c in classes], w, label="zero-shot", color="#90caf9")
            ax.bar(x + w / 2, [pc_a[c] for c in classes], w, label="+LoRA", color="#1565c0")
            ax.set_xticks(x); ax.set_xticklabels(classes, rotation=20, ha="right")
            ax.set_ylabel("IoU"); ax.set_title("VDD per-class: zero-shot vs SAM2+LoRA"); ax.legend()
            fig.tight_layout(); fig.savefig(exp.artifact_path("per_class.png"), dpi=140); plt.close(fig)

        results = {"val_regions": len(val), "table": table,
                   "lora": {"before": b, "mean": round(lora_mean, 4), "std": round(lora_std, 4),
                            "seeds": afters, "lift": round(lora_mean - b, 4)},
                   "per_class_before": pc_b, "per_class_after": pc_a}
        exp.save_json("landuse_results.json", results)
        exp.save_summary(results)

        print("\n=== Headline land-use experiment (VDD val) ===")
        print(f"{'method':28s} {'IoU':>8} {'FPS':>7} {'params':>9}")
        print(f"{'Classical GrabCut':28s} {cl.mean_iou:8.3f} {cl.fps:7.1f} {0:>9}")
        print(f"{'SAM2 zero-shot (box)':28s} {zs.mean_iou:8.3f} {zs.fps:7.1f} {0:>9}")
        print(f"{'SAM2 + LoRA (3-seed)':28s} {lora_mean:8.3f} {zs.fps:7.1f} {'167,936':>9}  (±{lora_std:.3f})")
        print(f"LoRA lift over zero-shot: +{lora_mean - b:.3f} ± {lora_std:.3f}  (0.21% trainable params)")
        print(f"artifacts: {exp.artifacts}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
