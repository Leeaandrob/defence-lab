#!/usr/bin/env python3
"""Phase 6 -- operational evaluation & benchmarking (the skeleton's closing piece).

Benchmarks segmentation methods on identical samples and emits the operational
metric suite (IoU / Dice / boundary-IoU / latency / FPS / peak memory), a
benchmark table (Markdown + CSV), comparison plots, and a final report. Default
methods: SAM2 (box), SAM2 (point), and a classical GrabCut baseline -- directly
probing the research question "promptable foundation model vs. traditional
pipeline". Dataset-agnostic: runs on synthetic data by default, or a real COCO
set, or any method you plug via CallableMethod.

Usage
-----
    python scripts/phase6_evaluate.py
    python scripts/phase6_evaluate.py n_samples=24
    python scripts/phase6_evaluate.py --coco-ann data/inst.json --coco-images data/imgs
    python scripts/phase6_evaluate.py --lora-adapter experiments/.../lora_adapter.safetensors
"""
from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from defense_lab.config.base import build_config, to_dict  # noqa: E402
from defense_lab.datasets.types import mask_to_xyxy  # noqa: E402
from defense_lab.evaluation.evaluator import EvalSample, compare_methods, report_rows  # noqa: E402
from defense_lab.evaluation.methods import (  # noqa: E402
    ClassicalBoxMethod,
    Sam2BoxMethod,
    Sam2LoraMethod,
    Sam2PointMethod,
)
from defense_lab.repro.experiment import Experiment  # noqa: E402
from defense_lab.segmentation.predictor import PromptableSegmenter, SegmenterConfig  # noqa: E402
from defense_lab.visualization.plots import bar_metric  # noqa: E402


@dataclass
class Phase6Config:
    seed: int = 1234
    size: int = 512
    n_samples: int = 16
    seg: SegmenterConfig = field(default_factory=SegmenterConfig)


def make_eval_sample(rng: np.random.Generator, size: int) -> EvalSample:
    import cv2

    yy, xx = np.mgrid[0:size, 0:size]
    bg = (50 + 45 * (xx / size) + 20 * (yy / size)).astype(np.float32)
    img = np.clip(np.stack([bg, bg * 0.9, bg * 0.85], -1) + rng.normal(0, 8, (size, size, 3)), 0, 255).astype(np.uint8)
    if rng.integers(0, 2) == 0:  # disc
        cx, cy = rng.integers(int(size * 0.3), int(size * 0.7), size=2)
        r = int(rng.integers(int(size * 0.08), int(size * 0.15)))
        cv2.circle(img, (int(cx), int(cy)), r, (225, 215, 200), -1)
        gt = np.linalg.norm(np.stack([xx - cx, yy - cy]), axis=0) <= r
    else:  # rectangle
        x0, y0 = rng.integers(int(size * 0.15), int(size * 0.5), size=2)
        x1 = x0 + int(rng.integers(int(size * 0.18), int(size * 0.3)))
        y1 = y0 + int(rng.integers(int(size * 0.18), int(size * 0.3)))
        cv2.rectangle(img, (int(x0), int(y0)), (int(x1), int(y1)), (50, 95, 200), -1)
        gt = np.zeros((size, size), bool); gt[y0:y1, x0:x1] = True
    img = cv2.GaussianBlur(img, (3, 3), 0)
    x0, y0, x1, y1 = mask_to_xyxy(gt)
    point = ((x0 + x1) / 2, (y0 + y1) / 2)
    return EvalSample(image=img, gt=gt, box=(x0, y0, x1, y1), point=point)


def load_coco_samples(ann: str, images: str, n: int) -> list[EvalSample]:
    from defense_lab.datasets.coco import CocoDataset

    ds = CocoDataset(ann, images)
    out: list[EvalSample] = []
    for i in range(min(n, len(ds))):
        s = ds[i]
        inst = next((x for x in s.instances if x.mask is not None), None)
        if inst is None:
            continue
        img = s.load_image()
        x0, y0, x1, y1 = inst.box
        out.append(EvalSample(img, inst.mask, (x0, y0, x1, y1), ((x0 + x1) / 2, (y0 + y1) / 2), s.image_id))
    return out


def md_table(rows: list[dict]) -> str:
    if not rows:
        return "_(no rows)_\n"
    cols = list(rows[0].keys())
    out = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    return "\n".join(out) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description="Phase 6 evaluation & benchmark")
    ap.add_argument("--config", type=str, default=None)
    ap.add_argument("--name", type=str, default="phase6_evaluate")
    ap.add_argument("--coco-ann", type=str, default=None)
    ap.add_argument("--coco-images", type=str, default=None)
    ap.add_argument("--lora-adapter", type=str, default=None)
    ap.add_argument("overrides", nargs="*")
    args = ap.parse_args()
    cfg = build_config(Phase6Config, args.config, list(args.overrides))

    with Experiment(args.name, config=cfg, seed=cfg.seed) as exp:
        if args.coco_ann and args.coco_images:
            dataset = load_coco_samples(args.coco_ann, args.coco_images, cfg.n_samples)
            source = f"COCO ({args.coco_ann})"
        else:
            rng = np.random.default_rng(cfg.seed)
            dataset = [make_eval_sample(rng, cfg.size) for _ in range(cfg.n_samples)]
            source = f"synthetic (n={cfg.n_samples}, size={cfg.size})"
        exp.logger.info("eval dataset: %s, %d samples", source, len(dataset))

        segmenter = PromptableSegmenter(cfg.seg)  # shared across SAM2 methods
        methods = [Sam2BoxMethod(segmenter), Sam2PointMethod(segmenter), ClassicalBoxMethod()]
        if args.lora_adapter:
            methods.append(Sam2LoraMethod(PromptableSegmenter(cfg.seg), args.lora_adapter))

        exp.logger.info("benchmarking %d methods ...", len(methods))
        reports = compare_methods(methods, dataset)
        rows = report_rows(reports)

        # table (markdown + csv)
        exp.artifact_path("benchmark_table.md").write_text(md_table(rows))
        with open(exp.artifact_path("benchmark_table.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

        # plots
        rep_list = list(reports.values())
        bar_metric(rep_list, "mean_iou", f"Segmentation IoU — {source}", "mean IoU", exp.artifact_path("iou_by_method.png"))
        bar_metric(rep_list, "fps", "Throughput by method", "FPS", exp.artifact_path("fps_by_method.png"))

        # final report
        best = max(rep_list, key=lambda r: r.mean_iou)
        report = ["# Phase 6 — Operational Segmentation Benchmark", "",
                  f"- dataset: **{source}**", f"- best IoU: **{best.name}** ({best.mean_iou})", "",
                  "## Results", "", md_table(rows), "",
                  "## Research question",
                  "_Can SAM2 (promptable, adaptable) match/beat a traditional pipeline?_",
                  f"On this set, SAM2 box-prompting reaches IoU {reports['SAM2 (box prompt)'].mean_iou} "
                  f"vs the classical GrabCut baseline at {reports['Classical (GrabCut+box)'].mean_iou}.",
                  "", "Plug a real dataset (`--coco-ann/--coco-images`) or an external detector "
                  "(`CallableMethod`) to extend the comparison.", ""]
        exp.artifact_path("phase6_report.md").write_text("\n".join(report))

        exp.save_json("phase6_results.json", {"config": to_dict(cfg), "source": source, "rows": rows})
        for r in rep_list:
            exp.log_metrics(**{f"{r.name}_iou": r.mean_iou, f"{r.name}_fps": r.fps})
        exp.save_summary({"source": source, "rows": rows, "best": best.name})

        print("\n=== Phase 6 benchmark ===  source:", source)
        print(md_table(rows))
        print(f"best IoU: {best.name} ({best.mean_iou})")
        print(f"artifacts: {exp.artifacts}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
