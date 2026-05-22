#!/usr/bin/env python3
"""Phase 3 -- SAM-style data engine, end to end.

Demonstrates the three data-engine stages on a synthetic multi-object scene
with known ground truth, then closes the loop:

  1) ASSISTED   : upgrade (jittered) boxes -> masks, measure IoU vs GT.
  2) AUTOMATIC  : class-agnostic AMG masks (open-world), with quality scores.
  3) GATE       : confidence-triage AMG masks into pseudo-labels vs HITL review.
  4) REFINE     : fix a deliberately coarse box with one corrective click (HITL).
  5) EXPORT     : write accepted pseudo-labels as COCO and reload to verify.

Optional real-data ingestion:
    --  data.coco_ann=/path/instances.json data.image_dir=/path/images
    --  data.video=/path/clip.mp4    (extracts frames for Phase-5 streaming)

Usage
-----
    python scripts/phase3_data_engine.py
    python scripts/phase3_data_engine.py auto.points_per_side=24 gate.iou_accept=0.9
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from defense_lab.annotations.auto_mask import AutoMaskConfig, AutoMaskLabeler  # noqa: E402
from defense_lab.annotations.engine import AssistedLabeler  # noqa: E402
from defense_lab.annotations.pseudo_label import PseudoLabelConfig, gate, review_manifest  # noqa: E402
from defense_lab.annotations.refinement import refine  # noqa: E402
from defense_lab.config.base import build_config, to_dict  # noqa: E402
from defense_lab.datasets.coco import CocoDataset, export_coco  # noqa: E402
from defense_lab.datasets.types import Sample  # noqa: E402
from defense_lab.datasets.video import extract_frames  # noqa: E402
from defense_lab.evaluation.metrics import mask_iou  # noqa: E402
from defense_lab.repro.experiment import Experiment  # noqa: E402
from defense_lab.segmentation.predictor import PromptableSegmenter, SegmenterConfig  # noqa: E402
from defense_lab.visualization.overlay import render_instances, render_panels  # noqa: E402


@dataclass
class DataConfig:
    coco_ann: str = ""
    image_dir: str = ""
    video: str = ""
    max_images: int = 3
    frame_stride: int = 10
    max_frames: int = 16


@dataclass
class Phase3Config:
    seed: int = 1234
    size: int = 768
    seg: SegmenterConfig = field(default_factory=SegmenterConfig)
    auto: AutoMaskConfig = field(default_factory=AutoMaskConfig)
    gate: PseudoLabelConfig = field(default_factory=PseudoLabelConfig)
    data: DataConfig = field(default_factory=DataConfig)


def synth_multi_scene(size: int, seed: int):
    """Return (image, gt) where gt = list of (mask, box_xyxy, category_id)."""
    import cv2

    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:size, 0:size]
    base = (50 + 50 * (xx / size) + 25 * (yy / size)).astype(np.float32)
    img = np.clip(np.stack([base, base * 0.9, base * 0.85], -1) + rng.normal(0, 5, (size, size, 3)), 0, 255).astype(np.uint8)

    specs = [
        ("disc", (int(size * 0.22), int(size * 0.30)), int(size * 0.09), (235, 225, 210), 1),
        ("disc", (int(size * 0.70), int(size * 0.68)), int(size * 0.10), (210, 235, 215), 1),
        ("rect", (int(size * 0.55), int(size * 0.15), int(size * 0.78), int(size * 0.40)), None, (40, 90, 200), 2),
        ("rect", (int(size * 0.12), int(size * 0.62), int(size * 0.34), int(size * 0.86)), None, (200, 80, 60), 2),
    ]
    gt = []
    for kind, geom, r, color, cid in specs:
        if kind == "disc":
            cx, cy = geom
            cv2.circle(img, (cx, cy), r, color, -1)
            mask = np.linalg.norm(np.stack([xx - cx, yy - cy]), axis=0) <= r
            box = (cx - r, cy - r, cx + r, cy + r)
        else:
            x0, y0, x1, y1 = geom
            cv2.rectangle(img, (x0, y0), (x1, y1), color, -1)
            mask = np.zeros((size, size), bool)
            mask[y0:y1, x0:x1] = True
            box = (x0, y0, x1, y1)
        gt.append((mask, tuple(map(float, box)), cid))
    img = cv2.GaussianBlur(img, (3, 3), 0)
    return img, gt


def main() -> int:
    ap = argparse.ArgumentParser(description="Phase 3 data engine")
    ap.add_argument("--config", type=str, default=None)
    ap.add_argument("--name", type=str, default="phase3_data_engine")
    ap.add_argument("overrides", nargs="*")
    args = ap.parse_args()
    cfg = build_config(Phase3Config, args.config, list(args.overrides))

    with Experiment(args.name, config=cfg, seed=cfg.seed) as exp:
        import cv2

        image, gt = synth_multi_scene(cfg.size, cfg.seed)
        img_area = cfg.size * cfg.size
        rng = np.random.default_rng(cfg.seed)
        results: dict = {}

        seg = PromptableSegmenter(cfg.seg)
        assisted = AssistedLabeler(seg)

        # ---- stage 1: assisted (boxes -> masks) ----
        jittered = [tuple(np.array(b) + rng.uniform(-6, 6, 4)) for (_, b, _) in gt]
        cats = [c for (_, _, c) in gt]
        a_inst = assisted.label_from_boxes(image, jittered, category_ids=cats)
        ious = [mask_iou(a_inst[i].mask, gt[i][0]) for i in range(len(gt))]
        results["assisted"] = {"n": len(a_inst), "mean_iou_vs_gt": round(float(np.mean(ious)), 4),
                               "per_object_iou": [round(x, 4) for x in ious]}
        render_instances(image, a_inst, exp.artifact_path("stage1_assisted.png"),
                         title=f"assisted (box→mask)  mean IoU={results['assisted']['mean_iou_vs_gt']:.3f}")

        # ---- stage 2: automatic (open-world AMG) ----
        amg = AutoMaskLabeler(cfg.auto)
        auto_inst = amg.generate(image)
        results["automatic"] = {
            "n_masks": len(auto_inst),
            "mean_pred_iou": round(float(np.mean([i.score for i in auto_inst])), 4) if auto_inst else None,
            "mean_stability": round(float(np.mean([i.stability for i in auto_inst])), 4) if auto_inst else None,
        }
        render_instances(image, auto_inst, exp.artifact_path("stage2_automatic.png"),
                         title=f"automatic (class-agnostic AMG)  {len(auto_inst)} masks")

        # ---- stage 3: confidence gate -> pseudo-labels vs review ----
        gres = gate(auto_inst, cfg.gate, img_area)
        results["gate"] = gres.stats(img_area)
        exp.save_json("review_manifest.json", review_manifest(gres, img_area))

        # ---- stage 4: HITL refinement of a coarse box ----
        gt_mask0, gt_box0, _ = gt[0]
        cx = (gt_box0[0] + gt_box0[2]) / 2
        cy = (gt_box0[1] + gt_box0[3]) / 2
        coarse = (gt_box0[0] + 25, gt_box0[1] + 25, gt_box0[2] - 5, gt_box0[3] - 5)  # too-tight, offset
        coarse_inst = assisted.label_from_boxes(image, [coarse])[0]
        iou_before = mask_iou(coarse_inst.mask, gt_mask0)
        refined, step_scores = refine(seg, image, box=coarse, corrections=[(cx, cy, True)])
        iou_after = mask_iou(refined.mask, gt_mask0)
        results["refine"] = {"iou_before": round(iou_before, 4), "iou_after": round(iou_after, 4),
                             "step_scores": [round(s, 4) for s in step_scores]}
        render_panels(image, [coarse_inst.mask, refined.mask],
                      [f"coarse box  IoU={iou_before:.3f}", f"+1 click  IoU={iou_after:.3f}"],
                      exp.artifact_path("stage4_refine.png"))

        # ---- stage 5: export accepted pseudo-labels to COCO + reload ----
        img_file = "synth_scene.jpg"
        cv2.imwrite(str(exp.artifacts / img_file), image[:, :, ::-1])
        sample = Sample(image_id=1, height=cfg.size, width=cfg.size, file_name=img_file,
                        image_dir=str(exp.artifacts), instances=gres.accepted or a_inst)
        coco_path = export_coco([sample], exp.artifacts / "pseudo_labels.coco.json",
                                categories={1: "disc", 2: "rect"})
        reloaded = CocoDataset(coco_path, exp.artifacts)
        results["export"] = {"coco_json": str(coco_path), "n_images": len(reloaded),
                             "n_instances": len(reloaded[0].instances)}

        # ---- optional: ingest a real COCO set ----
        if cfg.data.coco_ann and cfg.data.image_dir:
            ds = CocoDataset(cfg.data.coco_ann, cfg.data.image_dir)
            real = []
            for i in range(min(cfg.data.max_images, len(ds))):
                s = ds[i]
                boxes = [ins.box for ins in s.instances]
                masks = assisted.label_from_boxes(s.load_image(), boxes) if boxes else []
                real.append({"image_id": s.image_id, "n_gt": len(s.instances), "n_relabeled": len(masks)})
            results["real_coco"] = {"n_categories": len(ds.categories), "samples": real}

        # ---- optional: extract video frames ----
        if cfg.data.video:
            frames = extract_frames(cfg.data.video, exp.artifacts / "frames",
                                    stride=cfg.data.frame_stride, max_frames=cfg.data.max_frames)
            results["video"] = {"n_frames_extracted": len(frames), "frames_dir": str(exp.artifacts / "frames")}

        exp.save_json("phase3_results.json", {"config": to_dict(cfg), "results": results})
        exp.log_metrics(assisted_iou=results["assisted"]["mean_iou_vs_gt"],
                        auto_masks=results["automatic"]["n_masks"],
                        accept_rate=results["gate"]["accept_rate"])
        exp.save_summary(results)

        print("\n=== Phase 3 data engine ===")
        print(f"assisted : {results['assisted']['n']} objs, mean IoU vs GT={results['assisted']['mean_iou_vs_gt']:.3f}")
        print(f"automatic: {results['automatic']['n_masks']} class-agnostic masks "
              f"(mean pred-IoU={results['automatic']['mean_pred_iou']}, stability={results['automatic']['mean_stability']})")
        print(f"gate     : accept {results['gate']['n_accepted']}/{results['gate']['n_total']} "
              f"(rate={results['gate']['accept_rate']:.2f}) → {results['gate']['n_review']} to HITL review")
        print(f"refine   : IoU {results['refine']['iou_before']:.3f} → {results['refine']['iou_after']:.3f} (+1 click)")
        print(f"export   : COCO reload OK — {results['export']['n_instances']} instances")
        print(f"artifacts: {exp.artifacts}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
