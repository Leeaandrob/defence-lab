#!/usr/bin/env python3
"""Phase 2 demo -- promptable segmentation: point, box, and interactive refine.

Builds a synthetic operational-style scene with two objects of *known* ground
truth (a bright disc and a rectangle), then segments them zero-shot with:
  1) a single foreground point,
  2) a bounding box,
  3) an interactive 2-click refinement loop,
reporting IoU/Dice against ground truth and saving annotated overlays. This
validates the encode-once / decode-many engine and the typed prompt contract.

Usage
-----
    python scripts/phase2_promptable_demo.py
    python scripts/phase2_promptable_demo.py seg.checkpoint=checkpoints/sam2.1_hiera_large.pt \
        seg.model_cfg=configs/sam2.1/sam2.1_hiera_l.yaml
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from defense_lab.config.base import build_config, to_dict  # noqa: E402
from defense_lab.evaluation.metrics import dice_coefficient, mask_iou  # noqa: E402
from defense_lab.prompting import PromptSet  # noqa: E402
from defense_lab.repro.experiment import Experiment  # noqa: E402
from defense_lab.segmentation import (  # noqa: E402
    InteractiveSession,
    PromptableSegmenter,
    SegmenterConfig,
)
from defense_lab.visualization.overlay import render_panels, render_result  # noqa: E402


@dataclass
class Phase2Config:
    seed: int = 1234
    size: int = 1024
    seg: SegmenterConfig = field(default_factory=SegmenterConfig)


def synth_scene(size: int, seed: int) -> tuple[np.ndarray, dict[str, np.ndarray], dict]:
    """Textured background + a bright disc and a rectangle. Returns (image, gt_masks, geom)."""
    import cv2

    rng = np.random.default_rng(seed)
    # smooth gradient background + mild noise (operational-ish, low contrast)
    yy, xx = np.mgrid[0:size, 0:size]
    base = (60 + 60 * (xx / size) + 30 * (yy / size)).astype(np.float32)
    img = np.stack([base, base * 0.9, base * 0.8], axis=-1)
    img += rng.normal(0, 6, img.shape)
    img = np.clip(img, 0, 255).astype(np.uint8)

    disc_c, disc_r = (int(size * 0.37), int(size * 0.52)), int(size * 0.11)
    rect_p0, rect_p1 = (int(size * 0.62), int(size * 0.29)), (int(size * 0.84), int(size * 0.55))

    cv2.circle(img, disc_c, disc_r, (235, 230, 220), thickness=-1)
    cv2.rectangle(img, rect_p0, rect_p1, (40, 90, 200), thickness=-1)
    img = cv2.GaussianBlur(img, (3, 3), 0)

    disc_gt = np.linalg.norm(np.stack([xx - disc_c[0], yy - disc_c[1]]), axis=0) <= disc_r
    rect_gt = np.zeros((size, size), bool)
    rect_gt[rect_p0[1]:rect_p1[1], rect_p0[0]:rect_p1[0]] = True

    geom = {"disc_c": disc_c, "disc_r": disc_r, "rect_p0": rect_p0, "rect_p1": rect_p1}
    return img, {"disc": disc_gt, "rect": rect_gt}, geom


def main() -> int:
    ap = argparse.ArgumentParser(description="Phase 2 promptable segmentation demo")
    ap.add_argument("--config", type=str, default=None)
    ap.add_argument("--name", type=str, default="phase2_promptable_demo")
    ap.add_argument("overrides", nargs="*")
    args = ap.parse_args()

    cfg = build_config(Phase2Config, args.config, list(args.overrides))

    with Experiment(args.name, config=cfg, seed=cfg.seed) as exp:
        image, gt, geom = synth_scene(cfg.size, cfg.seed)
        seg = PromptableSegmenter(cfg.seg)
        seg.set_image(image)
        exp.logger.info("image encoded; running prompt modes ...")

        results: dict = {}

        # 1) point prompt on the disc
        cx, cy = geom["disc_c"]
        pt = PromptSet.from_point(cx, cy, foreground=True)
        r_pt = seg.predict(pt)
        m_pt, s_pt, _ = r_pt.best()
        results["point_disc"] = {"iou": mask_iou(m_pt, gt["disc"]), "dice": dice_coefficient(m_pt, gt["disc"]), "score": s_pt}
        render_result(image, m_pt, exp.artifact_path("point_disc.png"), pt,
                      title=f"point → disc  IoU={results['point_disc']['iou']:.3f}")

        # 2) box prompt on the rectangle
        (rx0, ry0), (rx1, ry1) = geom["rect_p0"], geom["rect_p1"]
        bx = PromptSet.from_box(rx0 - 8, ry0 - 8, rx1 + 8, ry1 + 8)
        r_bx = seg.predict(bx)
        m_bx, s_bx, _ = r_bx.best()
        results["box_rect"] = {"iou": mask_iou(m_bx, gt["rect"]), "dice": dice_coefficient(m_bx, gt["rect"]), "score": s_bx}
        render_result(image, m_bx, exp.artifact_path("box_rect.png"), bx,
                      title=f"box → rect  IoU={results['box_rect']['iou']:.3f}")

        # 3) interactive refinement on the disc (center click, then edge correction)
        sess = InteractiveSession(seg)  # image already encoded on `seg`
        r1 = sess.add_point(cx, cy, True)
        iou1 = mask_iou(r1.best()[0], gt["disc"])
        r2 = sess.add_point(cx + geom["disc_r"] - 6, cy, True)  # near the boundary
        iou2 = mask_iou(r2.best()[0], gt["disc"])
        results["interactive_disc"] = {"iou_1click": iou1, "iou_2click": iou2, "num_clicks": sess.num_clicks}
        render_panels(
            image,
            [r1.best()[0], r2.best()[0]],
            [f"1 click  IoU={iou1:.3f}", f"2 clicks  IoU={iou2:.3f}"],
            exp.artifact_path("interactive_disc.png"),
            prompts=[PromptSet.from_point(cx, cy), None],
        )

        exp.save_json("phase2_results.json", {"config": to_dict(cfg), "results": results})
        exp.log_metrics(point_iou=results["point_disc"]["iou"], box_iou=results["box_rect"]["iou"])
        exp.save_summary(results)

        print("\n=== Phase 2 promptable segmentation ===")
        print(f"point→disc : IoU={results['point_disc']['iou']:.3f}  Dice={results['point_disc']['dice']:.3f}")
        print(f"box→rect   : IoU={results['box_rect']['iou']:.3f}  Dice={results['box_rect']['dice']:.3f}")
        print(f"interactive: 1-click IoU={iou1:.3f} → 2-click IoU={iou2:.3f}")
        print(f"overlays   : {exp.artifacts}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
