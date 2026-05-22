#!/usr/bin/env python3
"""Phase 5 -- temporal memory & streaming video segmentation.

Prompts two generic moving objects on frame 0 (a drifting disc via a point, a
descending rectangle via a box), then propagates masks across the clip with
SAM2's temporal memory -- no re-prompting. Reports per-object IoU vs known GT
across frames, temporal consistency (flicker), object-presence rate and
streaming FPS, and saves tracked-mask overlays. Content is fully synthetic and
class-agnostic (no person/identity notion).

Usage
-----
    python scripts/phase5_temporal.py
    python scripts/phase5_temporal.py n_frames=48 size=512
    python scripts/phase5_temporal.py video.offload_video_to_cpu=true
"""
from __future__ import annotations

import argparse
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from defense_lab.config.base import build_config, to_dict  # noqa: E402
from defense_lab.datasets.types import Instance, mask_to_xyxy  # noqa: E402
from defense_lab.evaluation.metrics import sequence_iou, temporal_consistency  # noqa: E402
from defense_lab.prompting import PromptSet, TemporalPrompt  # noqa: E402
from defense_lab.repro.experiment import Experiment  # noqa: E402
from defense_lab.temporal.video_segmenter import VideoSegmenter, VideoSegmenterConfig  # noqa: E402
from defense_lab.tracking.tracker import run_tracking  # noqa: E402
from defense_lab.visualization.overlay import render_instances  # noqa: E402


@dataclass
class Phase5Config:
    seed: int = 1234
    size: int = 512
    n_frames: int = 24
    video: VideoSegmenterConfig = field(default_factory=VideoSegmenterConfig)


def synth_moving_clip(size: int, n: int, seed: int):
    """Returns (frames[list HxWx3], gt[obj_id -> list[mask per frame]], prompts info)."""
    import cv2

    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:size, 0:size]
    r = int(size * 0.08)
    rh, rw = int(size * 0.13), int(size * 0.16)
    frames, gt_disc, gt_rect = [], [], []
    disc_centers, rect_tops = [], []
    for i in range(n):
        t = i / max(n - 1, 1)
        bg = (45 + 40 * (xx / size) + 20 * (yy / size)).astype(np.float32)
        img = np.clip(np.stack([bg, bg * 0.9, bg * 0.85], -1) + rng.normal(0, 5, (size, size, 3)), 0, 255).astype(np.uint8)
        # disc drifts left -> right at mid-height
        cx = int(size * (0.18 + 0.64 * t)); cy = int(size * 0.40)
        cv2.circle(img, (cx, cy), r, (235, 228, 215), -1)
        disc = np.linalg.norm(np.stack([xx - cx, yy - cy]), axis=0) <= r
        # rectangle descends top -> bottom at a fixed column
        x0 = int(size * 0.60); y0 = int(size * (0.10 + 0.45 * t))
        x1, y1 = x0 + rw, y0 + rh
        cv2.rectangle(img, (x0, y0), (x1, y1), (40, 90, 200), -1)
        rect = np.zeros((size, size), bool); rect[y0:y1, x0:x1] = True
        img = cv2.GaussianBlur(img, (3, 3), 0)
        frames.append(img); gt_disc.append(disc); gt_rect.append(rect)
        disc_centers.append((cx, cy)); rect_tops.append((x0, y0, x1, y1))
    prompts = {"disc_point": disc_centers[0], "rect_box": rect_tops[0]}
    return frames, {1: gt_disc, 2: gt_rect}, prompts


def main() -> int:
    ap = argparse.ArgumentParser(description="Phase 5 temporal video segmentation")
    ap.add_argument("--config", type=str, default=None)
    ap.add_argument("--name", type=str, default="phase5_temporal")
    ap.add_argument("overrides", nargs="*")
    args = ap.parse_args()
    cfg = build_config(Phase5Config, args.config, list(args.overrides))

    with Experiment(args.name, config=cfg, seed=cfg.seed) as exp:
        import cv2

        frames, gt, prompts = synth_moving_clip(cfg.size, cfg.n_frames, cfg.seed)

        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            for i, f in enumerate(frames):
                cv2.imwrite(str(tmp_dir / f"{i:05d}.jpg"), f[:, :, ::-1])

            seg = VideoSegmenter(cfg.video)
            (px, py) = prompts["disc_point"]
            (bx0, by0, bx1, by1) = prompts["rect_box"]
            temporal_prompts = [
                TemporalPrompt(frame_idx=0, obj_id=1, prompt=PromptSet.from_point(px, py, True)),
                TemporalPrompt(frame_idx=0, obj_id=2, prompt=PromptSet.from_box(bx0, by0, bx1, by1)),
            ]
            exp.logger.info("propagating %d objects over %d frames ...", len(temporal_prompts), cfg.n_frames)
            result, timing = run_tracking(seg, tmp_dir, temporal_prompts)

        # ---- evaluation ----
        names = {1: "disc", 2: "rect"}
        per_obj = {}
        for oid, track in result.tracks.items():
            fi = track.frame_indices
            preds = [track.mask_at(f) for f in fi]
            gts = [gt[oid][f] for f in fi]
            per_obj[names.get(oid, str(oid))] = {
                "mean_iou_vs_gt": round(sequence_iou(preds, gts), 4),
                "temporal_consistency": round(temporal_consistency(preds), 4),
                "presence_rate": round(track.presence_rate(), 4),
                "frames": len(fi),
            }

        results = {"config": to_dict(cfg), "timing": timing, "per_object": per_obj}
        exp.save_json("phase5_results.json", results)
        for oid, m in per_obj.items():
            exp.log_metrics(**{f"{oid}_iou": m["mean_iou_vs_gt"], f"{oid}_tc": m["temporal_consistency"]})
        exp.log_metrics(streaming_fps=timing["streaming_fps"])
        exp.save_summary(results)

        # ---- tracked-mask overlays at start / middle / end ----
        for f in (0, cfg.n_frames // 2, cfg.n_frames - 1):
            insts = [Instance(box=mask_to_xyxy(m), mask=m, obj_id=oid)
                     for oid, m in result.per_frame(f).items()]
            render_instances(frames[f], insts, exp.artifact_path(f"frame_{f:03d}.png"),
                             title=f"frame {f}  ({len(insts)} tracked objects)")

        print("\n=== Phase 5 temporal video segmentation ===")
        for oid, m in per_obj.items():
            print(f"{oid:5s}: IoU vs GT={m['mean_iou_vs_gt']:.3f}  temporal-consistency={m['temporal_consistency']:.3f}  "
                  f"presence={m['presence_rate']:.2f} over {m['frames']} frames")
        print(f"streaming: {timing['streaming_fps']} FPS ({timing['frames']} frames in {timing['propagation_s']}s)")
        print(f"artifacts: {exp.artifacts}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
