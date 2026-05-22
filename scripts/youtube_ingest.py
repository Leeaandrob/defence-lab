#!/usr/bin/env python3
"""YouTube -> frames -> SAM2 masks -> (optional) Claude-vision class labels -> COCO.

Data-engine, CC-only. Downloads a *Creative-Commons* video section (aborts
otherwise), extracts frames, runs SAM2 automatic mask generation, gates by
confidence, optionally labels each region's semantic class with Claude vision
(`--vlm-label`), and exports a class-tagged COCO. Pseudo-labels = weak
supervision / LoRA-warmup material, NOT verified benchmark ground truth.
Neutral scenes only.

Usage:
    python scripts/youtube_ingest.py --url https://youtu.be/8Ds1LfZHzeA \
        --section "*0:30-0:45" --stride 30 --max-frames 4 --vlm-label
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from defense_lab.annotations.auto_mask import AutoMaskConfig, AutoMaskLabeler  # noqa: E402
from defense_lab.annotations.pseudo_label import PseudoLabelConfig, gate  # noqa: E402
from defense_lab.annotations.vlm_labeler import ClaudeVLMLabeler, VLMLabelerConfig  # noqa: E402
from defense_lab.datasets.coco import export_coco  # noqa: E402
from defense_lab.datasets.types import Sample  # noqa: E402
from defense_lab.datasets.video import extract_frames  # noqa: E402
from defense_lab.repro.experiment import Experiment  # noqa: E402
from defense_lab.visualization.overlay import render_instances  # noqa: E402


def _license(url: str) -> str:
    out = subprocess.run(["yt-dlp", url, "--skip-download", "--no-warnings", "--print", "%(license)s"],
                         capture_output=True, text=True, timeout=120)
    return out.stdout.strip()


def _download(url: str, section: str, dest: Path) -> Path:
    cmd = ["yt-dlp", "-f", "bv*[height<=720]+ba/b[height<=720]/b[height<=720]/b",
           "-o", str(dest / "vid.%(ext)s"), "--merge-output-format", "mp4", "--no-warnings"]
    if section:
        cmd += ["--download-sections", section, "--force-keyframes-at-cuts"]
    cmd += [url]
    subprocess.run(cmd, check=True, timeout=900)
    vids = list(dest.glob("vid.*"))
    if not vids:
        raise RuntimeError("download produced no file")
    return vids[0]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--section", default="*0:30-1:00")
    ap.add_argument("--stride", type=int, default=60)
    ap.add_argument("--max-frames", type=int, default=12)
    ap.add_argument("--points-per-side", type=int, default=16)
    ap.add_argument("--vlm-label", action="store_true", help="assign classes with Claude vision")
    ap.add_argument("--vlm-model", default="haiku")
    ap.add_argument("--vlm-top-k", type=int, default=8)
    ap.add_argument("--name", default="youtube_ingest")
    args = ap.parse_args()

    lic = _license(args.url)
    if "creative commons" not in lic.lower():
        print(f"ABORT: license is '{lic}', not Creative Commons. Refusing to ingest copyrighted video.")
        return 2
    print(f"license OK: {lic}")

    with Experiment(args.name, config={"url": args.url, "section": args.section, "license": lic,
                                       "stride": args.stride, "max_frames": args.max_frames,
                                       "vlm_label": args.vlm_label, "vlm_model": args.vlm_model}, seed=1234) as exp:
        import cv2

        frames_dir = exp.artifacts / "frames"
        vid = _download(args.url, args.section, exp.artifacts)
        exp.logger.info("downloaded %s (%.1f MB)", vid.name, vid.stat().st_size / 1e6)
        frame_paths = extract_frames(vid, frames_dir, stride=args.stride, max_frames=args.max_frames)
        vid.unlink(missing_ok=True)
        exp.logger.info("extracted %d frames", len(frame_paths))

        labeler = AutoMaskLabeler(AutoMaskConfig(points_per_side=args.points_per_side))
        vlm = ClaudeVLMLabeler(VLMLabelerConfig(model=args.vlm_model, top_k=args.vlm_top_k)) if args.vlm_label else None
        gcfg = PseudoLabelConfig()

        samples, n_auto, n_pseudo, total_cost = [], 0, 0, 0.0
        class_hist: Counter = Counter()
        for i, fp in enumerate(frame_paths):
            img = np.ascontiguousarray(cv2.imread(str(fp))[:, :, ::-1])
            insts = labeler.generate(img)
            res = gate(insts, gcfg, img.shape[0] * img.shape[1])
            n_auto += len(insts)
            final = res.accepted
            if vlm and final:
                final, cost = vlm.label(img, final)
                total_cost += cost
                class_hist.update(ins.category_name for ins in final)
            n_pseudo += len(final)
            samples.append(Sample(image_id=i, height=img.shape[0], width=img.shape[1],
                                  file_name=fp.name, image_dir=str(frames_dir), instances=final))
            if i < 3:
                render_instances(img, final, exp.artifact_path(f"frame_{i}_labels.png"),
                                 title=f"frame {i}: {len(final)} pseudo-labels")

        # map class names -> ids for COCO
        names = sorted(class_hist) or ["object"]
        name2id = {n: k + 1 for k, n in enumerate(names)}
        for s in samples:
            for ins in s.instances:
                ins.category_id = name2id.get(ins.category_name, 1)
        coco = export_coco(samples, exp.artifact_path("pseudo_labels.coco.json"),
                           categories={v: k for k, v in name2id.items()})

        results = {"license": lic, "n_frames": len(frame_paths), "n_auto_masks": n_auto,
                   "n_pseudo_labels": n_pseudo, "vlm": args.vlm_label, "vlm_cost_usd": round(total_cost, 4),
                   "class_histogram": dict(class_hist), "coco": str(coco)}
        exp.save_json("ingest_results.json", results)
        exp.save_summary(results)

        print("\n=== YouTube data-engine ingest ===")
        print(f"license   : {lic}")
        print(f"frames    : {len(frame_paths)}   auto masks: {n_auto}   pseudo-labels: {n_pseudo}")
        if args.vlm_label:
            print(f"vlm cost  : ${total_cost:.4f} ({args.vlm_model})")
            print(f"classes   : {dict(class_hist)}")
        print(f"coco      : {coco}")
        print(f"artifacts : {exp.artifacts}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
