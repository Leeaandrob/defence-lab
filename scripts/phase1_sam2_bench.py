#!/usr/bin/env python3
"""Phase 1 (model) -- SAM2 inference & video-throughput benchmark.

Benchmarks the actual SAM2 model on this GH200:
  * image predictor: encode-once latency vs per-prompt decode latency (the
    SAM design seam) + interactive prompt FPS + peak memory;
  * video predictor: streaming propagation FPS over a synthetic clip.

Writes a reproducible experiment dir (config + env + metrics + markdown report).

Usage
-----
    python scripts/phase1_sam2_bench.py
    python scripts/phase1_sam2_bench.py --quick
    python scripts/phase1_sam2_bench.py image.checkpoint=checkpoints/sam2.1_hiera_large.pt \
        image.model_cfg=configs/sam2.1/sam2.1_hiera_l.yaml
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from defense_lab.benchmarks import sam2_inference, sam2_video  # noqa: E402
from defense_lab.config.base import build_config, to_dict  # noqa: E402
from defense_lab.repro.experiment import Experiment  # noqa: E402


@dataclass
class Sam2BenchSuite:
    seed: int = 1234
    image: sam2_inference.Sam2BenchConfig = field(default_factory=sam2_inference.Sam2BenchConfig)
    video: sam2_video.Sam2VideoBenchConfig = field(default_factory=sam2_video.Sam2VideoBenchConfig)


def _markdown(img: dict, vid: dict) -> str:
    lines = ["# Phase 1 (model) -- SAM2 Benchmark", ""]
    lines += ["## Image predictor (encode-once / decode-many)", ""]
    if img.get("available"):
        lines += [
            f"- device/dtype: **{img['device']} / {img['dtype']}**, image {img['image_size']}px",
            f"- encode (set_image): **{img['encode_ms_per_image']} ms** ({img['encode_fps']} img/s)",
            f"- decode (per prompt): **{img['decode_ms_per_prompt']} ms** "
            f"({img['interactive_prompt_fps']} prompts/s)",
            f"- peak GPU mem: **{img['peak_mem_gib']} GiB**",
        ]
    else:
        lines += [f"- _unavailable_: {img.get('reason')} ({img.get('hint','')})"]
    lines += ["", "## Video predictor (streaming propagation)", ""]
    if vid.get("available"):
        lines += [
            f"- clip: {vid['num_frames']} frames @ {vid['frame_size']}px, {vid['dtype']}",
            f"- init_state: **{vid['init_state_s']} s**",
            f"- propagation: **{vid['streaming_fps']} FPS** "
            f"({vid['frames_propagated']} frames in {vid['propagation_s']} s)",
            f"- peak GPU mem: **{vid['peak_mem_gib']} GiB**",
        ]
    else:
        lines += [f"- _unavailable_: {vid.get('reason')} ({vid.get('hint','')})"]
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description="SAM2 inference & video benchmark")
    ap.add_argument("--config", type=str, default=None)
    ap.add_argument("--name", type=str, default="phase1_sam2_bench")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("overrides", nargs="*")
    args = ap.parse_args()

    overrides = list(args.overrides)
    if args.quick:
        overrides += ["image.iters=5", "image.warmup=2", "video.num_frames=8", "video.frame_size=512"]

    cfg = build_config(Sam2BenchSuite, args.config, overrides)
    with Experiment(args.name, config=cfg, seed=cfg.seed) as exp:
        exp.logger.info("benchmarking SAM2 image predictor ...")
        img = sam2_inference.run(cfg.image)
        exp.logger.info("image result: %s", img)
        exp.logger.info("benchmarking SAM2 video predictor ...")
        vid = sam2_video.run(cfg.video)
        exp.logger.info("video result: %s", vid)

        results = {"config": to_dict(cfg), "image": img, "video": vid}
        exp.save_json("sam2_bench_results.json", results)
        md = _markdown(img, vid)
        exp.artifact_path("sam2_bench_report.md").write_text(md)
        if img.get("available"):
            exp.log_metrics(encode_ms=img["encode_ms_per_image"], decode_ms=img["decode_ms_per_prompt"])
        if vid.get("available"):
            exp.log_metrics(streaming_fps=vid["streaming_fps"])
        exp.save_summary({
            "image_decode_ms": img.get("decode_ms_per_prompt"),
            "image_peak_gib": img.get("peak_mem_gib"),
            "video_streaming_fps": vid.get("streaming_fps"),
            "video_peak_gib": vid.get("peak_mem_gib"),
        })
        print("\n" + md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
