"""SAM2 video streaming-throughput benchmark (Phase 1 / Phase 5 precursor).

Exercises the SAM2 *video predictor* -- the temporal-memory path that makes
SAM2 more than an image model: prompt object(s) on frame 0, then propagate the
masklet across the clip via frame-memory attention. We report propagation FPS
(the streaming-inference number that bounds operational video workloads) and
peak memory.

Frames are synthesized to a temp dir (a moving bright blob on noise) so the
benchmark is self-contained and needs no dataset. ``offload_video_to_cpu``
keeps the shared GPU happy on longer clips.
"""
from __future__ import annotations

import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch


@dataclass
class Sam2VideoBenchConfig:
    model_cfg: str = "configs/sam2.1/sam2.1_hiera_b+.yaml"
    checkpoint: str = "checkpoints/sam2.1_hiera_base_plus.pt"
    num_frames: int = 24
    frame_size: int = 1024
    dtype: str = "bf16"
    offload_video_to_cpu: bool = True
    device: str = "cuda"


def _synthesize_clip(out_dir: Path, n: int, size: int) -> None:
    import cv2

    rng = np.random.default_rng(0)
    for i in range(n):
        img = rng.integers(0, 60, size=(size, size, 3), dtype=np.uint8)  # dark noise
        cx = int(size * (0.2 + 0.6 * i / max(n - 1, 1)))  # blob drifts across frame
        cy = size // 2
        cv2.circle(img, (cx, cy), size // 12, (230, 230, 230), thickness=-1)
        cv2.imwrite(str(out_dir / f"{i:05d}.jpg"), img)


def run(cfg: Sam2VideoBenchConfig | None = None) -> dict[str, Any]:
    cfg = cfg or Sam2VideoBenchConfig()
    try:
        from sam2.build_sam import build_sam2_video_predictor
    except Exception as ex:
        return {"available": False, "reason": f"sam2 not importable: {ex}", "hint": "Run scripts/install_sam2.sh"}
    if not Path(cfg.checkpoint).exists():
        return {"available": False, "reason": f"checkpoint not found: {cfg.checkpoint}", "hint": "Run scripts/download_checkpoints.sh"}

    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    td = torch.bfloat16 if cfg.dtype == "bf16" else (torch.float16 if cfg.dtype == "fp16" else torch.float32)
    predictor = build_sam2_video_predictor(cfg.model_cfg, cfg.checkpoint, device=str(device))

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        _synthesize_clip(tmp_dir, cfg.num_frames, cfg.frame_size)

        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()

        with torch.inference_mode(), torch.autocast(device.type, dtype=td) if device.type == "cuda" else _null():
            t0 = time.perf_counter()
            state = predictor.init_state(
                video_path=str(tmp_dir),
                offload_video_to_cpu=cfg.offload_video_to_cpu,
            )
            if device.type == "cuda":
                torch.cuda.synchronize()
            init_s = time.perf_counter() - t0

            # prompt the drifting blob on frame 0 with a single positive point
            c = cfg.frame_size
            predictor.add_new_points_or_box(
                state,
                frame_idx=0,
                obj_id=1,
                points=np.array([[int(c * 0.2), c // 2]], dtype=np.float32),
                labels=np.array([1], dtype=np.int32),
            )

            t0 = time.perf_counter()
            propagated = 0
            for _frame_idx, _obj_ids, _mask_logits in predictor.propagate_in_video(state):
                propagated += 1
            if device.type == "cuda":
                torch.cuda.synchronize()
            prop_s = time.perf_counter() - t0

    peak_gib = (torch.cuda.max_memory_allocated() / 2**30) if device.type == "cuda" else None
    fps = propagated / prop_s if prop_s > 0 else 0.0
    return {
        "available": True,
        "device": str(device),
        "dtype": cfg.dtype,
        "num_frames": cfg.num_frames,
        "frame_size": cfg.frame_size,
        "init_state_s": round(init_s, 3),
        "propagation_s": round(prop_s, 3),
        "frames_propagated": propagated,
        "streaming_fps": round(fps, 2),
        "peak_mem_gib": round(peak_gib, 3) if peak_gib is not None else None,
    }


def _null():
    import contextlib

    return contextlib.nullcontext()
