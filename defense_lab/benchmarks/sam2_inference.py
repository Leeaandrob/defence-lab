"""SAM2 image-predictor inference benchmark (Phase 1 / Phase 6).

Measures end-to-end promptable-segmentation latency and throughput: image
embedding (the expensive, amortizable step) vs per-prompt mask decoding (the
cheap, interactive step). This separation is the whole point of SAM's design --
encode once, prompt many times -- so we report them independently.

The import of ``sam2`` is lazy and guarded: if the package or checkpoint is
missing, ``run`` returns an actionable status dict instead of crashing, so the
Phase-1 pipeline stays runnable on a fresh box. Install with
``scripts/install_sam2.sh`` and fetch weights with
``scripts/download_checkpoints.sh``.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch


@dataclass
class Sam2BenchConfig:
    model_cfg: str = "configs/sam2.1/sam2.1_hiera_b+.yaml"  # resolved by sam2 package
    checkpoint: str = "checkpoints/sam2.1_hiera_base_plus.pt"
    image_size: int = 1024
    iters: int = 30
    warmup: int = 5
    dtype: str = "bf16"
    device: str = "cuda"


def _autocast(device: str, dtype: str):
    if device == "cuda" and dtype in ("bf16", "fp16"):
        td = torch.bfloat16 if dtype == "bf16" else torch.float16
        return torch.autocast("cuda", dtype=td)
    import contextlib

    return contextlib.nullcontext()


def run(cfg: Sam2BenchConfig | None = None) -> dict[str, Any]:
    cfg = cfg or Sam2BenchConfig()
    try:
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
    except Exception as ex:
        return {
            "available": False,
            "reason": f"sam2 not importable: {type(ex).__name__}: {ex}",
            "hint": "Run scripts/install_sam2.sh",
        }
    if not Path(cfg.checkpoint).exists():
        return {
            "available": False,
            "reason": f"checkpoint not found: {cfg.checkpoint}",
            "hint": "Run scripts/download_checkpoints.sh",
        }

    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    model = build_sam2(cfg.model_cfg, cfg.checkpoint, device=str(device))
    predictor = SAM2ImagePredictor(model)

    # synthetic operational-style frame + a single foreground point prompt
    rng = np.random.default_rng(0)
    image = rng.integers(0, 255, size=(cfg.image_size, cfg.image_size, 3), dtype=np.uint8)
    point = np.array([[cfg.image_size // 2, cfg.image_size // 2]], dtype=np.float32)
    label = np.array([1], dtype=np.int32)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    def _sync():
        if device.type == "cuda":
            torch.cuda.synchronize()

    with torch.inference_mode(), _autocast(device.type, cfg.dtype):
        # ---- image embedding (encode once) ----
        for _ in range(cfg.warmup):
            predictor.set_image(image)
        _sync()
        t = time.perf_counter()
        for _ in range(cfg.iters):
            predictor.set_image(image)
        _sync()
        encode_ms = (time.perf_counter() - t) / cfg.iters * 1e3

        # ---- mask decode (prompt many) ----
        predictor.set_image(image)
        for _ in range(cfg.warmup):
            predictor.predict(point_coords=point, point_labels=label, multimask_output=True)
        _sync()
        t = time.perf_counter()
        for _ in range(cfg.iters):
            predictor.predict(point_coords=point, point_labels=label, multimask_output=True)
        _sync()
        decode_ms = (time.perf_counter() - t) / cfg.iters * 1e3

    peak_gib = (torch.cuda.max_memory_allocated() / 2**30) if device.type == "cuda" else None
    return {
        "available": True,
        "device": str(device),
        "dtype": cfg.dtype,
        "image_size": cfg.image_size,
        "encode_ms_per_image": round(encode_ms, 3),
        "decode_ms_per_prompt": round(decode_ms, 3),
        "encode_fps": round(1e3 / encode_ms, 2),
        "interactive_prompt_fps": round(1e3 / decode_ms, 2),
        "peak_mem_gib": round(peak_gib, 3) if peak_gib is not None else None,
    }
