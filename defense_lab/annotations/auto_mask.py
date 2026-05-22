"""Fully-automatic, class-agnostic mask generation (SAM data-engine stage 3).

Wraps ``SAM2AutomaticMaskGenerator``: a regular grid of point prompts is decoded
into *every* salient mask, with no class vocabulary -- the open-world labeling
that seeded SA-1B. Each mask carries a predicted-IoU and stability score, which
the pseudo-label gate (see :mod:`defense_lab.annotations.pseudo_label`) uses to
decide auto-accept vs human review.
"""
from __future__ import annotations

import contextlib
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from defense_lab.datasets.types import Instance

_DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}


@dataclass
class AutoMaskConfig:
    model_cfg: str = "configs/sam2.1/sam2.1_hiera_b+.yaml"
    checkpoint: str = "checkpoints/sam2.1_hiera_base_plus.pt"
    device: str = "cuda"
    dtype: str = "bf16"
    points_per_side: int = 16          # 16 -> 256 prompts; modest on a shared GPU
    points_per_batch: int = 64
    pred_iou_thresh: float = 0.7
    stability_score_thresh: float = 0.9
    min_mask_region_area: int = 64


class AutoMaskLabeler:
    def __init__(self, cfg: Optional[AutoMaskConfig] = None) -> None:
        self.cfg = cfg or AutoMaskConfig()
        self.device = torch.device(self.cfg.device if torch.cuda.is_available() else "cpu")
        self._dtype = _DTYPES[self.cfg.dtype]
        self._amg = self._build()

    def _build(self):
        from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
        from sam2.build_sam import build_sam2

        if not Path(self.cfg.checkpoint).exists():
            raise FileNotFoundError(f"checkpoint missing: {self.cfg.checkpoint}")
        model = build_sam2(self.cfg.model_cfg, self.cfg.checkpoint, device=str(self.device),
                           apply_postprocessing=False)
        return SAM2AutomaticMaskGenerator(
            model,
            points_per_side=self.cfg.points_per_side,
            points_per_batch=self.cfg.points_per_batch,
            pred_iou_thresh=self.cfg.pred_iou_thresh,
            stability_score_thresh=self.cfg.stability_score_thresh,
            min_mask_region_area=self.cfg.min_mask_region_area,
        )

    def _autocast(self):
        if self.device.type == "cuda" and self._dtype in (torch.bfloat16, torch.float16):
            return torch.autocast("cuda", dtype=self._dtype)
        return contextlib.nullcontext()

    def generate(self, image: np.ndarray) -> list[Instance]:
        """Return class-agnostic Instances for one HxWx3 RGB uint8 image."""
        with torch.inference_mode(), self._autocast():
            records = self._amg.generate(image)
        out: list[Instance] = []
        for r in records:
            seg = np.asarray(r["segmentation"], dtype=bool)
            x, y, w, h = r["bbox"]  # xywh
            out.append(
                Instance(
                    box=(float(x), float(y), float(x + w), float(y + h)),
                    mask=seg,
                    score=float(r.get("predicted_iou", 0.0)),
                    stability=float(r.get("stability_score", 0.0)),
                    source="auto",
                )
            )
        return out
