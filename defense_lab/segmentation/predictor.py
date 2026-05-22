"""Promptable segmentation engine -- SAM2 image predictor, encode-once/decode-many.

``PromptableSegmenter`` is the Phase-2 façade over SAM2's image predictor. It
embodies SAM's central design: an expensive image embedding computed *once*
(:meth:`set_image`), then many cheap promptable :meth:`predict` calls. bf16 +
``inference_mode`` are handled internally so callers stay backend-agnostic and
work only with the typed prompts from :mod:`defense_lab.prompting`.
"""
from __future__ import annotations

import contextlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch

from defense_lab.prompting.prompts import MaskPrompt, PromptSet

_DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}


@dataclass
class SegmenterConfig:
    model_cfg: str = "configs/sam2.1/sam2.1_hiera_b+.yaml"
    checkpoint: str = "checkpoints/sam2.1_hiera_base_plus.pt"
    device: str = "cuda"
    dtype: str = "bf16"


@dataclass
class SegmentationResult:
    """Output of a single promptable query.

    masks: (K, H, W) bool;  scores: (K,) predicted IoU;  low_res_logits: (K,256,256).
    With ``multimask_output=True``, K=3 (SAM resolves prompt ambiguity into
    whole/part/subpart hypotheses); pick with :meth:`best`.
    """

    masks: np.ndarray
    scores: np.ndarray
    low_res_logits: np.ndarray

    def best_index(self) -> int:
        return int(np.argmax(self.scores))

    def best(self) -> tuple[np.ndarray, float, np.ndarray]:
        i = self.best_index()
        return self.masks[i], float(self.scores[i]), self.low_res_logits[i]

    def best_mask_prompt(self) -> MaskPrompt:
        """The best hypothesis as a refinement mask-prompt for the next decode."""
        return MaskPrompt(self.low_res_logits[self.best_index()])


class PromptableSegmenter:
    def __init__(self, cfg: Optional[SegmenterConfig] = None) -> None:
        self.cfg = cfg or SegmenterConfig()
        self.device = torch.device(self.cfg.device if torch.cuda.is_available() else "cpu")
        self._dtype = _DTYPES[self.cfg.dtype]
        self._predictor = self._build()
        self._has_image = False

    def _build(self):
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        if not Path(self.cfg.checkpoint).exists():
            raise FileNotFoundError(
                f"SAM2 checkpoint missing: {self.cfg.checkpoint}. "
                "Run scripts/download_checkpoints.sh"
            )
        model = build_sam2(self.cfg.model_cfg, self.cfg.checkpoint, device=str(self.device))
        return SAM2ImagePredictor(model)

    def _autocast(self):
        if self.device.type == "cuda" and self._dtype in (torch.bfloat16, torch.float16):
            return torch.autocast("cuda", dtype=self._dtype)
        return contextlib.nullcontext()

    # -- encode once -------------------------------------------------------- #
    def set_image(self, image: np.ndarray) -> "PromptableSegmenter":
        """Compute and cache the image embedding. ``image`` is HxWx3 uint8 RGB."""
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"Expected HxWx3 RGB image, got shape {image.shape}")
        with torch.inference_mode(), self._autocast():
            self._predictor.set_image(image)
        self._has_image = True
        return self

    # -- decode many -------------------------------------------------------- #
    def predict(self, prompt: PromptSet, multimask_output: bool = True) -> SegmentationResult:
        if not self._has_image:
            raise RuntimeError("Call set_image(...) before predict(...).")
        if prompt.is_empty():
            raise ValueError("Empty PromptSet: provide at least a point, box, or mask.")
        kwargs: dict[str, Any] = prompt.to_predictor_kwargs()
        with torch.inference_mode(), self._autocast():
            masks, scores, low_res = self._predictor.predict(
                multimask_output=multimask_output, return_logits=False, **kwargs
            )
        return SegmentationResult(
            masks=np.asarray(masks, dtype=bool),
            scores=np.asarray(scores, dtype=np.float32),
            low_res_logits=np.asarray(low_res, dtype=np.float32),
        )

    @property
    def image_size(self) -> Optional[tuple[int, int]]:
        return getattr(self._predictor, "_orig_hw", [None])[0] if self._has_image else None
