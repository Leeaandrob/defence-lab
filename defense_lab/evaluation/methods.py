"""Benchmarkable segmentation methods (Phase 6).

Each class satisfies the ``SegMethod`` protocol. SAM2 variants share one
predictor (encode once per image). ``ClassicalBoxMethod`` (GrabCut seeded by the
box) is the "traditional pipeline" baseline for the research question, needing no
extra deps. ``CallableMethod`` is the adapter seam to plug an external baseline
(YOLO-seg / DETR / Mask R-CNN) you provide -- nothing of that kind is bundled.
"""
from __future__ import annotations

from typing import Callable, Optional

import numpy as np

from defense_lab.evaluation.evaluator import EvalSample
from defense_lab.prompting.prompts import BoxPrompt, PointPrompt, PromptSet
from defense_lab.segmentation.predictor import PromptableSegmenter


class Sam2BoxMethod:
    name = "SAM2 (box prompt)"

    def __init__(self, segmenter: PromptableSegmenter) -> None:
        self.seg = segmenter

    def predict(self, s: EvalSample) -> np.ndarray:
        self.seg.set_image(s.image)
        return self.seg.predict(PromptSet.from_box(*s.box)).best()[0]


class Sam2PointMethod:
    name = "SAM2 (point prompt)"

    def __init__(self, segmenter: PromptableSegmenter) -> None:
        self.seg = segmenter

    def predict(self, s: EvalSample) -> np.ndarray:
        self.seg.set_image(s.image)
        return self.seg.predict(PromptSet.from_point(s.point[0], s.point[1], True)).best()[0]


class Sam2BoxPointMethod:
    """Best single-shot SAM2 prompt: box + a positive centroid point together."""

    name = "SAM2 (box + point)"

    def __init__(self, segmenter: PromptableSegmenter) -> None:
        self.seg = segmenter

    def predict(self, s: EvalSample) -> np.ndarray:
        self.seg.set_image(s.image)
        ps = PromptSet(points=PointPrompt().add(s.point[0], s.point[1], True), box=BoxPrompt(*s.box))
        return self.seg.predict(ps).best()[0]


class ClassicalBoxMethod:
    """Traditional baseline: GrabCut initialized from the prompt box."""

    name = "Classical (GrabCut+box)"

    def __init__(self, iters: int = 5) -> None:
        self.iters = iters

    def predict(self, s: EvalSample) -> np.ndarray:
        import cv2

        img = np.ascontiguousarray(s.image)
        h, w = img.shape[:2]
        mask = np.zeros((h, w), np.uint8)
        bgd, fgd = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
        x0, y0, x1, y1 = (int(v) for v in s.box)
        rect = (max(0, x0), max(0, y0), max(1, x1 - x0), max(1, y1 - y0))
        try:
            cv2.grabCut(img, mask, rect, bgd, fgd, self.iters, cv2.GC_INIT_WITH_RECT)
        except Exception:
            return np.zeros((h, w), bool)
        return np.isin(mask, [cv2.GC_FGD, cv2.GC_PR_FGD])


class CallableMethod:
    """Adapter to benchmark any external method: pass a name and a callable
    ``fn(sample) -> HxW bool mask``. Lets you drop in YOLO-seg/DETR/Mask R-CNN
    predictions (which you supply) without touching the harness."""

    def __init__(self, name: str, fn: Callable[[EvalSample], np.ndarray]) -> None:
        self.name = name
        self.fn = fn

    def predict(self, s: EvalSample) -> np.ndarray:
        return self.fn(s)


class Sam2LoraMethod:
    """SAM2 with a trained LoRA adapter loaded (decoder-transformer targets by default)."""

    name = "SAM2 + LoRA"

    def __init__(self, segmenter: PromptableSegmenter, adapter_path: str, lora_cfg=None) -> None:
        from defense_lab.lora.inject import inject_lora, load_lora
        from defense_lab.lora.layers import LoRAConfig

        model = segmenter._predictor.model
        inject_lora(model, lora_cfg or LoRAConfig())
        model.to(segmenter.device)
        load_lora(model, adapter_path)
        self.seg = segmenter

    def predict(self, s: EvalSample) -> np.ndarray:
        self.seg.set_image(s.image)
        return self.seg.predict(PromptSet.from_box(*s.box)).best()[0]
