"""Method-agnostic segmentation evaluation harness (Phase 6).

Evaluates any object implementing the :class:`SegMethod` protocol over a list of
:class:`EvalSample` and reports the operational metric suite (IoU / Dice /
boundary-IoU / latency / FPS / peak memory). Because methods are duck-typed, the
same harness benchmarks SAM2 variants *and* external baselines (a classical
pipeline, or a plugged-in detector) on identical footing -- the apparatus for
the project's research question. Nothing here is class- or domain-specific.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Optional, Protocol, runtime_checkable

import numpy as np
import torch

from defense_lab.evaluation.metrics import boundary_iou, dice_coefficient, mask_iou


@dataclass
class EvalSample:
    image: np.ndarray                 # HxWx3 RGB uint8
    gt: np.ndarray                    # HxW bool
    box: Optional[tuple] = None       # xyxy
    point: Optional[tuple] = None     # (x, y)
    sample_id: Any = None


@runtime_checkable
class SegMethod(Protocol):
    name: str

    def predict(self, sample: EvalSample) -> np.ndarray:  # -> HxW bool
        ...


@dataclass
class MethodReport:
    name: str
    n: int
    mean_iou: float
    mean_dice: float
    mean_boundary_iou: float
    mean_latency_ms: float
    fps: float
    peak_mem_gib: Optional[float]
    per_sample_iou: list


def evaluate_method(method: SegMethod, dataset: list[EvalSample], warmup: bool = True) -> MethodReport:
    cuda = torch.cuda.is_available()
    if warmup and dataset:
        method.predict(dataset[0])  # exclude one-time compile/alloc from timing
    if cuda:
        torch.cuda.reset_peak_memory_stats()
    ious, dices, bious, lat = [], [], [], []
    for s in dataset:
        t0 = time.perf_counter()
        pred = method.predict(s)
        if cuda:
            torch.cuda.synchronize()
        lat.append((time.perf_counter() - t0) * 1e3)
        ious.append(mask_iou(pred, s.gt))
        dices.append(dice_coefficient(pred, s.gt))
        bious.append(boundary_iou(pred, s.gt))
    peak = round(torch.cuda.max_memory_allocated() / 2**30, 3) if cuda else None
    mlat = float(np.mean(lat)) if lat else 0.0
    return MethodReport(
        name=method.name,
        n=len(dataset),
        mean_iou=round(float(np.mean(ious)), 4) if ious else 0.0,
        mean_dice=round(float(np.mean(dices)), 4) if dices else 0.0,
        mean_boundary_iou=round(float(np.mean(bious)), 4) if bious else 0.0,
        mean_latency_ms=round(mlat, 3),
        fps=round(1000.0 / mlat, 2) if mlat > 0 else 0.0,
        peak_mem_gib=peak,
        per_sample_iou=[round(x, 4) for x in ious],
    )


def compare_methods(methods: list[SegMethod], dataset: list[EvalSample]) -> dict[str, MethodReport]:
    return {m.name: evaluate_method(m, dataset) for m in methods}


def report_rows(reports: dict[str, MethodReport]) -> list[dict]:
    return [
        {
            "method": r.name, "IoU": r.mean_iou, "Dice": r.mean_dice,
            "boundary_IoU": r.mean_boundary_iou, "latency_ms": r.mean_latency_ms,
            "FPS": r.fps, "peak_mem_GiB": r.peak_mem_gib, "n": r.n,
        }
        for r in reports.values()
    ]
