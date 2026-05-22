"""Pseudo-label gating + human-in-the-loop queue (the data-engine loop).

The SAM data engine is iterative: the model proposes masks, high-confidence ones
become training labels, low-confidence ones are routed to a human. ``gate``
implements that triage via predicted-IoU / stability / area thresholds; the
accepted set is the pseudo-label yield, the review set is the HITL work queue.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Optional

from defense_lab.datasets.types import Instance


@dataclass
class PseudoLabelConfig:
    iou_accept: float = 0.85        # min predicted IoU to auto-accept
    stability_accept: float = 0.90  # min stability score to auto-accept
    min_area: int = 64              # reject specks
    max_area_frac: float = 0.95     # reject near-whole-image masks (usually background)


@dataclass
class GateResult:
    accepted: list[Instance] = field(default_factory=list)
    review: list[Instance] = field(default_factory=list)

    def stats(self, image_area: Optional[int] = None) -> dict:
        n = len(self.accepted) + len(self.review)
        return {
            "n_total": n,
            "n_accepted": len(self.accepted),
            "n_review": len(self.review),
            "accept_rate": (len(self.accepted) / n) if n else 0.0,
            "mean_accepted_score": _mean([i.score for i in self.accepted]),
            "mean_review_score": _mean([i.score for i in self.review]),
        }


def _mean(xs: list[Optional[float]]) -> Optional[float]:
    vals = [x for x in xs if x is not None]
    return round(sum(vals) / len(vals), 4) if vals else None


def gate(instances: list[Instance], cfg: PseudoLabelConfig, image_area: int) -> GateResult:
    res = GateResult()
    for ins in instances:
        score_ok = ins.score is None or ins.score >= cfg.iou_accept
        stab_ok = ins.stability is None or ins.stability >= cfg.stability_accept
        area = ins.area
        area_ok = cfg.min_area <= area <= cfg.max_area_frac * image_area
        accepted = score_ok and stab_ok and area_ok
        ins.source = "pseudo" if accepted else (ins.source or "auto")
        (res.accepted if accepted else res.review).append(ins)
    return res


def review_manifest(result: GateResult, image_area: int) -> dict:
    """Compact, JSON-safe summary of one image's gating decision (no mask arrays)."""
    def row(i: Instance, decision: str) -> dict:
        return {
            "box": [round(v, 1) for v in i.box],
            "area": i.area,
            "score": i.score,
            "stability": i.stability,
            "decision": decision,
        }

    return {
        "stats": result.stats(image_area),
        "accepted": [row(i, "accept") for i in result.accepted],
        "review": [row(i, "review") for i in result.review],
    }
