"""Publication-ready benchmark plots (Phase 6). Headless (Agg)."""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

_BLUE = "#1976D2"


def bar_metric(reports: Sequence, attr: str, title: str, ylabel: str, path: str | Path, dpi: int = 130) -> Path:
    names = [r.name for r in reports]
    vals = [getattr(r, attr) or 0.0 for r in reports]
    fig, ax = plt.subplots(figsize=(max(5.0, 1.6 * len(names)), 4.2))
    bars = ax.bar(names, vals, color=_BLUE)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=15, ha="right")
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.3g}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    return path


def loss_curve(history: list[dict], path: str | Path, dpi: int = 130) -> Path:
    steps = [h["step"] for h in history]
    loss = [h["loss"] for h in history]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(steps, loss, marker="o", color=_BLUE)
    ax.set_xlabel("step")
    ax.set_ylabel("loss")
    ax.set_title("Training loss")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    return path
