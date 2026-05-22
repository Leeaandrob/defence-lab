"""Mask / prompt overlay rendering for qualitative + paper-ready figures.

Uses the Agg backend so it runs headless on a server. ``render_result`` produces
a single annotated panel (image + translucent mask + prompt markers + score);
``render_panels`` lays several side by side for ablation/figure use.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from defense_lab.prompting.prompts import PromptSet  # noqa: E402

_MASK_COLOR = np.array([0.13, 0.59, 0.95])  # blue


def _overlay(ax, image: np.ndarray, mask: Optional[np.ndarray], alpha: float = 0.5) -> None:
    ax.imshow(image)
    if mask is not None:
        h, w = mask.shape
        rgba = np.zeros((h, w, 4), dtype=np.float32)
        rgba[..., :3] = _MASK_COLOR
        rgba[..., 3] = mask.astype(np.float32) * alpha
        ax.imshow(rgba)
    ax.set_xticks([])
    ax.set_yticks([])


def _draw_prompts(ax, prompt: Optional[PromptSet]) -> None:
    if prompt is None:
        return
    if prompt.points is not None:
        for p in prompt.points.points:
            ax.scatter(
                p.x, p.y, c=("lime" if p.label == 1 else "red"),
                marker="*", s=160, edgecolors="black", linewidths=0.8, zorder=3,
            )
    if prompt.box is not None:
        b = prompt.box
        ax.add_patch(
            plt.Rectangle((b.x0, b.y0), b.x1 - b.x0, b.y1 - b.y0,
                          fill=False, edgecolor="yellow", linewidth=2.0, zorder=2)
        )


def render_result(
    image: np.ndarray,
    mask: Optional[np.ndarray],
    path: str | Path,
    prompt: Optional[PromptSet] = None,
    title: Optional[str] = None,
    dpi: int = 130,
) -> Path:
    fig, ax = plt.subplots(figsize=(6, 6))
    _overlay(ax, image, mask)
    _draw_prompts(ax, prompt)
    if title:
        ax.set_title(title, fontsize=11)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path


def render_instances(
    image: np.ndarray,
    instances,
    path: str | Path,
    title: Optional[str] = None,
    alpha: float = 0.55,
    dpi: int = 130,
) -> Path:
    """Overlay many masks, each a distinct color -- for open-world AMG output."""
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.imshow(image)
    rng = np.random.default_rng(0)
    for ins in instances:
        if getattr(ins, "mask", None) is None:
            continue
        m = ins.mask
        rgba = np.zeros((*m.shape, 4), dtype=np.float32)
        rgba[..., :3] = rng.random(3)
        rgba[..., 3] = m.astype(np.float32) * alpha
        ax.imshow(rgba)
    ax.set_xticks([])
    ax.set_yticks([])
    if title:
        ax.set_title(title, fontsize=11)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path


def render_panels(
    image: np.ndarray,
    masks: Sequence[Optional[np.ndarray]],
    titles: Sequence[str],
    path: str | Path,
    prompts: Optional[Sequence[Optional[PromptSet]]] = None,
    dpi: int = 130,
) -> Path:
    n = len(masks)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))
    if n == 1:
        axes = [axes]
    for i, ax in enumerate(axes):
        _overlay(ax, image, masks[i])
        _draw_prompts(ax, prompts[i] if prompts else None)
        ax.set_title(titles[i], fontsize=11)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path
