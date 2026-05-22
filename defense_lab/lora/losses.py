"""Segmentation training losses (SAM's recipe: focal + dice, ~20:1).

Operates on raw mask logits at the original image resolution. Vectorized over
the batch/mask dim so it works for single-mask supervision (training) and
multi-mask outputs alike.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def dice_loss(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1.0) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    dims = (-1, -2)
    num = 2.0 * (probs * targets).sum(dim=dims)
    den = probs.sum(dim=dims) + targets.sum(dim=dims) + eps
    return (1.0 - (num + eps) / den).mean()


def sigmoid_focal_loss(
    logits: torch.Tensor, targets: torch.Tensor, alpha: float = 0.25, gamma: float = 2.0
) -> torch.Tensor:
    p = torch.sigmoid(logits)
    ce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    pt = p * targets + (1 - p) * (1 - targets)
    loss = ce * ((1 - pt) ** gamma)
    if alpha >= 0:
        loss = (alpha * targets + (1 - alpha) * (1 - targets)) * loss
    return loss.mean()


def seg_loss(
    logits: torch.Tensor, targets: torch.Tensor, w_focal: float = 20.0, w_dice: float = 1.0
) -> torch.Tensor:
    return w_focal * sigmoid_focal_loss(logits, targets) + w_dice * dice_loss(logits, targets)
