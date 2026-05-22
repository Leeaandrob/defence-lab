"""Low-Rank Adaptation (Hu et al., 2021) -- a minimal, typed, dependency-free impl.

We adapt SAM2 by freezing all pretrained weights and learning a low-rank delta
``ΔW = (alpha/r) · B·A`` per targeted ``nn.Linear``. ``B`` is zero-initialized so
the adapted model *starts identical to the base* -- training only ever moves it
away on purpose. This is the efficiency lever of Phase 4: <1% trainable params,
no full retrain, and adapters are tiny to checkpoint and compose.

We roll our own (rather than depend on PEFT, which is installed) so dtype/grad
behaviour under bf16 autocast is fully under our control and SAM2's custom
modules wrap cleanly.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
import torch.nn as nn


@dataclass
class LoRAConfig:
    r: int = 8
    alpha: int = 16
    dropout: float = 0.0
    # substrings matched against qualified module names; an nn.Linear matches if
    # any substring is in its name. Default targets the SAM2 mask-decoder transformer.
    target_substrings: tuple[str, ...] = ("sam_mask_decoder.transformer",)


class LoRALinear(nn.Module):
    """Wraps a frozen ``nn.Linear`` with a trainable low-rank update."""

    def __init__(self, base: nn.Linear, r: int = 8, alpha: int = 16, dropout: float = 0.0) -> None:
        super().__init__()
        if r <= 0:
            raise ValueError("LoRA rank r must be positive")
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.r = r
        self.scaling = alpha / r
        # fp32 adapters for stable optimization regardless of autocast dtype
        self.lora_A = nn.Parameter(torch.zeros(r, base.in_features, dtype=torch.float32))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, r, dtype=torch.float32))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))  # B stays zero -> ΔW=0 at init
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        xa = self.dropout(x).to(self.lora_A.dtype)
        delta = (xa @ self.lora_A.t()) @ self.lora_B.t()
        return out + (self.scaling * delta).to(out.dtype)

    def extra_repr(self) -> str:
        return f"r={self.r}, scaling={self.scaling:.3f}"
