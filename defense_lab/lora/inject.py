"""Inject / mark / save / load LoRA adapters on an arbitrary nn.Module.

These utilities are model-agnostic: they locate target ``nn.Linear`` layers by
name, swap in :class:`LoRALinear`, freeze everything except the adapters, and
serialize *only* the adapter tensors (a few hundred KB) via safetensors -- the
"lightweight, composable adapter" property that makes domain adaptation cheap.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterator

import torch
import torch.nn as nn

from defense_lab.lora.layers import LoRAConfig, LoRALinear


def _get_parent(model: nn.Module, qualified: str) -> tuple[nn.Module, str]:
    parts = qualified.split(".")
    parent = model
    for p in parts[:-1]:
        parent = getattr(parent, p)
    return parent, parts[-1]


def inject_lora(model: nn.Module, cfg: LoRAConfig) -> list[str]:
    """Replace matching nn.Linear layers with LoRALinear. Returns wrapped names."""
    targets = [
        (name, mod)
        for name, mod in model.named_modules()
        if isinstance(mod, nn.Linear) and any(s in name for s in cfg.target_substrings)
    ]
    for name, lin in targets:
        parent, attr = _get_parent(model, name)
        setattr(parent, attr, LoRALinear(lin, cfg.r, cfg.alpha, cfg.dropout))
    return [name for name, _ in targets]


def mark_only_lora_trainable(model: nn.Module) -> None:
    for n, p in model.named_parameters():
        p.requires_grad_(("lora_A" in n) or ("lora_B" in n))


def lora_parameters(model: nn.Module) -> list[nn.Parameter]:
    return [p for n, p in model.named_parameters() if ("lora_A" in n) or ("lora_B" in n)]


def count_parameters(model: nn.Module) -> dict[str, float]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "total": int(total),
        "trainable": int(trainable),
        "trainable_pct": round(100.0 * trainable / total, 4) if total else 0.0,
    }


def lora_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    return {
        n: p.detach().cpu()
        for n, p in model.named_parameters()
        if ("lora_A" in n) or ("lora_B" in n)
    }


def save_lora(model: nn.Module, path: str | Path) -> Path:
    from safetensors.torch import save_file

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    save_file(lora_state_dict(model), str(path))
    return path


def load_lora(model: nn.Module, path: str | Path) -> tuple[int, int]:
    """Load adapter tensors into an already-LoRA-injected model. Returns (loaded, missing)."""
    from safetensors.torch import load_file

    sd = load_file(str(path))
    result = model.load_state_dict(sd, strict=False)
    loaded = len(sd)
    # any adapter param not present in the file is "missing"
    missing = sum(1 for k in result.missing_keys if ("lora_A" in k) or ("lora_B" in k))
    return loaded, missing
