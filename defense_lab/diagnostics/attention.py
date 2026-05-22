"""Flash Attention / SDPA backend validation (Phase 1).

SAM2's image encoder and memory-attention blocks lean on fused attention.
Before benchmarking the model we verify *which* scaled-dot-product-attention
backend the box can actually run in bf16:

  * FLASH_ATTENTION  -- the fast path we want on Hopper
  * EFFICIENT_ATTENTION (mem-efficient)
  * MATH             -- the always-correct fallback

We probe each backend by forcing it and timing a forward+backward pass, and we
separately import the standalone ``flash_attn`` package (if present) to confirm
the kernels load against this CUDA/torch build.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F


@dataclass
class AttentionConfig:
    batch: int = 2
    heads: int = 8
    seq_len: int = 2048
    head_dim: int = 64
    iters: int = 20
    warmup: int = 5


def _backends():
    """Return {name: SDPBackend} across torch versions (2.7 path + legacy)."""
    try:
        from torch.nn.attention import SDPBackend

        return {
            "FLASH_ATTENTION": SDPBackend.FLASH_ATTENTION,
            "EFFICIENT_ATTENTION": SDPBackend.EFFICIENT_ATTENTION,
            "MATH": SDPBackend.MATH,
        }
    except Exception:
        return {}


def _sdpa_kernel_ctx(backend):
    from torch.nn.attention import sdpa_kernel

    return sdpa_kernel(backend)


def _bench_backend(cfg: AttentionConfig, backend) -> dict[str, Any]:
    dev = torch.device("cuda")
    shape = (cfg.batch, cfg.heads, cfg.seq_len, cfg.head_dim)
    q = torch.randn(*shape, device=dev, dtype=torch.bfloat16, requires_grad=True)
    k = torch.randn(*shape, device=dev, dtype=torch.bfloat16, requires_grad=True)
    v = torch.randn(*shape, device=dev, dtype=torch.bfloat16, requires_grad=True)
    try:
        with _sdpa_kernel_ctx(backend):
            for _ in range(cfg.warmup):
                out = F.scaled_dot_product_attention(q, k, v)
                out.sum().backward()
            torch.cuda.synchronize()
            s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            s.record()
            for _ in range(cfg.iters):
                out = F.scaled_dot_product_attention(q, k, v)
                out.sum().backward()
            e.record()
            torch.cuda.synchronize()
            ms = s.elapsed_time(e) / cfg.iters
        res = {"supported": True, "fwd_bwd_latency_ms": round(ms, 4)}
    except Exception as ex:
        res = {"supported": False, "error": f"{type(ex).__name__}: {ex}"}
    finally:
        del q, k, v
        torch.cuda.empty_cache()
    return res


def _flash_attn_pkg() -> dict[str, Any]:
    try:
        import flash_attn
        from flash_attn import flash_attn_func

        dev = torch.device("cuda")
        # flash_attn expects (B, S, H, D), fp16/bf16
        q = torch.randn(2, 1024, 8, 64, device=dev, dtype=torch.bfloat16)
        out = flash_attn_func(q, q, q)
        ok = tuple(out.shape) == (2, 1024, 8, 64)
        del q, out
        torch.cuda.empty_cache()
        return {"installed": True, "version": getattr(flash_attn, "__version__", "?"), "kernel_ran": ok}
    except Exception as ex:
        return {"installed": False, "error": f"{type(ex).__name__}: {ex}"}


def run(cfg: AttentionConfig | None = None) -> dict[str, Any]:
    if not torch.cuda.is_available():
        return {"available": False, "reason": "CUDA not available"}
    cfg = cfg or AttentionConfig()
    out: dict[str, Any] = {
        "available": True,
        "config": {"batch": cfg.batch, "heads": cfg.heads, "seq_len": cfg.seq_len, "head_dim": cfg.head_dim},
        "sdpa_backends": {},
    }
    for name, backend in _backends().items():
        out["sdpa_backends"][name] = _bench_backend(cfg, backend)
    out["flash_attn_package"] = _flash_attn_pkg()
    # which backend torch would pick by default (no forcing)
    try:
        dev = torch.device("cuda")
        q = torch.randn(1, cfg.heads, 512, cfg.head_dim, device=dev, dtype=torch.bfloat16)
        _ = F.scaled_dot_product_attention(q, q, q)
        torch.cuda.synchronize()
        out["default_path_ran_bf16"] = True
        del q
        torch.cuda.empty_cache()
    except Exception as ex:
        out["default_path_ran_bf16"] = False
        out["default_path_error"] = str(ex)
    return out
