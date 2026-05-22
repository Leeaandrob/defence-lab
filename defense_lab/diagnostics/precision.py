"""Precision validation + matmul throughput (Phase 1).

Validates that bf16 -- the working precision for the whole lab on Hopper -- is
numerically sane vs an fp32 reference, then measures realized matmul throughput
for fp32 / tf32 / fp16 / bf16 so we know the hardware ceiling before profiling
SAM2. On a GH200 (Hopper) bf16 should land in the hundreds of TFLOPS.

Good-citizen note: the GPU on this box is shared. Default matrix size is modest
and every tensor is freed + cache emptied between dtypes.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch


@dataclass
class PrecisionConfig:
    matmul_size: int = 4096          # square matmul dimension
    iters: int = 30
    warmup: int = 10
    dtypes: list[str] = field(default_factory=lambda: ["fp32", "tf32", "fp16", "bf16"])
    bf16_check_size: int = 2048      # smaller, for correctness vs fp32 reference


_DTYPE = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32, "tf32": torch.float32}


def _time_matmul(n: int, dtype: torch.dtype, iters: int, warmup: int, tf32: bool) -> float:
    torch.backends.cuda.matmul.allow_tf32 = tf32
    torch.backends.cudnn.allow_tf32 = tf32
    dev = torch.device("cuda")
    a = torch.randn(n, n, device=dev, dtype=dtype)
    b = torch.randn(n, n, device=dev, dtype=dtype)
    for _ in range(warmup):
        c = a @ b
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        c = a @ b
    end.record()
    torch.cuda.synchronize()
    ms = start.elapsed_time(end) / iters
    del a, b, c
    torch.cuda.empty_cache()
    return ms


def validate_bf16(n: int) -> dict[str, Any]:
    """Compare a bf16 matmul against an fp32 reference; report relative error."""
    dev = torch.device("cuda")
    a = torch.randn(n, n, device=dev, dtype=torch.float32)
    b = torch.randn(n, n, device=dev, dtype=torch.float32)
    ref = a @ b
    got = (a.bfloat16() @ b.bfloat16()).float()
    rel_err = (torch.linalg.norm(got - ref) / torch.linalg.norm(ref)).item()
    del a, b, ref, got
    torch.cuda.empty_cache()
    # bf16 has ~8 bits of mantissa; ~1e-2 relative error on a 2k-dim matmul is expected.
    return {"size": n, "relative_frobenius_error": rel_err, "passed": rel_err < 5e-2}


def run(cfg: PrecisionConfig | None = None) -> dict[str, Any]:
    if not torch.cuda.is_available():
        return {"available": False, "reason": "CUDA not available"}
    cfg = cfg or PrecisionConfig()
    n = cfg.matmul_size
    flops = 2.0 * n**3
    results: dict[str, Any] = {"available": True, "matmul_size": n, "throughput": {}}

    for name in cfg.dtypes:
        tf32 = name == "tf32"
        ms = _time_matmul(n, _DTYPE[name], cfg.iters, cfg.warmup, tf32=tf32)
        tflops = flops / (ms * 1e-3) / 1e12
        results["throughput"][name] = {"latency_ms": round(ms, 4), "tflops": round(tflops, 2)}

    # restore tf32 to a sane default for downstream code
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    results["bf16_validation"] = validate_bf16(cfg.bf16_check_size)
    results["bf16_supported"] = torch.cuda.is_bf16_supported()
    return results
