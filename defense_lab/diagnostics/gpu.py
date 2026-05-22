"""GPU diagnostics + memory-bandwidth micro-benchmarks (Phase 1).

Reports device properties and current memory headroom (this card is shared, so
headroom matters), then measures host<->device and device<->device copy
bandwidth -- the numbers that bound streaming-video inference throughput.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass
class GpuConfig:
    copy_mib: int = 256        # transfer size per bandwidth test
    iters: int = 20
    warmup: int = 5
    headroom_warn_gib: float = 8.0


def device_report(index: int = 0) -> dict[str, Any]:
    p = torch.cuda.get_device_properties(index)
    free, total = torch.cuda.mem_get_info(index)
    return {
        "index": index,
        "name": p.name,
        "capability": f"{p.major}.{p.minor}",
        "total_mem_gib": round(p.total_memory / 2**30, 3),
        "free_mem_gib": round(free / 2**30, 3),
        "used_by_others_gib": round((total - free) / 2**30, 3),
        "multiprocessors": p.multi_processor_count,
        "torch_reserved_gib": round(torch.cuda.memory_reserved(index) / 2**30, 3),
        "torch_allocated_gib": round(torch.cuda.memory_allocated(index) / 2**30, 3),
    }


def _bandwidth(cfg: GpuConfig) -> dict[str, float]:
    dev = torch.device("cuda")
    n = cfg.copy_mib * 1024 * 1024 // 4  # float32 elements
    nbytes = n * 4
    host = torch.empty(n, dtype=torch.float32, pin_memory=True)
    dev_a = torch.empty(n, dtype=torch.float32, device=dev)
    dev_b = torch.empty(n, dtype=torch.float32, device=dev)

    def _bw(fn) -> float:
        for _ in range(cfg.warmup):
            fn()
        torch.cuda.synchronize()
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        for _ in range(cfg.iters):
            fn()
        e.record()
        torch.cuda.synchronize()
        ms = s.elapsed_time(e) / cfg.iters
        return round(nbytes / (ms * 1e-3) / 1e9, 2)  # GB/s

    out = {
        "h2d_gb_s": _bw(lambda: dev_a.copy_(host, non_blocking=True)),
        "d2h_gb_s": _bw(lambda: host.copy_(dev_a, non_blocking=True)),
        "d2d_gb_s": _bw(lambda: dev_b.copy_(dev_a, non_blocking=True)),
    }
    del host, dev_a, dev_b
    torch.cuda.empty_cache()
    return out


def run(cfg: GpuConfig | None = None) -> dict[str, Any]:
    if not torch.cuda.is_available():
        return {"available": False, "reason": "CUDA not available"}
    cfg = cfg or GpuConfig()
    rep = device_report(0)
    rep_all = {
        "available": True,
        "devices": [device_report(i) for i in range(torch.cuda.device_count())],
        "bandwidth_mib_per_copy": cfg.copy_mib,
        "bandwidth": _bandwidth(cfg),
    }
    if rep["free_mem_gib"] < cfg.headroom_warn_gib:
        rep_all["warning"] = (
            f"Low GPU headroom: {rep['free_mem_gib']} GiB free "
            f"({rep['used_by_others_gib']} GiB used by other processes). "
            "Keep batch sizes small to avoid OOM-ing co-tenants."
        )
    return rep_all
