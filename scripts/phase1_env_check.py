#!/usr/bin/env python3
"""Phase 1 -- Foundation Environment validation & benchmarking.

Validates CUDA, bf16 and Flash Attention, then benchmarks matmul throughput,
attention backends and memory bandwidth on this machine. Everything is written
to a reproducible experiment directory with full env provenance.

Usage
-----
    python scripts/phase1_env_check.py
    python scripts/phase1_env_check.py --config configs/phase1.yaml
    python scripts/phase1_env_check.py --quick
    python scripts/phase1_env_check.py precision.matmul_size=8192 attention.seq_len=4096

Any trailing ``key.subpath=value`` tokens are Hydra-style overrides.
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path

# allow running as a plain script without installing the package
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from defense_lab.config.base import build_config, to_dict  # noqa: E402
from defense_lab.diagnostics import attention as attn_diag  # noqa: E402
from defense_lab.diagnostics import gpu as gpu_diag  # noqa: E402
from defense_lab.diagnostics import precision as prec_diag  # noqa: E402
from defense_lab.repro.experiment import Experiment  # noqa: E402


@dataclass
class Phase1Config:
    seed: int = 1234
    deterministic: bool = False
    gpu: gpu_diag.GpuConfig = field(default_factory=gpu_diag.GpuConfig)
    precision: prec_diag.PrecisionConfig = field(default_factory=prec_diag.PrecisionConfig)
    attention: attn_diag.AttentionConfig = field(default_factory=attn_diag.AttentionConfig)


def _markdown_report(results: dict) -> str:
    lines = ["# Phase 1 -- Foundation Environment Report", ""]
    p = results.get("precision", {})
    if p.get("available"):
        lines += [f"## Matmul throughput (n={p['matmul_size']})", "",
                  "| dtype | latency (ms) | TFLOPS |", "|---|---|---|"]
        for d, v in p["throughput"].items():
            lines.append(f"| {d} | {v['latency_ms']} | {v['tflops']} |")
        bf = p.get("bf16_validation", {})
        lines += ["", f"- bf16 supported: **{p.get('bf16_supported')}**",
                  f"- bf16 vs fp32 relative error: **{bf.get('relative_frobenius_error'):.3e}** "
                  f"(passed: {bf.get('passed')})", ""]
    a = results.get("attention", {})
    if a.get("available"):
        lines += ["## Attention (SDPA) backends -- bf16 fwd+bwd", "",
                  "| backend | supported | latency (ms) |", "|---|---|---|"]
        for name, v in a["sdpa_backends"].items():
            lines.append(f"| {name} | {v.get('supported')} | {v.get('fwd_bwd_latency_ms', '-')} |")
        fa = a.get("flash_attn_package", {})
        lines += ["", f"- flash_attn package: **{fa.get('installed')}** "
                  f"(v{fa.get('version', '-')}, kernel_ran={fa.get('kernel_ran')})", ""]
    g = results.get("gpu", {})
    if g.get("available"):
        d0 = g["devices"][0]
        bw = g["bandwidth"]
        lines += ["## GPU & memory bandwidth", "",
                  f"- device: **{d0['name']}** (cc {d0['capability']}, {d0['multiprocessors']} SMs)",
                  f"- VRAM: {d0['total_mem_gib']} GiB total, {d0['free_mem_gib']} GiB free, "
                  f"{d0['used_by_others_gib']} GiB used by other processes",
                  f"- H2D: {bw['h2d_gb_s']} GB/s | D2H: {bw['d2h_gb_s']} GB/s | D2D: {bw['d2d_gb_s']} GB/s"]
        if "warning" in g:
            lines += ["", f"> ⚠️ {g['warning']}"]
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description="Phase 1 foundation environment check")
    ap.add_argument("--config", type=str, default=None, help="YAML config path")
    ap.add_argument("--name", type=str, default="phase1_env_check")
    ap.add_argument("--quick", action="store_true", help="tiny sizes for a fast smoke test")
    ap.add_argument("overrides", nargs="*", help="key.subpath=value Hydra-style overrides")
    args = ap.parse_args()

    overrides = list(args.overrides)
    if args.quick:
        overrides += ["precision.matmul_size=1024", "precision.iters=5", "precision.warmup=2",
                      "attention.seq_len=512", "attention.iters=3", "gpu.iters=3"]

    cfg = build_config(Phase1Config, args.config, overrides)

    with Experiment(args.name, config=cfg, seed=cfg.seed, deterministic=cfg.deterministic) as exp:
        exp.logger.info("running GPU diagnostics ...")
        gpu_res = gpu_diag.run(cfg.gpu)
        exp.logger.info("running precision diagnostics ...")
        prec_res = prec_diag.run(cfg.precision)
        exp.logger.info("running attention diagnostics ...")
        attn_res = attn_diag.run(cfg.attention)

        results = {"config": to_dict(cfg), "gpu": gpu_res, "precision": prec_res, "attention": attn_res}
        exp.save_json("phase1_results.json", results)

        md = _markdown_report(results)
        report_path = exp.artifact_path("phase1_report.md")
        report_path.write_text(md)
        exp.logger.info("wrote report -> %s", report_path)

        # log headline scalars to metrics
        if prec_res.get("available"):
            for d, v in prec_res["throughput"].items():
                exp.log_metrics(dtype_tflops=v["tflops"], step=None, **{f"{d}_tflops": v["tflops"]})
        exp.save_summary({
            "bf16_tflops": prec_res.get("throughput", {}).get("bf16", {}).get("tflops"),
            "bf16_valid": prec_res.get("bf16_validation", {}).get("passed"),
            "flash_attn_ok": attn_res.get("flash_attn_package", {}).get("kernel_ran"),
            "gpu": gpu_res.get("devices", [{}])[0].get("name"),
            "free_vram_gib": gpu_res.get("devices", [{}])[0].get("free_mem_gib"),
        })
        print("\n" + md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
