"""Environment metadata capture for reproducibility & provenance.

Every run records *exactly* the machine it ran on: arch, CPU, RAM, GPU(s),
driver, CUDA/cuDNN, key library versions, relevant env vars, and the git
revision of this codebase. This is what lets a benchmark number be trusted and
an experiment be re-run months later -- a hard requirement for a research lab.
"""
from __future__ import annotations

import datetime as _dt
import importlib.metadata as _md
import json
import os
import platform
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any


def _run(cmd: list[str]) -> str | None:
    exe = shutil.which(cmd[0])
    if exe is None:
        return None
    try:
        out = subprocess.run(
            [exe, *cmd[1:]], capture_output=True, text=True, timeout=20, check=False
        )
        return out.stdout.strip() or None
    except Exception:
        return None


def _meminfo() -> dict[str, float]:
    info: dict[str, float] = {}
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            k, _, rest = line.partition(":")
            kb = float(rest.strip().split()[0])
            if k in ("MemTotal", "MemAvailable"):
                info[k] = round(kb / 1024 / 1024, 2)  # GiB
    except Exception:
        pass
    return info


def _cpu_model() -> str | None:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            # x86 uses "model name"; ARM Neoverse exposes "CPU implementer"/part.
            if line.lower().startswith(("model name", "cpu part")):
                return line.split(":", 1)[1].strip()
    except Exception:
        pass
    return platform.processor() or None


def _git_info() -> dict[str, Any]:
    sha = _run(["git", "rev-parse", "HEAD"])
    status = _run(["git", "status", "--porcelain"])
    branch = _run(["git", "rev-parse", "--abbrev-ref", "HEAD"])
    return {
        "sha": sha,  # None if no commits yet
        "branch": branch,
        "dirty": bool(status) if status is not None else None,
    }


def _lib_versions() -> dict[str, str]:
    libs = [
        "torch", "torchvision", "numpy", "opencv-python", "pillow",
        "matplotlib", "pandas", "einops", "flash-attn", "sam2",
        "pycocotools", "decord", "av", "hydra-core", "omegaconf",
    ]
    out: dict[str, str] = {}
    for lib in libs:
        try:
            out[lib] = _md.version(lib)
        except Exception:
            out[lib] = "not-installed"
    return out


def _gpu_info() -> list[dict[str, Any]]:
    gpus: list[dict[str, Any]] = []
    try:
        import torch

        if not torch.cuda.is_available():
            return gpus
        for i in range(torch.cuda.device_count()):
            p = torch.cuda.get_device_properties(i)
            free, total = torch.cuda.mem_get_info(i)
            gpus.append(
                {
                    "index": i,
                    "name": p.name,
                    "capability": f"{p.major}.{p.minor}",
                    "total_mem_gib": round(p.total_memory / 2**30, 2),
                    "free_mem_gib": round(free / 2**30, 2),
                    "used_by_others_gib": round((total - free) / 2**30, 2),
                    "multiprocessors": p.multi_processor_count,
                }
            )
    except Exception:
        pass
    return gpus


def _nvidia_smi() -> dict[str, Any]:
    raw = _run(
        [
            "nvidia-smi",
            "--query-gpu=driver_version,utilization.gpu,memory.used,memory.total,temperature.gpu",
            "--format=csv,noheader,nounits",
        ]
    )
    out: dict[str, Any] = {}
    if raw:
        rows = []
        for line in raw.splitlines():
            drv, util, mused, mtot, temp = [x.strip() for x in line.split(",")]
            rows.append(
                {
                    "driver": drv,
                    "util_pct": float(util),
                    "mem_used_mib": float(mused),
                    "mem_total_mib": float(mtot),
                    "temp_c": float(temp),
                }
            )
        out["per_gpu"] = rows
        out["driver"] = rows[0]["driver"] if rows else None
    return out


# env vars that materially affect numerics / performance reproducibility
_RELEVANT_ENV = [
    "CUDA_VISIBLE_DEVICES", "PYTORCH_CUDA_ALLOC_CONF", "CUBLAS_WORKSPACE_CONFIG",
    "OMP_NUM_THREADS", "PYTHONHASHSEED", "TORCH_CUDNN_V8_API_ENABLED",
    "NVIDIA_TF32_OVERRIDE", "HF_HOME", "TORCH_HOME",
]


def capture_environment() -> dict[str, Any]:
    """Return a JSON-serializable snapshot of the runtime environment."""
    snap: dict[str, Any] = {
        "captured_at": _dt.datetime.now().astimezone().isoformat(),
        "hostname": platform.node(),
        "platform": platform.platform(),
        "arch": platform.machine(),
        "python": platform.python_version(),
        "cpu": {"model": _cpu_model(), "logical_cores": os.cpu_count()},
        "memory_gib": _meminfo(),
        "git": _git_info(),
        "libraries": _lib_versions(),
        "env_vars": {k: os.environ.get(k) for k in _RELEVANT_ENV},
    }
    try:
        import torch

        snap["torch"] = {
            "version": torch.__version__,
            "cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "cuda_available": torch.cuda.is_available(),
            "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        }
    except Exception as e:  # pragma: no cover
        snap["torch"] = {"error": str(e)}

    snap["gpus"] = _gpu_info()
    snap["nvidia_smi"] = _nvidia_smi()
    return snap


def save_environment(path: str | Path) -> dict[str, Any]:
    snap = capture_environment()
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        json.dump(snap, f, indent=2, sort_keys=True)
    return snap


def summarize(snap: dict[str, Any]) -> str:
    """One-screen human summary for console output."""
    g = snap.get("gpus") or [{}]
    g0 = g[0]
    mem = snap.get("memory_gib", {})
    return re.sub(
        r"\n +", "\n",
        f"""
        host={snap.get('hostname')}  arch={snap.get('arch')}  py={snap.get('python')}
        cpu={snap.get('cpu', {}).get('logical_cores')} cores  ram={mem.get('MemTotal')} GiB (avail {mem.get('MemAvailable')})
        torch={snap.get('torch', {}).get('version')}  cuda={snap.get('torch', {}).get('cuda')}  cudnn={snap.get('torch', {}).get('cudnn')}
        gpu={g0.get('name')}  cc={g0.get('capability')}  vram={g0.get('total_mem_gib')} GiB (free {g0.get('free_mem_gib')}, used-by-others {g0.get('used_by_others_gib')})
        git={snap.get('git', {}).get('sha')}  dirty={snap.get('git', {}).get('dirty')}
        """.strip(),
    )
