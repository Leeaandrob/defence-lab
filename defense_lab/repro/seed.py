"""Global seeding for reproducible foundation-model experiments.

``seed_everything`` covers Python, NumPy and PyTorch (CPU + all CUDA devices).
The ``deterministic`` switch trades throughput for bit-reproducibility -- on a
GH200 we usually keep cuDNN autotuning *on* for benchmarking and only flip
determinism on for the ablations that must be exactly reproducible.
"""
from __future__ import annotations

import os
import random

import numpy as np
import torch


def seed_everything(seed: int = 1234, deterministic: bool = False) -> int:
    """Seed all RNGs. Returns the seed so callers can log it.

    Args:
        seed: the integer seed.
        deterministic: if True, force deterministic kernels (slower) and set the
            cuBLAS workspace config required for deterministic matmuls.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        # cuBLAS needs this set *before* the first CUDA call for determinism.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True
    return seed
