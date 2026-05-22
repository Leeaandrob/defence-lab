"""Minimal, dependency-free structured logging.

A research run should emit the same human-readable stream to the console and to
a per-experiment ``run.log`` file, so we attach an optional ``FileHandler``.
We avoid ``rich``/external deps to keep the ARM64 footprint small.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
_DATEFMT = "%Y-%m-%d %H:%M:%S"


def get_logger(
    name: str = "defense_lab",
    logfile: str | Path | None = None,
    level: int = logging.INFO,
) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False

    has_stream = any(isinstance(h, logging.StreamHandler) for h in logger.handlers)
    if not has_stream:
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(logging.Formatter(_FORMAT, _DATEFMT))
        logger.addHandler(sh)

    if logfile is not None:
        logfile = Path(logfile)
        logfile.parent.mkdir(parents=True, exist_ok=True)
        already = any(
            isinstance(h, logging.FileHandler)
            and Path(getattr(h, "baseFilename", "")) == logfile.resolve()
            for h in logger.handlers
        )
        if not already:
            fh = logging.FileHandler(logfile)
            fh.setFormatter(logging.Formatter(_FORMAT, _DATEFMT))
            logger.addHandler(fh)
    return logger
