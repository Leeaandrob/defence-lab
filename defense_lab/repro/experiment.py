"""Experiment directory management -- the unit of reproducibility.

An ``Experiment`` materializes a self-contained run directory:

    experiments/<name>/<timestamp>__<cfghash>/
        config.yaml      # fully-resolved config
        env.json         # machine + library provenance (see repro.env)
        run.log          # full console log
        metrics.jsonl    # one JSON record per logged step/scalar
        metrics_summary.json
        artifacts/       # plots, tables, masks, checkpoints, reports

This mirrors how SAM/SAM2 research was run at scale: every datapoint a run
produced is traceable to the exact config, seed and environment that made it.
"""
from __future__ import annotations

import json
import time
from contextlib import AbstractContextManager
from datetime import datetime
from pathlib import Path
from types import TracebackType
from typing import Any

from defense_lab.config.base import config_hash, save_config, to_dict
from defense_lab.logging_utils import get_logger
from defense_lab.repro.env import save_environment, summarize
from defense_lab.repro.seed import seed_everything


class Experiment(AbstractContextManager):
    def __init__(
        self,
        name: str,
        config: Any | None = None,
        seed: int = 1234,
        deterministic: bool = False,
        root: str | Path = "experiments",
    ) -> None:
        self.name = name
        self.config = config
        self.seed = seed
        self.deterministic = deterministic
        cfg_tag = config_hash(config) if config is not None else "noconfig"
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.dir = Path(root) / name / f"{ts}__{cfg_tag}"
        self.artifacts = self.dir / "artifacts"
        self._metrics_fp = None
        self._t0 = 0.0
        self.logger = None

    # -- lifecycle ---------------------------------------------------------- #
    def __enter__(self) -> "Experiment":
        self.artifacts.mkdir(parents=True, exist_ok=True)
        self.logger = get_logger(f"exp.{self.name}", logfile=self.dir / "run.log")
        seed_everything(self.seed, self.deterministic)
        if self.config is not None:
            save_config(self.config, self.dir / "config.yaml")
        env = save_environment(self.dir / "env.json")
        self._metrics_fp = open(self.dir / "metrics.jsonl", "a")
        self._t0 = time.time()
        self.logger.info("=== experiment '%s' ===", self.name)
        self.logger.info("run dir: %s", self.dir)
        self.logger.info("seed=%d deterministic=%s", self.seed, self.deterministic)
        for line in summarize(env).splitlines():
            self.logger.info("env | %s", line.strip())
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> bool:
        elapsed = time.time() - self._t0
        if exc is not None:
            self.logger.error("experiment failed after %.1fs: %s", elapsed, exc)
        else:
            self.logger.info("experiment finished in %.1fs", elapsed)
        if self._metrics_fp is not None:
            self._metrics_fp.close()
        return False  # never suppress exceptions

    # -- recording ---------------------------------------------------------- #
    def log_metrics(self, step: int | None = None, **scalars: float) -> None:
        record = {"t": round(time.time() - self._t0, 4), "step": step, **scalars}
        self._metrics_fp.write(json.dumps(record) + "\n")
        self._metrics_fp.flush()
        msg = "  ".join(f"{k}={v}" for k, v in scalars.items())
        self.logger.info("metrics | step=%s %s", step, msg)

    def save_summary(self, summary: dict[str, Any]) -> Path:
        path = self.dir / "metrics_summary.json"
        with open(path, "w") as f:
            json.dump(summary, f, indent=2, sort_keys=True, default=str)
        self.logger.info("wrote summary -> %s", path)
        return path

    def save_json(self, name: str, obj: Any) -> Path:
        path = self.artifacts / name
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(obj, f, indent=2, sort_keys=True, default=str)
        return path

    def artifact_path(self, name: str) -> Path:
        p = self.artifacts / name
        p.parent.mkdir(parents=True, exist_ok=True)
        return p
