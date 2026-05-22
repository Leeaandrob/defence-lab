"""VLM-assisted semantic labeling of class-agnostic masks (Claude vision).

SAM2/AMG produce class-agnostic regions; this assigns a semantic class to each
via Claude vision, closing the "AMG has no class" gap in the data engine. All
top-K regions of a frame are numbered onto one overlay and labeled in a single
call (amortizes per-call overhead). Vocabulary-constrained, JSON-parsed.
Pseudo-labels — verify a subset before trusting.

Backends:
  * "cli": shells `claude -p --model <m>` (Claude Code auth, no API key; ~$0.03-0.07
    per call from session context — so batch many regions per call).
  * "api": anthropic SDK + image base64 (needs ANTHROPIC_API_KEY; ~$0.002/call).
"""
from __future__ import annotations

import base64
import json
import re
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

DEFAULT_VOCAB = ["road", "building", "rooftop", "vegetation", "forest", "water",
                 "vehicle", "farmland", "bare_soil", "other"]
_MODEL_FULL = {"haiku": "claude-haiku-4-5", "sonnet": "claude-sonnet-4-6", "opus": "claude-opus-4-7"}


@dataclass
class VLMLabelerConfig:
    model: str = "haiku"
    backend: str = "cli"            # "cli" | "api"
    vocab: tuple = tuple(DEFAULT_VOCAB)
    top_k: int = 10
    max_budget_usd: float = 0.20


class ClaudeVLMLabeler:
    def __init__(self, cfg: VLMLabelerConfig | None = None) -> None:
        self.cfg = cfg or VLMLabelerConfig()

    def _overlay(self, image: np.ndarray, instances: list) -> Path:
        import cv2

        vis = np.ascontiguousarray(image[:, :, ::-1]).copy()  # RGB->BGR for cv2
        for k, ins in enumerate(instances):
            m = ins.mask.astype(np.uint8)
            cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(vis, cnts, -1, (0, 0, 255), 2)
            ys, xs = np.where(ins.mask)
            cx, cy = int(xs.mean()), int(ys.mean())
            cv2.putText(vis, str(k), (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 255), 3)
        tmp = Path(tempfile.mkstemp(suffix=".png")[1])
        cv2.imwrite(str(tmp), vis)
        return tmp

    def _prompt(self, n: int, path: Path | None) -> str:
        vocab = ", ".join(self.cfg.vocab)
        head = (f"Use the Read tool to open the image at {path}. " if path is not None
                else "The attached image shows ")
        return (head + f"It shows {n} regions outlined in red, each tagged with a yellow "
                f"number 0..{n-1}. For EACH number pick the single best class from: [{vocab}]. "
                f'Output ONLY a JSON object number->class, e.g. {{"0":"farmland","1":"road"}}. No prose.')

    def _run_cli(self, prompt: str) -> tuple[str, float]:
        cmd = ["claude", "-p", prompt, "--model", self.cfg.model, "--output-format", "json",
               "--allowedTools", "Read", "--max-budget-usd", str(self.cfg.max_budget_usd)]
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        try:
            d = json.loads(out.stdout)
        except Exception:
            return "", 0.0
        return (d.get("result") or ""), float(d.get("total_cost_usd") or 0.0)

    def _run_api(self, prompt: str, img_path: Path) -> tuple[str, float]:
        import anthropic

        client = anthropic.Anthropic()
        b64 = base64.b64encode(img_path.read_bytes()).decode()
        model = _MODEL_FULL.get(self.cfg.model, self.cfg.model)
        msg = client.messages.create(
            model=model, max_tokens=1024,
            messages=[{"role": "user", "content": [
                {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": b64}},
                {"type": "text", "text": prompt}]}],
        )
        txt = "".join(b.text for b in msg.content if getattr(b, "type", None) == "text")
        return txt, 0.0

    @staticmethod
    def _parse(text: str) -> dict:
        m = re.search(r"\{.*\}", text, re.S)
        if not m:
            return {}
        try:
            return json.loads(m.group(0))
        except Exception:
            return {}

    def label(self, image: np.ndarray, instances: list) -> tuple[list, float]:
        """Assign category_name to the top-K instances by area. Returns (labeled, cost_usd)."""
        insts = sorted(instances, key=lambda i: i.area, reverse=True)[: self.cfg.top_k]
        if not insts:
            return [], 0.0
        path = self._overlay(image, insts)
        try:
            prompt = self._prompt(len(insts), path if self.cfg.backend == "cli" else None)
            text, cost = (self._run_api(prompt, path) if self.cfg.backend == "api"
                          else self._run_cli(prompt))
        finally:
            path.unlink(missing_ok=True)
        labels = self._parse(text)
        for k, ins in enumerate(insts):
            ins.category_name = labels.get(str(k)) or labels.get(k) or "unlabeled"
        return insts, cost
