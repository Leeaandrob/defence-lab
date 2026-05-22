#!/usr/bin/env python3
"""Phase 4 -- LoRA domain adaptation of SAM2.

Adapts SAM2 to a synthetic *low-SNR operational* domain (faint, low-contrast
elliptical "vehicles" on noisy background -- a stand-in for IR/aerial imagery
where zero-shot SAM2 has headroom). We freeze the whole model, train only LoRA
adapters on the mask decoder (add `--` adapt_encoder=true for encoder LoRA),
and report held-out IoU before vs after, the loss curve, and verify the saved
adapter reloads to the same accuracy.

Usage
-----
    python scripts/phase4_lora_finetune.py
    python scripts/phase4_lora_finetune.py steps=200 lora.r=16 adapt_encoder=true
    python scripts/phase4_lora_finetune.py contrast=14 n_train=12
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from defense_lab.config.base import build_config, to_dict  # noqa: E402
from defense_lab.lora.inject import load_lora, save_lora  # noqa: E402
from defense_lab.lora.trainer import LoraFinetuneConfig, Sam2LoraTrainer  # noqa: E402
from defense_lab.repro.experiment import Experiment  # noqa: E402
from defense_lab.visualization.overlay import render_panels  # noqa: E402


@dataclass
class Phase4Config:
    seed: int = 1234
    size: int = 768
    contrast: float = 5.0      # ellipse-vs-background delta (low => harder => more headroom)
    noise: float = 16.0
    n_train: int = 16
    n_val: int = 8
    finetune: LoraFinetuneConfig = field(default_factory=LoraFinetuneConfig)


def make_sample(rng: np.random.Generator, size: int, contrast: float, noise: float) -> dict:
    """A faint ellipse on noisy background; prompt = center point; GT = ellipse."""
    import cv2

    bg = float(rng.uniform(55, 85))
    img = np.clip(rng.normal(bg, noise, (size, size, 3)), 0, 255).astype(np.float32)
    cx, cy = rng.integers(int(size * 0.3), int(size * 0.7), size=2)
    ax, ay = rng.integers(int(size * 0.10), int(size * 0.18), size=2)
    angle = float(rng.uniform(0, 180))
    gt = np.zeros((size, size), np.uint8)
    cv2.ellipse(gt, (int(cx), int(cy)), (int(ax), int(ay)), angle, 0, 360, 1, -1)
    tint = rng.uniform(-0.3, 0.3, 3)
    img[gt.astype(bool)] += contrast * (1.0 + tint)
    img = cv2.GaussianBlur(np.clip(img, 0, 255).astype(np.uint8), (5, 5), 0)
    return {"image": img, "gt": gt.astype(bool), "point": (float(cx), float(cy)), "box": None}


def main() -> int:
    ap = argparse.ArgumentParser(description="Phase 4 LoRA domain adaptation")
    ap.add_argument("--config", type=str, default=None)
    ap.add_argument("--name", type=str, default="phase4_lora_finetune")
    ap.add_argument("overrides", nargs="*")
    args = ap.parse_args()
    cfg = build_config(Phase4Config, args.config, list(args.overrides))

    with Experiment(args.name, config=cfg, seed=cfg.seed) as exp:
        rng = np.random.default_rng(cfg.seed)
        train_set = [make_sample(rng, cfg.size, cfg.contrast, cfg.noise) for _ in range(cfg.n_train)]
        val_set = [make_sample(rng, cfg.size, cfg.contrast, cfg.noise) for _ in range(cfg.n_val)]

        trainer = Sam2LoraTrainer(cfg.finetune)
        exp.logger.info("param stats: %s", trainer.param_stats)

        before = trainer.evaluate(val_set)
        before_train = trainer.evaluate(train_set)
        exp.logger.info("BEFORE  val IoU=%.4f  train IoU=%.4f", before["mean_iou"], before_train["mean_iou"])

        # capture zero-shot masks now (adapters are identity at init) for before/after figures
        viz_idx = list(range(min(2, len(val_set))))
        masks_before = [trainer.predict_mask(val_set[i]) for i in viz_idx]

        history = trainer.train(train_set)
        for h in history:
            exp.log_metrics(step=h["step"], loss=h["loss"])

        after = trainer.evaluate(val_set)
        after_train = trainer.evaluate(train_set)
        exp.logger.info("AFTER   val IoU=%.4f  train IoU=%.4f", after["mean_iou"], after_train["mean_iou"])

        # save adapter + verify reload reproduces accuracy
        adapter_path = exp.artifact_path("lora_adapter.safetensors")
        save_lora(trainer.model, adapter_path)
        size_kb = adapter_path.stat().st_size / 1024
        fresh = Sam2LoraTrainer(cfg.finetune)
        n_loaded, n_missing = load_lora(fresh.model, adapter_path)
        reloaded = fresh.evaluate(val_set)

        # qualitative before/after figures from cached zero-shot vs adapted masks
        for i in viz_idx:
            s = val_set[i]
            render_panels(
                s["image"], [s["gt"], masks_before[i], trainer.predict_mask(s)],
                ["ground truth", "base SAM2 (zero-shot)", "LoRA-adapted"],
                exp.artifact_path(f"val_{i}_before_after.png"),
            )

        results = {
            "config": to_dict(cfg),
            "param_stats": trainer.param_stats,
            "val_iou_before": before["mean_iou"], "val_iou_after": after["mean_iou"],
            "train_iou_before": before_train["mean_iou"], "train_iou_after": after_train["mean_iou"],
            "val_dice_before": before["mean_dice"], "val_dice_after": after["mean_dice"],
            "loss_history": history,
            "adapter_kb": round(size_kb, 1),
            "reload": {"loaded_tensors": n_loaded, "missing_adapters": n_missing,
                       "reloaded_val_iou": reloaded["mean_iou"], "matches_after": reloaded["mean_iou"] == after["mean_iou"]},
        }
        exp.save_json("phase4_results.json", results)
        exp.save_summary(results)

        print("\n=== Phase 4 LoRA domain adaptation ===")
        ps = trainer.param_stats
        print(f"adapters : {ps['n_wrapped_linears']} Linear layers wrapped; "
              f"trainable {ps['trainable']:,}/{ps['total']:,} params ({ps['trainable_pct']}%)")
        print(f"val  IoU : {before['mean_iou']:.4f} → {after['mean_iou']:.4f}")
        print(f"train IoU: {before_train['mean_iou']:.4f} → {after_train['mean_iou']:.4f}")
        print(f"loss     : {history[0]['loss']:.3f} → {history[-1]['loss']:.3f}")
        print(f"adapter  : {results['adapter_kb']} KB; reload IoU={reloaded['mean_iou']:.4f} "
              f"(matches: {results['reload']['matches_after']})")
        print(f"artifacts: {exp.artifacts}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
