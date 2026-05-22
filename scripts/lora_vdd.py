#!/usr/bin/env python3
"""LoRA domain adaptation on real VDD drone data — measure IoU lift.

Trains decoder LoRA (box-prompted, class-region samples) on the VDD train split,
evaluates on the VDD val split (same regions the zero-shot benchmark used).
Reports overall + per-class IoU before/after, loss curve, and saves the adapter.
Neutral land-cover classes only.

Usage:
    python scripts/lora_vdd.py --train-imgs 40 --val-imgs 20 --steps 200
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from defense_lab.datasets.vdd import VDD_CLASSES, load_vdd_eval_samples  # noqa: E402
from defense_lab.evaluation.metrics import mask_iou  # noqa: E402
from defense_lab.lora.inject import save_lora  # noqa: E402
from defense_lab.lora.layers import LoRAConfig  # noqa: E402
from defense_lab.lora.trainer import LoraFinetuneConfig, Sam2LoraTrainer  # noqa: E402
from defense_lab.repro.experiment import Experiment  # noqa: E402
from defense_lab.visualization.plots import loss_curve  # noqa: E402


def to_sample(es) -> dict:
    return {"image": es.image, "gt": es.gt, "box": list(es.box), "point": None, "cls": es.sample_id[1]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/home/ubuntu/.cache/vdd/VDD")
    ap.add_argument("--train-imgs", type=int, default=40)
    ap.add_argument("--val-imgs", type=int, default=20)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--name", default="lora_vdd")
    args = ap.parse_args()

    cfg = LoraFinetuneConfig(steps=args.steps, lr=args.lr, seed=args.seed,
                             lora=LoRAConfig(r=args.rank, alpha=2 * args.rank))
    with Experiment(args.name, config={"finetune": {"steps": args.steps, "lr": args.lr},
                                       "train_imgs": args.train_imgs, "val_imgs": args.val_imgs}, seed=args.seed) as exp:
        train = [to_sample(e) for e in load_vdd_eval_samples(args.root, "train", n_images=args.train_imgs, seed=args.seed)]
        val = [to_sample(e) for e in load_vdd_eval_samples(args.root, "val", n_images=args.val_imgs, seed=1234)]
        exp.logger.info("train regions=%d  val regions=%d", len(train), len(val))

        tr = Sam2LoraTrainer(cfg)
        exp.logger.info("param stats: %s", tr.param_stats)

        def per_class(samps):
            g = defaultdict(list)
            for s in samps:
                g[s["cls"]].append(mask_iou(tr.predict_mask(s), s["gt"]))
            return {c: round(float(np.mean(v)), 4) for c, v in sorted(g.items())}

        before = tr.evaluate(val)
        pc_before = per_class(val)
        exp.logger.info("BEFORE val IoU=%.4f", before["mean_iou"])

        history = tr.train(train)
        for h in history:
            exp.log_metrics(step=h["step"], loss=h["loss"])

        after = tr.evaluate(val)
        pc_after = per_class(val)
        exp.logger.info("AFTER  val IoU=%.4f", after["mean_iou"])

        adapter = exp.artifact_path("vdd_lora_adapter.safetensors")
        save_lora(tr.model, adapter)
        loss_curve(history, exp.artifact_path("loss_curve.png"))

        results = {
            "param_stats": tr.param_stats,
            "val_iou_before": before["mean_iou"], "val_iou_after": after["mean_iou"],
            "per_class_before": {VDD_CLASSES.get(c, str(c)): v for c, v in pc_before.items()},
            "per_class_after": {VDD_CLASSES.get(c, str(c)): v for c, v in pc_after.items()},
            "loss_history": history, "adapter_kb": round(adapter.stat().st_size / 1024, 1),
        }
        exp.save_json("lora_vdd_results.json", results)
        exp.save_summary(results)

        print("\n=== LoRA on VDD (real drone) ===")
        ps = tr.param_stats
        print(f"trainable {ps['trainable']:,}/{ps['total']:,} ({ps['trainable_pct']}%)  adapter {results['adapter_kb']} KB")
        print(f"overall val IoU: {before['mean_iou']:.4f} -> {after['mean_iou']:.4f}")
        print(f"loss: {history[0]['loss']:.3f} -> {history[-1]['loss']:.3f}")
        print("per-class IoU (before -> after):")
        for c in sorted(set(pc_before) | set(pc_after)):
            nm = VDD_CLASSES.get(c, str(c))
            print(f"  {nm:11s} {pc_before.get(c, float('nan')):.3f} -> {pc_after.get(c, float('nan')):.3f}")
        print(f"artifacts: {exp.artifacts}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
