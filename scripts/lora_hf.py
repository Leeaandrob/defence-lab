#!/usr/bin/env python3
"""LoRA domain adaptation on any HF image+mask segmentation dataset — IoU lift.

Loads box-prompted class/foreground-region samples, splits train/eval, trains
decoder LoRA, reports IoU before/after + loss curve + adapter. Neutral
land-feature segmentation.

Usage:
    python scripts/lora_hf.py --repo ls-da3m0ns/amazon_airstrips --train 100 --eval 40 --steps 250
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from defense_lab.datasets.hf_seg import load_hf_eval_samples  # noqa: E402
from defense_lab.lora.inject import save_lora  # noqa: E402
from defense_lab.lora.trainer import LoraFinetuneConfig, Sam2LoraTrainer  # noqa: E402
from defense_lab.repro.experiment import Experiment  # noqa: E402
from defense_lab.visualization.plots import loss_curve  # noqa: E402


def to_sample(es) -> dict:
    return {"image": es.image, "gt": es.gt, "box": list(es.box), "point": None}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--split", default="train")
    ap.add_argument("--train", type=int, default=100)
    ap.add_argument("--eval", type=int, default=40)
    ap.add_argument("--steps", type=int, default=250)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--image-col", default="image")
    ap.add_argument("--label-col", default="label")
    ap.add_argument("--fg-thresh", type=int, default=128)
    ap.add_argument("--name", default="lora_hf")
    args = ap.parse_args()

    with Experiment(args.name, config={"repo": args.repo, "train": args.train, "eval": args.eval,
                                       "steps": args.steps, "lr": args.lr}, seed=1234) as exp:
        n = args.train + args.eval
        samples = load_hf_eval_samples(args.repo, split=args.split, n=n, min_area=300,
                                       image_col=args.image_col, label_col=args.label_col,
                                       foreground_threshold=args.fg_thresh)
        if len(samples) < args.train + 5:
            exp.logger.error("only %d usable samples", len(samples)); return 1
        train = [to_sample(e) for e in samples[: args.train]]
        val = [to_sample(e) for e in samples[args.train:]]
        exp.logger.info("train=%d val=%d", len(train), len(val))

        tr = Sam2LoraTrainer(LoraFinetuneConfig(steps=args.steps, lr=args.lr))
        exp.logger.info("params: %s", tr.param_stats)
        before = tr.evaluate(val)
        history = tr.train(train)
        for h in history:
            exp.log_metrics(step=h["step"], loss=h["loss"])
        after = tr.evaluate(val)

        adapter = exp.artifact_path("adapter.safetensors")
        save_lora(tr.model, adapter)
        loss_curve(history, exp.artifact_path("loss_curve.png"))
        results = {"repo": args.repo, "param_stats": tr.param_stats,
                   "val_iou_before": before["mean_iou"], "val_iou_after": after["mean_iou"],
                   "val_dice_before": before["mean_dice"], "val_dice_after": after["mean_dice"],
                   "adapter_kb": round(adapter.stat().st_size / 1024, 1), "loss_history": history}
        exp.save_json("results.json", results)
        exp.save_summary(results)

        print(f"\n=== LoRA on {args.repo} ===")
        ps = tr.param_stats
        print(f"trainable {ps['trainable']:,}/{ps['total']:,} ({ps['trainable_pct']}%)  adapter {results['adapter_kb']} KB")
        print(f"val IoU : {before['mean_iou']:.4f} -> {after['mean_iou']:.4f}")
        print(f"val Dice: {before['mean_dice']:.4f} -> {after['mean_dice']:.4f}")
        print(f"loss    : {history[0]['loss']:.3f} -> {history[-1]['loss']:.3f}")
        print(f"artifacts: {exp.artifacts}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
