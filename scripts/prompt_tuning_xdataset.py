#!/usr/bin/env python3
"""Cross-dataset prompt-tuning: SAM2 zero-shot vs +prompt-tokens, same protocol.

Runs visual prompt-tuning (K learnable decoder tokens, ~2k params) on VDD,
satellite, and morocco-buildings; reports before/after box-prompted IoU per
dataset. Direct comparison to the LoRA cross-dataset table.

Usage:
    python scripts/prompt_tuning_xdataset.py --steps 400
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from defense_lab.datasets.hf_seg import load_hf_eval_samples  # noqa: E402
from defense_lab.datasets.vdd import load_vdd_eval_samples  # noqa: E402
from defense_lab.lora.prompt_tuning import PromptTuneConfig, Sam2PromptTuner  # noqa: E402
from defense_lab.repro.experiment import Experiment  # noqa: E402


def to_sample(e):
    return {"image": e.image, "gt": e.gt, "box": list(e.box), "point": None}


def vdd_sets():
    tr = [to_sample(e) for e in load_vdd_eval_samples("/home/ubuntu/.cache/vdd/VDD", "train", n_images=280, seed=1234)]
    ev = [to_sample(e) for e in load_vdd_eval_samples("/home/ubuntu/.cache/vdd/VDD", "val", n_images=20, seed=1234)]
    return tr, ev


def hf_sets(repo, image_col="image", fg=128, n=260, tr_n=200):
    s = load_hf_eval_samples(repo, split="train", n=n, min_area=300, image_col=image_col, foreground_threshold=fg)
    s = [to_sample(e) for e in s]
    return s[:tr_n], s[tr_n:]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--tokens", type=int, default=8)
    ap.add_argument("--name", default="prompt_tuning_xdataset")
    args = ap.parse_args()

    specs = [
        ("VDD drone land-cover", vdd_sets),
        ("satellite binary", lambda: hf_sets("saidines12/satellite-imagery-segmentation")),
        ("morocco buildings", lambda: hf_sets("tferhan/morocco_satellite_buildings_semantic_segmentation_512_v2",
                                              image_col="pixel_values", fg=1)),
    ]

    with Experiment(args.name, config={"steps": args.steps, "tokens": args.tokens}, seed=1234) as exp:
        rows = []
        for name, getter in specs:
            try:
                tr, ev = getter()
                exp.logger.info("%s: train=%d eval=%d", name, len(tr), len(ev))
                pt = Sam2PromptTuner(PromptTuneConfig(steps=args.steps, num_tokens=args.tokens))
                before = pt.evaluate(ev)
                hist = pt.train(tr)
                after = pt.evaluate(ev)
                rows.append({"dataset": name, "before": before["mean_iou"], "after": after["mean_iou"],
                             "delta": round(after["mean_iou"] - before["mean_iou"], 4),
                             "loss": [hist[0]["loss"], hist[-1]["loss"]], "trainable": pt.param_stats["trainable"]})
                exp.logger.info("%s: %.4f -> %.4f", name, before["mean_iou"], after["mean_iou"])
                del pt
                import torch
                torch.cuda.empty_cache()
            except Exception as e:
                rows.append({"dataset": name, "error": f"{type(e).__name__}: {e}"})
                exp.logger.error("%s failed: %s", name, e)

        exp.save_json("prompt_tuning_results.json", {"rows": rows})
        exp.save_summary({"rows": rows})

        print("\n=== Cross-dataset prompt-tuning (box-prompted IoU) ===")
        print(f"{'dataset':24s} {'before':>7} {'after':>7} {'delta':>7}")
        for r in rows:
            if "error" in r:
                print(f"{r['dataset']:24s}  ERROR: {r['error'][:40]}")
            else:
                print(f"{r['dataset']:24s} {r['before']:7.3f} {r['after']:7.3f} {r['delta']:+7.3f}")
        print(f"trainable params/dataset: {args.tokens}*256 = {args.tokens*256}")
        print(f"artifacts: {exp.artifacts}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
