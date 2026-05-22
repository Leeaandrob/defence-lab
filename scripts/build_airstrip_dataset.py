#!/usr/bin/env python3
"""Build the clandestine-airstrip segmentation dataset (S1-AAD HF mirror).

Materializes ls-da3m0ns/amazon_airstrips into a clean train/test layout with
positives (airstrip present) and negatives (none), + a manifest. This is the
has-U-Net-baseline task for the SIGE objective. The full 1040-image Mendeley
S1-AAD (DOI 10.17632/x7rn78ymtn.1, CC BY 4.0) is a manual scale-up: drop its
zip and re-point this script.

Usage:
    python scripts/build_airstrip_dataset.py --out /home/ubuntu/.cache/airstrip_s1aad --test-frac 0.2
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="ls-da3m0ns/amazon_airstrips")
    ap.add_argument("--split", default="train")
    ap.add_argument("--out", default="/home/ubuntu/.cache/airstrip_s1aad")
    ap.add_argument("--test-frac", type=float, default=0.2)
    ap.add_argument("--min-area", type=int, default=200)
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    import cv2
    from datasets import load_dataset

    ds = load_dataset(args.repo, split=args.split)
    out = Path(args.out)
    rng = np.random.default_rng(args.seed)

    # classify pos/neg + stratified train/test split
    recs = []
    for i in range(len(ds)):
        lab = np.array(ds[i]["label"].convert("L"))
        area = int((lab >= 128).sum())
        recs.append({"idx": i, "positive": area >= args.min_area, "area": area})
    pos = [r for r in recs if r["positive"]]
    neg = [r for r in recs if not r["positive"]]
    for grp in (pos, neg):
        rng.shuffle(grp)
    def split(grp):
        k = int(round(len(grp) * args.test_frac))
        return grp[k:], grp[:k]
    pos_tr, pos_te = split(pos)
    neg_tr, neg_te = split(neg)
    assign = {}
    for r in pos_tr + neg_tr: assign[r["idx"]] = "train"
    for r in pos_te + neg_te: assign[r["idx"]] = "test"

    manifest = []
    for sp in ("train", "test"):
        (out / sp / "images").mkdir(parents=True, exist_ok=True)
        (out / sp / "masks").mkdir(parents=True, exist_ok=True)
    for r in recs:
        i = r["idx"]; sp = assign[i]
        img = np.array(ds[i]["image"].convert("RGB"))
        msk = (np.array(ds[i]["label"].convert("L")) >= 128).astype(np.uint8) * 255
        fn = f"{i:04d}.png"
        cv2.imwrite(str(out / sp / "images" / fn), img[:, :, ::-1])
        cv2.imwrite(str(out / sp / "masks" / fn), msk)
        manifest.append({"file": fn, "split": sp, "positive": r["positive"],
                         "area": r["area"], "h": int(img.shape[0]), "w": int(img.shape[1])})

    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    stats = {"total": len(recs), "positives": len(pos), "negatives": len(neg),
             "train": {"pos": len(pos_tr), "neg": len(neg_tr)},
             "test": {"pos": len(pos_te), "neg": len(neg_te)},
             "image_size": [manifest[0]["h"], manifest[0]["w"]] if manifest else None,
             "out": str(out), "source": args.repo}
    (out / "stats.json").write_text(json.dumps(stats, indent=2))

    print("\n=== airstrip dataset built ===")
    print(json.dumps(stats, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
