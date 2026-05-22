#!/usr/bin/env python3
"""No-oracle evaluation: automatic prompting (AMG) -> match GT -> detection metrics.

Removes the oracle box. SAM2's automatic mask generator proposes class-agnostic
masks; we greedily match them to GT instances and report the operational metrics
that actually matter: recall@0.5, precision@0.5, false-positives per image, and
mean IoU on matched instances. This measures the real bottleneck (automatic
localization), not box-given mask refinement.

Usage:
    python scripts/no_oracle_eval.py --n 15
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from defense_lab.annotations.auto_mask import AutoMaskConfig, AutoMaskLabeler  # noqa: E402
from defense_lab.evaluation.metrics import mask_iou  # noqa: E402
from defense_lab.repro.experiment import Experiment  # noqa: E402


def _resize(img, mask, long_side=1024):
    import cv2

    h, w = mask.shape
    s = long_side / max(h, w)
    if s < 1:
        img = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
        mask = cv2.resize(mask, (int(w * s), int(h * s)), interpolation=cv2.INTER_NEAREST)
    return img, mask


def gt_instances(mask, ignore=0, min_area=200, binary=False, thresh=128):
    import cv2

    insts = []
    classes = [1] if binary else [int(c) for c in np.unique(mask) if int(c) != ignore]
    for c in classes:
        m = (mask >= thresh) if binary else (mask == c)
        n, lab, stats, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8), connectivity=8)
        for i in range(1, n):
            if stats[i, cv2.CC_STAT_AREA] >= min_area:
                insts.append(lab == i)
    return insts


def vdd_images(n, seed=1234):
    import cv2

    base = Path("/home/ubuntu/.cache/vdd/VDD/val")
    stems = sorted(p.stem for p in (base / "gt").glob("*.png"))
    np.random.default_rng(seed).shuffle(stems)
    for stem in stems[:n]:
        srcs = list((base / "src").glob(stem + ".*"))
        if not srcs:
            continue
        img = np.array(__import__("PIL.Image", fromlist=["Image"]).open(srcs[0]).convert("RGB"))
        m = np.array(__import__("PIL.Image", fromlist=["Image"]).open(base / "gt" / f"{stem}.png"))
        if m.ndim == 3:
            m = m[..., 0]
        img, m = _resize(img, m)
        yield np.ascontiguousarray(img), gt_instances(m, ignore=0, min_area=200)


def hf_images(repo, n, image_col="image", binary_thresh=128, seed=1234):
    from datasets import load_dataset

    ds = load_dataset(repo, split="train")
    idx = np.random.default_rng(seed).permutation(len(ds))[:n]
    for i in idx:
        s = ds[int(i)]
        img = np.array(s[image_col].convert("RGB"))
        m = np.array(s["label"].convert("L"))
        img, m = _resize(img, m)
        yield np.ascontiguousarray(img), gt_instances(m, min_area=200, binary=True, thresh=binary_thresh)


def match(proposals, gts, thr=0.5):
    props = sorted(proposals, key=lambda p: -int(p.sum()))
    used = [False] * len(gts)
    tp, fp, ious = 0, 0, []
    for p in props:
        best, bj = 0.0, -1
        for j, g in enumerate(gts):
            if used[j]:
                continue
            v = mask_iou(p, g)
            if v > best:
                best, bj = v, j
        if best >= thr and bj >= 0:
            used[bj] = True; tp += 1; ious.append(best)
        else:
            fp += 1
    fn = used.count(False)
    return tp, fp, fn, ious


def run_dataset(name, gen, amg):
    TP = FP = FN = 0; IOU = []; nimg = 0
    for img, gts in gen:
        if not gts:
            continue
        props = [ins.mask for ins in amg.generate(img)]
        tp, fp, fn, ious = match(props, gts)
        TP += tp; FP += fp; FN += fn; IOU += ious; nimg += 1
    rec = TP / (TP + FN) if TP + FN else 0.0
    prec = TP / (TP + FP) if TP + FP else 0.0
    return {"dataset": name, "n_img": nimg, "recall@0.5": round(rec, 3), "precision@0.5": round(prec, 3),
            "fp_per_img": round(FP / max(nimg, 1), 2), "mean_matched_iou": round(float(np.mean(IOU)), 3) if IOU else 0.0,
            "TP": TP, "FP": FP, "FN": FN}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=15)
    ap.add_argument("--name", default="no_oracle_eval")
    args = ap.parse_args()

    with Experiment(args.name, config={"n": args.n, "matcher_thr": 0.5}, seed=1234) as exp:
        amg = AutoMaskLabeler(AutoMaskConfig(points_per_side=16))
        specs = [
            ("VDD drone land-cover", vdd_images(args.n)),
            ("satellite binary", hf_images("saidines12/satellite-imagery-segmentation", args.n)),
            ("morocco buildings", hf_images("tferhan/morocco_satellite_buildings_semantic_segmentation_512_v2",
                                            args.n, image_col="pixel_values", binary_thresh=1)),
        ]
        rows = []
        for nm, gen in specs:
            try:
                r = run_dataset(nm, gen, amg)
                rows.append(r); exp.logger.info("%s: %s", nm, r)
            except Exception as e:
                rows.append({"dataset": nm, "error": f"{type(e).__name__}: {e}"})
                exp.logger.error("%s: %s", nm, e)
        exp.save_json("no_oracle_results.json", {"rows": rows})
        exp.save_summary({"rows": rows})

        print("\n=== No-oracle automatic-prompting eval (AMG -> match GT) ===")
        print(f"{'dataset':24s} {'recall':>7} {'prec':>6} {'FP/img':>7} {'matchIoU':>9}")
        for r in rows:
            if "error" in r:
                print(f"{r['dataset']:24s}  ERR {r['error'][:36]}")
            else:
                print(f"{r['dataset']:24s} {r['recall@0.5']:7.2f} {r['precision@0.5']:6.2f} "
                      f"{r['fp_per_img']:7.1f} {r['mean_matched_iou']:9.2f}")
        print(f"artifacts: {exp.artifacts}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
