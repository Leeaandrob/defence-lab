#!/usr/bin/env python3
"""Learned localizer vs AMG: supervised foreground -> proposals -> SAM2 refine.

Tests the project's reformulated thesis fix: a light supervised foreground UNet
proposes regions (low FP because trained), SAM2 refines each into a clean mask.
Compared head-to-head with class-agnostic AMG proposals on the SAME GT matching,
reporting recall / precision / FP-per-image / matched-IoU. If the learned
localizer raises precision / cuts FP at comparable recall, it confirms that the
operational lever is localization, not the mask backbone.

Usage:
    python scripts/learned_localizer.py --n-train 120 --n-test 12 --epochs 60
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE.parent))

import no_oracle_eval as noe  # noqa: E402  (gt_instances, match, _resize)
from airstrip_benchmark import UNet  # noqa: E402
from defense_lab.annotations.auto_mask import AutoMaskConfig, AutoMaskLabeler  # noqa: E402
from defense_lab.datasets.types import mask_to_xyxy  # noqa: E402
from defense_lab.evaluation.metrics import mask_iou  # noqa: E402
from defense_lab.prompting.prompts import PromptSet  # noqa: E402
from defense_lab.repro.experiment import Experiment  # noqa: E402
from defense_lab.segmentation.predictor import PromptableSegmenter, SegmenterConfig  # noqa: E402


def vdd_examples(split, n, seed=1234):
    from PIL import Image

    base = Path("/home/ubuntu/.cache/vdd/VDD") / split
    stems = sorted(p.stem for p in (base / "gt").glob("*.png"))
    np.random.default_rng(seed).shuffle(stems)
    out = []
    for stem in stems[:n]:
        srcs = list((base / "src").glob(stem + ".*"))
        if not srcs:
            continue
        img = np.array(Image.open(srcs[0]).convert("RGB"))
        m = np.array(Image.open(base / "gt" / f"{stem}.png"))
        if m.ndim == 3:
            m = m[..., 0]
        img, m = noe._resize(img, m)
        out.append((np.ascontiguousarray(img), m > 0, noe.gt_instances(m, ignore=0, min_area=200)))
    return out


def hf_examples(repo, n, image_col="image", fg_thresh=128, seed=1234):
    from datasets import load_dataset

    ds = load_dataset(repo, split="train")
    idx = np.random.default_rng(seed).permutation(len(ds))[:n]
    out = []
    for i in idx:
        s = ds[int(i)]
        img = np.array(s[image_col].convert("RGB"))
        m = np.array(s["label"].convert("L"))
        img, m = noe._resize(img, m)
        fg = m >= fg_thresh
        out.append((np.ascontiguousarray(img), fg, noe.gt_instances(m, binary=True, thresh=fg_thresh, min_area=200)))
    return out


def _to256(img, fg):
    import cv2

    return (cv2.resize(img, (256, 256)).transpose(2, 0, 1) / 255.0,
            cv2.resize(fg.astype(np.uint8), (256, 256), interpolation=cv2.INTER_NEAREST))


def train_unet(train_ex, epochs, dev):
    import cv2

    X = np.array([_to256(i, f)[0] for i, f, _ in train_ex], np.float32)
    Y = np.array([_to256(i, f)[1] for i, f, _ in train_ex], np.float32)
    X = torch.tensor(X, device=dev); Y = torch.tensor(Y, device=dev)
    net = UNet().to(dev); opt = torch.optim.Adam(net.parameters(), 1e-3)
    bce = torch.nn.BCEWithLogitsLoss(); n = len(X); bs = 8
    for _ in range(epochs):
        perm = torch.randperm(n, device=dev); net.train()
        for i in range(0, n, bs):
            idx = perm[i:i + bs]; logit = net(X[idx])[:, 0]; p = torch.sigmoid(logit)
            dice = 1 - (2 * (p * Y[idx]).sum() + 1) / (p.sum() + Y[idx].sum() + 1)
            loss = bce(logit, Y[idx]) + dice
            opt.zero_grad(); loss.backward(); opt.step()
    net.eval()
    return net


def unet_proposals(net, img, seg, dev, min_area=300):
    import cv2

    h, w = img.shape[:2]
    xi = cv2.resize(img, (256, 256)).transpose(2, 0, 1) / 255.0
    with torch.no_grad():
        pr = (torch.sigmoid(net(torch.tensor(xi[None], dtype=torch.float32, device=dev))[0, 0]) > 0.5).cpu().numpy()
    pr = cv2.resize(pr.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
    nlab, lab, stats, _ = cv2.connectedComponentsWithStats(pr, connectivity=8)
    seg.set_image(img)
    masks = []
    for i in range(1, nlab):
        if stats[i, cv2.CC_STAT_AREA] < min_area:
            continue
        comp = lab == i
        box = mask_to_xyxy(comp)
        masks.append(seg.predict(PromptSet.from_box(*box)).best()[0])
    return masks


def evalset(name, test_ex, amg, net, seg, dev):
    out = {}
    for tag, propfn in [("AMG", lambda im: [p.mask for p in amg.generate(im)]),
                        ("learned+SAM2", lambda im: unet_proposals(net, im, seg, dev))]:
        TP = FP = FN = 0; IOU = []
        for img, _fg, gts in test_ex:
            if not gts:
                continue
            tp, fp, fn, ious = noe.match(propfn(img), gts)
            TP += tp; FP += fp; FN += fn; IOU += ious
        n = len([1 for _, _, g in test_ex if g])
        out[tag] = {"recall": round(TP / (TP + FN), 3) if TP + FN else 0.0,
                    "precision": round(TP / (TP + FP), 3) if TP + FP else 0.0,
                    "fp_img": round(FP / max(n, 1), 1),
                    "iou": round(float(np.mean(IOU)), 3) if IOU else 0.0}
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-train", type=int, default=120)
    ap.add_argument("--n-test", type=int, default=12)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--name", default="learned_localizer")
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    with Experiment(args.name, config={"n_train": args.n_train, "n_test": args.n_test, "epochs": args.epochs}, seed=1234) as exp:
        amg = AutoMaskLabeler(AutoMaskConfig(points_per_side=16))
        seg = PromptableSegmenter(SegmenterConfig())

        datasets = {
            "VDD": (lambda: vdd_examples("train", args.n_train), lambda: vdd_examples("val", args.n_test)),
            "satellite": (lambda: hf_examples("saidines12/satellite-imagery-segmentation", args.n_train + args.n_test)[:args.n_train],
                          lambda: hf_examples("saidines12/satellite-imagery-segmentation", args.n_train + args.n_test)[args.n_train:]),
            "morocco": (lambda: hf_examples("tferhan/morocco_satellite_buildings_semantic_segmentation_512_v2", args.n_train + args.n_test, image_col="pixel_values", fg_thresh=1)[:args.n_train],
                        lambda: hf_examples("tferhan/morocco_satellite_buildings_semantic_segmentation_512_v2", args.n_train + args.n_test, image_col="pixel_values", fg_thresh=1)[args.n_train:]),
        }
        results = {}
        for name, (get_tr, get_te) in datasets.items():
            try:
                tr, te = get_tr(), get_te()
                exp.logger.info("%s: train=%d test=%d", name, len(tr), len(te))
                net = train_unet(tr, args.epochs, dev)
                results[name] = evalset(name, te, amg, net, seg, dev)
                exp.logger.info("%s: %s", name, results[name])
                del net; torch.cuda.empty_cache()
            except Exception as e:
                results[name] = {"error": f"{type(e).__name__}: {e}"}
                exp.logger.error("%s: %s", name, e)

        exp.save_json("learned_localizer.json", results)
        exp.save_summary(results)

        print("\n=== Learned localizer vs AMG (recall / precision / FP-img / IoU) ===")
        for name, r in results.items():
            if "error" in r:
                print(f"{name}: ERROR {r['error'][:50]}"); continue
            print(f"\n{name}:")
            for tag in ("AMG", "learned+SAM2"):
                m = r[tag]
                print(f"  {tag:14s} recall={m['recall']:.2f} prec={m['precision']:.2f} FP/img={m['fp_img']:5.1f} IoU={m['iou']:.2f}")
        print(f"\nartifacts: {exp.artifacts}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
