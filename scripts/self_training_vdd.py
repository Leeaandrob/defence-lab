#!/usr/bin/env python3
"""Honest 'self-training' experiment: VLM-labeled corpus -> class head -> eval VDD.

Self-supervising SAM2's box-IoU on its OWN AMG masks is circular, so instead we
use the *new* information the VLM added (class labels). We pool SAM2's image
embedding over each labeled region into a feature vector, train a light linear
classifier on the YouTube CC corpus (VLM classes -> VDD classes), and evaluate
class accuracy on the real VDD validation regions (authoritative GT). Honest:
class names are noisy pseudo-labels, VDD 'wall' has no corpus support.

Usage:
    python scripts/self_training_vdd.py --vdd /home/ubuntu/.cache/vdd/VDD --val-imgs 20
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from defense_lab.datasets.vdd import VDD_CLASSES, load_vdd_eval_samples  # noqa: E402
from defense_lab.repro.experiment import Experiment  # noqa: E402
from defense_lab.segmentation.predictor import PromptableSegmenter, SegmenterConfig  # noqa: E402

# VLM vocabulary -> VDD class id (0 other,1 wall,2 road,3 veg,4 vehicle,5 roof,6 water)
VLM2VDD = {"road": 2, "vegetation": 3, "forest": 3, "farmland": 3, "water": 6,
           "vehicle": 4, "building": 5, "rooftop": 5, "bare_soil": 0, "other": 0}


def _pool_features(seg: PromptableSegmenter, image: np.ndarray, masks: list[np.ndarray]) -> list[np.ndarray]:
    """Set image once, return mean-pooled 256-d SAM2 embedding per mask."""
    import cv2

    seg.set_image(image)
    emb = seg._predictor._features["image_embed"][0].float().cpu().numpy()  # (256,64,64)
    feats = []
    for m in masks:
        m64 = cv2.resize(m.astype(np.uint8), (emb.shape[2], emb.shape[1]), interpolation=cv2.INTER_NEAREST).astype(bool)
        v = emb[:, m64].mean(axis=1) if m64.any() else emb.reshape(emb.shape[0], -1).mean(axis=1)
        feats.append(v)
    return feats


def load_corpus(seg, glob_pat="experiments/corpus_*/*/artifacts/pseudo_labels.coco.json"):
    from pycocotools import mask as mask_utils

    X, y = [], []
    for cf in glob.glob(glob_pat):
        coco = json.load(open(cf))
        frames_dir = Path(cf).parent / "frames"
        id2name = {c["id"]: c["name"] for c in coco["categories"]}
        imgs = {im["id"]: im for im in coco["images"]}
        anns_by_img = defaultdict(list)
        for a in coco["annotations"]:
            anns_by_img[a["image_id"]].append(a)
        import cv2

        for img_id, anns in anns_by_img.items():
            fp = frames_dir / imgs[img_id]["file_name"]
            if not fp.exists():
                continue
            image = np.ascontiguousarray(cv2.imread(str(fp))[:, :, ::-1])
            masks, labels = [], []
            for a in anns:
                rle = dict(a["segmentation"]); rle["counts"] = rle["counts"].encode("ascii")
                m = mask_utils.decode(rle).astype(bool)
                cls = VLM2VDD.get(id2name.get(a["category_id"], "other"))
                if cls is None or m.sum() < 50:
                    continue
                masks.append(m); labels.append(cls)
            if masks:
                X += _pool_features(seg, image, masks)
                y += labels
    return np.array(X, np.float32), np.array(y, np.int64)


def load_vdd(seg, root, val_imgs):
    samples = load_vdd_eval_samples(root, "val", n_images=val_imgs, seed=1234)
    by_stem = defaultdict(list)
    for s in samples:
        by_stem[s.sample_id[0]].append(s)
    X, y = [], []
    for stem, ss in by_stem.items():
        feats = _pool_features(seg, ss[0].image, [s.gt for s in ss])
        X += feats
        y += [s.sample_id[1] for s in ss]
    return np.array(X, np.float32), np.array(y, np.int64)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vdd", default="/home/ubuntu/.cache/vdd/VDD")
    ap.add_argument("--val-imgs", type=int, default=20)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--name", default="self_training_vdd")
    args = ap.parse_args()

    with Experiment(args.name, config={"vdd": args.vdd, "val_imgs": args.val_imgs}, seed=1234) as exp:
        seg = PromptableSegmenter(SegmenterConfig())
        exp.logger.info("extracting corpus features ...")
        Xc, yc = load_corpus(seg)
        exp.logger.info("corpus: %d regions, classes=%s", len(yc), dict(Counter(yc.tolist())))
        if len(yc) < 10:
            exp.logger.error("corpus too small (%d)", len(yc)); return 1
        exp.logger.info("extracting VDD val features ...")
        Xv, yv = load_vdd(seg, args.vdd, args.val_imgs)
        exp.logger.info("vdd val: %d regions, classes=%s", len(yv), dict(Counter(yv.tolist())))

        # standardize on corpus stats
        mu, sd = Xc.mean(0), Xc.std(0) + 1e-6
        Xc_n = (Xc - mu) / sd
        Xv_n = (Xv - mu) / sd

        # tiny linear classifier (torch logistic regression), 7 VDD classes
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        Xt = torch.tensor(Xc_n, device=dev); yt = torch.tensor(yc, device=dev)
        clf = torch.nn.Linear(Xc.shape[1], 7).to(dev)
        opt = torch.optim.Adam(clf.parameters(), lr=1e-2, weight_decay=1e-4)
        lossf = torch.nn.CrossEntropyLoss()
        for _ in range(args.steps):
            opt.zero_grad(); loss = lossf(clf(Xt), yt); loss.backward(); opt.step()

        with torch.no_grad():
            pred = clf(torch.tensor(Xv_n, device=dev)).argmax(1).cpu().numpy()
        acc = float((pred == yv).mean())
        # majority baseline (most common corpus class)
        maj = Counter(yc.tolist()).most_common(1)[0][0]
        base = float((yv == maj).mean())
        per_class = {}
        for c in sorted(set(yv.tolist())):
            mask = yv == c
            per_class[VDD_CLASSES.get(c, str(c))] = {"n": int(mask.sum()),
                                                     "acc": round(float((pred[mask] == c).mean()), 3)}

        results = {"corpus_regions": int(len(yc)), "vdd_regions": int(len(yv)),
                   "class_head_acc": round(acc, 4), "majority_baseline_acc": round(base, 4),
                   "per_class": per_class, "vlm2vdd": VLM2VDD}
        exp.save_json("self_training_results.json", results)
        exp.save_summary(results)

        print("\n=== Honest self-training: VLM-corpus class head -> VDD GT ===")
        print(f"corpus regions: {len(yc)}   VDD val regions: {len(yv)}")
        print(f"class-head acc : {acc:.3f}   (majority baseline {base:.3f})")
        print("per-class acc (VDD GT):")
        for k, v in per_class.items():
            print(f"  {k:11s} n={v['n']:3d} acc={v['acc']}")
        print(f"artifacts: {exp.artifacts}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
