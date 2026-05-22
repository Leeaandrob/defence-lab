#!/usr/bin/env python3
"""Airstrip benchmark (SIGE core): SAM2 zero-shot vs SAM2+LoRA vs U-Net.

Trains on the airstrip train split, evaluates on the held-out test positives.
SAM2 uses a box prompt (bbox of the GT airstrip); U-Net is trained from scratch
on 256px images. Reports test IoU/Dice for all three. Honest: test set is small
(few dozen positives) -> treat numbers as indicative, report variance later.

Usage:
    python scripts/airstrip_benchmark.py --root /home/ubuntu/.cache/airstrip_s1aad --lora-steps 400 --unet-epochs 60
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from defense_lab.datasets.types import mask_to_xyxy  # noqa: E402
from defense_lab.evaluation.metrics import dice_coefficient, mask_iou  # noqa: E402
from defense_lab.lora.trainer import LoraFinetuneConfig, Sam2LoraTrainer  # noqa: E402
from defense_lab.repro.experiment import Experiment  # noqa: E402


def _load(root: Path, split: str, fn: str):
    import cv2

    img = np.ascontiguousarray(cv2.imread(str(root / split / "images" / fn))[:, :, ::-1])
    msk = cv2.imread(str(root / split / "masks" / fn), 0) >= 128
    return img, msk


def sam2_samples(root, items, split):
    out = []
    for m in items:
        if not m["positive"]:
            continue
        img, msk = _load(root, split, m["file"])
        box = mask_to_xyxy(msk)
        out.append({"image": img, "gt": msk, "box": list(box), "point": None})
    return out


# ---------------- tiny U-Net ----------------
class _Block(nn.Module):
    def __init__(self, ci, co):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(ci, co, 3, padding=1), nn.BatchNorm2d(co), nn.ReLU(True),
                                 nn.Conv2d(co, co, 3, padding=1), nn.BatchNorm2d(co), nn.ReLU(True))

    def forward(self, x):
        return self.net(x)


class UNet(nn.Module):
    def __init__(self, ch=(32, 64, 128, 256)):
        super().__init__()
        self.d1 = _Block(3, ch[0]); self.d2 = _Block(ch[0], ch[1])
        self.d3 = _Block(ch[1], ch[2]); self.b = _Block(ch[2], ch[3])
        self.pool = nn.MaxPool2d(2)
        self.u3 = nn.ConvTranspose2d(ch[3], ch[2], 2, 2); self.c3 = _Block(ch[3], ch[2])
        self.u2 = nn.ConvTranspose2d(ch[2], ch[1], 2, 2); self.c2 = _Block(ch[2], ch[1])
        self.u1 = nn.ConvTranspose2d(ch[1], ch[0], 2, 2); self.c1 = _Block(ch[1], ch[0])
        self.out = nn.Conv2d(ch[0], 1, 1)

    def forward(self, x):
        d1 = self.d1(x); d2 = self.d2(self.pool(d1)); d3 = self.d3(self.pool(d2))
        b = self.b(self.pool(d3))
        x = self.c3(torch.cat([self.u3(b), d3], 1))
        x = self.c2(torch.cat([self.u2(x), d2], 1))
        x = self.c1(torch.cat([self.u1(x), d1], 1))
        return self.out(x)


def _stack(root, items, split, size=256):
    import cv2

    X, Y = [], []
    for m in items:
        img, msk = _load(root, split, m["file"])
        X.append(cv2.resize(img, (size, size)).transpose(2, 0, 1) / 255.0)
        Y.append(cv2.resize(msk.astype(np.uint8), (size, size), interpolation=cv2.INTER_NEAREST))
    return np.array(X, np.float32), np.array(Y, np.float32)


def train_unet(root, train_items, test_items, epochs, dev, size=256):
    Xtr, Ytr = _stack(root, train_items, "train", size)
    Xtr = torch.tensor(Xtr, device=dev); Ytr = torch.tensor(Ytr, device=dev)
    net = UNet().to(dev)
    opt = torch.optim.Adam(net.parameters(), 1e-3)
    bce = nn.BCEWithLogitsLoss()
    n = len(Xtr); bs = 8
    for ep in range(epochs):
        perm = torch.randperm(n, device=dev)
        net.train()
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            logit = net(Xtr[idx])[:, 0]
            p = torch.sigmoid(logit)
            dice = 1 - (2 * (p * Ytr[idx]).sum() + 1) / (p.sum() + Ytr[idx].sum() + 1)
            loss = bce(logit, Ytr[idx]) + dice
            opt.zero_grad(); loss.backward(); opt.step()
    # eval test positives at full res
    import cv2

    net.eval()
    ious, dices = [], []
    with torch.no_grad():
        for m in test_items:
            if not m["positive"]:
                continue
            img, msk = _load(root, "test", m["file"])
            xi = cv2.resize(img, (size, size)).transpose(2, 0, 1) / 255.0
            logit = net(torch.tensor(xi[None], dtype=torch.float32, device=dev))[0, 0]
            pr = (torch.sigmoid(logit) > 0.5).cpu().numpy().astype(np.uint8)
            pr = cv2.resize(pr, (msk.shape[1], msk.shape[0]), interpolation=cv2.INTER_NEAREST).astype(bool)
            ious.append(mask_iou(pr, msk)); dices.append(dice_coefficient(pr, msk))
    return float(np.mean(ious)), float(np.mean(dices)), sum(p.numel() for p in net.parameters())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/home/ubuntu/.cache/airstrip_s1aad")
    ap.add_argument("--lora-steps", type=int, default=400)
    ap.add_argument("--unet-epochs", type=int, default=60)
    ap.add_argument("--name", default="airstrip_benchmark")
    args = ap.parse_args()
    root = Path(args.root)
    man = json.load(open(root / "manifest.json"))
    train_items = [m for m in man if m["split"] == "train"]
    test_items = [m for m in man if m["split"] == "test"]
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    with Experiment(args.name, config={"root": str(root), "lora_steps": args.lora_steps,
                                       "unet_epochs": args.unet_epochs}, seed=1234) as exp:
        tr_s = sam2_samples(root, train_items, "train")
        te_s = sam2_samples(root, test_items, "test")
        exp.logger.info("SAM2 train pos=%d test pos=%d", len(tr_s), len(te_s))

        # SAM2 zero-shot (box) + LoRA
        trainer = Sam2LoraTrainer(LoraFinetuneConfig(steps=args.lora_steps, seed=1234))
        zs = trainer.evaluate(te_s)
        exp.logger.info("SAM2 zero-shot test IoU=%.4f", zs["mean_iou"])
        trainer.train(tr_s)
        lo = trainer.evaluate(te_s)
        exp.logger.info("SAM2+LoRA test IoU=%.4f", lo["mean_iou"])

        # U-Net from scratch
        u_iou, u_dice, u_params = train_unet(root, train_items, test_items, args.unet_epochs, dev)
        exp.logger.info("U-Net test IoU=%.4f (params %d)", u_iou, u_params)

        rows = [
            {"method": "SAM2 zero-shot (box)", "test_IoU": zs["mean_iou"], "test_Dice": zs["mean_dice"], "trainable": 0},
            {"method": "SAM2 + LoRA (box)", "test_IoU": lo["mean_iou"], "test_Dice": lo["mean_dice"],
             "trainable": trainer.param_stats["trainable"]},
            {"method": "U-Net (scratch)", "test_IoU": round(u_iou, 4), "test_Dice": round(u_dice, 4), "trainable": u_params},
        ]
        exp.save_json("airstrip_results.json", {"rows": rows, "n_test_pos": len(te_s), "n_train_pos": len(tr_s)})
        exp.save_summary({"rows": rows, "n_test_pos": len(te_s)})

        print("\n=== Airstrip benchmark (test split) ===")
        print(f"train pos={len(tr_s)}  test pos={len(te_s)}")
        print(f"{'method':24s} {'IoU':>7} {'Dice':>7} {'trainable':>11}")
        for r in rows:
            print(f"{r['method']:24s} {r['test_IoU']:7.3f} {r['test_Dice']:7.3f} {r['trainable']:>11,}")
        print(f"artifacts: {exp.artifacts}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
