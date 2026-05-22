"""LoRA fine-tuning trainer for SAM2 (Phase 4 domain adaptation).

Freezes the pretrained model, injects LoRA into the configured targets (decoder
transformer by default; image-encoder attention too when ``adapt_encoder``),
and optimizes only the adapters with AdamW under bf16 autocast. Tiny trainable
footprint, fast, and reproducible -- the "efficient adaptation, not full
retrain" mandate.

A training/eval sample is a dict: ``{"image": HxWx3 uint8, "gt": HxW bool,
"point": (x,y) | None, "box": (x0,y0,x1,y1) | None}``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import torch

from defense_lab.lora.inject import (
    count_parameters,
    inject_lora,
    lora_parameters,
    mark_only_lora_trainable,
)
from defense_lab.lora.layers import LoRAConfig
from defense_lab.lora.losses import seg_loss
from defense_lab.segmentation.predictor import SegmenterConfig
from defense_lab.segmentation.trainable import TrainableSam2

# encoder attention linears to additionally target when adapting the encoder
_ENCODER_TARGETS = ("image_encoder.trunk", "attn.qkv", "attn.proj")


@dataclass
class LoraFinetuneConfig:
    seg: SegmenterConfig = field(default_factory=SegmenterConfig)
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    adapt_encoder: bool = False
    lr: float = 1e-3
    weight_decay: float = 0.0
    steps: int = 200
    w_focal: float = 20.0
    w_dice: float = 1.0
    grad_clip: float = 1.0
    seed: int = 1234
    log_every: int = 20


def _prompt_kwargs(sample: dict[str, Any]) -> dict[str, Any]:
    kw: dict[str, Any] = {}
    if sample.get("box") is not None:
        kw["box"] = np.asarray(sample["box"], dtype=np.float32)
    if sample.get("point") is not None:
        kw["point_coords"] = np.asarray([sample["point"]], dtype=np.float32)
        kw["point_labels"] = np.asarray([1], dtype=np.int32)
    return kw


class Sam2LoraTrainer:
    def __init__(self, cfg: Optional[LoraFinetuneConfig] = None) -> None:
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        self.cfg = cfg or LoraFinetuneConfig()
        self.device = torch.device(self.cfg.seg.device if torch.cuda.is_available() else "cpu")
        model = build_sam2(self.cfg.seg.model_cfg, self.cfg.seg.checkpoint, device=str(self.device))

        lora_cfg = self.cfg.lora
        if self.cfg.adapt_encoder:
            lora_cfg = LoRAConfig(
                r=lora_cfg.r, alpha=lora_cfg.alpha, dropout=lora_cfg.dropout,
                target_substrings=tuple(lora_cfg.target_substrings) + _ENCODER_TARGETS,
            )
        self.wrapped = inject_lora(model, lora_cfg)
        mark_only_lora_trainable(model)
        model.to(self.device)  # move freshly-created adapter params onto the GPU
        self.model = model
        self.predictor = SAM2ImagePredictor(model)
        self.tf = TrainableSam2(model, self.predictor)
        self.param_stats = count_parameters(model)
        self.param_stats["n_wrapped_linears"] = len(self.wrapped)
        self.opt = torch.optim.AdamW(lora_parameters(model), lr=self.cfg.lr, weight_decay=self.cfg.weight_decay)

    def _autocast(self):
        if self.device.type == "cuda":
            return torch.autocast("cuda", dtype=torch.bfloat16)
        import contextlib

        return contextlib.nullcontext()

    def _forward_loss(self, sample: dict[str, Any]) -> torch.Tensor:
        feats = self.tf.encode(sample["image"], grad=self.cfg.adapt_encoder)
        masks, _iou, _low = self.tf.decode(feats, multimask=False, **_prompt_kwargs(sample))
        target = torch.from_numpy(sample["gt"].astype(np.float32))[None, None].to(self.device)
        return seg_loss(masks, target, self.cfg.w_focal, self.cfg.w_dice)

    def train(self, dataset: list[dict[str, Any]]) -> list[dict[str, float]]:
        self.model.eval()  # LoRA train w/ base in eval (no dropout/BN drift); only adapters learn
        rng = np.random.default_rng(self.cfg.seed)
        history: list[dict[str, float]] = []
        for step in range(1, self.cfg.steps + 1):
            sample = dataset[int(rng.integers(len(dataset)))]
            self.opt.zero_grad(set_to_none=True)
            with self._autocast():
                loss = self._forward_loss(sample)
            loss.backward()
            if self.cfg.grad_clip:
                torch.nn.utils.clip_grad_norm_(lora_parameters(self.model), self.cfg.grad_clip)
            self.opt.step()
            if step == 1 or step % self.cfg.log_every == 0 or step == self.cfg.steps:
                history.append({"step": step, "loss": float(loss.detach())})
        return history

    @torch.no_grad()
    def evaluate(self, dataset: list[dict[str, Any]]) -> dict[str, Any]:
        from defense_lab.evaluation.metrics import dice_coefficient, mask_iou

        self.model.eval()
        ious, dices = [], []
        for sample in dataset:
            with self._autocast():
                feats = self.tf.encode(sample["image"], grad=False)
                masks, _iou, _low = self.tf.decode(feats, multimask=False, **_prompt_kwargs(sample))
            pred = (masks[0, 0].float() > self.predictor.mask_threshold).cpu().numpy()
            ious.append(mask_iou(pred, sample["gt"]))
            dices.append(dice_coefficient(pred, sample["gt"]))
        return {
            "mean_iou": round(float(np.mean(ious)), 4),
            "mean_dice": round(float(np.mean(dices)), 4),
            "per_sample_iou": [round(x, 4) for x in ious],
        }

    def predict_mask(self, sample: dict[str, Any]) -> np.ndarray:
        with torch.no_grad(), self._autocast():
            feats = self.tf.encode(sample["image"], grad=False)
            masks, _iou, _low = self.tf.decode(feats, multimask=False, **_prompt_kwargs(sample))
        return (masks[0, 0].float() > self.predictor.mask_threshold).cpu().numpy()
