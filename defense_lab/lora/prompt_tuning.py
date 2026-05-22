"""Visual prompt-tuning for SAM2 (Phase 4 alt; lit: ~2k params, robust on tiny data).

Freezes the entire model and learns only K extra prompt tokens that are appended
to the mask decoder's sparse-prompt embeddings. Far fewer trainable params than
LoRA (K*256, e.g. 2048 for K=8) and less prone to overfit on small datasets
(Hu et al. prompt-tuning; cf. SAMed LoRA overfit). Same train/eval interface as
:class:`Sam2LoraTrainer` so experiments are drop-in comparable.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import torch

from defense_lab.lora.losses import seg_loss
from defense_lab.lora.trainer import _prompt_kwargs
from defense_lab.segmentation.predictor import SegmenterConfig
from defense_lab.segmentation.trainable import TrainableSam2


@dataclass
class PromptTuneConfig:
    seg: SegmenterConfig = field(default_factory=SegmenterConfig)
    num_tokens: int = 8
    lr: float = 1e-2
    weight_decay: float = 0.0
    steps: int = 400
    w_focal: float = 20.0
    w_dice: float = 1.0
    grad_clip: float = 1.0
    seed: int = 1234
    log_every: int = 50


class Sam2PromptTuner:
    def __init__(self, cfg: Optional[PromptTuneConfig] = None) -> None:
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        self.cfg = cfg or PromptTuneConfig()
        self.device = torch.device(self.cfg.seg.device if torch.cuda.is_available() else "cpu")
        model = build_sam2(self.cfg.seg.model_cfg, self.cfg.seg.checkpoint, device=str(self.device))
        for p in model.parameters():
            p.requires_grad_(False)
        self.model = model
        self.predictor = SAM2ImagePredictor(model)
        self.tf = TrainableSam2(model, self.predictor)

        dim = getattr(model.sam_prompt_encoder, "embed_dim", 256)
        self.tokens = torch.nn.Parameter(
            torch.randn(1, self.cfg.num_tokens, dim, device=self.device) * 0.02
        )
        self.opt = torch.optim.AdamW([self.tokens], lr=self.cfg.lr, weight_decay=self.cfg.weight_decay)
        self.param_stats = {"total": sum(p.numel() for p in model.parameters()),
                            "trainable": int(self.tokens.numel()),
                            "trainable_pct": round(100 * self.tokens.numel() / sum(p.numel() for p in model.parameters()), 5),
                            "num_tokens": self.cfg.num_tokens}

    def _autocast(self):
        if self.device.type == "cuda":
            return torch.autocast("cuda", dtype=torch.bfloat16)
        import contextlib

        return contextlib.nullcontext()

    def _forward_loss(self, sample: dict[str, Any]) -> torch.Tensor:
        feats = self.tf.encode(sample["image"], grad=False)
        masks, _i, _l = self.tf.decode(feats, multimask=False, extra_sparse=self.tokens, **_prompt_kwargs(sample))
        target = torch.from_numpy(sample["gt"].astype(np.float32))[None, None].to(self.device)
        return seg_loss(masks, target, self.cfg.w_focal, self.cfg.w_dice)

    def train(self, dataset: list[dict[str, Any]]) -> list[dict[str, float]]:
        self.model.eval()
        rng = np.random.default_rng(self.cfg.seed)
        hist = []
        for step in range(1, self.cfg.steps + 1):
            s = dataset[int(rng.integers(len(dataset)))]
            self.opt.zero_grad(set_to_none=True)
            with self._autocast():
                loss = self._forward_loss(s)
            loss.backward()
            if self.cfg.grad_clip:
                torch.nn.utils.clip_grad_norm_([self.tokens], self.cfg.grad_clip)
            self.opt.step()
            if step == 1 or step % self.cfg.log_every == 0 or step == self.cfg.steps:
                hist.append({"step": step, "loss": float(loss.detach())})
        return hist

    @torch.no_grad()
    def evaluate(self, dataset: list[dict[str, Any]]) -> dict[str, Any]:
        from defense_lab.evaluation.metrics import dice_coefficient, mask_iou

        self.model.eval()
        ious, dices = [], []
        for s in dataset:
            with self._autocast():
                feats = self.tf.encode(s["image"], grad=False)
                masks, _i, _l = self.tf.decode(feats, multimask=False, extra_sparse=self.tokens, **_prompt_kwargs(s))
            pred = (masks[0, 0].float() > self.predictor.mask_threshold).cpu().numpy()
            ious.append(mask_iou(pred, s["gt"])); dices.append(dice_coefficient(pred, s["gt"]))
        return {"mean_iou": round(float(np.mean(ious)), 4), "mean_dice": round(float(np.mean(dices)), 4)}
