"""Gradient-enabled SAM2 image forward (for LoRA fine-tuning).

The stock ``SAM2ImagePredictor.set_image`` / ``_predict`` are ``@torch.no_grad``,
so they cannot be used for training. ``TrainableSam2`` replays exactly the same
computation (verified against the predictor source) but lets gradients flow,
reusing the predictor's ``_transforms`` / ``_bb_feat_sizes`` / ``_prep_prompts``
so preprocessing is bit-identical to inference.

``encode(..., grad=False)`` keeps the (frozen) image encoder in ``no_grad`` for
cheap, low-memory decoder-only adaptation; ``grad=True`` enables encoder LoRA.
"""
from __future__ import annotations

import contextlib
from typing import Any, Optional

import numpy as np
import torch


class TrainableSam2:
    def __init__(self, model, predictor) -> None:
        self.model = model
        self.p = predictor  # SAM2ImagePredictor, used only for transforms/helpers
        self.device = model.device

    def encode(self, image: np.ndarray, grad: bool) -> dict[str, Any]:
        self.p._orig_hw = [image.shape[:2]]
        ctx = contextlib.nullcontext() if grad else torch.no_grad()
        with ctx:
            inp = self.p._transforms(image)[None, ...].to(self.device)
            backbone_out = self.model.forward_image(inp)
            _, vision_feats, _, _ = self.model._prepare_backbone_features(backbone_out)
            if self.model.directly_add_no_mem_embed:
                vision_feats[-1] = vision_feats[-1] + self.model.no_mem_embed
            feats = [
                feat.permute(1, 2, 0).view(1, -1, *fs)
                for feat, fs in zip(vision_feats[::-1], self.p._bb_feat_sizes[::-1])
            ][::-1]
        return {"image_embed": feats[-1], "high_res_feats": feats[:-1], "orig_hw": tuple(image.shape[:2])}

    def decode(
        self,
        features: dict[str, Any],
        *,
        point_coords: Optional[np.ndarray] = None,
        point_labels: Optional[np.ndarray] = None,
        box: Optional[np.ndarray] = None,
        mask_input: Optional[np.ndarray] = None,
        multimask: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (masks_full_res_logits[1,C,H,W], iou[1,C], low_res[1,C,256,256])."""
        self.p._orig_hw = [features["orig_hw"]]
        mask_in, coords, labels, ubox = self.p._prep_prompts(
            point_coords, point_labels, box, mask_input, normalize_coords=True
        )
        concat_points = (coords, labels) if coords is not None else None
        if ubox is not None:
            box_coords = ubox.reshape(-1, 2, 2)
            box_labels = torch.tensor([[2, 3]], dtype=torch.int, device=ubox.device).repeat(ubox.size(0), 1)
            if concat_points is not None:
                concat_points = (
                    torch.cat([box_coords, concat_points[0]], dim=1),
                    torch.cat([box_labels, concat_points[1]], dim=1),
                )
            else:
                concat_points = (box_coords, box_labels)

        sparse, dense = self.model.sam_prompt_encoder(points=concat_points, boxes=None, masks=mask_in)
        batched_mode = concat_points is not None and concat_points[0].shape[0] > 1
        low_res, iou, _, _ = self.model.sam_mask_decoder(
            image_embeddings=features["image_embed"],
            image_pe=self.model.sam_prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse,
            dense_prompt_embeddings=dense,
            multimask_output=multimask,
            repeat_image=batched_mode,
            high_res_features=features["high_res_feats"],
        )
        masks = self.p._transforms.postprocess_masks(low_res, features["orig_hw"])
        return masks, iou, low_res
