"""SAM2 temporal video segmentation (Phase 5).

A typed façade over SAM2's video predictor that exposes the temporal-memory
machinery cleanly: ``init_state`` builds the per-frame memory bank, prompts are
attached to a (frame, object) via :class:`TemporalPrompt`, and ``propagate``
streams masks forward using SAM2's memory-attention -- giving object persistence
across frames without re-prompting. Class-agnostic by construction (you segment
whatever you point at); no identity/biometric notion.
"""
from __future__ import annotations

import contextlib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional

import numpy as np
import torch

from defense_lab.prompting.prompts import TemporalPrompt

_DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}


@dataclass
class VideoSegmenterConfig:
    model_cfg: str = "configs/sam2.1/sam2.1_hiera_b+.yaml"
    checkpoint: str = "checkpoints/sam2.1_hiera_base_plus.pt"
    device: str = "cuda"
    dtype: str = "bf16"
    offload_video_to_cpu: bool = True   # keep the shared GPU lean on long clips
    offload_state_to_cpu: bool = False


class VideoSegmenter:
    def __init__(self, cfg: Optional[VideoSegmenterConfig] = None) -> None:
        from sam2.build_sam import build_sam2_video_predictor

        self.cfg = cfg or VideoSegmenterConfig()
        if not Path(self.cfg.checkpoint).exists():
            raise FileNotFoundError(
                f"SAM2 checkpoint missing: {self.cfg.checkpoint}. Run scripts/download_checkpoints.sh"
            )
        self.device = torch.device(self.cfg.device if torch.cuda.is_available() else "cpu")
        self._dtype = _DTYPES[self.cfg.dtype]
        self.predictor = build_sam2_video_predictor(
            self.cfg.model_cfg, self.cfg.checkpoint, device=str(self.device)
        )

    def _autocast(self):
        if self.device.type == "cuda" and self._dtype in (torch.bfloat16, torch.float16):
            return torch.autocast("cuda", dtype=self._dtype)
        return contextlib.nullcontext()

    # -- memory bank -------------------------------------------------------- #
    def init_state(self, frames_dir: str | Path):
        with torch.inference_mode(), self._autocast():
            return self.predictor.init_state(
                video_path=str(frames_dir),
                offload_video_to_cpu=self.cfg.offload_video_to_cpu,
                offload_state_to_cpu=self.cfg.offload_state_to_cpu,
            )

    def reset(self, state) -> None:
        self.predictor.reset_state(state)

    # -- prompting ---------------------------------------------------------- #
    def add_prompt(self, state, tp: TemporalPrompt) -> None:
        ps = tp.prompt
        coords, labels, box = None, None, None
        if ps.points is not None and ps.points.points:
            coords, labels = ps.points.to_arrays()
        if ps.box is not None:
            box = ps.box.to_array()
        if coords is None and box is None:
            raise ValueError("TemporalPrompt has neither points nor a box")
        with torch.inference_mode(), self._autocast():
            self.predictor.add_new_points_or_box(
                state, frame_idx=tp.frame_idx, obj_id=tp.obj_id,
                points=coords, labels=labels, box=box,
            )

    # -- streaming propagation --------------------------------------------- #
    def propagate(self, state) -> Iterator[tuple[int, dict[int, np.ndarray]]]:
        """Yield (frame_idx, {obj_id: bool mask}) for every frame in order."""
        with torch.inference_mode(), self._autocast():
            for frame_idx, obj_ids, mask_logits in self.predictor.propagate_in_video(state):
                masks = {
                    int(obj_ids[i]): (mask_logits[i, 0] > 0.0).cpu().numpy()
                    for i in range(len(obj_ids))
                }
                yield int(frame_idx), masks
