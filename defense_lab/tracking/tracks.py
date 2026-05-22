"""Track containers for multi-object video segmentation (Phase 5).

A ``Track`` is one object's mask through time (object persistence); a
``MultiObjectResult`` bundles all tracks for a clip. Pure data + simple derived
series (presence, area) -- evaluation and rendering consume these.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Track:
    obj_id: int
    masks: dict[int, np.ndarray] = field(default_factory=dict)  # frame_idx -> bool mask

    def add(self, frame_idx: int, mask: np.ndarray) -> None:
        self.masks[frame_idx] = mask

    @property
    def frame_indices(self) -> list[int]:
        return sorted(self.masks)

    def mask_at(self, frame_idx: int) -> np.ndarray | None:
        return self.masks.get(frame_idx)

    def present(self, frame_idx: int, min_area: int = 1) -> bool:
        m = self.masks.get(frame_idx)
        return m is not None and int(m.sum()) >= min_area

    def presence_rate(self, min_area: int = 1) -> float:
        if not self.masks:
            return 0.0
        return float(np.mean([self.present(f, min_area) for f in self.frame_indices]))

    def area_series(self) -> list[int]:
        return [int(self.masks[f].sum()) for f in self.frame_indices]

    def mask_sequence(self) -> list[np.ndarray]:
        return [self.masks[f] for f in self.frame_indices]


@dataclass
class MultiObjectResult:
    tracks: dict[int, Track] = field(default_factory=dict)
    num_frames: int = 0

    def per_frame(self, frame_idx: int) -> dict[int, np.ndarray]:
        out = {}
        for oid, t in self.tracks.items():
            m = t.mask_at(frame_idx)
            if m is not None:
                out[oid] = m
        return out

    @property
    def obj_ids(self) -> list[int]:
        return sorted(self.tracks)
