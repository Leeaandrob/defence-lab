"""High-level multi-object tracking over a video clip (Phase 5).

Ties prompts -> propagation -> tracks: initialize the memory bank on a frame
directory, attach one or more :class:`TemporalPrompt` (each a frame+object+
prompt), stream propagation, and collect per-object :class:`Track`s. Also times
streaming throughput. The segmenter is injected so this stays decoupled from the
SAM2 backend.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Sequence

from defense_lab.prompting.prompts import TemporalPrompt
from defense_lab.tracking.tracks import MultiObjectResult, Track


def run_tracking(
    segmenter: Any,
    frames_dir: str | Path,
    prompts: Sequence[TemporalPrompt],
) -> tuple[MultiObjectResult, dict[str, float]]:
    """Returns (result, timing). ``segmenter`` is a VideoSegmenter-like object."""
    state = segmenter.init_state(frames_dir)
    for tp in prompts:
        segmenter.add_prompt(state, tp)

    result = MultiObjectResult(tracks={tp.obj_id: Track(tp.obj_id) for tp in prompts})
    n_frames = 0
    t0 = time.perf_counter()
    for frame_idx, masks in segmenter.propagate(state):
        for oid, mask in masks.items():
            result.tracks.setdefault(oid, Track(oid)).add(frame_idx, mask)
        n_frames += 1
    elapsed = time.perf_counter() - t0

    result.num_frames = n_frames
    segmenter.reset(state)
    timing = {
        "frames": n_frames,
        "propagation_s": round(elapsed, 4),
        "streaming_fps": round(n_frames / elapsed, 2) if elapsed > 0 else 0.0,
    }
    return result, timing
