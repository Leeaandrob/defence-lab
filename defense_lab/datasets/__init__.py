"""Dataset ingestion + the SAM-style data engine (COCO, video, pseudo-labels, HITL loops).
"""
from defense_lab.datasets.coco import CocoDataset, export_coco
from defense_lab.datasets.types import Instance, Sample, mask_to_xyxy
from defense_lab.datasets.video import VideoFrameSource, extract_frames

__all__ = [
    "Instance",
    "Sample",
    "mask_to_xyxy",
    "CocoDataset",
    "export_coco",
    "VideoFrameSource",
    "extract_frames",
]
