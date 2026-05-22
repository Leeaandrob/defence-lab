"""COCO ingestion + export (the labeled-data on-ramp for the data engine).

``CocoDataset`` wraps pycocotools to yield typed :class:`Sample` objects with
decoded masks. ``export_coco`` writes data-engine outputs (pseudo-labels) back
out in COCO instances format, so the loop's products are reusable by any COCO
tool and by Phase-4 LoRA training.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np

from defense_lab.datasets.types import Instance, Sample


class CocoDataset:
    def __init__(self, ann_file: str | Path, image_dir: str | Path, *, with_masks: bool = True) -> None:
        from pycocotools.coco import COCO

        self.ann_file = str(ann_file)
        self.image_dir = str(image_dir)
        self.with_masks = with_masks
        self.coco = COCO(self.ann_file)
        self.ids = sorted(self.coco.getImgIds())
        self.categories = {c["id"]: c["name"] for c in self.coco.loadCats(self.coco.getCatIds())}

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, i: int) -> Sample:
        img_id = self.ids[i]
        info = self.coco.loadImgs([img_id])[0]
        anns = self.coco.loadAnns(self.coco.getAnnIds(imgIds=[img_id]))
        instances: list[Instance] = []
        for a in anns:
            mask = None
            if self.with_masks and a.get("segmentation") is not None:
                mask = self.coco.annToMask(a).astype(bool)
            x, y, w, h = a["bbox"]
            instances.append(
                Instance(
                    box=(float(x), float(y), float(x + w), float(y + h)),
                    mask=mask,
                    category_id=a.get("category_id"),
                    category_name=self.categories.get(a.get("category_id")),
                    obj_id=a.get("id"),
                    source="gt",
                )
            )
        return Sample(
            image_id=img_id,
            height=int(info["height"]),
            width=int(info["width"]),
            file_name=info.get("file_name"),
            image_dir=self.image_dir,
            instances=instances,
        )

    def __iter__(self) -> Iterable[Sample]:
        for i in range(len(self)):
            yield self[i]


def _rle(mask: np.ndarray) -> dict[str, Any]:
    from pycocotools import mask as mask_utils

    rle = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
    rle["counts"] = rle["counts"].decode("ascii")  # JSON-safe
    return rle


def export_coco(
    samples: Iterable[Sample],
    path: str | Path,
    categories: Optional[dict[int, str]] = None,
) -> Path:
    """Write samples (with mask Instances) to a COCO instances JSON file."""
    from pycocotools import mask as mask_utils

    images, annotations = [], []
    cat_ids: set[int] = set()
    ann_id = 1
    for s in samples:
        images.append({"id": s.image_id, "file_name": s.file_name, "height": s.height, "width": s.width})
        for ins in s.instances:
            if ins.mask is None:
                continue
            seg = _rle(ins.mask)
            x0, y0, x1, y1 = ins.box
            cid = ins.category_id if ins.category_id is not None else 1
            cat_ids.add(cid)
            annotations.append({
                "id": ann_id,
                "image_id": s.image_id,
                "category_id": cid,
                "segmentation": seg,
                "area": float(mask_utils.area(mask_utils.encode(np.asfortranarray(ins.mask.astype(np.uint8))))),
                "bbox": [x0, y0, x1 - x0, y1 - y0],
                "iscrowd": 0,
                "score": ins.score,
                "source": ins.source,
            })
            ann_id += 1
    cats = categories or {cid: f"class_{cid}" for cid in sorted(cat_ids)} or {1: "object"}
    coco = {
        "images": images,
        "annotations": annotations,
        "categories": [{"id": k, "name": v} for k, v in cats.items()],
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(coco, f)
    return path
