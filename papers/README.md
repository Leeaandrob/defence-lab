# Foundational papers & how they map to this codebase

## Segment Anything (SAM) — Kirillov et al., ICCV 2023
The anchor of this project. Introduces **promptable segmentation**, a foundation
model for vision (SAM = image encoder + prompt encoder + lightweight mask
decoder), **zero-shot transfer** to new tasks via prompting, and the **data
engine** (model-assisted annotation at scale → SA-1B, 1.1B masks).

Principles → modules:
- promptable segmentation → `defense_lab/prompting`, `defense_lab/segmentation`
- visual prompt engineering → `defense_lab/prompting`
- data engine / HITL → `defense_lab/datasets`, `defense_lab/annotations`
- zero-shot transfer & generalized understanding → `defense_lab/evaluation`

```bibtex
@inproceedings{kirillov2023segment,
  title     = {Segment Anything},
  author    = {Kirillov, Alexander and Mintun, Eric and Ravi, Nikhila and Mao, Hanzi
               and Rolland, Chloe and Gustafson, Laura and Xiao, Tete and Whitehead, Spencer
               and Berg, Alexander C. and Lo, Wan-Yen and Doll{\'a}r, Piotr and Girshick, Ross},
  booktitle = {ICCV},
  year      = {2023}
}
```

## SAM 2 — Ravi et al., 2024
Extends SAM to **video** with a **streaming memory** module: per-frame memory
encoding, memory attention, and mask propagation for object persistence across
frames. Drives Phases 5 (temporal memory) and parts of Phase 6 (temporal metrics).

Principles → modules:
- frame memory / propagation / streaming → `defense_lab/temporal`
- object persistence / tracking → `defense_lab/tracking`

```bibtex
@article{ravi2024sam2,
  title   = {SAM 2: Segment Anything in Images and Videos},
  author  = {Ravi, Nikhila and Gabeur, Valentin and Hu, Yuan-Ting and Hu, Ronghang and others},
  journal = {arXiv preprint arXiv:2408.00714},
  year    = {2024}
}
```

## Adjacent directions (future)
- **LoRA** (Hu et al., 2021) — Phase 4 efficient domain adaptation.
- **GeoChat / remote-sensing VLMs** — future visual grounding & spatial reasoning.
