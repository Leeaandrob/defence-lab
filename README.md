# Defense-Lab — Foundation-Model-Driven Spatial Understanding

A research-grade, SAM/SAM2-inspired **promptable visual foundation** stack for
**operational visual understanding** (aerial, surveillance, urban, disaster
response, infrastructure). This is **not** a fixed-class object detector — it is
a reusable visual foundation layer adapted to new domains through *prompting*
and *lightweight tuning*.

> Design follows Kirillov et al., **"Segment Anything"** (ICCV 2023) and the
> SAM2 video extension. See [`papers/`](papers/README.md).

## Architectural reasoning

SAM's central bet is **decoupling a heavy image encoder from a cheap, promptable
mask decoder**: encode an image once, then answer many point/box/mask prompts
interactively. We mirror that separation throughout:

| Layer | SAM principle | Module |
|---|---|---|
| Prompts as first-class input | promptable segmentation | `defense_lab/prompting` |
| Encode-once / decode-many | foundation model, interactive | `defense_lab/segmentation` |
| Frame memory & propagation | SAM2 temporal memory | `defense_lab/temporal` |
| Object persistence over time | streaming inference | `defense_lab/tracking` |
| Data engine / HITL labeling | scalable data engine | `defense_lab/datasets`, `defense_lab/annotations` |
| Efficient domain transfer | downstream adaptability | `defense_lab/lora` *(Phase 4)* |
| Zero-shot quality measurement | generalized understanding | `defense_lab/evaluation` |

**Research tradeoffs baked in.** We default to **bf16** (Hopper's sweet spot:
~800 TFLOPS measured here, vs ~50 for fp32) and the **Flash-Attention** SDPA path
(~6× faster than the math fallback). We avoid a hard Hydra dependency in favor of
typed dataclass configs — fewer moving parts on ARM64, while keeping Hydra-style
`key.path=value` overrides for fast ablation sweeps. LoRA-first adaptation (Phase
4) is chosen over full fine-tuning to maximize experiment velocity and
reproducibility.

## Layout

```
defense_lab/            # importable package (typed, modular)
  config/   repro/      # structured configs + reproducibility core
  diagnostics/          # Phase 1: CUDA / bf16 / FlashAttn / GPU / bandwidth
  prompting/ segmentation/ temporal/ tracking/   # the foundation stack
  datasets/ annotations/                          # SAM-style data engine
  evaluation/ benchmarks/ visualization/          # metrics & reporting
configs/    scripts/    experiments/   # YAML, entrypoints, run outputs
checkpoints/  notebooks/  papers/
```
*(Code lives under the importable `defense_lab/` package to avoid name clashes;
assets/outputs stay at the top level, matching the SAM2 repo convention.)*

## Quickstart

```bash
# Phase 1 — validate & benchmark the foundation environment
python scripts/phase1_env_check.py                 # full run
python scripts/phase1_env_check.py --quick         # fast smoke test
python scripts/phase1_env_check.py precision.matmul_size=8192   # override

# (optional) install SAM2 + weights for the model benchmark
bash scripts/install_sam2.sh
bash scripts/download_checkpoints.sh base_plus
```

Each run writes a self-contained directory under `experiments/<name>/<ts>__<cfghash>/`
with `config.yaml`, `env.json`, `run.log`, `metrics.jsonl` and `artifacts/`.

## Reproducibility

Every experiment records the resolved config, the seed, full environment
provenance (arch, CPU/RAM, GPU, driver, CUDA/cuDNN, library versions, git SHA),
metrics as JSONL, and a content hash of the config in the run-dir name. Re-run an
experiment by pointing `--config` at the saved `config.yaml`.

## Profiling guidance

- **Throughput ceiling first.** `phase1_env_check` reports matmul TFLOPS per
  dtype and attention-backend latency — your hardware ceiling before touching SAM2.
- **Kernel-level.** Wrap a benchmark in `nsys profile -o trace python ...` or use
  `torch.profiler` with `record_shapes=True` to attribute time to encode vs decode.
- **Memory.** `torch.cuda.max_memory_allocated()` is logged by the SAM2 benchmark;
  use `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` for long streaming runs.

## Debugging guidance

- **Shared GPU OOM.** This card is co-tenanted (~26 GiB free of 94.5). Diagnostics
  warn under 8 GiB headroom; keep batch sizes small. Set `CUDA_VISIBLE_DEVICES`.
- **Determinism vs speed.** `deterministic=true` flips on deterministic kernels +
  `CUBLAS_WORKSPACE_CONFIG=:4096:8`; expect a throughput hit.
- **Unknown config key** errors are intentional — typos in overrides fail loudly.

## Phase roadmap — all 6 implemented & verified ✅

1. **Foundation environment** — ✅ `scripts/phase1_env_check.py` (+ `phase1_sam2_bench.py`).
2. **Promptable segmentation** — ✅ point/box/mask + interactive (`phase2_promptable_demo.py`).
3. **Data engine** — ✅ COCO + video ingestion, SAM-assisted labeling, pseudo-label/HITL (`phase3_data_engine.py`).
4. **LoRA domain adaptation** — ✅ adapters + partial freezing, no full retrain (`phase4_lora_finetune.py`).
5. **Temporal memory** — ✅ frame memory, propagation, persistence, streaming (`phase5_temporal.py`).
6. **Operational evaluation** — ✅ IoU/Dice/boundary-F, temporal consistency, FPS, benchmark table+plots (`phase6_evaluate.py`).

**Scope:** class-agnostic scene understanding / segmentation for neutral operational
domains (disaster response, infrastructure, environmental/aerial). The pipeline does
**not** perform person/identity tracking, profiling, biometrics, or weapon/threat
classification — those are explicitly out of scope. Benchmarks run on synthetic data
by default; point Phase 6 at a real dataset (`--coco-ann/--coco-images`) to extend.

## Future directions

The prompt/encoder/decoder seams are designed for VLM and geospatial integration
(GeoChat-style grounding, remote-sensing foundation models, Gaussian-splatting and
drone imagery) without re-architecting the core.

## Acknowledgments

Developed as part of the Data Science postgraduate program at the
[STAR Research Institute](https://starresearch.institute/), within its defense
research group — with thanks to **Carlos Melo** for the mentorship and the group.
