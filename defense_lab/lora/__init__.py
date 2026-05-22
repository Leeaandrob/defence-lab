"""LoRA domain adaptation for SAM2: freeze the base, learn tiny low-rank deltas.

Efficient adaptation (NOT full retrain): inject adapters into targeted Linear
layers, train only those, checkpoint just the adapters.
"""
from defense_lab.lora.inject import (
    count_parameters,
    inject_lora,
    load_lora,
    lora_parameters,
    mark_only_lora_trainable,
    save_lora,
)
from defense_lab.lora.layers import LoRAConfig, LoRALinear
from defense_lab.lora.trainer import LoraFinetuneConfig, Sam2LoraTrainer

__all__ = [
    "LoRAConfig",
    "LoRALinear",
    "inject_lora",
    "mark_only_lora_trainable",
    "lora_parameters",
    "count_parameters",
    "save_lora",
    "load_lora",
    "LoraFinetuneConfig",
    "Sam2LoraTrainer",
]
