"""Visual prompt engineering: point / box / mask / region / temporal prompts.
Prompts are first-class; the system never assumes fixed classes only.
"""
from defense_lab.prompting.prompts import (
    BACKGROUND,
    FOREGROUND,
    BoxPrompt,
    MaskPrompt,
    Point,
    PointPrompt,
    PromptSet,
    TemporalPrompt,
    region_prompt,
)

__all__ = [
    "FOREGROUND",
    "BACKGROUND",
    "Point",
    "PointPrompt",
    "BoxPrompt",
    "MaskPrompt",
    "PromptSet",
    "TemporalPrompt",
    "region_prompt",
]
