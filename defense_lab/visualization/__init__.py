"""Publication-ready plotting and mask/prompt overlay rendering.
"""
from defense_lab.visualization.overlay import render_instances, render_panels, render_result
from defense_lab.visualization.plots import bar_metric, loss_curve

__all__ = [
    "render_result", "render_panels", "render_instances", "bar_metric", "loss_curve",
]
