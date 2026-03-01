"""
Evaluation Module for MIL Training.

Recent Updates:
    - [2026-01-06] Refactor: Phase 7 - Decoupled evaluation modules
        - WarmupEvaluator: P0 criteria for warmup→stable transition
        - FinalEvaluator: Post-training complete evaluation
        - MILVisualizer: Unified visualization (tile distribution + spatial heatmap)

Modules:
    - warmup_evaluator: Warmup phase evaluation (P0 metrics)
    - final_evaluator: Post-training evaluation (metrics + heatmaps)
    - heatmap_visualizer: Unified visualization module

Usage:
    >>> from src.evaluation import WarmupEvaluator, FinalEvaluator, MILVisualizer
"""

from src.evaluation.warmup_evaluator import WarmupEvaluator
from src.evaluation.final_evaluator import FinalEvaluator
from src.evaluation.heatmap_visualizer import MILVisualizer, CLASS_NAMES, CLASS_NAMES_SHORT

__all__ = [
    'WarmupEvaluator',
    'FinalEvaluator',
    'MILVisualizer',
    'CLASS_NAMES',
    'CLASS_NAMES_SHORT',
]
