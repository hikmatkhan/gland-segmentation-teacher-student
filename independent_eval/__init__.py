"""
Independent Model Evaluator for GlandSegModels nnU-Net
======================================================

This package provides tools for independent evaluation of trained GlandSegModels nnU-Net models.

Features:
- Multi-architecture support (baseline_unet, nnunet)
- Multi-dataset evaluation (mixed, mag5x, mag10x, mag20x, mag40x)
- Comprehensive metrics (Dice, IoU, Pixel Accuracy, Classification Accuracy)
- Rich visualizations (100 samples per split)
- Multi-split evaluation (train/val/test)

Usage:
    From shell script (recommended):
        ./test_independent_evaluator.sh

    From command line:
        python independent_evaluator.py --experiment_path <path> --architecture <arch> --dataset_key <key> ...

Author: Claude Code - Generated for OSU CRC Research
Date: 2025-09-18
"""

__version__ = "1.0.0"
__author__ = "Claude Code"
__email__ = "noreply@anthropic.com"

# Export main components
from .independent_evaluator import IndependentModelEvaluator

__all__ = ["IndependentModelEvaluator"]