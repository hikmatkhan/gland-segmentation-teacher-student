#!/usr/bin/env python3
"""
Segmentation Metrics for 4-Class Gland Segmentation
Computes Dice Score, IoU, and Pixel Accuracy metrics
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
from torch import Tensor

class SegmentationMetrics:
    """
    Comprehensive segmentation metrics for 4-class gland segmentation
    Handles Background(0), Benign(1), Malignant(2), PDC(3) classes
    """

    def __init__(self, num_classes: int = 4, ignore_background: bool = True, smooth: float = 1e-5):
        """
        Args:
            num_classes: Number of segmentation classes (default: 4)
            ignore_background: Whether to exclude background class from metrics
            smooth: Smoothing factor to avoid division by zero
        """
        self.num_classes = num_classes
        self.ignore_background = ignore_background
        self.smooth = smooth
        self.class_names = {
            0: "Background",
            1: "Benign Glands",
            2: "Malignant Glands",
            3: "PDC"
        }

    def dice_score(self, predictions: Tensor, targets: Tensor) -> Dict[str, float]:
        """
        Compute Dice Score (Dice Coefficient) for each class

        Args:
            predictions: [B, C, H, W] logits or [B, H, W] class predictions
            targets: [B, H, W] ground truth class indices

        Returns:
            Dict with per-class and mean Dice scores
        """
        # Convert logits to predictions if needed
        if predictions.dim() == 4 and predictions.size(1) == self.num_classes:
            predictions = torch.argmax(predictions, dim=1)  # [B, H, W]

        predictions = predictions.long()
        targets = targets.long()

        dice_scores = {}
        all_scores = []

        # Determine classes to evaluate
        start_class = 1 if self.ignore_background else 0

        for c in range(start_class, self.num_classes):
            pred_c = (predictions == c).float()
            target_c = (targets == c).float()

            intersection = (pred_c * target_c).sum()
            union = pred_c.sum() + target_c.sum()

            dice = (2 * intersection + self.smooth) / (union + self.smooth)

            dice_scores[f"dice_{self.class_names[c].lower().replace(' ', '_')}"] = dice.item()
            all_scores.append(dice.item())

        # Mean Dice (excluding background if specified)
        dice_scores["dice_mean"] = np.mean(all_scores) if all_scores else 0.0

        return dice_scores

    def iou_score(self, predictions: Tensor, targets: Tensor) -> Dict[str, float]:
        """
        Compute IoU (Intersection over Union) for each class

        Args:
            predictions: [B, C, H, W] logits or [B, H, W] class predictions
            targets: [B, H, W] ground truth class indices

        Returns:
            Dict with per-class and mean IoU scores
        """
        # Convert logits to predictions if needed
        if predictions.dim() == 4 and predictions.size(1) == self.num_classes:
            predictions = torch.argmax(predictions, dim=1)  # [B, H, W]

        predictions = predictions.long()
        targets = targets.long()

        iou_scores = {}
        all_scores = []

        # Determine classes to evaluate
        start_class = 1 if self.ignore_background else 0

        for c in range(start_class, self.num_classes):
            pred_c = (predictions == c).float()
            target_c = (targets == c).float()

            intersection = (pred_c * target_c).sum()
            union = pred_c.sum() + target_c.sum() - intersection

            iou = (intersection + self.smooth) / (union + self.smooth)

            iou_scores[f"iou_{self.class_names[c].lower().replace(' ', '_')}"] = iou.item()
            all_scores.append(iou.item())

        # Mean IoU (excluding background if specified)
        iou_scores["iou_mean"] = np.mean(all_scores) if all_scores else 0.0

        return iou_scores

    def pixel_accuracy(self, predictions: Tensor, targets: Tensor) -> Dict[str, float]:
        """
        Compute pixel-wise accuracy metrics

        Args:
            predictions: [B, C, H, W] logits or [B, H, W] class predictions
            targets: [B, H, W] ground truth class indices

        Returns:
            Dict with overall and per-class pixel accuracy
        """
        # Convert logits to predictions if needed
        if predictions.dim() == 4 and predictions.size(1) == self.num_classes:
            predictions = torch.argmax(predictions, dim=1)  # [B, H, W]

        predictions = predictions.long()
        targets = targets.long()

        # Overall pixel accuracy
        correct = (predictions == targets).float()
        overall_acc = correct.mean().item()

        acc_scores = {"pixel_accuracy_overall": overall_acc}

        # Per-class accuracy
        for c in range(self.num_classes):
            if self.ignore_background and c == 0:
                continue

            # Only compute accuracy where this class is present in ground truth
            class_mask = (targets == c)
            if class_mask.sum() > 0:
                class_correct = (predictions == targets)[class_mask]
                class_acc = class_correct.float().mean().item()
            else:
                class_acc = 0.0  # No pixels of this class present

            acc_scores[f"pixel_accuracy_{self.class_names[c].lower().replace(' ', '_')}"] = class_acc

        return acc_scores

    def compute_all_metrics(self, predictions: Tensor, targets: Tensor) -> Dict[str, float]:
        """
        Compute all segmentation metrics in one call

        Args:
            predictions: [B, C, H, W] logits or [B, H, W] class predictions
            targets: [B, H, W] ground truth class indices

        Returns:
            Dict with all metrics combined
        """
        metrics = {}

        # Compute all metric types
        metrics.update(self.dice_score(predictions, targets))
        metrics.update(self.iou_score(predictions, targets))
        metrics.update(self.pixel_accuracy(predictions, targets))

        return metrics

    def format_metrics_for_logging(self, metrics: Dict[str, float], prefix: str = "") -> str:
        """
        Format metrics for nice logging display

        Args:
            metrics: Dictionary of computed metrics
            prefix: Optional prefix for the metrics (e.g., "train_", "val_")

        Returns:
            Formatted string for logging
        """
        lines = []

        # Group metrics by type
        dice_metrics = {k: v for k, v in metrics.items() if k.startswith('dice_')}
        iou_metrics = {k: v for k, v in metrics.items() if k.startswith('iou_')}
        acc_metrics = {k: v for k, v in metrics.items() if k.startswith('pixel_accuracy_')}

        if dice_metrics:
            lines.append(f"   📊 {prefix}Dice Scores:")
            for key, value in dice_metrics.items():
                display_name = key.replace('dice_', '').replace('_', ' ').title()
                lines.append(f"      {display_name}: {value:.4f}")

        if iou_metrics:
            lines.append(f"   🎯 {prefix}IoU Scores:")
            for key, value in iou_metrics.items():
                display_name = key.replace('iou_', '').replace('_', ' ').title()
                lines.append(f"      {display_name}: {value:.4f}")

        if acc_metrics:
            lines.append(f"   ✅ {prefix}Pixel Accuracy:")
            for key, value in acc_metrics.items():
                display_name = key.replace('pixel_accuracy_', '').replace('_', ' ').title()
                lines.append(f"      {display_name}: {value:.4f}")

        return "\n".join(lines)

def compute_segmentation_metrics(predictions: Tensor, targets: Tensor,
                               num_classes: int = 4, ignore_background: bool = True) -> Dict[str, float]:
    """
    Convenience function to compute all segmentation metrics

    Args:
        predictions: [B, C, H, W] logits or [B, H, W] class predictions
        targets: [B, H, W] ground truth class indices
        num_classes: Number of classes (default: 4)
        ignore_background: Whether to exclude background from metrics

    Returns:
        Dictionary with all computed metrics
    """
    metrics_calculator = SegmentationMetrics(num_classes=num_classes, ignore_background=ignore_background)
    return metrics_calculator.compute_all_metrics(predictions, targets)

def test_metrics():
    """Test the metrics implementation"""
    print("🧪 Testing Segmentation Metrics...")

    # Create sample data
    batch_size, height, width = 2, 64, 64
    num_classes = 4

    # Random predictions and targets
    predictions = torch.randn(batch_size, num_classes, height, width)
    targets = torch.randint(0, num_classes, (batch_size, height, width))

    # Test metrics
    metrics_calc = SegmentationMetrics(num_classes=num_classes, ignore_background=True)

    # Test individual metrics
    dice_scores = metrics_calc.dice_score(predictions, targets)
    iou_scores = metrics_calc.iou_score(predictions, targets)
    acc_scores = metrics_calc.pixel_accuracy(predictions, targets)

    print("✅ Individual metrics computed successfully")

    # Test combined metrics
    all_metrics = metrics_calc.compute_all_metrics(predictions, targets)
    print(f"✅ Combined metrics computed: {len(all_metrics)} metrics")

    # Test formatting
    formatted = metrics_calc.format_metrics_for_logging(all_metrics, "test_")
    print("✅ Metrics formatting working")

    # Test convenience function
    convenience_metrics = compute_segmentation_metrics(predictions, targets)
    print("✅ Convenience function working")

    print("🎉 All metrics tests passed!")

if __name__ == "__main__":
    test_metrics()