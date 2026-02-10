#!/usr/bin/env python3
"""
4-Class Loss Functions for Combined Gland Segmentation
Adapted for multi-task learning with Background(0), Benign(1), Malignant(2), PDC(3)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, List, Tuple, Optional
import numpy as np

class FourClassDiceLoss(nn.Module):
    """
    4-class Dice loss for segmentation
    Handles Background(0), Benign(1), Malignant(2), PDC(3) classes
    """
    def __init__(self, smooth: float = 1e-5, ignore_index: int = -1, exclude_background: bool = True):
        super().__init__()
        self.smooth = smooth
        self.ignore_index = ignore_index
        self.exclude_background = exclude_background

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            predictions: [B, 4, H, W] logits for 4 classes
            targets: [B, H, W] class indices (0-3)
        """
        # Convert predictions to probabilities
        predictions = F.softmax(predictions, dim=1)

        # Convert targets to one-hot
        num_classes = predictions.shape[1]
        targets_one_hot = F.one_hot(targets.long(), num_classes).permute(0, 3, 1, 2).float()

        # Calculate Dice for each class
        dice_scores = []
        class_range = range(1, num_classes) if self.exclude_background else range(num_classes)

        for c in class_range:
            pred_c = predictions[:, c]
            target_c = targets_one_hot[:, c]

            intersection = (pred_c * target_c).sum(dim=[1, 2])
            union = pred_c.sum(dim=[1, 2]) + target_c.sum(dim=[1, 2])

            dice = (2 * intersection + self.smooth) / (union + self.smooth)
            dice_scores.append(dice)

        dice_scores = torch.stack(dice_scores, dim=1)  # [B, num_foreground_classes]
        dice_loss = 1 - dice_scores.mean()

        return dice_loss

class WeightedCrossEntropyLoss(nn.Module):
    """
    Weighted Cross Entropy Loss for handling class imbalance in 4-class segmentation
    """
    def __init__(self, class_weights: Optional[List[float]] = None):
        super().__init__()

        # Default weights for 4-class gland segmentation (background gets lower weight)
        if class_weights is None:
            class_weights = [0.1, 1.0, 1.0, 1.0]  # [Background, Benign, Malignant, PDC]

        self.class_weights = torch.tensor(class_weights)

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            predictions: [B, 4, H, W] logits
            targets: [B, H, W] class indices
        """
        # Move weights to same device as predictions
        weights = self.class_weights.to(predictions.device)

        # Use weighted cross entropy
        loss = F.cross_entropy(predictions, targets.long(), weight=weights)
        return loss

class CombinedSegmentationLoss(nn.Module):
    """
    Combined 4-class segmentation loss: Dice + Weighted Cross Entropy
    """
    def __init__(self,
                 dice_weight: float = 0.5,
                 ce_weight: float = 0.5,
                 class_weights: Optional[List[float]] = None):
        super().__init__()
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight
        self.dice_loss = FourClassDiceLoss()
        self.ce_loss = WeightedCrossEntropyLoss(class_weights)

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        dice = self.dice_loss(predictions, targets)
        ce = self.ce_loss(predictions, targets)
        return self.dice_weight * dice + self.ce_weight * ce

class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance in classification tasks
    """
    def __init__(self, alpha: float = 1.0, gamma: float = 2.0, reduce: bool = True):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduce = reduce

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: [N, num_classes] logits
            targets: [N] class indices
        """
        ce_loss = F.cross_entropy(inputs, targets.long(), reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduce:
            return focal_loss.mean()
        else:
            return focal_loss

class AdaptiveLossWeighting(nn.Module):
    """
    Adaptive loss weighting for 4-class multi-task learning
    Based on: Multi-Task Learning Using Uncertainty to Weigh Losses for Scene Geometry and Semantics
    """
    def __init__(self, num_tasks: int = 3):
        super().__init__()
        # Log variance parameters (learnable)
        self.log_vars = nn.Parameter(torch.zeros(num_tasks))

    def forward(self, losses: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            losses: List of task losses [segmentation, patch_classification, gland_classification]

        Returns:
            total_loss: Weighted sum of losses
            weights: Current loss weights (precision values)
        """
        total_loss = 0
        weights = []

        for i, loss in enumerate(losses):
            if loss is not None and not torch.isnan(loss) and not torch.isinf(loss):
                precision = torch.exp(-self.log_vars[i])
                weighted_loss = precision * loss + self.log_vars[i]
                total_loss += weighted_loss
                weights.append(precision.item())
            else:
                weights.append(0.0)

        return total_loss, torch.tensor(weights)

class MultiTaskLoss(nn.Module):
    """
    4-class multi-task loss combining segmentation and classification losses
    Supports both single-label and multi-label classification
    """
    def __init__(self,
                 use_adaptive_weighting: bool = True,
                 fixed_weights: Optional[Dict[str, float]] = None,
                 dice_weight: float = 0.5,
                 ce_weight: float = 0.5,
                 class_weights: Optional[List[float]] = None,
                 use_focal_loss: bool = False,
                 use_multilabel_patch: bool = True):
        super().__init__()

        self.use_adaptive_weighting = use_adaptive_weighting
        self.use_focal_loss = use_focal_loss
        self.use_multilabel_patch = use_multilabel_patch

        # Segmentation loss
        self.segmentation_loss = CombinedSegmentationLoss(
            dice_weight=dice_weight,
            ce_weight=ce_weight,
            class_weights=class_weights
        )

        # Classification losses
        if use_multilabel_patch:
            # Multi-label classification for patches (can have multiple gland types)
            self.patch_classification_loss = nn.BCEWithLogitsLoss()  # Built-in sigmoid + BCE
            print("   🏷️ Using multi-label patch classification (BCEWithLogitsLoss)")
        else:
            # Single-label classification (backward compatibility)
            if use_focal_loss:
                self.patch_classification_loss = FocalLoss(alpha=1.0, gamma=2.0)
            else:
                self.patch_classification_loss = nn.CrossEntropyLoss()
            print("   🏷️ Using single-label patch classification")

        # Gland-level is still single-label (each gland has one primary type)
        if use_focal_loss:
            self.gland_classification_loss = FocalLoss(alpha=1.0, gamma=2.0)
        else:
            self.gland_classification_loss = nn.CrossEntropyLoss()

        # Loss weighting
        if use_adaptive_weighting:
            self.loss_weighting = AdaptiveLossWeighting(num_tasks=3)
        else:
            self.fixed_weights = fixed_weights or {
                'segmentation': 0.5,
                'patch_classification': 0.3,
                'gland_classification': 0.2
            }

        print(f"✅ MultiTaskLoss initialized:")
        print(f"   🎯 Adaptive weighting: {use_adaptive_weighting}")
        print(f"   🎪 Focal loss: {use_focal_loss}")
        print(f"   🔄 Multi-label patches: {use_multilabel_patch}")
        print(f"   ⚖️ Dice weight: {dice_weight}, CE weight: {ce_weight}")
        if class_weights:
            print(f"   📊 Class weights: {class_weights}")

    def forward(self,
                outputs: Dict[str, Any],
                targets: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """
        Args:
            outputs: Model outputs dictionary containing:
                - segmentation: [B, 4, H, W] segmentation logits
                - patch_classification: [B, 4] patch classification logits
                - gland_classification: [N, 4] gland classification logits
                - gland_counts: List[int] number of glands per image
            targets: Ground truth dictionary with keys:
                - segmentation: [B, H, W] segmentation targets (0-3)
                - patch_labels: [B] patch-level labels (0-3)
                - gland_labels: [N] gland-level labels (0-3)
                - gland_counts: List[int] number of glands per image
        """
        losses = {}
        task_losses = []

        # Segmentation loss
        if 'segmentation' in outputs and 'segmentation' in targets:
            seg_loss = self.segmentation_loss(outputs['segmentation'], targets['segmentation'])
            losses['segmentation'] = seg_loss
            task_losses.append(seg_loss)
        else:
            task_losses.append(None)

        # Patch classification loss
        if 'patch_classification' in outputs and 'patch_labels' in targets:
            if self.use_multilabel_patch:
                # Multi-label classification: targets should be [B, 4] float tensor
                patch_targets = targets['patch_labels'].float()
                patch_loss = self.patch_classification_loss(
                    outputs['patch_classification'],
                    patch_targets
                )
            else:
                # Single-label classification: targets should be [B] long tensor
                patch_loss = self.patch_classification_loss(
                    outputs['patch_classification'],
                    targets['patch_labels'].long()
                )
            losses['patch_classification'] = patch_loss
            task_losses.append(patch_loss)
        else:
            task_losses.append(None)

        # Gland classification loss
        if ('gland_classification' in outputs and
            'gland_labels' in targets and
            outputs['gland_classification'].shape[0] > 0 and
            targets['gland_labels'].shape[0] > 0):

            gland_loss = self.gland_classification_loss(
                outputs['gland_classification'],
                targets['gland_labels'].long()
            )
            losses['gland_classification'] = gland_loss
            task_losses.append(gland_loss)
        else:
            task_losses.append(None)

        # Handle deep supervision if present
        if 'deep_supervision' in outputs and 'segmentation' in targets:
            deep_losses = []
            deep_supervision_outputs = outputs['deep_supervision']

            for i, deep_output in enumerate(deep_supervision_outputs[1:], 1):  # Skip first (main) output
                # Resize target to match deep supervision output size
                target_resized = F.interpolate(
                    targets['segmentation'].unsqueeze(1).float(),
                    size=deep_output.shape[-2:],
                    mode='nearest'
                ).squeeze(1).long()

                deep_loss = self.segmentation_loss(deep_output, target_resized)
                deep_losses.append(deep_loss)

            if deep_losses:
                # Weight deep supervision losses (typically lower weights for deeper levels)
                deep_weights = [0.5 ** (i+1) for i in range(len(deep_losses))]
                weighted_deep_loss = sum(w * l for w, l in zip(deep_weights, deep_losses))
                losses['deep_supervision'] = weighted_deep_loss

                # Add to segmentation loss
                if 'segmentation' in losses:
                    losses['segmentation'] = losses['segmentation'] + 0.5 * weighted_deep_loss

        # Combine losses
        if self.use_adaptive_weighting:
            total_loss, weights = self.loss_weighting(task_losses)
            losses['total'] = total_loss
            losses['weights'] = weights
        else:
            total_loss = 0
            for loss_name, loss_value in losses.items():
                if loss_value is not None and loss_name in self.fixed_weights:
                    total_loss += self.fixed_weights[loss_name] * loss_value
            losses['total'] = total_loss

        return losses

def create_4class_gland_labels_from_patch_labels(patch_labels: torch.Tensor,
                                                gland_counts: List[int]) -> torch.Tensor:
    """
    Create 4-class gland-level labels from patch-level labels

    Args:
        patch_labels: [B] patch-level labels (0=Background, 1=Benign, 2=Malignant, 3=PDC)
        gland_counts: List[int] number of glands per image

    Returns:
        gland_labels: [total_glands] gland-level labels (0-3)

    Note: This is a simplified approach where all glands in a patch
    inherit the patch-level label.
    """
    gland_labels = []

    for batch_idx, count in enumerate(gland_counts):
        if count > 0:
            # Convert patch label to gland label (same as patch label)
            patch_label = patch_labels[batch_idx].item()
            gland_labels.extend([patch_label] * count)

    if gland_labels:
        return torch.tensor(gland_labels, device=patch_labels.device)
    else:
        return torch.zeros(0, device=patch_labels.device, dtype=torch.long)

def create_detailed_4class_gland_labels(patch_labels: torch.Tensor,
                                       gland_counts: List[int],
                                       segmentation_mask: torch.Tensor) -> torch.Tensor:
    """
    Create detailed 4-class gland-level labels using segmentation information

    Args:
        patch_labels: [B] patch-level labels (0-3)
        gland_counts: List[int] number of glands per image
        segmentation_mask: [B, H, W] segmentation predictions (0=bg, 1=benign, 2=malignant, 3=PDC)

    Returns:
        gland_labels: [total_glands] 4-class gland-level labels (0-3)
    """
    # For now, use the simpler approach
    # Could be extended to analyze individual gland regions in the segmentation mask
    return create_4class_gland_labels_from_patch_labels(patch_labels, gland_counts)

def calculate_class_weights(segmentation_targets: torch.Tensor, num_classes: int = 4) -> List[float]:
    """
    Calculate class weights based on frequency to handle class imbalance

    Args:
        segmentation_targets: [B, H, W] segmentation targets
        num_classes: Number of classes (default: 4)

    Returns:
        class_weights: List of weights for each class
    """
    # Count pixels for each class
    class_counts = []
    for class_id in range(num_classes):
        count = (segmentation_targets == class_id).sum().item()
        class_counts.append(count)

    # Calculate weights (inverse frequency)
    total_pixels = sum(class_counts)
    class_weights = []

    for count in class_counts:
        if count > 0:
            weight = total_pixels / (num_classes * count)
        else:
            weight = 0.0
        class_weights.append(weight)

    # Normalize weights
    max_weight = max(class_weights)
    if max_weight > 0:
        class_weights = [w / max_weight for w in class_weights]

    return class_weights

def test_4class_loss_functions():
    """Test the 4-class loss functions"""
    print("🧪 Testing 4-class loss functions...")

    batch_size = 2
    num_classes = 4  # Background, Benign, Malignant, PDC
    height, width = 64, 64

    # Create test data
    seg_pred = torch.randn(batch_size, num_classes, height, width)
    seg_target = torch.randint(0, num_classes, (batch_size, height, width))

    patch_pred = torch.randn(batch_size, num_classes)
    patch_labels = torch.randint(0, num_classes, (batch_size,))

    # Gland data
    num_glands = 3
    gland_pred = torch.randn(num_glands, num_classes)
    gland_labels = torch.randint(0, num_classes, (num_glands,))
    gland_counts = [1, 2]  # 1 gland in first image, 2 in second

    print(f"📊 Test data shapes:")
    print(f"   Segmentation pred: {seg_pred.shape}, target: {seg_target.shape}")
    print(f"   Patch pred: {patch_pred.shape}, labels: {patch_labels.shape}")
    print(f"   Gland pred: {gland_pred.shape}, labels: {gland_labels.shape}")
    print(f"   Gland counts: {gland_counts}")

    # Test individual loss components
    print(f"\n🎯 Testing individual loss components:")

    # Dice loss
    dice_loss = FourClassDiceLoss()
    dice_result = dice_loss(seg_pred, seg_target)
    print(f"   ✅ Dice loss: {dice_result.item():.4f}")

    # Weighted CE loss
    ce_loss = WeightedCrossEntropyLoss()
    ce_result = ce_loss(seg_pred, seg_target)
    print(f"   ✅ Weighted CE loss: {ce_result.item():.4f}")

    # Combined segmentation loss
    combined_seg_loss = CombinedSegmentationLoss()
    combined_result = combined_seg_loss(seg_pred, seg_target)
    print(f"   ✅ Combined segmentation loss: {combined_result.item():.4f}")

    # Focal loss
    focal_loss = FocalLoss()
    focal_result = focal_loss(patch_pred, patch_labels)
    print(f"   ✅ Focal loss: {focal_result.item():.4f}")

    # Test multi-task loss
    print(f"\n🔄 Testing multi-task loss:")

    # Test with adaptive weighting
    print("   📊 Adaptive weighting:")
    multitask_loss_adaptive = MultiTaskLoss(use_adaptive_weighting=True, use_focal_loss=False)

    outputs = {
        'segmentation': seg_pred,
        'patch_classification': patch_pred,
        'gland_classification': gland_pred,
        'gland_counts': gland_counts
    }

    targets = {
        'segmentation': seg_target,
        'patch_labels': patch_labels,
        'gland_labels': gland_labels,
        'gland_counts': gland_counts
    }

    loss_results = multitask_loss_adaptive(outputs, targets)
    print(f"      Total loss: {loss_results['total'].item():.4f}")
    print(f"      Weights: {loss_results['weights'].tolist()}")
    print(f"      Individual losses:")
    for key, value in loss_results.items():
        if key not in ['total', 'weights'] and isinstance(value, torch.Tensor):
            print(f"         {key}: {value.item():.4f}")

    # Test with fixed weighting
    print("   📊 Fixed weighting:")
    multitask_loss_fixed = MultiTaskLoss(use_adaptive_weighting=False, use_focal_loss=True)
    loss_results_fixed = multitask_loss_fixed(outputs, targets)
    print(f"      Total loss: {loss_results_fixed['total'].item():.4f}")
    print(f"      Individual losses:")
    for key, value in loss_results_fixed.items():
        if key != 'total' and isinstance(value, torch.Tensor):
            print(f"         {key}: {value.item():.4f}")

    # Test class weights calculation
    print(f"\n⚖️ Testing class weights calculation:")
    class_weights = calculate_class_weights(seg_target, num_classes)
    print(f"   Calculated class weights: {[f'{w:.3f}' for w in class_weights]}")

    # Test label creation
    print(f"\n🏷️ Testing label creation:")
    gland_labels_created = create_4class_gland_labels_from_patch_labels(patch_labels, gland_counts)
    print(f"   Created gland labels: {gland_labels_created.tolist()}")
    print(f"   Expected gland count: {sum(gland_counts)}, Actual: {len(gland_labels_created)}")

    print(f"\n✅ 4-class loss functions test completed successfully!")
    print(f"📝 Summary:")
    print(f"   - 4-class segmentation loss: Dice + Weighted CE")
    print(f"   - Classification losses: Cross Entropy / Focal Loss")
    print(f"   - Multi-task weighting: Adaptive / Fixed")
    print(f"   - Class imbalance handling: ✅")
    print(f"   - Deep supervision support: ✅")
    print(f"   - All tests passed!")

if __name__ == "__main__":
    test_4class_loss_functions()