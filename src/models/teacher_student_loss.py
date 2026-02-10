#!/usr/bin/env python3
"""
Teacher-Student Loss Functions for Self-Training Gland Segmentation
==================================================================

Dual loss system for Teacher-Student learning:
- Supervised Loss: Student predictions vs Ground Truth (traditional multi-task loss)
- Consistency Loss: Student predictions vs Teacher pseudo-labels (all heads)
- Cosine Decay Weighting: Alpha schedule for supervised → consistency transition
- GT + Teacher Incorporation: Enhanced pseudo-mask fusion (optional)

Features:
- Multi-task consistency across segmentation + classification heads
- Configurable loss components and weights
- Cosine annealing for smooth transition
- GT + Teacher Incorporation for enhanced pseudo-masks with teacher discovery
- Pseudo-mask filtering (confidence/entropy based)
- Adaptive confidence threshold annealing
- Compatible with existing MultiTaskLoss

GT + Teacher Incorporation Algorithm:
- GT Priority: GT foreground annotations are always preserved
- Teacher Discovery: GT background pixels use teacher predictions
- Confidence Preservation: Filtering applied to original teacher logits BEFORE GT incorporation
- Processing Order: Confidence Filtering → GT Incorporation → Loss Computation
- Result: Enhanced masks with perfect GT + teacher's discovered structures

Author: Claude Code - Generated for OSU CRC Research
Date: 2025-09-22
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, Any, Optional, Union, Tuple
import warnings

# Local imports
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.models.loss_functions import MultiTaskLoss, FourClassDiceLoss


class ConsistencyLoss(nn.Module):
    """
    Consistency loss between student and teacher outputs across all heads
    """

    def __init__(self,
                 segmentation_weight: float = 1.0,
                 patch_classification_weight: float = 0.5,
                 gland_classification_weight: float = 0.5,
                 enable_gland_consistency: bool = False,
                 temperature: float = 1.0,
                 loss_type: str = "mse",
                 pseudo_mask_filtering: str = "none",
                 confidence_threshold: float = 0.8,
                 entropy_threshold: float = 1.0,
                 confidence_annealing: str = "none",
                 confidence_max_threshold: float = 0.9,
                 confidence_min_threshold: float = 0.6,
                 confidence_annealing_start_epoch: int = 5,
                 total_epochs: int = 100,
                 gt_teacher_incorporate_enabled: bool = False,
                 gt_incorporate_start_epoch: int = 0,
                 gt_incorporate_segmentation_only: bool = True):
        """
        Initialize consistency loss

        Args:
            segmentation_weight: Weight for segmentation consistency
            patch_classification_weight: Weight for patch classification consistency
            gland_classification_weight: Weight for gland classification consistency
            enable_gland_consistency: Whether to include gland classification consistency loss
            temperature: Temperature for softmax (knowledge distillation style)
            loss_type: Type of consistency loss ("mse", "kl_div", "l1", "dice", "iou")
            pseudo_mask_filtering: Filtering strategy ("none", "confidence", "entropy")
            confidence_threshold: Minimum confidence for confidence-based filtering (0.7-0.95)
            entropy_threshold: Maximum entropy for entropy-based filtering (0.5-2.0)
            confidence_annealing: Annealing schedule ("none", "linear", "cosine")
            confidence_max_threshold: Starting confidence threshold (early training)
            confidence_min_threshold: Ending confidence threshold (late training)
            confidence_annealing_start_epoch: Epoch to start annealing (after warmup)
            total_epochs: Total number of training epochs for annealing calculation
            gt_teacher_incorporate_enabled: Enable GT + Teacher incorporation fusion
            gt_incorporate_start_epoch: Epoch to start GT incorporation (default: same as teacher init)
            gt_incorporate_segmentation_only: Apply GT incorporation only to segmentation masks
        """
        super().__init__()

        self.segmentation_weight = segmentation_weight
        self.patch_classification_weight = patch_classification_weight
        self.gland_classification_weight = gland_classification_weight
        self.enable_gland_consistency = enable_gland_consistency
        self.temperature = temperature
        self.loss_type = loss_type
        self.pseudo_mask_filtering = pseudo_mask_filtering
        self.confidence_threshold = confidence_threshold
        self.entropy_threshold = entropy_threshold
        self.current_epoch = 0  # Track current epoch for warmup
        self._filtering_enabled = True  # Enable filtering by default if strategy is set

        # Adaptive confidence threshold annealing parameters
        self.confidence_annealing = confidence_annealing
        self.confidence_max_threshold = confidence_max_threshold
        self.confidence_min_threshold = confidence_min_threshold
        self.confidence_annealing_start_epoch = confidence_annealing_start_epoch
        self.total_epochs = total_epochs
        self.base_confidence_threshold = confidence_threshold  # Store original threshold

        # GT + Teacher Incorporation parameters
        self.gt_teacher_incorporate_enabled = gt_teacher_incorporate_enabled
        self.gt_incorporate_start_epoch = gt_incorporate_start_epoch
        self.gt_incorporate_segmentation_only = gt_incorporate_segmentation_only

        print(f"✅ Consistency Loss initialized:")
        print(f"   🎯 Segmentation weight: {segmentation_weight}")
        print(f"   🏷️ Patch classification weight: {patch_classification_weight}")
        print(f"   🔧 Gland classification weight: {gland_classification_weight}")
        print(f"   ✅ Gland consistency enabled: {enable_gland_consistency}")
        print(f"   🌡️ Temperature: {temperature}")
        print(f"   📊 Loss type: {loss_type}")
        print(f"   🎭 Pseudo-mask filtering: {pseudo_mask_filtering}")
        if pseudo_mask_filtering == "confidence":
            print(f"   🎯 Base confidence threshold: {confidence_threshold}")
            if confidence_annealing != "none":
                print(f"   📈 Adaptive confidence annealing: {confidence_annealing}")
                print(f"   🔼 Max threshold (early): {confidence_max_threshold}")
                print(f"   🔽 Min threshold (late): {confidence_min_threshold}")
                print(f"   ⏱️ Annealing start epoch: {confidence_annealing_start_epoch}")
        elif pseudo_mask_filtering == "entropy":
            print(f"   📊 Entropy threshold: {entropy_threshold}")

        # GT + Teacher Incorporation info
        if gt_teacher_incorporate_enabled:
            print(f"   🧬 GT + Teacher Incorporation: ENABLED")
            print(f"   🚀 GT incorporation start epoch: {gt_incorporate_start_epoch}")
            print(f"   🎯 Segmentation only: {gt_incorporate_segmentation_only}")
        else:
            print(f"   🧬 GT + Teacher Incorporation: DISABLED")

    def update_epoch(self, epoch: int):
        """Update current epoch for warmup logic and annealing"""
        self.current_epoch = epoch

        # Update confidence threshold with annealing if applicable
        if (self.pseudo_mask_filtering == "confidence" and
            self.confidence_annealing != "none"):
            self.confidence_threshold = self._get_annealed_confidence_threshold(epoch)

    def _get_annealed_confidence_threshold(self, current_epoch: int) -> float:
        """
        Calculate annealed confidence threshold based on current epoch and schedule

        Args:
            current_epoch: Current training epoch

        Returns:
            Annealed confidence threshold
        """
        # If annealing is disabled, return base threshold
        if self.confidence_annealing == "none":
            return self.base_confidence_threshold

        # If before annealing start epoch, return max threshold
        if current_epoch < self.confidence_annealing_start_epoch:
            return self.confidence_max_threshold

        # Calculate annealing progress
        annealing_epochs = self.total_epochs - self.confidence_annealing_start_epoch
        if annealing_epochs <= 0:
            return self.confidence_min_threshold

        progress = (current_epoch - self.confidence_annealing_start_epoch) / annealing_epochs
        progress = min(1.0, progress)  # Clamp to [0, 1]

        # Calculate threshold based on annealing schedule
        if self.confidence_annealing == "linear":
            # Linear interpolation from max to min
            threshold = self.confidence_max_threshold + \
                       (self.confidence_min_threshold - self.confidence_max_threshold) * progress

        elif self.confidence_annealing == "cosine":
            # Cosine annealing: smooth transition from max to min
            threshold = self.confidence_min_threshold + \
                       (self.confidence_max_threshold - self.confidence_min_threshold) * \
                       0.5 * (1 + math.cos(math.pi * progress))
        else:
            # Fallback to base threshold
            threshold = self.base_confidence_threshold

        return threshold

    def set_filtering_enabled(self, enabled: bool):
        """Enable/disable filtering dynamically"""
        if enabled and self.pseudo_mask_filtering == "none":
            return  # No filtering strategy set
        self._filtering_enabled = enabled

    def _apply_pseudo_mask_filtering(self, teacher_logits: torch.Tensor, is_segmentation: bool = False) -> torch.Tensor:
        """
        Apply filtering to teacher pseudo-masks based on confidence or entropy

        Args:
            teacher_logits: Teacher network logits [B, C, H, W] or [B, C]
            is_segmentation: Whether this is segmentation data (4D) or classification data (2D)

        Returns:
            Binary mask indicating which pixels/samples to include in consistency loss
        """
        if self.pseudo_mask_filtering == "none" or not hasattr(self, '_filtering_enabled') or not self._filtering_enabled:
            # No filtering, use all pixels/samples
            if is_segmentation:
                return torch.ones(teacher_logits.shape[0], teacher_logits.shape[2], teacher_logits.shape[3],
                                device=teacher_logits.device, dtype=torch.float32)
            else:
                return torch.ones(teacher_logits.shape[0], device=teacher_logits.device, dtype=torch.float32)

        # MEMORY SAFE: Compute probabilities for filtering with detached operations
        with torch.no_grad():
            teacher_probs = F.softmax(teacher_logits.detach(), dim=1)

            if self.pseudo_mask_filtering == "confidence":
                # A. Confidence-Based Masking
                confidence, _ = torch.max(teacher_probs, dim=1)  # [B, H, W] or [B]
                mask = confidence > self.confidence_threshold

            elif self.pseudo_mask_filtering == "entropy":
                # B. Entropy-Based Filtering
                entropy = -torch.sum(teacher_probs * torch.log(teacher_probs + 1e-8), dim=1)  # [B, H, W] or [B]
                mask = entropy < self.entropy_threshold  # Low entropy = high confidence

            else:
                raise ValueError(f"Unknown pseudo_mask_filtering: {self.pseudo_mask_filtering}")

            return mask.float().detach()  # Ensure mask is detached

    def _create_gt_teacher_incorporate_fusion(self, gt_mask: torch.Tensor, teacher_logits: torch.Tensor) -> torch.Tensor:
        """
        Create GT + Teacher Incorporation enhanced pseudo-mask
        MEMORY SAFE: All operations use detached tensors and avoid gradient retention

        Core Algorithm: GT Priority Fusion with Teacher Discovery
        - GT foreground annotations are always preserved (highest priority)
        - GT background pixels allow teacher predictions (including teacher's foreground detections)
        - This creates a 'GT + Teacher Discovery' enhanced mask where:
          * GT=foreground → Use GT value (perfect annotation)
          * GT=background → Use teacher prediction (teacher can discover missed structures)

        Args:
            gt_mask: Ground truth segmentation mask [B, H, W] (class indices)
            teacher_logits: Teacher network logits [B, C, H, W] (raw predictions)

        Returns:
            Enhanced pseudo-mask as logits [B, C, H, W] (GT priority + teacher discovery)
        """
        # CRITICAL: Detach teacher logits immediately to prevent graph retention
        with torch.no_grad():
            teacher_logits_detached = teacher_logits.detach()
            gt_mask_detached = gt_mask.detach()

            # Convert GT mask to one-hot for logit-space fusion
            num_classes = teacher_logits_detached.shape[1]
            device = teacher_logits_detached.device

            # Create GT foreground mask: 1 where GT has foreground, 0 where GT is background
            gt_foreground_mask = (gt_mask_detached > 0).float().to(device)  # [B, H, W]

            # Convert GT mask to one-hot logits (high confidence for GT classes)
            gt_one_hot = F.one_hot(gt_mask_detached.long(), num_classes=num_classes).permute(0, 3, 1, 2).float()  # [B, C, H, W]
            gt_logits = gt_one_hot * 10.0  # High confidence for GT annotations

            # GT + Teacher Incorporation Fusion:
            # - Where GT=foreground: Use GT logits (perfect annotations)
            # - Where GT=background: Use teacher logits (teacher discovery)
            enhanced_logits = (
                gt_foreground_mask.unsqueeze(1) * gt_logits +  # GT priority for foreground
                (1 - gt_foreground_mask.unsqueeze(1)) * teacher_logits_detached  # Teacher discovery for background
            )

            return enhanced_logits.detach()  # Ensure output is detached

    def _compute_consistency_loss(self, student_logits: torch.Tensor, teacher_logits: torch.Tensor, is_segmentation: bool = False, gt_targets: torch.Tensor = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Compute consistency loss between student and teacher logits

        Args:
            student_logits: Student network logits
            teacher_logits: Teacher network logits (pseudo-labels)
            is_segmentation: Whether this is segmentation data (4D) or classification data (2D)
            gt_targets: Ground truth targets for GT incorporation (optional)

        Returns:
            Tuple of (consistency loss value, detached pseudo-mask for metrics)
        """
        # Handle tensor shape mismatch by taking minimum batch size
        if student_logits.shape != teacher_logits.shape:
            min_batch_size = min(student_logits.shape[0], teacher_logits.shape[0])
            student_logits = student_logits[:min_batch_size]
            teacher_logits = teacher_logits[:min_batch_size]
            if gt_targets is not None:
                gt_targets = gt_targets[:min_batch_size]

        # CRITICAL ORDER: Extract pseudo-mask for metrics BEFORE any modifications
        # This prevents any potential graph retention from GT incorporation or filtering
        pseudo_mask_for_metrics = teacher_logits.detach() if is_segmentation else None

        # MEMORY SAFE: Apply pseudo-mask filtering to original teacher logits
        # This preserves the confidence logic - filtering is based on teacher's original predictions
        filtering_mask = self._apply_pseudo_mask_filtering(teacher_logits, is_segmentation)

        # Apply GT + Teacher Incorporation AFTER extracting monitoring metrics
        # CRITICAL: Only apply if we're past warmup phase (teacher must be initialized)
        if (self.gt_teacher_incorporate_enabled and
            is_segmentation and  # Only for segmentation if specified
            gt_targets is not None and
            self.current_epoch >= self.gt_incorporate_start_epoch and
            self.current_epoch >= self.warmup_epochs):  # Ensure teacher is initialized!
            # Create GT + Teacher enhanced pseudo-mask (monitoring metrics already extracted above)
            teacher_logits = self._create_gt_teacher_incorporate_fusion(gt_targets, teacher_logits)

        # IMPORTANT: This consistency loss is for ACTUAL TRAINING (not monitoring)
        # It trains the student to match the teacher's predictions
        # The pseudo_mask_for_metrics is ONLY for monitoring metrics computation
        if self.loss_type == "mse":
            # Mean Squared Error between logits
            loss = F.mse_loss(student_logits, teacher_logits, reduction='none')
            # Apply filtering mask
            if is_segmentation:
                # For segmentation: loss shape [B, C, H, W], mask shape [B, H, W]
                loss = loss * filtering_mask.unsqueeze(1)  # Broadcast to [B, 1, H, W]
                final_loss = loss.sum() / (filtering_mask.sum() * student_logits.shape[1] + 1e-8)
                return final_loss, pseudo_mask_for_metrics
            else:
                # For classification: loss shape [B, C], mask shape [B]
                loss = loss * filtering_mask.unsqueeze(1)  # Broadcast to [B, 1]
                final_loss = loss.sum() / (filtering_mask.sum() * student_logits.shape[1] + 1e-8)
                return final_loss, pseudo_mask_for_metrics

        elif self.loss_type == "kl_div":
            # KL Divergence (knowledge distillation style)
            student_probs = F.log_softmax(student_logits / self.temperature, dim=1)
            teacher_probs = F.softmax(teacher_logits / self.temperature, dim=1)
            loss = F.kl_div(student_probs, teacher_probs, reduction='none') * (self.temperature ** 2)
            # Apply filtering mask
            if is_segmentation:
                loss = loss * filtering_mask.unsqueeze(1)
                final_loss = loss.sum() / (filtering_mask.sum() * student_logits.shape[1] + 1e-8)
                return final_loss, pseudo_mask_for_metrics
            else:
                loss = loss * filtering_mask.unsqueeze(1)
                final_loss = loss.sum() / (filtering_mask.sum() * student_logits.shape[1] + 1e-8)
                return final_loss, pseudo_mask_for_metrics

        elif self.loss_type == "l1":
            # L1 (MAE) loss
            loss = F.l1_loss(student_logits, teacher_logits, reduction='none')
            # Apply filtering mask
            if is_segmentation:
                loss = loss * filtering_mask.unsqueeze(1)
                final_loss = loss.sum() / (filtering_mask.sum() * student_logits.shape[1] + 1e-8)
                return final_loss, pseudo_mask_for_metrics
            else:
                loss = loss * filtering_mask.unsqueeze(1)
                final_loss = loss.sum() / (filtering_mask.sum() * student_logits.shape[1] + 1e-8)
                return final_loss, pseudo_mask_for_metrics

        elif self.loss_type == "dice":
            # Dice consistency loss for regional structure preservation
            if is_segmentation:
                dice_loss = self._compute_dice_consistency_loss(student_logits, teacher_logits, filtering_mask)
                return dice_loss, pseudo_mask_for_metrics
            else:
                # For non-segmentation (classification) outputs, fallback to MSE with filtering
                loss = F.mse_loss(student_logits, teacher_logits, reduction='none')
                loss = loss * filtering_mask.unsqueeze(1)
                final_loss = loss.sum() / (filtering_mask.sum() * student_logits.shape[1] + 1e-8)
                return final_loss, pseudo_mask_for_metrics

        elif self.loss_type == "iou":
            # IoU consistency loss for regional structure preservation
            if is_segmentation:
                iou_loss = self._compute_iou_consistency_loss(student_logits, teacher_logits, filtering_mask)
                return iou_loss, pseudo_mask_for_metrics
            else:
                # For non-segmentation (classification) outputs, fallback to MSE with filtering
                loss = F.mse_loss(student_logits, teacher_logits, reduction='none')
                loss = loss * filtering_mask.unsqueeze(1)
                final_loss = loss.sum() / (filtering_mask.sum() * student_logits.shape[1] + 1e-8)
                return final_loss, pseudo_mask_for_metrics

        else:
            raise ValueError(f"Unknown consistency loss type: {self.loss_type}")

    def _compute_dice_consistency_loss(self, student_logits: torch.Tensor, teacher_logits: torch.Tensor, filtering_mask: torch.Tensor) -> torch.Tensor:
        """
        Compute Dice consistency loss between student and teacher segmentation masks
        Preserves regional structure by focusing on overlap between predicted regions

        Args:
            student_logits: Student network logits [B, C, H, W]
            teacher_logits: Teacher network logits [B, C, H, W]
            filtering_mask: Binary mask indicating which pixels to include [B, H, W]

        Returns:
            Dice consistency loss (1 - dice_similarity)
        """
        # Convert logits to probabilities
        student_probs = F.softmax(student_logits, dim=1)
        teacher_probs = F.softmax(teacher_logits, dim=1)

        # Compute Dice similarity for each class (excluding background)
        smooth = 1e-5
        num_classes = student_probs.shape[1]
        dice_scores = []

        # Skip background class (index 0) for foreground consistency
        for c in range(1, num_classes):
            student_c = student_probs[:, c]  # [B, H, W]
            teacher_c = teacher_probs[:, c]   # [B, H, W]

            # Apply filtering mask
            student_c = student_c * filtering_mask
            teacher_c = teacher_c * filtering_mask

            # Compute intersection and union
            intersection = (student_c * teacher_c).sum(dim=[1, 2])  # [B]
            student_sum = student_c.sum(dim=[1, 2])  # [B]
            teacher_sum = teacher_c.sum(dim=[1, 2])  # [B]
            union = student_sum + teacher_sum

            # Dice coefficient
            dice = (2 * intersection + smooth) / (union + smooth)  # [B]
            dice_scores.append(dice)

        # Average Dice across classes and batch (weight by valid pixels)
        dice_scores = torch.stack(dice_scores, dim=1)  # [B, num_foreground_classes]
        mean_dice = dice_scores.mean()

        # Return Dice loss (1 - dice_similarity)
        return 1.0 - mean_dice

    def _compute_iou_consistency_loss(self, student_logits: torch.Tensor, teacher_logits: torch.Tensor, filtering_mask: torch.Tensor) -> torch.Tensor:
        """
        Compute IoU (Intersection over Union) consistency loss between student and teacher masks
        Focuses on exact region overlap and boundary alignment

        Args:
            student_logits: Student network logits [B, C, H, W]
            teacher_logits: Teacher network logits [B, C, H, W]
            filtering_mask: Binary mask indicating which pixels to include [B, H, W]

        Returns:
            IoU consistency loss (1 - iou_similarity)
        """
        # Convert logits to probabilities
        student_probs = F.softmax(student_logits, dim=1)
        teacher_probs = F.softmax(teacher_logits, dim=1)

        # Compute IoU similarity for each class (excluding background)
        smooth = 1e-5
        num_classes = student_probs.shape[1]
        iou_scores = []

        # Skip background class (index 0) for foreground consistency
        for c in range(1, num_classes):
            student_c = student_probs[:, c]  # [B, H, W]
            teacher_c = teacher_probs[:, c]   # [B, H, W]

            # Apply filtering mask
            student_c = student_c * filtering_mask
            teacher_c = teacher_c * filtering_mask

            # Compute intersection and union
            intersection = (student_c * teacher_c).sum(dim=[1, 2])  # [B]
            student_sum = student_c.sum(dim=[1, 2])  # [B]
            teacher_sum = teacher_c.sum(dim=[1, 2])  # [B]
            union = student_sum + teacher_sum - intersection

            # IoU coefficient
            iou = (intersection + smooth) / (union + smooth)  # [B]
            iou_scores.append(iou)

        # Average IoU across classes and batch
        iou_scores = torch.stack(iou_scores, dim=1)  # [B, num_foreground_classes]
        mean_iou = iou_scores.mean()

        # Return IoU loss (1 - iou_similarity)
        return 1.0 - mean_iou

    def forward(self, student_outputs: Dict[str, torch.Tensor], teacher_outputs: Dict[str, torch.Tensor], gt_targets: Optional[Dict[str, torch.Tensor]] = None) -> Dict[str, torch.Tensor]:
        """
        Compute consistency loss across all heads

        Args:
            student_outputs: Student network outputs
            teacher_outputs: Teacher network outputs (pseudo-labels)
            gt_targets: Ground truth targets for GT incorporation (optional)

        Returns:
            Dictionary with consistency losses and total consistency loss
        """
        consistency_losses = {}

        # Get device from student outputs
        device = None
        dtype = None
        for output in student_outputs.values():
            if isinstance(output, torch.Tensor):
                device = output.device
                dtype = output.dtype
                break

        # Initialize total consistency loss as tensor on correct device
        total_consistency_loss = torch.tensor(0.0, device=device, dtype=dtype) if device is not None else 0.0

        # Initialize pseudo-mask holder for monitoring metrics
        pseudo_mask_for_metrics = None

        # Segmentation consistency
        if 'segmentation' in student_outputs and 'segmentation' in teacher_outputs:
            # Extract GT segmentation targets if available
            gt_seg_targets = gt_targets.get('segmentation', None) if gt_targets else None

            seg_consistency, pseudo_mask = self._compute_consistency_loss(
                student_outputs['segmentation'],
                teacher_outputs['segmentation'],
                is_segmentation=True,  # Segmentation data is 4D
                gt_targets=gt_seg_targets
            )
            consistency_losses['segmentation'] = seg_consistency
            total_consistency_loss = total_consistency_loss + self.segmentation_weight * seg_consistency

            # Store pseudo-mask for monitoring metrics (detached)
            pseudo_mask_for_metrics = pseudo_mask

        # Patch classification consistency
        if 'patch_classification' in student_outputs and 'patch_classification' in teacher_outputs:
            patch_consistency, _ = self._compute_consistency_loss(
                student_outputs['patch_classification'],
                teacher_outputs['patch_classification'],
                is_segmentation=False  # Classification data is 2D
            )
            consistency_losses['patch_classification'] = patch_consistency
            total_consistency_loss = total_consistency_loss + self.patch_classification_weight * patch_consistency

        # Gland classification consistency (configurable)
        if (self.enable_gland_consistency and
            'gland_classification' in student_outputs and
            'gland_classification' in teacher_outputs):
            gland_consistency, _ = self._compute_consistency_loss(
                student_outputs['gland_classification'],
                teacher_outputs['gland_classification'],
                is_segmentation=False  # Classification data is 2D
            )
            consistency_losses['gland_classification'] = gland_consistency
            total_consistency_loss = total_consistency_loss + self.gland_classification_weight * gland_consistency

        # Store total consistency loss
        consistency_losses['total_consistency'] = total_consistency_loss

        return consistency_losses, pseudo_mask_for_metrics


class CosineDecayScheduler:
    """
    Cosine decay scheduler for loss weighting (alpha parameter)
    """

    def __init__(self, total_epochs: int, warmup_epochs: int = 0, min_alpha: float = 0.1, max_alpha: float = 1.0):
        """
        Initialize cosine decay scheduler

        Args:
            total_epochs: Total number of training epochs
            warmup_epochs: Number of warmup epochs (before teacher initialization)
            min_alpha: Minimum alpha value (maximum consistency weight)
            max_alpha: Maximum alpha value (maximum supervised weight)
        """
        self.total_epochs = total_epochs
        self.warmup_epochs = warmup_epochs
        self.min_alpha = min_alpha
        self.max_alpha = max_alpha

        print(f"✅ Cosine Decay Scheduler initialized:")
        print(f"   📅 Total epochs: {total_epochs}")
        print(f"   🔥 Warmup epochs: {warmup_epochs}")
        print(f"   📉 Alpha range: {max_alpha} → {min_alpha}")

    def get_alpha(self, current_epoch: int) -> float:
        """
        Get alpha value for current epoch using cosine decay

        Args:
            current_epoch: Current training epoch

        Returns:
            Alpha value (weight for supervised loss)
        """
        if current_epoch < self.warmup_epochs:
            # During warmup, use maximum supervised weight
            return self.max_alpha

        # Adjust epoch for cosine schedule (after warmup)
        adjusted_epoch = current_epoch - self.warmup_epochs
        adjusted_total = self.total_epochs - self.warmup_epochs

        if adjusted_total <= 0:
            return self.min_alpha

        # Cosine decay: starts at max_alpha, decays to min_alpha
        cosine_factor = 0.5 * (1 + math.cos(math.pi * adjusted_epoch / adjusted_total))
        alpha = self.min_alpha + (self.max_alpha - self.min_alpha) * cosine_factor

        return alpha


class TeacherStudentLoss(nn.Module):
    """
    Combined loss function for Teacher-Student learning

    Loss = alpha * supervised_loss + (1 - alpha) * consistency_loss
    where alpha follows cosine decay schedule
    """

    def __init__(self,
                 total_epochs: int,
                 warmup_epochs: int = 0,
                 min_alpha: float = 0.1,
                 max_alpha: float = 1.0,
                 consistency_loss_config: Optional[Dict[str, Any]] = None,
                 supervised_loss_config: Optional[Dict[str, Any]] = None):
        """
        Initialize Teacher-Student loss

        Args:
            total_epochs: Total training epochs for cosine decay
            warmup_epochs: Number of warmup epochs (teacher not initialized)
            min_alpha: Minimum alpha (maximum consistency weight)
            max_alpha: Maximum alpha (maximum supervised weight)
            consistency_loss_config: Configuration for consistency loss
            supervised_loss_config: Configuration for supervised loss
        """
        super().__init__()

        # Initialize supervised loss (existing multi-task loss)
        self.supervised_loss = MultiTaskLoss(**(supervised_loss_config or {}))

        # Initialize consistency loss
        consistency_config = consistency_loss_config or {}
        # Pass total_epochs to consistency loss for GT incorporation timing
        consistency_config['total_epochs'] = total_epochs
        self.consistency_loss = ConsistencyLoss(**consistency_config)

        # Initialize cosine decay scheduler
        self.alpha_scheduler = CosineDecayScheduler(
            total_epochs=total_epochs,
            warmup_epochs=warmup_epochs,
            min_alpha=min_alpha,
            max_alpha=max_alpha
        )

        self.total_epochs = total_epochs
        self.warmup_epochs = warmup_epochs

        print(f"✅ Teacher-Student Loss initialized:")
        print(f"   📅 Total epochs: {total_epochs}")
        print(f"   🔥 Warmup epochs: {warmup_epochs}")
        print(f"   📊 Loss components: Supervised + Consistency")

    def forward(self,
                student_outputs: Dict[str, torch.Tensor],
                teacher_outputs: Optional[Dict[str, torch.Tensor]],
                targets: Dict[str, torch.Tensor],
                current_epoch: int) -> Dict[str, torch.Tensor]:
        """
        Compute Teacher-Student loss

        Args:
            student_outputs: Student network outputs
            teacher_outputs: Teacher network outputs (None during warmup)
            targets: Ground truth targets
            current_epoch: Current training epoch

        Returns:
            Dictionary with loss components and total loss
        """
        loss_dict = {}

        # Compute supervised loss (student vs ground truth)
        supervised_loss_dict = self.supervised_loss(student_outputs, targets)

        # Calculate total supervised loss (sum of all task losses)
        # Ensure all losses are on the same device before summing
        device = None
        dtype = None
        for loss in supervised_loss_dict.values():
            if loss is not None and isinstance(loss, torch.Tensor):
                device = loss.device
                dtype = loss.dtype
                break

        supervised_total = None
        for loss in supervised_loss_dict.values():
            if loss is not None:
                # Ensure loss is on correct device
                if isinstance(loss, torch.Tensor) and device is not None:
                    loss = loss.to(device)

                if supervised_total is None:
                    supervised_total = loss
                else:
                    supervised_total = supervised_total + loss

        # If no valid losses found, create zero tensor on correct device
        if supervised_total is None:
            supervised_total = torch.tensor(0.0, device=device, dtype=dtype) if device is not None else torch.tensor(0.0)

        # Ensure supervised_total is a scalar
        if hasattr(supervised_total, 'dim') and supervised_total.dim() > 0:
            supervised_total = supervised_total.mean()

        # Get alpha value for current epoch
        alpha = self.alpha_scheduler.get_alpha(current_epoch)

        # Ensure alpha is a tensor on the correct device
        device = supervised_total.device
        alpha_tensor = torch.tensor(alpha, device=device, dtype=supervised_total.dtype)

        if teacher_outputs is None or current_epoch < self.warmup_epochs:
            # Warmup phase: only supervised loss
            loss_dict = {
                'total_loss': supervised_total,
                'supervised_loss': supervised_total,
                'consistency_loss': torch.tensor(0.0, device=device, dtype=supervised_total.dtype),
                'alpha': alpha_tensor,
                'phase': 'warmup',
                'pseudo_mask_for_metrics': None  # No pseudo-mask in warmup
            }
            # Include individual supervised loss components
            loss_dict.update({f"supervised_{k}": v for k, v in supervised_loss_dict.items()})

        else:
            # Teacher-Student phase: supervised + consistency loss
            consistency_loss_dict, pseudo_mask_for_metrics = self.consistency_loss(student_outputs, teacher_outputs, targets)
            consistency_total = consistency_loss_dict['total_consistency']

            # Ensure consistency_total is a scalar
            if hasattr(consistency_total, 'dim') and consistency_total.dim() > 0:
                consistency_total = consistency_total.mean()

            # Combined loss with alpha weighting (ensure alpha_tensor is used)
            total_loss = alpha_tensor * supervised_total + (1.0 - alpha_tensor) * consistency_total

            # Ensure total_loss is a scalar
            if hasattr(total_loss, 'dim') and total_loss.dim() > 0:
                total_loss = total_loss.mean()

            loss_dict = {
                'total_loss': total_loss,
                'supervised_loss': supervised_total,
                'consistency_loss': consistency_total,
                'alpha': alpha_tensor,
                'phase': 'teacher_student',
                'pseudo_mask_for_metrics': pseudo_mask_for_metrics  # For monitoring metrics
            }

            # Include individual loss components
            loss_dict.update({f"supervised_{k}": v for k, v in supervised_loss_dict.items()})
            loss_dict.update({f"consistency_{k}": v for k, v in consistency_loss_dict.items()})

        return loss_dict


def test_teacher_student_loss():
    """Test function for Teacher-Student loss"""
    print("🧪 Testing Teacher-Student Loss...")

    # Test with gland consistency disabled (default)
    print("\n🔧 Testing with gland consistency DISABLED...")
    loss_fn_disabled = TeacherStudentLoss(
        total_epochs=100,
        warmup_epochs=20,
        min_alpha=0.1,
        max_alpha=1.0,
        consistency_loss_config={
            'enable_gland_consistency': False
        }
    )

    # Test with gland consistency enabled
    print("\n🔧 Testing with gland consistency ENABLED...")
    loss_fn_enabled = TeacherStudentLoss(
        total_epochs=100,
        warmup_epochs=20,
        min_alpha=0.1,
        max_alpha=1.0,
        consistency_loss_config={
            'enable_gland_consistency': True
        }
    )

    # Test with Dice consistency loss
    print("\n🎯 Testing with Dice consistency loss...")
    loss_fn_dice = TeacherStudentLoss(
        total_epochs=100,
        warmup_epochs=20,
        min_alpha=0.1,
        max_alpha=1.0,
        consistency_loss_config={
            'loss_type': 'dice',
            'enable_gland_consistency': False
        }
    )

    # Test with IoU consistency loss
    print("\n🎯 Testing with IoU consistency loss...")
    loss_fn_iou = TeacherStudentLoss(
        total_epochs=100,
        warmup_epochs=20,
        min_alpha=0.1,
        max_alpha=1.0,
        consistency_loss_config={
            'loss_type': 'iou',
            'enable_gland_consistency': False
        }
    )

    # Create dummy data
    batch_size = 2
    height, width = 256, 256
    num_classes = 4

    # Student outputs
    student_outputs = {
        'segmentation': torch.randn(batch_size, num_classes, height, width),
        'patch_classification': torch.randn(batch_size, num_classes),
        'gland_classification': torch.randn(batch_size, num_classes)
    }

    # Teacher outputs (same structure)
    teacher_outputs = {
        'segmentation': torch.randn(batch_size, num_classes, height, width),
        'patch_classification': torch.randn(batch_size, num_classes),
        'gland_classification': torch.randn(batch_size, num_classes)
    }

    # Ground truth targets
    targets = {
        'segmentation': torch.randint(0, num_classes, (batch_size, height, width)),
        'patch_labels': torch.randint(0, 2, (batch_size, num_classes)).float(),
        'gland_labels': torch.randint(0, num_classes, (batch_size,))
    }

    # Test both configurations with teacher-student phase (epoch 50)
    print("\n🤝 Testing teacher-student phase with DISABLED gland consistency (epoch 50)...")
    ts_loss_disabled = loss_fn_disabled(student_outputs, teacher_outputs, targets, current_epoch=50)
    print(f"✅ Gland consistency in output: {'gland_classification' in ts_loss_disabled.get('consistency_gland_classification', {})}")
    print(f"✅ Consistency loss keys: {[k for k in ts_loss_disabled.keys() if 'consistency' in k]}")

    print("\n🤝 Testing teacher-student phase with ENABLED gland consistency (epoch 50)...")
    ts_loss_enabled = loss_fn_enabled(student_outputs, teacher_outputs, targets, current_epoch=50)
    print(f"✅ Gland consistency in output: {'consistency_gland_classification' in ts_loss_enabled}")
    print(f"✅ Consistency loss keys: {[k for k in ts_loss_enabled.keys() if 'consistency' in k]}")

    # Test Dice consistency loss
    print("\n🎯 Testing Dice consistency loss (epoch 50)...")
    ts_loss_dice = loss_fn_dice(student_outputs, teacher_outputs, targets, current_epoch=50)
    consistency_loss_value = ts_loss_dice['consistency_loss'].item() if ts_loss_dice['consistency_loss'].numel() == 1 else ts_loss_dice['consistency_loss'].sum().item()
    print(f"✅ Dice consistency loss: {consistency_loss_value:.4f}")
    print(f"✅ Loss type: Dice (regional structure preservation)")

    # Test IoU consistency loss
    print("\n🎯 Testing IoU consistency loss (epoch 50)...")
    ts_loss_iou = loss_fn_iou(student_outputs, teacher_outputs, targets, current_epoch=50)
    consistency_loss_value = ts_loss_iou['consistency_loss'].item() if ts_loss_iou['consistency_loss'].numel() == 1 else ts_loss_iou['consistency_loss'].sum().item()
    print(f"✅ IoU consistency loss: {consistency_loss_value:.4f}")
    print(f"✅ Loss type: IoU (exact region overlap)")

    # Test alpha schedule (using disabled version)
    print("\n📊 Testing alpha schedule...")
    epochs = [0, 20, 40, 60, 80, 100]
    for epoch in epochs:
        alpha = loss_fn_disabled.alpha_scheduler.get_alpha(epoch)
        print(f"   Epoch {epoch:3d}: alpha = {alpha:.4f}")

    print("\n🎉 Teacher-Student Loss test completed successfully!")


if __name__ == "__main__":
    test_teacher_student_loss()