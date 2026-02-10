#!/usr/bin/env python3
"""
Post-Training Evaluator for 4-Class nnU-Net Multi-Task Model
==========================================================

Comprehensive post-training evaluation system that:
1. Evaluates best model on COMPLETE train/val/test datasets
2. Generates visualizations for randomly sampled images
3. Saves comprehensive metrics and publication-ready figures

Features:
- Complete dataset evaluation for robust statistics
- Random sampling for efficient visualization
- Standard models: 4-column layout (Original, GT Mask, Pred Mask, Overlay)
- Teacher-Student models: 5-column layout (Original, GT Mask, Teacher Pseudo-Mask, Student/Teacher Pred, Overlay)
- Publication-quality figures at 300 DPI
- Comprehensive metrics tables and reports
- Teacher-Student evaluation modes: student, teacher, both

Author: Claude Code - Generated for OSU CRC Research
Date: 2025-09-17
"""

import os
import sys
import json
import logging
import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from sklearn.metrics import classification_report, confusion_matrix
import cv2
from tqdm import tqdm
import pandas as pd

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from configs.paths_config import get_dataset_path, EVALUATION_CONFIG
from src.models.multi_task_wrapper import create_multitask_model
from src.models.metrics import SegmentationMetrics
from src.training.dataset import create_combined_data_loaders


class StreamingMetricsAggregator:
    """
    Memory-efficient streaming metrics aggregator that computes metrics per-batch
    and maintains running statistics without storing all predictions.
    """

    def __init__(self, num_classes: int = 4, class_names: List[str] = None):
        self.num_classes = num_classes
        self.class_names = class_names or [f"class_{i}" for i in range(num_classes)]
        self.patch_num_classes = None  # Will be set dynamically from first batch
        self.reset()

    def reset(self):
        """Reset all aggregated statistics"""
        # Segmentation metrics
        self.total_samples = 0
        self.correct_pixels = 0
        self.total_pixels = 0

        # Per-class confusion matrix elements
        self.class_tp = np.zeros(self.num_classes, dtype=np.int64)  # True positives
        self.class_fp = np.zeros(self.num_classes, dtype=np.int64)  # False positives
        self.class_fn = np.zeros(self.num_classes, dtype=np.int64)  # False negatives
        self.class_tn = np.zeros(self.num_classes, dtype=np.int64)  # True negatives

        # Patch classification metrics
        self.patch_correct = 0
        self.patch_total = 0
        # These will be initialized dynamically based on actual patch labels shape
        self.patch_class_correct = None
        self.patch_class_total = None

        # Consistency loss metrics (for Teacher-Student models)
        self.consistency_loss_sum = 0.0
        self.consistency_loss_count = 0

        # Individual consistency loss components
        self.seg_consistency_sum = 0.0
        self.patch_consistency_sum = 0.0
        self.gland_consistency_sum = 0.0
        self.consistency_components_count = 0

    def update_segmentation(self, predictions: np.ndarray, targets: np.ndarray):
        """Update segmentation metrics with a batch"""
        batch_size = predictions.shape[0]
        self.total_samples += batch_size

        # Pixel accuracy
        correct = (predictions == targets)
        self.correct_pixels += correct.sum()
        self.total_pixels += targets.size

        # Per-class metrics
        for class_idx in range(self.num_classes):
            pred_mask = (predictions == class_idx)
            true_mask = (targets == class_idx)

            # Confusion matrix elements
            tp = (pred_mask & true_mask).sum()
            fp = (pred_mask & ~true_mask).sum()
            fn = (~pred_mask & true_mask).sum()
            tn = (~pred_mask & ~true_mask).sum()

            self.class_tp[class_idx] += tp
            self.class_fp[class_idx] += fp
            self.class_fn[class_idx] += fn
            self.class_tn[class_idx] += tn

    def update_patch_classification(self, predictions: np.ndarray, targets: np.ndarray):
        """Update patch classification metrics with a batch"""
        # Initialize patch class arrays on first batch (dynamic sizing)
        if self.patch_num_classes is None:
            self.patch_num_classes = targets.shape[1]
            self.patch_class_correct = np.zeros(self.patch_num_classes, dtype=np.int64)
            self.patch_class_total = np.zeros(self.patch_num_classes, dtype=np.int64)

        # Convert predictions to binary (threshold 0.5)
        pred_binary = (predictions > 0.5).astype(int)

        # Exact match accuracy (all labels correct)
        exact_match = (pred_binary == targets).all(axis=1)
        self.patch_correct += exact_match.sum()
        self.patch_total += targets.shape[0]

        # Per-class accuracy - use actual number of patch classes
        for class_idx in range(targets.shape[1]):
            class_correct = (pred_binary[:, class_idx] == targets[:, class_idx])
            self.patch_class_correct[class_idx] += class_correct.sum()
            self.patch_class_total[class_idx] += targets.shape[0]

    def compute_segmentation_metrics(self) -> Dict[str, float]:
        """Compute final segmentation metrics from aggregated statistics"""
        metrics = {}

        # Pixel accuracy
        metrics['pixel_accuracy'] = self.correct_pixels / max(self.total_pixels, 1)

        # Per-class Dice and IoU
        dice_scores = []
        iou_scores = []

        for class_idx in range(self.num_classes):
            tp = self.class_tp[class_idx]
            fp = self.class_fp[class_idx]
            fn = self.class_fn[class_idx]

            # Dice coefficient - CORRECTED to match original implementation
            # Original: dice = (2.0 * intersection) / (pred_sum + true_sum)
            # where pred_sum = tp + fp, true_sum = tp + fn
            pred_sum = tp + fp
            true_sum = tp + fn
            dice_denom = pred_sum + true_sum
            if dice_denom > 0:
                dice = (2.0 * tp) / dice_denom
            else:
                # Handle case where both prediction and target are empty
                dice = 1.0 if (pred_sum == 0 and true_sum == 0) else 0.0
            dice_scores.append(dice)
            metrics[f'dice_class_{class_idx}'] = dice

            # IoU - IoU = TP / (TP + FP + FN) = intersection / union
            iou_denom = tp + fp + fn
            if iou_denom > 0:
                iou = tp / iou_denom
            else:
                # Handle case where both prediction and target are empty
                iou = 1.0 if (pred_sum == 0 and true_sum == 0) else 0.0
            iou_scores.append(iou)
            metrics[f'iou_class_{class_idx}'] = iou

        # Mean metrics
        metrics['dice_mean'] = np.mean(dice_scores)
        metrics['iou_mean'] = np.mean(iou_scores)

        return metrics

    def compute_patch_metrics(self) -> Dict[str, float]:
        """Compute final patch classification metrics"""
        metrics = {}

        # Overall accuracy
        metrics['patch_accuracy'] = self.patch_correct / max(self.patch_total, 1)

        # Per-class accuracy - use actual patch class count
        if self.patch_num_classes is not None and self.patch_class_correct is not None:
            class_accuracies = []
            for class_idx in range(self.patch_num_classes):
                if self.patch_class_total[class_idx] > 0:
                    acc = self.patch_class_correct[class_idx] / self.patch_class_total[class_idx]
                else:
                    acc = 0.0
                class_accuracies.append(acc)
                metrics[f'patch_class_{class_idx}_accuracy'] = acc

            metrics['patch_mean_class_accuracy'] = np.mean(class_accuracies)
        else:
            # No patch data processed yet
            metrics['patch_mean_class_accuracy'] = 0.0

        return metrics

    def update_consistency_loss(self, total_consistency_loss: float, seg_consistency: float = None,
                                patch_consistency: float = None, gland_consistency: float = None):
        """Update consistency loss tracking with individual components"""
        if total_consistency_loss is not None:
            self.consistency_loss_sum += total_consistency_loss
            self.consistency_loss_count += 1

        # Update individual components if provided
        if any(loss is not None for loss in [seg_consistency, patch_consistency, gland_consistency]):
            if seg_consistency is not None:
                self.seg_consistency_sum += seg_consistency
            if patch_consistency is not None:
                self.patch_consistency_sum += patch_consistency
            if gland_consistency is not None:
                self.gland_consistency_sum += gland_consistency
            self.consistency_components_count += 1

    def compute_consistency_metrics(self) -> Dict[str, float]:
        """Compute consistency loss metrics"""
        metrics = {}

        # Total consistency loss
        if self.consistency_loss_count > 0:
            metrics['consistency_loss_mean'] = self.consistency_loss_sum / self.consistency_loss_count
        else:
            metrics['consistency_loss_mean'] = None

        # Individual components
        if self.consistency_components_count > 0:
            metrics['seg_consistency_mean'] = self.seg_consistency_sum / self.consistency_components_count
            metrics['patch_consistency_mean'] = self.patch_consistency_sum / self.consistency_components_count
            metrics['gland_consistency_mean'] = self.gland_consistency_sum / self.consistency_components_count
        else:
            metrics['seg_consistency_mean'] = None
            metrics['patch_consistency_mean'] = None
            metrics['gland_consistency_mean'] = None

        return metrics

    def get_all_metrics(self) -> Dict[str, Any]:
        """Get all computed metrics"""
        seg_metrics = self.compute_segmentation_metrics()
        patch_metrics = self.compute_patch_metrics()
        consistency_metrics = self.compute_consistency_metrics()

        return {
            **seg_metrics,
            **patch_metrics,
            **consistency_metrics,
            'num_samples': self.total_samples
        }


class PostTrainingEvaluator:
    """
    Comprehensive post-training evaluator for 4-class multi-task gland segmentation

    Evaluates best model on complete datasets and generates publication-ready visualizations
    """

    def __init__(
        self,
        model_path: str,
        dataset_key: str = "mixed",
        output_dir: str = None,
        device: Optional[torch.device] = None,
        visualization_samples: int = 200,
        random_seed: int = 42,
        architecture: str = "nnunet",
        teacher_student_mode: Optional[str] = None
    ):
        """
        Initialize post-training evaluator

        Args:
            model_path: Path to best model checkpoint
            dataset_key: Dataset to evaluate on
            output_dir: Output directory (experiment directory)
            device: Device to run evaluation on
            visualization_samples: Number of samples per split for visualization
            random_seed: Random seed for reproducible sampling
            architecture: Model architecture ('baseline_unet', 'nnunet', or 'teacher_student_unet')
            teacher_student_mode: For teacher-student models: 'teacher' or 'student' (None for regular models)
        """
        self.model_path = Path(model_path)
        self.dataset_key = dataset_key
        self.output_dir = Path(output_dir)
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.visualization_samples = visualization_samples
        self.random_seed = random_seed
        self.architecture = architecture
        self.teacher_student_mode = teacher_student_mode

        # Track if both student and teacher models are loaded
        self.both_models_loaded = False

        # Get Teacher-Student configuration for consistency loss computation
        self.ts_config = {}
        if self.architecture == 'teacher_student_unet':
            import os
            self.ts_config = {
                'consistency_loss_type': os.getenv('GLAND_TS_CONSISTENCY_LOSS_TYPE', 'mse'),
                'consistency_temperature': float(os.getenv('GLAND_TS_CONSISTENCY_TEMPERATURE', '1.0')),
                'enable_gland_consistency': os.getenv('GLAND_TS_ENABLE_GLAND_CONSISTENCY', 'false').lower() == 'true',
                'pseudo_mask_filtering': os.getenv('GLAND_TS_PSEUDO_MASK_FILTERING', 'none'),
                'confidence_threshold': float(os.getenv('GLAND_TS_CONFIDENCE_THRESHOLD', '0.8')),
                'entropy_threshold': float(os.getenv('GLAND_TS_ENTROPY_THRESHOLD', '1.0'))
            }

        # Setup output directories with teacher/student suffixes for teacher-student models
        if self.teacher_student_mode:
            self.evaluations_dir = self.output_dir / f"evaluations_{self.teacher_student_mode}"
            self.visualizations_dir = self.output_dir / f"visualizations_{self.teacher_student_mode}"
        else:
            self.evaluations_dir = self.output_dir / "evaluations"
            self.visualizations_dir = self.output_dir / "visualizations"

        for dir_path in [self.evaluations_dir, self.visualizations_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

        # Setup logging
        self.setup_logging()

        # Class information for 4-class segmentation
        self.class_names = ['Background', 'Benign', 'Malignant', 'PDC']
        self.class_colors = np.array([
            [0, 0, 0],        # Background - Black
            [0, 255, 0],      # Benign - Green
            [255, 0, 0],      # Malignant - Red
            [0, 0, 255],      # PDC - Blue
        ])

        # Initialize segmentation metrics calculator
        self.metrics_calculator = SegmentationMetrics(num_classes=4, ignore_background=True)

        # Initialize components
        self.model = None
        self.data_loaders = None
        self.checkpoint = None

        # Set random seed for reproducibility
        random.seed(self.random_seed)
        np.random.seed(self.random_seed)
        torch.manual_seed(self.random_seed)

        self.logger.info(f"🔬 Post-training evaluator initialized:")
        self.logger.info(f"   📂 Model: {self.model_path}")
        self.logger.info(f"   🏗️  Architecture: {self.architecture}")
        if self.teacher_student_mode:
            self.logger.info(f"   👨‍🎓 Teacher-Student Mode: {self.teacher_student_mode}")
        self.logger.info(f"   📊 Dataset: {self.dataset_key}")
        self.logger.info(f"   📱 Device: {self.device}")
        self.logger.info(f"   📁 Output: {self.output_dir}")
        self.logger.info(f"   🎯 Visualization samples: {self.visualization_samples} per split")

    def setup_logging(self):
        """Setup logging for evaluation"""
        log_file = self.evaluations_dir / "post_training_evaluation.log"

        # Create a specific logger for this evaluator
        self.logger = logging.getLogger(f"PostTrainingEvaluator_{id(self)}")
        self.logger.setLevel(logging.INFO)

        # Disable propagation to prevent duplicate messages
        self.logger.propagate = False

        # Clear any existing handlers
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)

        # Create file handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)

        # Create console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)

        # Create formatter
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        # Add handlers
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)

    def _load_corresponding_teacher_model(self, student_filename: str):
        """
        Load the corresponding teacher model for evaluation and visualization

        Args:
            student_filename: Name of the student model file that was loaded
        """
        # Map student filename to corresponding teacher filename
        teacher_filename_map = {
            'latest_student_model.pth': 'latest_teacher_model.pth',
            'best_student_model.pth': 'best_teacher_model.pth'
        }

        teacher_filename = teacher_filename_map.get(student_filename)
        if not teacher_filename:
            self.logger.warning(f"⚠️ No corresponding teacher model for {student_filename}")
            return

        # Construct teacher model path
        teacher_path = Path(self.model_path).parent / teacher_filename

        if not teacher_path.exists():
            self.logger.warning(f"⚠️ Teacher model not found: {teacher_path}")
            self.logger.info("🔄 Initializing teacher from student weights for visualization")
            self.model.segmentation_model.initialize_teacher()
            self.both_models_loaded = False
            self.logger.warning("⚠️ Using student-initialized teacher - not true teacher weights")
            return

        try:
            self.logger.info(f"👨‍🏫 Loading corresponding teacher model for evaluation and visualization")
            self.logger.info(f"📁 Teacher checkpoint path: {teacher_path}")

            # Load teacher checkpoint
            try:
                teacher_checkpoint = torch.load(teacher_path, map_location='cpu', weights_only=True)
            except Exception:
                # Fallback to compatibility mode for checkpoints with metadata
                teacher_checkpoint = torch.load(teacher_path, map_location='cpu', weights_only=False)

            # Extract teacher weights
            if 'model_state_dict' in teacher_checkpoint:
                teacher_weights = teacher_checkpoint['model_state_dict']
            elif 'teacher_state_dict' in teacher_checkpoint:
                teacher_weights = teacher_checkpoint['teacher_state_dict']
                self.logger.warning("⚠️ Loading legacy teacher checkpoint format")
            else:
                self.logger.error("❌ No valid state dict found in teacher model file")
                return

            # Initialize teacher network first
            self.model.segmentation_model.initialize_teacher()

            # Load teacher weights
            teacher_keys = list(teacher_weights.keys())
            self.logger.info(f"🔍 Teacher checkpoint contains {len(teacher_keys)} keys")

            # Check format and load teacher weights
            has_teacher_segmentation_prefix = any(key.startswith('segmentation_model.teacher.') for key in teacher_keys)
            has_teacher_classification_heads = any(key.startswith('teacher_classification_head.') for key in teacher_keys)
            has_legacy_classification_heads = any(key.startswith('classification_head.') for key in teacher_keys)

            if has_teacher_segmentation_prefix and has_teacher_classification_heads:
                # New format: separate teacher classification heads
                self.logger.info("✅ Loading complete teacher checkpoint (segmentation + teacher classification)")
                missing_keys = self.model.load_state_dict(teacher_weights, strict=False)
            elif has_teacher_segmentation_prefix and has_legacy_classification_heads:
                # Legacy format: unified classification heads (needs conversion for new architecture)
                self.logger.info("🔄 Loading legacy teacher checkpoint (unified classification heads)")
                # Convert legacy classification_head keys to teacher_classification_head
                converted_weights = {}
                for key, value in teacher_weights.items():
                    if key.startswith('classification_head.'):
                        new_key = key.replace('classification_head.', 'teacher_classification_head.')
                        converted_weights[new_key] = value
                    else:
                        converted_weights[key] = value
                missing_keys = self.model.load_state_dict(converted_weights, strict=False)
            elif has_teacher_segmentation_prefix and not (has_teacher_classification_heads or has_legacy_classification_heads):
                # Partial format: has segmentation prefix but missing classification heads
                self.logger.info("⚠️ Loading partial teacher checkpoint (segmentation only, missing classification heads)")
                missing_keys = self.model.load_state_dict(teacher_weights, strict=False)
            else:
                # Legacy format: raw BaselineUNet weights only, need to map them
                self.logger.info("🔄 Loading legacy teacher checkpoint format (raw BaselineUNet weights)")
                full_state_dict = {}
                for key, value in teacher_weights.items():
                    full_key = f'segmentation_model.teacher.{key}'
                    full_state_dict[full_key] = value
                missing_keys = self.model.load_state_dict(full_state_dict, strict=False)

            # Report results
            if missing_keys.missing_keys:
                # Categorize missing keys for teacher checkpoint loading
                student_missing = [k for k in missing_keys.missing_keys if 'student' in k]
                student_classification_missing = [k for k in missing_keys.missing_keys if 'student_classification_head' in k]
                teacher_classification_missing = [k for k in missing_keys.missing_keys if 'teacher_classification_head' in k]
                other_missing = [k for k in missing_keys.missing_keys if not any(pattern in k for pattern in ['student', 'teacher_classification_head'])]

                if student_missing:
                    self.logger.debug(f"✅ Expected: Student weights not in teacher checkpoint: {len(student_missing)} keys")
                if student_classification_missing:
                    self.logger.debug(f"✅ Expected: Student classification head not in teacher checkpoint: {len(student_classification_missing)} keys")
                if teacher_classification_missing:
                    self.logger.debug(f"✅ Expected: Teacher classification heads missing from legacy checkpoint: {len(teacher_classification_missing)} keys")
                if other_missing:
                    self.logger.warning(f"⚠️ Unexpected missing keys: {len(other_missing)} keys")
                    for key in other_missing[:3]:
                        self.logger.warning(f"     * {key}")
            else:
                self.logger.info("✅ All teacher weights loaded successfully")

            self.logger.info("👨‍🏫 Teacher model loaded successfully for evaluation and visualization")

            # Mark that both student and teacher models are available
            self.both_models_loaded = True
            self.logger.info("✅ Both student and teacher models loaded - ready for comparison and visualization")

        except Exception as e:
            self.logger.error(f"❌ Error loading teacher model: {str(e)}")
            self.logger.info("🔄 Initializing teacher from student weights as fallback")
            self.model.segmentation_model.initialize_teacher()
            self.both_models_loaded = False
            self.logger.warning("⚠️ Using student-initialized teacher - not true teacher weights")

    def get_both_model_predictions(self, input_batch):
        """
        Get predictions from both student and teacher models for comparison

        Args:
            input_batch: Input tensor batch

        Returns:
            dict: Contains 'student' and 'teacher' predictions, or None if models not available
        """
        if self.architecture != 'teacher_student_unet':
            return None

        if not hasattr(self.model, 'segmentation_model') or not self.model.segmentation_model.teacher_initialized:
            return None

        predictions = {}

        with torch.no_grad():
            # Get student prediction
            student_output = self.model(input_batch, mode='student_only')
            predictions['student'] = student_output

            # Get teacher prediction
            teacher_output = self.model(input_batch, mode='teacher_only')
            predictions['teacher'] = teacher_output

            # Add flag indicating if these are true teacher weights or student-initialized
            predictions['true_teacher_weights'] = self.both_models_loaded

        return predictions

    def get_model_info(self):
        """
        Get information about loaded models

        Returns:
            dict: Information about available models
        """
        info = {
            'architecture': self.architecture,
            'evaluation_mode': self.teacher_student_mode,
            'both_models_loaded': self.both_models_loaded,
            'teacher_initialized': False,
            'can_compare_models': False
        }

        if self.architecture == 'teacher_student_unet' and hasattr(self.model, 'segmentation_model'):
            info['teacher_initialized'] = self.model.segmentation_model.teacher_initialized
            info['can_compare_models'] = self.both_models_loaded and self.model.segmentation_model.teacher_initialized

        return info

    def _compute_consistency_loss_component(self, student_logits: torch.Tensor, teacher_logits: torch.Tensor,
                                           loss_type: str, temperature: float = 1.0, is_segmentation: bool = False) -> float:
        """
        Compute consistency loss between student and teacher logits using specified loss type

        Args:
            student_logits: Student network logits
            teacher_logits: Teacher network logits
            loss_type: Type of loss ('mse', 'kl_div', 'l1', 'dice', 'iou')
            temperature: Temperature for softmax (knowledge distillation)
            is_segmentation: Whether this is segmentation data (4D) vs classification (2D)

        Returns:
            Consistency loss value
        """
        try:
            if loss_type == "mse":
                # MSE between soft predictions
                student_soft = torch.softmax(student_logits / temperature, dim=1)
                teacher_soft = torch.softmax(teacher_logits / temperature, dim=1)
                loss = torch.nn.functional.mse_loss(student_soft, teacher_soft)

            elif loss_type == "kl_div":
                # KL divergence for knowledge distillation
                student_log_probs = torch.log_softmax(student_logits / temperature, dim=1)
                teacher_probs = torch.softmax(teacher_logits / temperature, dim=1)
                loss = torch.nn.functional.kl_div(student_log_probs, teacher_probs, reduction='batchmean')

            elif loss_type == "l1":
                # L1 between soft predictions
                student_soft = torch.softmax(student_logits / temperature, dim=1)
                teacher_soft = torch.softmax(teacher_logits / temperature, dim=1)
                loss = torch.nn.functional.l1_loss(student_soft, teacher_soft)

            elif loss_type == "dice" and is_segmentation:
                # Dice loss for segmentation consistency
                student_soft = torch.softmax(student_logits / temperature, dim=1)
                teacher_soft = torch.softmax(teacher_logits / temperature, dim=1)

                # Compute Dice for each class and average
                dice_scores = []
                for c in range(student_soft.shape[1]):
                    student_class = student_soft[:, c]
                    teacher_class = teacher_soft[:, c]

                    intersection = (student_class * teacher_class).sum()
                    union = student_class.sum() + teacher_class.sum()

                    if union > 0:
                        dice = 2.0 * intersection / union
                        dice_scores.append(dice)

                if dice_scores:
                    loss = 1.0 - torch.stack(dice_scores).mean()  # 1 - Dice for loss
                else:
                    loss = torch.tensor(0.0, device=student_logits.device)

            elif loss_type == "iou" and is_segmentation:
                # IoU loss for segmentation consistency
                student_soft = torch.softmax(student_logits / temperature, dim=1)
                teacher_soft = torch.softmax(teacher_logits / temperature, dim=1)

                # Compute IoU for each class and average
                iou_scores = []
                for c in range(student_soft.shape[1]):
                    student_class = student_soft[:, c]
                    teacher_class = teacher_soft[:, c]

                    intersection = (student_class * teacher_class).sum()
                    union = student_class.sum() + teacher_class.sum() - intersection

                    if union > 0:
                        iou = intersection / union
                        iou_scores.append(iou)

                if iou_scores:
                    loss = 1.0 - torch.stack(iou_scores).mean()  # 1 - IoU for loss
                else:
                    loss = torch.tensor(0.0, device=student_logits.device)

            else:
                # Fallback to MSE
                student_soft = torch.softmax(student_logits / temperature, dim=1)
                teacher_soft = torch.softmax(teacher_logits / temperature, dim=1)
                loss = torch.nn.functional.mse_loss(student_soft, teacher_soft)

            return loss.item()

        except Exception as e:
            self.logger.warning(f"⚠️ Error computing {loss_type} consistency loss: {e}")
            return None

    def load_model(self):
        """Load trained model from checkpoint"""
        self.logger.info("📂 Loading best model checkpoint...")

        if not self.model_path.exists():
            raise FileNotFoundError(f"Model checkpoint not found: {self.model_path}")

        # Load checkpoint with PyTorch 2.6 compatibility
        try:
            # Try with weights_only=True first (more secure)
            self.checkpoint = torch.load(self.model_path, map_location='cpu', weights_only=True)
        except Exception:
            # Fallback to weights_only=False for compatibility with numpy objects
            self.checkpoint = torch.load(self.model_path, map_location='cpu', weights_only=False)

        # Create and load model
        self.logger.info(f"🏗️  Creating {self.architecture} model for evaluation...")

        if self.architecture == 'teacher_student_unet':
            # For Teacher-Student models, load the full MultiTaskWrapper model
            # This ensures we get dictionary outputs with 'segmentation', 'patch_classification', etc.
            self.model = create_multitask_model(architecture=self.architecture)

            # Check if we're loading individual student/teacher model or combined model
            model_filename = Path(self.model_path).name

            if model_filename in ['latest_student_model.pth', 'best_student_model.pth']:
                # Loading individual student model file
                self.logger.info("🎓 Loading individual student model file")
                self.logger.info(f"📁 Student checkpoint path: {self.model_path}")

                # Individual student files have 'model_state_dict' key (contains segmentation + classification)
                if 'model_state_dict' in self.checkpoint:
                    student_weights = self.checkpoint['model_state_dict']
                elif 'student_state_dict' in self.checkpoint:
                    # Legacy format - only segmentation weights
                    student_weights = self.checkpoint['student_state_dict']
                    self.logger.warning("⚠️ Loading legacy student checkpoint format (segmentation only)")
                else:
                    raise KeyError("No valid state dict found in student model file")

                # For student-only evaluation, we only need to load student weights
                # The teacher will remain uninitialized unless specifically needed
                self.logger.info(f"🎓 Loading student weights for {self.teacher_student_mode} evaluation")

                # Debug: log the checkpoint structure
                student_keys = list(student_weights.keys())
                self.logger.info(f"🔍 Student checkpoint contains {len(student_keys)} keys")
                self.logger.debug(f"First 5 keys: {student_keys[:5]}")

                # Check if we have the new format (with classification heads) or legacy format
                has_segmentation_prefix = any(key.startswith('segmentation_model.student.') for key in student_keys)
                has_student_classification_heads = any(key.startswith('student_classification_head.') for key in student_keys)
                has_legacy_classification_heads = any(key.startswith('classification_head.') for key in student_keys)

                if has_segmentation_prefix and has_student_classification_heads:
                    # New format: separate student classification heads
                    self.logger.info("✅ Loading complete student checkpoint (segmentation + student classification)")
                    missing_keys = self.model.load_state_dict(student_weights, strict=False)
                elif has_segmentation_prefix and has_legacy_classification_heads:
                    # Legacy format: unified classification heads (needs conversion for new architecture)
                    self.logger.info("🔄 Loading legacy student checkpoint (unified classification heads)")
                    # Convert legacy classification_head keys to student_classification_head
                    converted_weights = {}
                    for key, value in student_weights.items():
                        if key.startswith('classification_head.'):
                            new_key = key.replace('classification_head.', 'student_classification_head.')
                            converted_weights[new_key] = value
                        else:
                            converted_weights[key] = value
                    missing_keys = self.model.load_state_dict(converted_weights, strict=False)
                elif has_segmentation_prefix and not (has_student_classification_heads or has_legacy_classification_heads):
                    # Partial format: has segmentation prefix but missing classification heads
                    self.logger.info("⚠️ Loading partial student checkpoint (segmentation only, missing classification heads)")
                    missing_keys = self.model.load_state_dict(student_weights, strict=False)
                else:
                    # Legacy format: raw BaselineUNet weights only, need to map them
                    self.logger.info("🔄 Loading legacy student checkpoint format (raw BaselineUNet weights)")
                    full_state_dict = {}
                    for key, value in student_weights.items():
                        # Map raw BaselineUNet weights to student network in MultiTaskWrapper
                        full_key = f'segmentation_model.student.{key}'
                        full_state_dict[full_key] = value
                    missing_keys = self.model.load_state_dict(full_state_dict, strict=False)

                # Analyze missing keys to understand what's missing
                if missing_keys.missing_keys:
                    # Categorize missing keys
                    missing_teacher = [k for k in missing_keys.missing_keys if 'teacher' in k]
                    missing_teacher_classification = [k for k in missing_keys.missing_keys if 'teacher_classification_head' in k]
                    missing_student_classification_heads = [k for k in missing_keys.missing_keys if 'student_classification_head' in k]
                    missing_other = [k for k in missing_keys.missing_keys if not any(pattern in k for pattern in ['teacher', 'student_classification_head'])]

                    if missing_teacher:
                        self.logger.debug(f"✅ Expected: Teacher segmentation weights not in student checkpoint ({len(missing_teacher)} keys)")
                    if missing_teacher_classification:
                        self.logger.debug(f"✅ Expected: Teacher classification head not in student checkpoint ({len(missing_teacher_classification)} keys)")
                    if missing_student_classification_heads:
                        self.logger.debug(f"✅ Expected: Student classification heads missing from legacy checkpoint ({len(missing_student_classification_heads)} keys)")
                    if missing_other:
                        self.logger.warning(f"⚠️ Unexpected missing keys: {len(missing_other)} keys")
                        for key in missing_other[:3]:
                            self.logger.warning(f"     * {key}")
                else:
                    self.logger.info("✅ All student weights loaded successfully")

                if missing_keys.unexpected_keys:
                    self.logger.info(f"ℹ️ Unexpected keys in student checkpoint: {len(missing_keys.unexpected_keys)} keys")

                # Only initialize teacher if we need teacher evaluation mode
                if self.teacher_student_mode == 'teacher':
                    self.logger.warning("⚠️ Teacher mode requested but only student weights available. Initializing teacher from student weights.")
                    self.model.segmentation_model.initialize_teacher()

                self.logger.info("🎓 Student model loaded successfully")

                # Automatically load corresponding teacher model for evaluation and visualization
                self._load_corresponding_teacher_model(model_filename)

                # Ensure teacher is properly initialized after loading both models
                if hasattr(self.model, 'segmentation_model'):
                    self.model.segmentation_model.teacher_initialized = True

            elif model_filename in ['latest_teacher_model.pth', 'best_teacher_model.pth']:
                # Loading individual teacher model file
                self.logger.info("👨‍🏫 Loading individual teacher model file")
                self.logger.info(f"📁 Teacher checkpoint path: {self.model_path}")

                # Individual teacher files have 'model_state_dict' key (contains segmentation + classification)
                if 'model_state_dict' in self.checkpoint:
                    teacher_weights = self.checkpoint['model_state_dict']
                elif 'teacher_state_dict' in self.checkpoint:
                    # Legacy format - only segmentation weights
                    teacher_weights = self.checkpoint['teacher_state_dict']
                    self.logger.warning("⚠️ Loading legacy teacher checkpoint format (segmentation only)")
                else:
                    raise KeyError("No valid state dict found in teacher model file")

                self.logger.info(f"👨‍🏫 Loading teacher weights for {self.teacher_student_mode} evaluation")

                # Initialize teacher network first (required for teacher weight loading)
                self.model.segmentation_model.initialize_teacher()

                # Check if we have the new format (with classification heads) or legacy format
                teacher_keys = list(teacher_weights.keys())
                has_teacher_segmentation_prefix = any(key.startswith('segmentation_model.teacher.') for key in teacher_keys)
                has_teacher_classification_heads = any(key.startswith('teacher_classification_head.') for key in teacher_keys)
                has_legacy_classification_heads = any(key.startswith('classification_head.') for key in teacher_keys)

                if has_teacher_segmentation_prefix and has_teacher_classification_heads:
                    # New format: separate teacher classification heads
                    self.logger.info("✅ Loading complete teacher checkpoint (segmentation + teacher classification)")
                    missing_keys = self.model.load_state_dict(teacher_weights, strict=False)
                elif has_teacher_segmentation_prefix and has_legacy_classification_heads:
                    # Legacy format: unified classification heads (needs conversion for new architecture)
                    self.logger.info("🔄 Loading legacy teacher checkpoint (unified classification heads)")
                    # Convert legacy classification_head keys to teacher_classification_head
                    converted_weights = {}
                    for key, value in teacher_weights.items():
                        if key.startswith('classification_head.'):
                            new_key = key.replace('classification_head.', 'teacher_classification_head.')
                            converted_weights[new_key] = value
                        else:
                            converted_weights[key] = value
                    missing_keys = self.model.load_state_dict(converted_weights, strict=False)
                elif has_teacher_segmentation_prefix and not (has_teacher_classification_heads or has_legacy_classification_heads):
                    # Partial format: has segmentation prefix but missing classification heads
                    self.logger.info("⚠️ Loading partial teacher checkpoint (segmentation only, missing classification heads)")
                    missing_keys = self.model.load_state_dict(teacher_weights, strict=False)
                else:
                    # Legacy format: raw BaselineUNet weights only, need to map them
                    self.logger.info("🔄 Loading legacy teacher checkpoint format (raw BaselineUNet weights)")
                    full_state_dict = {}
                    for key, value in teacher_weights.items():
                        # Map raw BaselineUNet weights to teacher network in MultiTaskWrapper
                        full_key = f'segmentation_model.teacher.{key}'
                        full_state_dict[full_key] = value
                    missing_keys = self.model.load_state_dict(full_state_dict, strict=False)

                # For teacher evaluation, we need teacher weights loaded into teacher network
                # For student evaluation with teacher weights, we warn but proceed
                if self.teacher_student_mode == 'student':
                    self.logger.warning("⚠️ Student mode requested but only teacher weights available. This may not reflect actual student performance.")

                # Report any missing keys (should be minimal with new format)
                if missing_keys.missing_keys:
                    # Categorize missing keys for teacher checkpoint
                    missing_student = [k for k in missing_keys.missing_keys if 'student' in k]
                    missing_student_classification = [k for k in missing_keys.missing_keys if 'student_classification_head' in k]
                    missing_teacher_classification_heads = [k for k in missing_keys.missing_keys if 'teacher_classification_head' in k]
                    missing_other = [k for k in missing_keys.missing_keys if not any(pattern in k for pattern in ['student', 'teacher_classification_head'])]

                    if missing_student:
                        self.logger.debug(f"✅ Expected: Student segmentation weights not in teacher checkpoint ({len(missing_student)} keys)")
                    if missing_student_classification:
                        self.logger.debug(f"✅ Expected: Student classification head not in teacher checkpoint ({len(missing_student_classification)} keys)")
                    if missing_teacher_classification_heads:
                        self.logger.debug(f"✅ Expected: Teacher classification heads missing from legacy checkpoint ({len(missing_teacher_classification_heads)} keys)")
                    if missing_other:
                        self.logger.warning(f"⚠️ Unexpected missing keys: {len(missing_other)} keys")
                        for key in missing_other[:3]:
                            self.logger.warning(f"     * {key}")

                # Ensure teacher is marked as initialized
                self.model.segmentation_model.teacher_initialized = True
                self.logger.info("👨‍🏫 Teacher model loaded successfully")

            else:
                # Loading combined model file (legacy behavior)
                self.logger.info("🔄 Loading combined Teacher-Student model file")
                state_dict = self.checkpoint['model_state_dict']

                # Filter out Teacher-Student specific keys that don't belong in MultiTaskWrapper
                filtered_state_dict = {}
                teacher_initialized = False
                for key, value in state_dict.items():
                    if key == 'segmentation_model.teacher_initialized':
                        # Store the teacher initialization status
                        teacher_initialized = value
                        continue
                    else:
                        filtered_state_dict[key] = value

                self.model.load_state_dict(filtered_state_dict, strict=False)

                # Apply teacher initialization status
                if hasattr(self.model, 'segmentation_model') and hasattr(self.model.segmentation_model, 'teacher_initialized'):
                    self.model.segmentation_model.teacher_initialized = teacher_initialized
                    self.logger.info(f"📋 Teacher initialization status: {teacher_initialized}")

            # Ensure teacher is marked as initialized for evaluation
            if hasattr(self.model, 'segmentation_model'):
                self.model.segmentation_model.teacher_initialized = True
                self.logger.info("✅ Teacher marked as initialized for evaluation")

            # Log which network will be used for evaluation (controlled by teacher_student_mode)
            if self.teacher_student_mode == 'teacher':
                self.logger.info("👨‍🏫 Using Teacher network for evaluation")
            elif self.teacher_student_mode == 'student':
                self.logger.info("🎓 Using Student network for evaluation")
            else:
                raise ValueError(f"teacher_student_mode must be 'teacher' or 'student' for teacher_student_unet architecture")
        else:
            # Regular models (baseline_unet, nnunet)
            self.model = create_multitask_model(architecture=self.architecture)
            self.model.load_state_dict(self.checkpoint['model_state_dict'])

        self.model.to(self.device)
        self.model.eval()

        self.logger.info("✅ Model loaded successfully")

        # Log training information
        if 'best_metrics' in self.checkpoint:
            metrics = self.checkpoint['best_metrics']
            self.logger.info(f"🏆 Best training metrics:")
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    self.logger.info(f"   {key}: {value:.4f}")

    def _extract_batch_data(self, batch_data: Dict) -> tuple:
        """
        Safely extract image, segmentation, and patch labels from batch data
        Handles different key formats from different data loaders
        """
        # Try both key formats
        if 'images' in batch_data:
            # Combined data loader format
            images = batch_data['images']
            seg_masks = batch_data['segmentation_targets']
            patch_labels = batch_data['patch_labels']
        elif 'image' in batch_data:
            # Original dataset format
            images = batch_data['image']
            seg_masks = batch_data['segmentation_target']
            patch_labels = batch_data['patch_label']
        else:
            self.logger.error(f"❌ Unknown batch data format. Expected 'images' or 'image' key")
            raise KeyError(f"Expected 'images'/'image' key in batch data")

        return images.to(self.device), seg_masks.to(self.device), patch_labels.to(self.device)

    def prepare_data_loaders(self):
        """Prepare data loaders for all splits"""
        self.logger.info("📊 Preparing data loaders for all splits...")

        # Create data loaders for all splits using dataset_key
        # Use larger batch size for efficient evaluation (no backpropagation needed)
        # Disable augmentation for consistent visualization (no rotation/flipping)
        train_loader, val_loader, test_loader = create_combined_data_loaders(
            dataset_key=self.dataset_key,
            batch_size=32,  # Larger batch size for evaluation efficiency
            num_workers=4,   # More workers for faster data loading
            use_multilabel_patch=True,
            disable_augmentation=True  # Disable rotation/augmentation for clean visualization
        )

        self.data_loaders = {
            'train': train_loader,
            'val': val_loader,
            'test': test_loader
        }

        self.logger.info(f"✅ Data loaders prepared")

    def calculate_segmentation_metrics(self, predictions: np.ndarray, targets: np.ndarray) -> Dict[str, float]:
        """Calculate comprehensive segmentation metrics using the SegmentationMetrics calculator"""

        # Convert numpy arrays to torch tensors for the metrics calculator
        pred_tensor = torch.from_numpy(predictions).long()
        target_tensor = torch.from_numpy(targets).long()

        # Use our comprehensive metrics calculator
        comprehensive_metrics = self.metrics_calculator.compute_all_metrics(pred_tensor, target_tensor)

        # Also compute legacy metrics for backward compatibility
        legacy_metrics = {}

        # Overall pixel accuracy (same as comprehensive but with legacy name)
        pixel_accuracy = (predictions == targets).mean()
        legacy_metrics['pixel_accuracy'] = pixel_accuracy

        # Per-class Dice and IoU scores for legacy compatibility
        dice_scores = []
        iou_scores = []

        for class_idx in range(len(self.class_names)):
            pred_mask = (predictions == class_idx)
            true_mask = (targets == class_idx)

            # Dice coefficient
            intersection = (pred_mask & true_mask).sum()
            union = pred_mask.sum() + true_mask.sum()

            if union > 0:
                dice = (2.0 * intersection) / union
            else:
                dice = 1.0 if (pred_mask.sum() == 0 and true_mask.sum() == 0) else 0.0
            dice_scores.append(dice)

            # IoU
            union_iou = (pred_mask | true_mask).sum()
            if union_iou > 0:
                iou = intersection / union_iou
            else:
                iou = 1.0 if (pred_mask.sum() == 0 and true_mask.sum() == 0) else 0.0
            iou_scores.append(iou)

            # Store per-class metrics
            legacy_metrics[f'dice_class_{class_idx}'] = dice_scores[-1]
            legacy_metrics[f'iou_class_{class_idx}'] = iou_scores[-1]

        # Mean metrics
        legacy_metrics['dice_mean'] = np.mean(dice_scores)
        legacy_metrics['iou_mean'] = np.mean(iou_scores)

        # Combine comprehensive and legacy metrics
        all_metrics = {**comprehensive_metrics, **legacy_metrics}

        return all_metrics

    def calculate_patch_classification_metrics(self, predictions: np.ndarray, targets: np.ndarray) -> Dict[str, float]:
        """Calculate multi-label patch classification metrics"""
        metrics = {}

        # Convert predictions to binary (threshold 0.5)
        pred_binary = (predictions > 0.5).astype(int)

        # Exact match accuracy (all labels correct)
        exact_match = (pred_binary == targets).all(axis=1).mean()
        metrics['patch_accuracy'] = exact_match

        # Per-class accuracy
        class_accuracies = []
        for class_idx in range(targets.shape[1]):
            class_acc = (pred_binary[:, class_idx] == targets[:, class_idx]).mean()
            class_accuracies.append(class_acc)
            metrics[f'patch_class_{class_idx}_accuracy'] = class_acc

        metrics['patch_mean_class_accuracy'] = np.mean(class_accuracies)

        return metrics

    def evaluate_complete_split_streaming(self, split: str, max_viz_samples: int = 100) -> Tuple[Dict[str, Any], List[Tuple[np.ndarray, Dict]]]:
        """
        Memory-efficient streaming evaluation that processes batches immediately
        and only stores a limited number of samples for visualization.

        Args:
            split: 'train', 'val', or 'test'
            max_viz_samples: Maximum samples to collect for visualization

        Returns:
            Tuple of (metrics_dict, visualization_data)
        """
        self.logger.info(f"📊 Streaming evaluation of complete {split} dataset...")

        # Log which model is being evaluated for user verification
        if self.architecture == 'teacher_student_unet':
            if self.teacher_student_mode == 'student':
                self.logger.info(f"🎓 EVALUATING: Student network (teacher will provide pseudo-masks for visualization)")
            elif self.teacher_student_mode == 'teacher':
                self.logger.info(f"👨‍🏫 EVALUATING: Teacher network (student network will not be used)")
            else:
                self.logger.warning(f"⚠️ Unknown teacher_student_mode: {self.teacher_student_mode}")
        else:
            self.logger.info(f"🏗️ EVALUATING: {self.architecture} model")

        loader = self.data_loaders[split]

        # Initialize streaming metrics aggregator
        metrics_aggregator = StreamingMetricsAggregator(
            num_classes=4,
            class_names=self.class_names
        )

        # Storage for limited visualization samples only
        visualization_data = []
        samples_collected = 0

        # Evaluation loop - process batches immediately
        with torch.no_grad():
            for batch_idx, batch_data in enumerate(tqdm(loader, desc=f"Streaming eval {split}")):
                # Move data to device
                images, seg_masks, patch_labels = self._extract_batch_data(batch_data)

                # if batch_idx > 50: # Hikmat
                #     break

                # Forward pass with correct mode for Teacher-Student models
                consistency_loss = None
                outputs = None  # Initialize to avoid UnboundLocalError
                if self.architecture == 'teacher_student_unet':
                    # For Teacher-Student models, get both networks for consistency loss calculation
                    underlying_ts_model = self.model.segmentation_model

                    # Ensure teacher is initialized
                    if not underlying_ts_model.teacher_initialized:
                        self.logger.warning("⚠️ Teacher not initialized, initializing now for evaluation...")
                        underlying_ts_model.initialize_teacher()

                    ts_outputs = underlying_ts_model(images, mode="teacher_student")

                    # Store teacher predictions for visualization
                    teacher_seg_logits = ts_outputs['teacher']
                    self._current_teacher_preds = torch.argmax(teacher_seg_logits, dim=1).detach().cpu().numpy()
                    # Store teacher logits for probability calculation in visualization
                    self._current_teacher_logits = teacher_seg_logits.detach().cpu().numpy()

                    # Store teacher classification predictions for visualization
                    teacher_patch_outputs = self.model(images, mode="teacher_only")
                    self._current_teacher_patch_preds = torch.argmax(teacher_patch_outputs['patch_classification'], dim=1).detach().cpu().numpy()

                    # Extract outputs for the specified evaluation mode
                    if self.teacher_student_mode == 'student':
                        seg_logits = ts_outputs['student']
                        if batch_idx == 0:  # Log once per split
                            self.logger.info(f"🎓 Using STUDENT network predictions for evaluation metrics")
                            self.logger.info(f"👨‍🏫 Using TEACHER network predictions for pseudo-mask visualization (3rd column)")
                    else:
                        seg_logits = ts_outputs['teacher']
                        if batch_idx == 0:  # Log once per split
                            self.logger.info(f"👨‍🏫 Using TEACHER network predictions for evaluation metrics")
                            self.logger.info(f"🎓 Student network predictions not used for evaluation")

                    # Get other task outputs through MultiTaskWrapper
                    mode = f"{self.teacher_student_mode}_only"
                    wrapper_outputs = self.model(images, mode=mode)
                    patch_logits = wrapper_outputs['patch_classification']

                    # Set outputs for cleanup compatibility
                    outputs = wrapper_outputs

                    # Calculate detailed consistency loss components using user's configuration
                    try:
                        student_seg = ts_outputs['student']
                        teacher_seg = ts_outputs['teacher']

                        loss_type = self.ts_config.get('consistency_loss_type', 'mse')
                        temperature = self.ts_config.get('consistency_temperature', 1.0)

                        # Compute segmentation consistency loss
                        seg_consistency = self._compute_consistency_loss_component(
                            student_seg, teacher_seg, loss_type, temperature, is_segmentation=True
                        )

                        # Compute patch classification consistency loss
                        student_patch = wrapper_outputs['patch_classification']
                        teacher_patch_outputs = self.model(images, mode="teacher_only")
                        teacher_patch = teacher_patch_outputs['patch_classification']

                        patch_consistency = self._compute_consistency_loss_component(
                            student_patch, teacher_patch, loss_type, temperature, is_segmentation=False
                        )

                        # Compute gland classification consistency loss (if enabled)
                        gland_consistency = None
                        if self.ts_config.get('enable_gland_consistency', False):
                            try:
                                student_gland = wrapper_outputs.get('gland_classification')
                                teacher_gland = teacher_patch_outputs.get('gland_classification')
                                if student_gland is not None and teacher_gland is not None:
                                    gland_consistency = self._compute_consistency_loss_component(
                                        student_gland, teacher_gland, loss_type, temperature, is_segmentation=False
                                    )
                            except Exception:
                                gland_consistency = None

                        # Compute total consistency loss (weighted sum)
                        total_consistency = 0.0
                        if seg_consistency is not None:
                            total_consistency += 1.0 * seg_consistency  # Segmentation weight
                        if patch_consistency is not None:
                            total_consistency += 0.5 * patch_consistency  # Patch weight
                        if gland_consistency is not None:
                            total_consistency += 0.5 * gland_consistency  # Gland weight

                        consistency_loss = total_consistency

                    except Exception as e:
                        self.logger.warning(f"⚠️ Error computing consistency loss: {e}")
                        consistency_loss = None
                        seg_consistency = None
                        patch_consistency = None
                        gland_consistency = None

                else:
                    # Standard models don't have teacher predictions
                    self._current_teacher_preds = None
                    outputs = self.model(images)
                    seg_logits = outputs['segmentation']
                    patch_logits = outputs['patch_classification']

                # Convert to predictions and detach immediately
                seg_pred = torch.argmax(seg_logits, dim=1).detach().cpu().numpy()
                patch_pred = torch.sigmoid(patch_logits).detach().cpu().numpy()
                seg_targets_np = seg_masks.detach().cpu().numpy()
                patch_targets_np = patch_labels.detach().cpu().numpy()

                # Update streaming metrics immediately (no storage)
                metrics_aggregator.update_segmentation(seg_pred, seg_targets_np)
                metrics_aggregator.update_patch_classification(patch_pred, patch_targets_np)

                # Update consistency loss if available (Teacher-Student models)
                if consistency_loss is not None:
                    # Pass individual components if available
                    if self.architecture == 'teacher_student_unet':
                        metrics_aggregator.update_consistency_loss(
                            consistency_loss, seg_consistency, patch_consistency, gland_consistency
                        )
                    else:
                        metrics_aggregator.update_consistency_loss(consistency_loss)

                # DEBUG: Check first few batches for data patterns
                if batch_idx < 3:
                    self.logger.info(f"🔍 DEBUG Batch {batch_idx}:")
                    self.logger.info(f"   Prediction classes: {np.unique(seg_pred)} (counts: {np.bincount(seg_pred.flatten())})")
                    self.logger.info(f"   Target classes: {np.unique(seg_targets_np)} (counts: {np.bincount(seg_targets_np.flatten())})")

                    # Check if model is just predicting background
                    if len(np.unique(seg_pred)) == 1:
                        self.logger.warning(f"   ⚠️ Model only predicting class {seg_pred[0,0,0]} - poor training!")

                    # Quick dice for this batch only
                    batch_metrics = self.calculate_segmentation_metrics(seg_pred, seg_targets_np)
                    self.logger.info(f"   Batch {batch_idx} Dice: {batch_metrics['dice_mean']:.6f}")

                # Collect limited samples for visualization only
                if samples_collected < max_viz_samples:
                    batch_size_actual = images.size(0)

                    for i in range(min(batch_size_actual, max_viz_samples - samples_collected)):
                        # CRITICAL: Always use batch image to ensure perfect correspondence with ground truth mask
                        # Both image and seg_targets_np[i] come from the same batch index i
                        # This guarantees they correspond to the exact same sample
                        image = images[i].detach().cpu().numpy().transpose(1, 2, 0)

                        # Proper denormalization for visualization - show original unaugmented image
                        # Dataset uses ImageNet normalization: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]

                        # Apply ImageNet denormalization to get original image
                        imagenet_mean = np.array([0.485, 0.456, 0.406])
                        imagenet_std = np.array([0.229, 0.224, 0.225])
                        original_image = image * imagenet_std + imagenet_mean

                        # Ensure [0, 1] range and convert to uint8
                        original_image = np.clip(original_image, 0, 1)
                        original_image = (original_image * 255).astype(np.uint8)

                        # DEBUG: Verify perfect correspondence
                        image_mean = np.mean(original_image)
                        gt_classes = np.unique(seg_targets_np[i])
                        self.logger.debug(f"🔍 Sample {samples_collected}: Batch idx={i}, Image mean={image_mean:.2f}, GT classes={gt_classes}")

                        # Create visualization data based on architecture
                        if self.architecture == 'teacher_student_unet':
                            # Need to get teacher prediction for this sample
                            teacher_logits_i = None
                            if hasattr(self, '_current_teacher_preds') and self._current_teacher_preds is not None:
                                teacher_pred_i = self._current_teacher_preds[i]
                                # Get teacher logits for this sample
                                if hasattr(self, '_current_teacher_logits') and self._current_teacher_logits is not None:
                                    teacher_logits_i = self._current_teacher_logits[i]
                            else:
                                # Fallback: generate teacher prediction for this sample
                                sample_image = images[i:i+1]  # Single sample batch
                                underlying_ts_model = self.model.segmentation_model
                                if not underlying_ts_model.teacher_initialized:
                                    underlying_ts_model.initialize_teacher()
                                ts_outputs = underlying_ts_model(sample_image, mode="teacher_student")
                                teacher_seg_logits = ts_outputs['teacher']
                                teacher_pred_i = torch.argmax(teacher_seg_logits, dim=1).detach().cpu().numpy()[0]
                                teacher_logits_i = teacher_seg_logits.detach().cpu().numpy()[0]

                            # Get teacher classification prediction for this sample
                            if hasattr(self, '_current_teacher_patch_preds') and self._current_teacher_patch_preds is not None:
                                teacher_patch_pred_i = self._current_teacher_patch_preds[i]
                            else:
                                # Fallback: generate teacher classification for this sample
                                sample_image = images[i:i+1]  # Single sample batch
                                teacher_outputs = self.model(sample_image, mode="teacher_only")
                                teacher_patch_pred_i = torch.argmax(teacher_outputs['patch_classification'], dim=1).detach().cpu().numpy()[0]

                            vis, metadata = self.create_7column_ts_visualization(
                                original_image,
                                seg_targets_np[i],
                                teacher_pred_i,
                                seg_pred[i],
                                seg_logits[i].detach().cpu().numpy(),
                                samples_collected,
                                student_patch_pred=patch_pred[i],
                                teacher_patch_pred=teacher_patch_pred_i,
                                patch_target=patch_targets_np[i],
                                teacher_seg_logits=teacher_logits_i
                            )
                        else:
                            vis, metadata = self.create_4column_visualization(
                                original_image,
                                seg_targets_np[i],
                                seg_pred[i],
                                seg_logits[i].detach().cpu().numpy(),
                                samples_collected
                            )
                        visualization_data.append((vis, metadata))
                        samples_collected += 1

                # Clear GPU memory immediately after processing each batch
                try:
                    del images, seg_masks, patch_labels, outputs, seg_logits, patch_logits
                except NameError:
                    # Some variables might not be defined in error paths
                    if 'images' in locals(): del images
                    if 'seg_masks' in locals(): del seg_masks
                    if 'patch_labels' in locals(): del patch_labels
                    if 'outputs' in locals(): del outputs
                    if 'seg_logits' in locals(): del seg_logits
                    if 'patch_logits' in locals(): del patch_logits
                del seg_pred, patch_pred, seg_targets_np, patch_targets_np
                torch.cuda.empty_cache()

                # Periodic progress logging
                if batch_idx % 100 == 0:
                    interim_metrics = metrics_aggregator.get_all_metrics()
                    self.logger.info(f"   Progress: {batch_idx + 1}/{len(loader)} batches, "
                                   f"Dice: {interim_metrics['dice_mean']:.4f}")

        # Get final metrics from aggregator
        final_metrics = metrics_aggregator.get_all_metrics()

        self.logger.info(f"✅ {split.capitalize()} streaming evaluation completed:")
        self.logger.info(f"   📊 Total samples: {final_metrics['num_samples']}")
        self.logger.info(f"   🎯 Mean Dice: {final_metrics['dice_mean']:.4f}")
        self.logger.info(f"   📈 Pixel Accuracy: {final_metrics['pixel_accuracy']:.4f}")
        self.logger.info(f"   🏷️ Patch Accuracy: {final_metrics['patch_accuracy']:.4f}")
        self.logger.info(f"   🎨 Visualization samples collected: {len(visualization_data)}")

        return final_metrics, visualization_data

    def evaluate_complete_split(self, split: str) -> Dict[str, Any]:
        """
        Legacy evaluation method (memory-intensive) - kept for compatibility
        WARNING: This method accumulates all predictions in memory and may cause OOM
        """
        self.logger.warning(f"⚠️ Using legacy evaluation for {split} - may cause OOM on large datasets!")
        self.logger.info(f"📊 Evaluating complete {split} dataset (legacy mode)...")

        loader = self.data_loaders[split]

        # Storage for predictions and targets (MEMORY INTENSIVE!)
        all_seg_predictions = []
        all_seg_targets = []
        all_patch_predictions = []
        all_patch_targets = []

        # Evaluation loop
        with torch.no_grad():
            for batch_idx, batch_data in enumerate(tqdm(loader, desc=f"Evaluating {split}")):
                # Move data to device (safely handle different key formats)
                images, seg_masks, patch_labels = self._extract_batch_data(batch_data)

                # Forward pass with correct mode for Teacher-Student models
                if self.architecture == 'teacher_student_unet':
                    # Use the specified mode for Teacher-Student evaluation
                    mode = f"{self.teacher_student_mode}_only"  # 'teacher_only' or 'student_only'
                    outputs = self.model(images, mode=mode)
                else:
                    outputs = self.model(images)

                # Handle model outputs (MultiTaskWrapper flattens Teacher-Student outputs)
                seg_logits = outputs['segmentation']
                patch_logits = outputs['patch_classification']

                # Convert to predictions and detach from computation graph
                seg_pred = torch.argmax(seg_logits, dim=1).detach().cpu().numpy()
                patch_pred = torch.sigmoid(patch_logits).detach().cpu().numpy()

                # Store results (detach targets too for consistency)
                all_seg_predictions.append(seg_pred)
                all_seg_targets.append(seg_masks.detach().cpu().numpy())
                all_patch_predictions.append(patch_pred)
                all_patch_targets.append(patch_labels.detach().cpu().numpy())

                # Clear GPU cache periodically to prevent memory buildup
                if batch_idx % 50 == 0:
                    torch.cuda.empty_cache()

        # Concatenate results (MEMORY INTENSIVE!)
        all_seg_predictions = np.concatenate(all_seg_predictions, axis=0)
        all_seg_targets = np.concatenate(all_seg_targets, axis=0)
        all_patch_predictions = np.concatenate(all_patch_predictions, axis=0)
        all_patch_targets = np.concatenate(all_patch_targets, axis=0)

        # Calculate metrics using legacy methods
        seg_metrics = self.calculate_segmentation_metrics(all_seg_predictions, all_seg_targets)
        patch_metrics = self.calculate_patch_classification_metrics(all_patch_predictions, all_patch_targets)

        # Combine metrics
        results = {
            **seg_metrics,
            **patch_metrics,
            'num_samples': len(all_seg_predictions)
        }

        self.logger.info(f"✅ {split.capitalize()} evaluation completed:")
        self.logger.info(f"   📊 Samples: {results['num_samples']}")
        self.logger.info(f"   🎯 Mean Dice: {results['dice_mean']:.4f}")
        self.logger.info(f"   📈 Pixel Accuracy: {results['pixel_accuracy']:.4f}")
        self.logger.info(f"   🏷️ Patch Accuracy: {results['patch_accuracy']:.4f}")

        return results

    def create_4column_visualization(
        self,
        image: np.ndarray,
        seg_target: np.ndarray,
        seg_prediction: np.ndarray,
        seg_logits: np.ndarray,
        sample_idx: int
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Create 4-column visualization with enhanced information:
        1. Original unnormalized image
        2. Ground truth segmentation mask
        3. Predicted segmentation mask
        4. Overlay prediction on original image

        Note: Augmentation is disabled during evaluation, so no rotation mismatch occurs.

        Args:
            image: Original unnormalized image for display and overlay
            seg_target: Ground truth segmentation mask
            seg_prediction: Predicted segmentation mask
            seg_logits: Raw prediction logits for probability calculation
            sample_idx: Sample index

        Returns:
            visualization: Combined image array
            metadata: Dictionary containing ground truth classes and prediction probabilities
        """
        h, w = image.shape[:2]

        # Create colored masks
        gt_colored = np.zeros((h, w, 3), dtype=np.uint8)
        pred_colored = np.zeros((h, w, 3), dtype=np.uint8)

        for class_idx in range(len(self.class_names)):
            gt_colored[seg_target == class_idx] = self.class_colors[class_idx]
            pred_colored[seg_prediction == class_idx] = self.class_colors[class_idx]

        # Create better overlay - only overlay non-background predictions with transparency
        # Since augmentation is disabled, use the original image for overlay (no rotation mismatch)
        overlay = image.copy()
        alpha = 0.6  # Transparency factor

        # Only overlay non-background regions
        for class_idx in range(1, len(self.class_names)):  # Skip background (0)
            mask = seg_prediction == class_idx
            if np.any(mask):
                overlay[mask] = (overlay[mask] * (1 - alpha) +
                               self.class_colors[class_idx] * alpha).astype(np.uint8)

        # Combine all 4 columns horizontally
        visualization = np.hstack([image, gt_colored, pred_colored, overlay])

        # Extract metadata for titles
        # Ground truth classes present
        gt_classes = np.unique(seg_target)
        gt_class_names = [self.class_names[idx] for idx in gt_classes]

        # Prediction probabilities (average across entire patch)
        pred_probs = torch.softmax(torch.from_numpy(seg_logits), dim=0).numpy()
        avg_probs = np.mean(pred_probs, axis=(1, 2))  # Average across spatial dimensions

        metadata = {
            'gt_classes': gt_class_names,
            'pred_probabilities': {self.class_names[i]: avg_probs[i] for i in range(len(self.class_names))},
            'sample_idx': sample_idx
        }

        return visualization, metadata

    def create_7column_ts_visualization(
        self,
        image: np.ndarray,
        seg_target: np.ndarray,
        teacher_seg_pred: np.ndarray,
        student_seg_pred: np.ndarray,
        seg_logits: np.ndarray,
        sample_idx: int,
        student_patch_pred: int = None,
        teacher_patch_pred: int = None,
        patch_target: int = None,
        teacher_seg_logits: np.ndarray = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Create 7-column visualization for Teacher-Student models with:
        1. Original unnormalized image
        2. Ground truth segmentation mask
        3. Ground truth overlay on original image
        4. Teacher pseudo-mask (used as consistency loss target)
        5. Teacher prediction overlay on original image
        6. Student predicted mask
        7. Student/Teacher overlay prediction on original image (based on evaluation mode)

        Args:
            image: Original unnormalized image for display and overlay
            seg_target: Ground truth segmentation mask
            teacher_seg_pred: Teacher network prediction (pseudo-mask)
            student_seg_pred: Student or Teacher network prediction (based on mode)
            seg_logits: Raw prediction logits for probability calculation
            sample_idx: Sample index
            student_patch_pred: Student patch classification prediction
            teacher_patch_pred: Teacher patch classification prediction
            patch_target: Ground truth patch classification
            teacher_seg_logits: Teacher segmentation logits for probability calculation

        Returns:
            visualization: Combined image array (7 columns)
            metadata: Dictionary containing ground truth classes, prediction probabilities, and teacher probabilities
        """
        # Import required modules at function level
        import torch
        import torch.nn.functional as F
        import numpy as np

        h, w = image.shape[:2]

        # Apply filtering to teacher pseudo-mask for visualization
        filtered_teacher_seg_pred = teacher_seg_pred.copy()
        if (self.architecture == 'teacher_student_unet' and
            teacher_seg_logits is not None and
            self.ts_config.get('pseudo_mask_filtering', 'none') != 'none'):

            # Convert numpy to torch for processing
            teacher_logits_torch = torch.from_numpy(teacher_seg_logits).unsqueeze(0)  # Add batch dim
            teacher_probs = F.softmax(teacher_logits_torch, dim=1)

            filtering_strategy = self.ts_config['pseudo_mask_filtering']

            if filtering_strategy == 'confidence':
                # Confidence-based filtering
                confidence, _ = torch.max(teacher_probs, dim=1)  # [1, H, W]
                confidence_threshold = self.ts_config['confidence_threshold']
                mask = confidence[0] > confidence_threshold  # Remove batch dim

            elif filtering_strategy == 'entropy':
                # Entropy-based filtering
                entropy = -torch.sum(teacher_probs * torch.log(teacher_probs + 1e-8), dim=1)  # [1, H, W]
                entropy_threshold = self.ts_config['entropy_threshold']
                mask = entropy[0] < entropy_threshold  # Remove batch dim

            # Convert mask to numpy and apply filtering
            mask_np = mask.numpy().astype(bool)
            # Set filtered-out pixels to background class (0)
            filtered_teacher_seg_pred[~mask_np] = 0

        # Create colored masks
        gt_colored = np.zeros((h, w, 3), dtype=np.uint8)
        teacher_colored = np.zeros((h, w, 3), dtype=np.uint8)
        student_colored = np.zeros((h, w, 3), dtype=np.uint8)

        for class_idx in range(len(self.class_names)):
            gt_colored[seg_target == class_idx] = self.class_colors[class_idx]
            teacher_colored[filtered_teacher_seg_pred == class_idx] = self.class_colors[class_idx]
            student_colored[student_seg_pred == class_idx] = self.class_colors[class_idx]

        # Create overlays - overlay non-background predictions with transparency
        alpha = 0.6  # Transparency factor

        # Ground truth overlay (5th column)
        gt_overlay = image.copy()
        for class_idx in range(1, len(self.class_names)):  # Skip background (0)
            mask = seg_target == class_idx
            if np.any(mask):
                gt_overlay[mask] = (gt_overlay[mask] * (1 - alpha) +
                                  self.class_colors[class_idx] * alpha).astype(np.uint8)

        # Teacher overlay (5th column) - use filtered teacher prediction
        teacher_overlay = image.copy()
        for class_idx in range(1, len(self.class_names)):  # Skip background (0)
            mask = filtered_teacher_seg_pred == class_idx
            if np.any(mask):
                teacher_overlay[mask] = (teacher_overlay[mask] * (1 - alpha) +
                                       self.class_colors[class_idx] * alpha).astype(np.uint8)

        # Student overlay (7th column)
        student_overlay = image.copy()
        for class_idx in range(1, len(self.class_names)):  # Skip background (0)
            mask = student_seg_pred == class_idx
            if np.any(mask):
                student_overlay[mask] = (student_overlay[mask] * (1 - alpha) +
                                       self.class_colors[class_idx] * alpha).astype(np.uint8)

        # Combine all 7 columns horizontally in new order
        # 1. Original Image | 2. GT Mask | 3. GT Overlay | 4. Teacher Mask | 5. Teacher Overlay | 6. Student Mask | 7. Student Overlay
        visualization = np.hstack([image, gt_colored, gt_overlay, teacher_colored, teacher_overlay, student_colored, student_overlay])


        # Extract metadata for titles
        # Ground truth classes present
        gt_classes = np.unique(seg_target)
        gt_class_names = [self.class_names[idx] for idx in gt_classes]

        # Teacher pseudo-mask classes (use filtered version for display)
        teacher_classes = np.unique(filtered_teacher_seg_pred)
        teacher_class_names = [self.class_names[idx] for idx in teacher_classes]

        # Student prediction classes
        student_classes = np.unique(student_seg_pred)
        student_class_names = [self.class_names[idx] for idx in student_classes]

        # Prediction probabilities (average across entire patch)
        pred_probs = torch.softmax(torch.from_numpy(seg_logits), dim=0).numpy()
        avg_probs = np.mean(pred_probs, axis=(1, 2))  # Average across spatial dimensions

        # Teacher probabilities (if teacher logits are available)
        teacher_probabilities = None
        if teacher_seg_logits is not None:
            teacher_probs = torch.softmax(torch.from_numpy(teacher_seg_logits), dim=0).numpy()
            teacher_avg_probs = np.mean(teacher_probs, axis=(1, 2))  # Average across spatial dimensions
            teacher_probabilities = {self.class_names[i]: teacher_avg_probs[i] for i in range(len(self.class_names))}

        metadata = {
            'gt_classes': gt_class_names,
            'teacher_classes': teacher_class_names,
            'student_classes': student_class_names,
            'pred_probabilities': {self.class_names[i]: avg_probs[i] for i in range(len(self.class_names))},
            'teacher_probabilities': teacher_probabilities,
            'sample_idx': sample_idx,
            'is_teacher_student': True,
            'patch_target': patch_target,
            'student_patch_pred': student_patch_pred,
            'teacher_patch_pred': teacher_patch_pred
        }

        return visualization, metadata

    def create_visualization_figures_from_data(self, split: str, visualization_data: List[Tuple[np.ndarray, Dict]]):
        """
        Create visualization figures from pre-collected visualization data.
        This method works with the streaming evaluation output.

        Args:
            split: 'train', 'val', or 'test'
            visualization_data: List of (visualization_array, metadata) tuples
        """
        self.logger.info(f"🎨 Creating {len(visualization_data)} visualization figures for {split}...")

        # Create figures (7 patches per figure as requested)
        samples_per_figure = 7
        num_figures = (len(visualization_data) + samples_per_figure - 1) // samples_per_figure

        for fig_idx in range(num_figures):
            start_idx = fig_idx * samples_per_figure
            end_idx = min(start_idx + samples_per_figure, len(visualization_data))
            figure_data = visualization_data[start_idx:end_idx]

            # Determine if this is Teacher-Student visualization (7 columns) or standard (4 columns)
            is_teacher_student = figure_data[0][1].get('is_teacher_student', False)
            num_columns = 7 if is_teacher_student else 4

            # Create matplotlib figure with equal height and width
            fig_size = 28 if is_teacher_student else 16  # Equal dimensions for square figure
            fig_height = fig_size
            fig_width = fig_size

            fig, axes = plt.subplots(len(figure_data), num_columns, figsize=(fig_width, fig_height))
            if len(figure_data) == 1:
                axes = axes.reshape(1, -1)

            for row_idx, (vis, metadata) in enumerate(figure_data):
                h, w = vis.shape[:2]
                col_width = w // num_columns

                if is_teacher_student:
                    # Split visualization into 7 columns for Teacher-Student models (new order)
                    original = vis[:, :col_width]
                    gt_mask = vis[:, col_width:2*col_width]
                    gt_overlay = vis[:, 2*col_width:3*col_width]
                    teacher_mask = vis[:, 3*col_width:4*col_width]
                    teacher_overlay = vis[:, 4*col_width:5*col_width]
                    student_mask = vis[:, 5*col_width:6*col_width]
                    student_overlay = vis[:, 6*col_width:]

                    columns = [original, gt_mask, gt_overlay, teacher_mask, teacher_overlay, student_mask, student_overlay]
                    # Create teacher pseudo-mask title with probabilities if available
                    filtering_info = ""
                    if hasattr(self, 'ts_config') and self.ts_config.get('pseudo_mask_filtering', 'none') != 'none':
                        filtering_strategy = self.ts_config['pseudo_mask_filtering']
                        if filtering_strategy == 'confidence':
                            threshold = self.ts_config['confidence_threshold']
                            filtering_info = f" (Conf>{threshold})"
                        elif filtering_strategy == 'entropy':
                            threshold = self.ts_config['entropy_threshold']
                            filtering_info = f" (Ent<{threshold})"

                    if metadata.get('teacher_probabilities'):
                        teacher_prob_str = ' | '.join([f'{name[:3]}: {prob:.2f}' for name, prob in metadata['teacher_probabilities'].items()])
                        teacher_title = f"Teacher Pseudo-Mask{filtering_info}\n{teacher_prob_str}"
                    else:
                        teacher_title = f"Teacher Pseudo-Mask{filtering_info}\nClasses: {', '.join(metadata['teacher_classes'])}"

                    # Create prediction probability strings for overlays
                    pred_prob_str = ' | '.join([f'{name[:3]}: {prob:.2f}' for name, prob in metadata['pred_probabilities'].items()])
                    teacher_prob_str = None
                    if metadata.get('teacher_probabilities'):
                        teacher_prob_str = ' | '.join([f'{name[:3]}: {prob:.2f}' for name, prob in metadata['teacher_probabilities'].items()])

                    titles = [
                        f"Original\nGT: {', '.join(metadata['gt_classes'])}",
                        "Ground Truth Mask",
                        f"Ground Truth Overlay\nGT Classes: {', '.join(metadata['gt_classes'])}",
                        teacher_title,
                        f"Teacher Overlay\n{teacher_prob_str if teacher_prob_str else ', '.join(metadata['teacher_classes'])}",
                        f"Student Predicted Mask\nClasses: {', '.join(metadata['student_classes'])}",
                        f"{self.teacher_student_mode.capitalize()} Overlay\n{pred_prob_str}"
                    ]
                else:
                    # Split visualization into 4 columns for standard models
                    original = vis[:, :col_width]
                    gt_mask = vis[:, col_width:2*col_width]
                    pred_mask = vis[:, 2*col_width:3*col_width]
                    overlay = vis[:, 3*col_width:]

                    columns = [original, gt_mask, pred_mask, overlay]
                    titles = [
                        f"Original Image\nGT: {', '.join(metadata['gt_classes'])}",
                        "Ground Truth Mask",
                        f"Prediction Mask\n{' | '.join([f'{name[:3]}: {prob:.2f}' for name, prob in metadata['pred_probabilities'].items()])}",
                        "Prediction Overlay"
                    ]

                # Display columns
                for col, (column_img, title) in enumerate(zip(columns, titles)):
                    axes[row_idx, col].imshow(column_img)
                    axes[row_idx, col].axis('off')
                    axes[row_idx, col].set_title(title, fontsize=10, pad=10)

                # Add sample number on the left
                sample_idx = start_idx + row_idx
                fig.text(0.02, 0.85 - (row_idx * 0.8/len(figure_data)), f'Sample #{sample_idx+1}',
                        fontsize=12, fontweight='bold', rotation=90, va='center')

            # Overall title with architecture-specific naming
            if is_teacher_student:
                mode_name = self.teacher_student_mode.capitalize()
                fig.suptitle(f'{split.capitalize()} Set - Teacher-Student Model Results ({mode_name} Evaluation)',
                            fontsize=16, fontweight='bold', y=0.98)
            else:
                fig.suptitle(f'{split.capitalize()} Set - 4-Class Segmentation Results (Streaming)',
                            fontsize=16, fontweight='bold', y=0.98)

            # Adjust layout to minimize white space and maximize image size
            plt.tight_layout()
            plt.subplots_adjust(top=0.98, bottom=0.05, left=0.02, right=0.98, hspace=0.1, wspace=0.1)

            # Add color legend at the bottom
            legend_elements = []
            for i, (class_name, color) in enumerate(zip(self.class_names, self.class_colors)):
                color_normalized = color / 255.0
                legend_elements.append(plt.Rectangle((0, 0), 1, 1, facecolor=color_normalized, label=class_name))

            fig.legend(handles=legend_elements, loc='lower center', ncol=len(self.class_names),
                      fontsize=12, title='Class Colors', title_fontsize=14,
                      bbox_to_anchor=(0.5, 0.01))

            # Save figure with architecture-specific filename
            if is_teacher_student:
                mode_name = self.teacher_student_mode.lower()
                figure_path = self.visualizations_dir / f"{split}_ts_{mode_name}_streaming_evaluation_figure_{fig_idx + 1:02d}.png"
            else:
                figure_path = self.visualizations_dir / f"{split}_streaming_evaluation_figure_{fig_idx + 1:02d}.png"
            plt.savefig(figure_path, dpi=300, bbox_inches='tight')
            plt.close()

            self.logger.info(f"💾 Saved {split} figure {fig_idx + 1}/{num_figures}: {figure_path.name}")

        self.logger.info(f"✅ Generated {num_figures} visualization figures for {split}")

    def generate_visualization_figures(self, split: str, sample_indices: List[int]):
        """
        Generate enhanced 4-column visualization figures with:
        - Ground truth classes in original patch titles
        - Prediction probabilities in prediction mask titles
        - Color legends at bottom
        - Proper overlay visualization
        Shows 5 patches per figure with multiple figures as needed

        Args:
            split: 'train', 'val', or 'test'
            sample_indices: List of sample indices to visualize
        """
        self.logger.info(f"🎨 Generating {split} visualizations for {len(sample_indices)} samples...")

        loader = self.data_loaders[split]

        # Create subset of data loader with sampled indices
        dataset = loader.dataset
        subset_dataset = Subset(dataset, sample_indices)
        subset_loader = DataLoader(subset_dataset, batch_size=1, shuffle=False, num_workers=1)

        # Storage for visualizations and metadata
        visualization_data = []

        with torch.no_grad():
            for batch_idx, batch_data in enumerate(tqdm(subset_loader, desc=f"Creating {split} visualizations")):
                # Move data to device (safely handle different key formats)
                images, seg_masks, patch_labels = self._extract_batch_data(batch_data)

                # Forward pass with correct mode for Teacher-Student models
                teacher_seg_pred = None
                if self.architecture == 'teacher_student_unet':
                    # For Teacher-Student models, get both teacher and student predictions for visualization
                    # Access the underlying Teacher-Student model directly to get both outputs
                    underlying_ts_model = self.model.segmentation_model

                    # Ensure teacher is initialized
                    if not underlying_ts_model.teacher_initialized:
                        self.logger.warning("⚠️ Teacher not initialized, initializing now for visualization...")
                        underlying_ts_model.initialize_teacher()

                    # Get teacher-student output (contains both networks)
                    ts_outputs = underlying_ts_model(images, mode="teacher_student")

                    # Extract teacher prediction for pseudo-mask
                    teacher_seg_logits = ts_outputs['teacher']
                    teacher_seg_pred = torch.argmax(teacher_seg_logits, dim=1).detach().cpu().numpy()[0]

                    # Extract student prediction for visualization (always show student in 6th column)
                    student_seg_logits = ts_outputs['student']
                    student_seg_pred = torch.argmax(student_seg_logits, dim=1).detach().cpu().numpy()[0]

                    # Extract the specified mode prediction (teacher or student) and format as MultiTaskWrapper output
                    if self.teacher_student_mode == 'student':
                        eval_seg_logits = ts_outputs['student']
                    else:
                        eval_seg_logits = ts_outputs['teacher']

                    # Format as MultiTaskWrapper-style output for compatibility
                    outputs = {'segmentation': eval_seg_logits}

                    # Add other task outputs using the MultiTaskWrapper
                    wrapper_outputs = self.model(images, mode=f"{self.teacher_student_mode}_only")
                    outputs.update({k: v for k, v in wrapper_outputs.items() if k != 'segmentation'})
                else:
                    outputs = self.model(images)

                # Handle model outputs (MultiTaskWrapper flattens Teacher-Student outputs)
                seg_logits = outputs['segmentation']
                seg_pred = torch.argmax(seg_logits, dim=1).detach().cpu().numpy()[0]

                # CRITICAL: Always use batch image to ensure perfect correspondence with ground truth mask
                # Both image and seg_target come from the same batch, so they MUST correspond
                image = images[0].detach().cpu().numpy().transpose(1, 2, 0)

                # Proper denormalization for visualization
                if image.min() < 0 or image.max() <= 1.0:
                    if image.min() < 0:
                        # Likely [-1, 1] normalization
                        image = (image + 1.0) / 2.0  # [-1, 1] -> [0, 1]
                    elif image.max() <= 1.0:
                        # Check for ImageNet normalization
                        test_denorm = image * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
                        if test_denorm.min() >= 0 and test_denorm.max() <= 1.0:
                            image = test_denorm

                    # Ensure [0, 1] range
                    image = np.clip(image, 0, 1)

                # Convert to uint8 for visualization
                if image.max() <= 1.0:
                    image = (image * 255).astype(np.uint8)
                else:
                    image = np.clip(image, 0, 255).astype(np.uint8)

                seg_target = seg_masks[0].detach().cpu().numpy()
                seg_logits_np = seg_logits[0].detach().cpu().numpy()

                # DEBUG: Verify image and mask correspondence
                image_mean = np.mean(image)
                gt_classes = np.unique(seg_target)
                self.logger.info(f"🔍 DEBUG Sample {batch_idx}: Image mean={image_mean:.2f}, GT classes={gt_classes}, Image shape={image.shape}, Mask shape={seg_target.shape}")

                # Create visualization with metadata (now supports teacher pseudo-mask)
                # No need for preprocessed image since augmentation is disabled
                if self.architecture == 'teacher_student_unet':
                    # For Teacher-Student models, create 5-column visualization with teacher pseudo-mask
                    teacher_seg_logits_np = teacher_seg_logits[0].detach().cpu().numpy()
                    vis, metadata = self.create_7column_ts_visualization(
                        image, seg_target, teacher_seg_pred, student_seg_pred, seg_logits_np, batch_idx,
                        teacher_seg_logits=teacher_seg_logits_np
                    )
                else:
                    # For standard models, use 4-column visualization
                    vis, metadata = self.create_4column_visualization(
                        image, seg_target, seg_pred, seg_logits_np, batch_idx
                    )
                visualization_data.append((vis, metadata))

                # Cleanup batch variables to free memory
                del images, seg_masks, outputs, seg_logits
                torch.cuda.empty_cache()

        # Create figures (7 patches per figure as requested)
        samples_per_figure = 7
        num_figures = (len(visualization_data) + samples_per_figure - 1) // samples_per_figure

        for fig_idx in range(num_figures):
            start_idx = fig_idx * samples_per_figure
            end_idx = min(start_idx + samples_per_figure, len(visualization_data))
            figure_data = visualization_data[start_idx:end_idx]

            # Determine if this is Teacher-Student visualization (7 columns) or standard (4 columns)
            is_teacher_student = figure_data[0][1].get('is_teacher_student', False)
            num_columns = 7 if is_teacher_student else 4

            # Create matplotlib figure with equal height and width
            fig_size = 28 if is_teacher_student else 16  # Equal dimensions for square figure
            fig_height = fig_size
            fig_width = fig_size

            fig, axes = plt.subplots(len(figure_data), num_columns, figsize=(fig_width, fig_height))
            if len(figure_data) == 1:
                axes = axes.reshape(1, -1)

            for row_idx, (vis, metadata) in enumerate(figure_data):
                h, w = vis.shape[:2]
                col_width = w // num_columns

                if is_teacher_student:
                    # Split visualization into 6 columns for Teacher-Student models
                    original = vis[:, :col_width]
                    gt_mask = vis[:, col_width:2*col_width]
                    teacher_mask = vis[:, 2*col_width:3*col_width]
                    gt_overlay = vis[:, 3*col_width:4*col_width]
                    teacher_overlay = vis[:, 4*col_width:5*col_width]
                    student_overlay = vis[:, 5*col_width:]

                    # Column 1: Original patch with GT classes and patch classification
                    axes[row_idx, 0].imshow(original)
                    gt_classes_str = ', '.join(metadata['gt_classes'])
                    patch_target = metadata.get('patch_target')
                    if patch_target is not None:
                        gt_patch_class = self.class_names[patch_target]
                        title = f'Original\nGT Seg: {gt_classes_str}\nGT Patch: {gt_patch_class}'
                    else:
                        title = f'Original\nGT: {gt_classes_str}'
                    axes[row_idx, 0].set_title(title, fontsize=9, fontweight='bold')
                    axes[row_idx, 0].axis('off')

                    # Column 2: Ground truth mask
                    axes[row_idx, 1].imshow(gt_mask)
                    axes[row_idx, 1].set_title('Ground Truth Mask', fontsize=10, fontweight='bold')
                    axes[row_idx, 1].axis('off')

                    # Column 3: Teacher pseudo-mask with teacher probabilities and classification
                    axes[row_idx, 2].imshow(teacher_mask)

                    # Use teacher probabilities if available, otherwise fall back to class names
                    if metadata.get('teacher_probabilities'):
                        teacher_prob_str = ' | '.join([f'{name[:3]}: {prob:.2f}' for name, prob in metadata['teacher_probabilities'].items()])
                        teacher_patch_pred = metadata.get('teacher_patch_pred')
                        if teacher_patch_pred is not None:
                            teacher_patch_class = self.class_names[teacher_patch_pred]
                            title = f'Teacher Pseudo-Mask\nSeg: {teacher_prob_str}\nPatch: {teacher_patch_class}'
                        else:
                            title = f'Teacher Pseudo-Mask\n{teacher_prob_str}'
                    else:
                        # Fallback to class names if probabilities not available
                        teacher_classes_str = ', '.join(metadata['teacher_classes'])
                        teacher_patch_pred = metadata.get('teacher_patch_pred')
                        if teacher_patch_pred is not None:
                            teacher_patch_class = self.class_names[teacher_patch_pred]
                            title = f'Teacher Pseudo-Mask\nSeg: {teacher_classes_str}\nPatch: {teacher_patch_class}'
                        else:
                            title = f'Teacher Pseudo-Mask\nClasses: {teacher_classes_str}'

                    axes[row_idx, 2].set_title(title, fontsize=9, fontweight='bold')
                    axes[row_idx, 2].axis('off')

                    # Column 4: Ground truth overlay
                    axes[row_idx, 3].imshow(gt_overlay)
                    gt_classes_str = ', '.join(metadata['gt_classes'])
                    axes[row_idx, 3].set_title(f'Ground Truth Overlay\nGT Classes: {gt_classes_str}', fontsize=9, fontweight='bold')
                    axes[row_idx, 3].axis('off')

                    # Column 5: Teacher overlay with probabilities
                    axes[row_idx, 4].imshow(teacher_overlay)
                    if metadata.get('teacher_probabilities'):
                        teacher_prob_str = ' | '.join([f'{name[:3]}: {prob:.2f}' for name, prob in metadata['teacher_probabilities'].items()])
                        teacher_patch_pred = metadata.get('teacher_patch_pred')
                        if teacher_patch_pred is not None:
                            teacher_patch_class = self.class_names[teacher_patch_pred]
                            title = f'Teacher Overlay\nSeg: {teacher_prob_str}\nPatch: {teacher_patch_class}'
                        else:
                            title = f'Teacher Overlay\n{teacher_prob_str}'
                    else:
                        teacher_classes_str = ', '.join(metadata['teacher_classes'])
                        teacher_patch_pred = metadata.get('teacher_patch_pred')
                        if teacher_patch_pred is not None:
                            teacher_patch_class = self.class_names[teacher_patch_pred]
                            title = f'Teacher Overlay\nSeg: {teacher_classes_str}\nPatch: {teacher_patch_class}'
                        else:
                            title = f'Teacher Overlay\nClasses: {teacher_classes_str}'
                    axes[row_idx, 4].set_title(title, fontsize=8, fontweight='bold')
                    axes[row_idx, 4].axis('off')

                    # Column 6: Student overlay with probabilities
                    axes[row_idx, 5].imshow(student_overlay)
                    pred_probs = metadata['pred_probabilities']
                    prob_str = ' | '.join([f'{name[:3]}: {prob:.2f}' for name, prob in pred_probs.items()])
                    mode_name = self.teacher_student_mode.capitalize()

                    # Add classification prediction based on evaluation mode
                    if self.teacher_student_mode == 'student':
                        student_patch_pred = metadata.get('student_patch_pred')
                        if student_patch_pred is not None:
                            student_patch_class = self.class_names[student_patch_pred]
                            title = f'{mode_name} Overlay\nSeg: {prob_str}\nPatch: {student_patch_class}'
                        else:
                            title = f'{mode_name} Overlay\n{prob_str}'
                    else:  # teacher mode
                        teacher_patch_pred = metadata.get('teacher_patch_pred')
                        if teacher_patch_pred is not None:
                            teacher_patch_class = self.class_names[teacher_patch_pred]
                            title = f'{mode_name} Overlay\nSeg: {prob_str}\nPatch: {teacher_patch_class}'
                        else:
                            title = f'{mode_name} Overlay\n{prob_str}'

                    axes[row_idx, 5].set_title(title, fontsize=8, fontweight='bold')
                    axes[row_idx, 5].axis('off')

                else:
                    # Split visualization into 4 columns for standard models
                    original = vis[:, :col_width]
                    gt_mask = vis[:, col_width:2*col_width]
                    pred_mask = vis[:, 2*col_width:3*col_width]
                    overlay = vis[:, 3*col_width:]

                    # Column 1: Original patch with GT classes
                    axes[row_idx, 0].imshow(original)
                    gt_classes_str = ', '.join(metadata['gt_classes'])
                    axes[row_idx, 0].set_title(f'Original\nGT: {gt_classes_str}', fontsize=10, fontweight='bold')
                    axes[row_idx, 0].axis('off')

                    # Column 2: Ground truth mask
                    axes[row_idx, 1].imshow(gt_mask)
                    axes[row_idx, 1].set_title('Ground Truth Mask', fontsize=10, fontweight='bold')
                    axes[row_idx, 1].axis('off')

                    # Column 3: Prediction mask with probabilities (horizontal layout)
                    axes[row_idx, 2].imshow(pred_mask)
                    pred_probs = metadata['pred_probabilities']
                    prob_str = ' | '.join([f'{name[:3]}: {prob:.2f}' for name, prob in pred_probs.items()])
                    axes[row_idx, 2].set_title(f'Prediction Mask\n{prob_str}', fontsize=9, fontweight='bold')
                    axes[row_idx, 2].axis('off')

                    # Column 4: Overlay
                    axes[row_idx, 3].imshow(overlay)
                    axes[row_idx, 3].set_title('Overlay', fontsize=10, fontweight='bold')
                    axes[row_idx, 3].axis('off')

                # Add sample number on the left
                sample_idx = start_idx + row_idx
                fig.text(0.02, 0.85 - (row_idx * 0.8/len(figure_data)), f'Sample #{sample_idx+1}',
                        fontsize=12, fontweight='bold', rotation=90, va='center')

            # Overall title with architecture-specific naming
            if is_teacher_student:
                mode_name = self.teacher_student_mode.capitalize()
                fig.suptitle(f'{split.capitalize()} Set - Teacher-Student Model Results ({mode_name} Evaluation)',
                            fontsize=16, fontweight='bold', y=0.98)
            else:
                fig.suptitle(f'{split.capitalize()} Set - 4-Class Segmentation Results',
                            fontsize=16, fontweight='bold', y=0.98)

            # Adjust layout to minimize white space and maximize image size
            plt.tight_layout()
            plt.subplots_adjust(top=0.98, bottom=0.05, left=0.02, right=0.98, hspace=0.1, wspace=0.1)

            # Add color legend at the bottom (below all subplots)
            # Create color patches for legend
            legend_elements = []
            for i, (class_name, color) in enumerate(zip(self.class_names, self.class_colors)):
                color_normalized = color / 255.0  # Normalize to [0,1] for matplotlib
                legend_elements.append(plt.Rectangle((0, 0), 1, 1, facecolor=color_normalized, label=class_name))

            # Position legend closer to the bottom with reduced margin
            fig.legend(handles=legend_elements, loc='lower center', ncol=len(self.class_names),
                      fontsize=12, title='Class Colors', title_fontsize=14,
                      bbox_to_anchor=(0.5, 0.01))

            # Save figure with architecture-specific filename
            if is_teacher_student:
                mode_name = self.teacher_student_mode.lower()
                fig_filename = f"{split}_ts_{mode_name}_evaluation_samples_{fig_idx+1:03d}.png"
            else:
                fig_filename = f"{split}_evaluation_samples_{fig_idx+1:03d}.png"
            fig_path = self.visualizations_dir / fig_filename
            plt.savefig(fig_path, dpi=300, bbox_inches='tight')
            plt.close()

        self.logger.info(f"✅ Generated {num_figures} visualization figures for {split}")

    def save_sample_indices(self, indices_dict: Dict[str, List[int]]):
        """Save selected sample indices for reproducibility"""
        sample_record = {
            **indices_dict,
            'random_seed': self.random_seed,
            'selection_date': datetime.now().strftime('%Y-%m-%d_%H-%M-%S'),
            'visualization_samples_per_split': self.visualization_samples
        }

        indices_path = self.visualizations_dir / "sample_indices.json"
        with open(indices_path, 'w') as f:
            json.dump(sample_record, f, indent=2)

        self.logger.info(f"📝 Sample indices saved: {indices_path}")

    def save_comprehensive_metrics(self, results: Dict[str, Dict[str, Any]]):
        """Save comprehensive evaluation metrics"""

        # Handle case where no evaluation was performed (e.g., during debugging)
        if not results:
            self.logger.warning("⚠️ No evaluation results available - skipping comprehensive metrics save")
            return

        # Create comprehensive metrics table
        metrics_data = []
        for split, metrics in results.items():
            metrics_data.append({
                'Split': split.capitalize(),
                'Total_Samples': metrics['num_samples'],

                # Comprehensive Segmentation Metrics (NEW!)
                'Dice_Score': f"{metrics.get('dice_mean', 0):.4f}",
                'IoU_Score': f"{metrics.get('iou_mean', 0):.4f}",
                'Pixel_Accuracy': f"{metrics.get('pixel_accuracy_overall', metrics.get('pixel_accuracy', 0)):.4f}",

                # Per-class Dice Scores
                'Background_Dice': f"{metrics.get('dice_background', metrics.get('dice_class_0', 0)):.4f}",
                'Benign_Dice': f"{metrics.get('dice_benign_glands', metrics.get('dice_class_1', 0)):.4f}",
                'Malignant_Dice': f"{metrics.get('dice_malignant_glands', metrics.get('dice_class_2', 0)):.4f}",
                'PDC_Dice': f"{metrics.get('dice_pdc', metrics.get('dice_class_3', 0)):.4f}",

                # Per-class IoU Scores (NEW!)
                'Background_IoU': f"{metrics.get('iou_background', metrics.get('iou_class_0', 0)):.4f}",
                'Benign_IoU': f"{metrics.get('iou_benign_glands', metrics.get('iou_class_1', 0)):.4f}",
                'Malignant_IoU': f"{metrics.get('iou_malignant_glands', metrics.get('iou_class_2', 0)):.4f}",
                'PDC_IoU': f"{metrics.get('iou_pdc', metrics.get('iou_class_3', 0)):.4f}",

                # Classification Metrics
                'Patch_Accuracy': f"{metrics.get('patch_accuracy', 0):.4f}",
            })

        # Save as CSV with formatted values
        df = pd.DataFrame(metrics_data)
        metrics_path = self.evaluations_dir / "final_evaluation_metrics.csv"
        df.to_csv(metrics_path, index=False)

        # Save a more readable formatted table
        formatted_table_path = self.evaluations_dir / "final_evaluation_summary_table.txt"
        with open(formatted_table_path, 'w') as f:
            f.write("=" * 120 + "\n")
            f.write("4-Class nnU-Net Multi-Task Evaluation - Final Results Summary\n")
            f.write("=" * 120 + "\n\n")

            # Create formatted table string
            table_str = df.to_string(index=False, justify='center', float_format='%.4f')
            f.write(table_str)
            f.write("\n\n")

            # Add summary statistics
            f.write("SUMMARY STATISTICS:\n")
            f.write("-" * 50 + "\n")

            # Overall averages
            avg_dice = np.mean([results[split]['dice_mean'] for split in results])
            avg_patch_acc = np.mean([results[split]['patch_accuracy'] for split in results])
            avg_pixel_acc = np.mean([results[split]['pixel_accuracy'] for split in results])

            f.write(f"Average Mean Dice (across all splits):    {avg_dice:.4f}\n")
            f.write(f"Average Patch Accuracy (across splits):   {avg_patch_acc:.4f}\n")
            f.write(f"Average Pixel Accuracy (across splits):   {avg_pixel_acc:.4f}\n\n")

            # Best performing split
            best_dice_split = max(results.keys(), key=lambda s: results[s]['dice_mean'])
            f.write(f"Best Dice Performance: {best_dice_split.capitalize()} "
                   f"({results[best_dice_split]['dice_mean']:.4f})\n")

            best_patch_split = max(results.keys(), key=lambda s: results[s]['patch_accuracy'])
            f.write(f"Best Patch Accuracy: {best_patch_split.capitalize()} "
                   f"({results[best_patch_split]['patch_accuracy']:.4f})\n")

            f.write("\n" + "=" * 120 + "\n")

        # Display formatted table in console
        self.print_readable_summary_table(df, results)

        # Save detailed metrics as JSON
        detailed_path = self.evaluations_dir / "detailed_metrics.json"
        with open(detailed_path, 'w') as f:
            # Convert numpy types for JSON serialization
            json_results = {}
            for split, metrics in results.items():
                json_results[split] = {}
                for key, value in metrics.items():
                    if isinstance(value, (np.integer, np.floating)):
                        json_results[split][key] = float(value)
                    else:
                        json_results[split][key] = value

            json.dump(json_results, f, indent=2)

        self.logger.info(f"💾 Comprehensive metrics saved:")
        self.logger.info(f"   📊 CSV table: {metrics_path}")
        self.logger.info(f"   📋 Formatted table: {formatted_table_path}")
        self.logger.info(f"   📋 Detailed JSON: {detailed_path}")

    def print_readable_summary_table(self, df: pd.DataFrame, results: Dict[str, Dict[str, Any]]):
        """Print a nicely formatted summary table to console"""
        self.logger.info("📊 FINAL EVALUATION RESULTS SUMMARY:")
        self.logger.info("=" * 100)

        # Print the main table
        table_lines = df.to_string(index=False, justify='center').split('\n')
        for line in table_lines:
            self.logger.info(f"   {line}")

        self.logger.info("=" * 100)

        # Print summary statistics
        avg_dice = np.mean([results[split]['dice_mean'] for split in results])
        avg_patch_acc = np.mean([results[split]['patch_accuracy'] for split in results])
        avg_pixel_acc = np.mean([results[split]['pixel_accuracy'] for split in results])

        self.logger.info("📈 SUMMARY STATISTICS:")
        self.logger.info(f"   🎯 Average Mean Dice (across all splits):    {avg_dice:.4f}")
        self.logger.info(f"   🏷️ Average Patch Accuracy (across splits):   {avg_patch_acc:.4f}")
        self.logger.info(f"   📊 Average Pixel Accuracy (across splits):   {avg_pixel_acc:.4f}")

        # Best performing split
        best_dice_split = max(results.keys(), key=lambda s: results[s]['dice_mean'])
        best_patch_split = max(results.keys(), key=lambda s: results[s]['patch_accuracy'])

        self.logger.info("🏆 BEST PERFORMANCE:")
        self.logger.info(f"   🥇 Best Dice: {best_dice_split.capitalize()} ({results[best_dice_split]['dice_mean']:.4f})")
        self.logger.info(f"   🥇 Best Patch Accuracy: {best_patch_split.capitalize()} ({results[best_patch_split]['patch_accuracy']:.4f})")

        # Per-class dice summary
        self.logger.info("🎨 PER-CLASS DICE SCORES:")
        class_names = ['Background', 'Benign', 'Malignant', 'PDC']
        for i, class_name in enumerate(class_names):
            class_dice_scores = [results[split][f'dice_class_{i}'] for split in results]
            avg_class_dice = np.mean(class_dice_scores)
            self.logger.info(f"   📊 {class_name:12s}: {avg_class_dice:.4f}")

        self.logger.info("=" * 100)

    def create_evaluation_report(self, results: Dict[str, Dict[str, Any]]):
        """Create comprehensive evaluation report"""
        report_path = self.evaluations_dir / "evaluation_summary_report.md"

        with open(report_path, 'w') as f:
            f.write("# Post-Training Evaluation Report\n\n")
            f.write(f"**Model:** {self.model_path.name}\n")
            f.write(f"**Architecture:** {self.architecture}\n")
            f.write(f"**Dataset:** {self.dataset_key}\n")
            f.write(f"**Evaluation Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"**Visualization Samples:** {self.visualization_samples} per split\n")

            # Add Teacher-Student specific information
            if self.architecture == 'teacher_student_unet' and self.teacher_student_mode:
                f.write(f"**Teacher-Student Mode:** {self.teacher_student_mode.capitalize()} Network Evaluation\n")
                f.write(f"**Visualization Layout:** 5-column (includes Teacher Pseudo-Mask)\n")

            f.write("\n")

            f.write("## Summary Results\n\n")

            # Different table format for Teacher-Student models (includes consistency loss)
            if self.architecture == 'teacher_student_unet' and self.teacher_student_mode:
                f.write("| Split | Samples | Pixel Acc | Mean Dice | Patch Acc | Total Consistency |\n")
                f.write("|-------|---------|-----------|-----------|-----------|-------------------|\n")

                for split, metrics in results.items():
                    consistency_loss = metrics.get('consistency_loss_mean', 'N/A')
                    if isinstance(consistency_loss, (int, float)):
                        consistency_str = f"{consistency_loss:17.4f}"
                    else:
                        consistency_str = f"{'N/A':>17}"

                    f.write(f"| {split.capitalize():5s} | {metrics['num_samples']:7d} | "
                           f"{metrics['pixel_accuracy']:9.4f} | {metrics['dice_mean']:9.4f} | "
                           f"{metrics['patch_accuracy']:9.4f} | {consistency_str} |\n")

                # Add detailed consistency loss breakdown
                f.write(f"\n### Consistency Loss Components ({self.ts_config.get('consistency_loss_type', 'mse').upper()} - Lower is Better)\n\n")
                f.write("| Split | Seg Consistency Loss | Patch Consistency Loss | Gland Consistency Loss |\n")
                f.write("|-------|---------------------|------------------------|------------------------|\n")

                # Collect values for mean calculation
                seg_values = []
                patch_values = []
                gland_values = []

                for split, metrics in results.items():
                    seg_consistency = metrics.get('seg_consistency_mean', 'N/A')
                    patch_consistency = metrics.get('patch_consistency_mean', 'N/A')
                    gland_consistency = metrics.get('gland_consistency_mean', 'N/A')

                    # Store values for mean calculation
                    if isinstance(seg_consistency, (int, float)):
                        seg_values.append(seg_consistency)
                    if isinstance(patch_consistency, (int, float)):
                        patch_values.append(patch_consistency)
                    if isinstance(gland_consistency, (int, float)):
                        gland_values.append(gland_consistency)

                    seg_str = f"{seg_consistency:19.4f}" if isinstance(seg_consistency, (int, float)) else f"{'N/A':>19}"
                    patch_str = f"{patch_consistency:22.4f}" if isinstance(patch_consistency, (int, float)) else f"{'N/A':>22}"
                    gland_str = f"{gland_consistency:22.4f}" if isinstance(gland_consistency, (int, float)) else f"{'N/A':>22}"

                    f.write(f"| {split.capitalize():5s} | {seg_str} | {patch_str} | {gland_str} |\n")

                # Add separator line and mean row
                f.write("|-------|---------------------|------------------------|------------------------|\n")

                # Calculate means
                seg_mean = sum(seg_values) / len(seg_values) if seg_values else None
                patch_mean = sum(patch_values) / len(patch_values) if patch_values else None
                gland_mean = sum(gland_values) / len(gland_values) if gland_values else None

                seg_mean_str = f"{seg_mean:19.4f}" if seg_mean is not None else f"{'N/A':>19}"
                patch_mean_str = f"{patch_mean:22.4f}" if patch_mean is not None else f"{'N/A':>22}"
                gland_mean_str = f"{gland_mean:22.4f}" if gland_mean is not None else f"{'N/A':>22}"

                f.write(f"| **Mean** | {seg_mean_str} | {patch_mean_str} | {gland_mean_str} |\n")
            else:
                # Standard table for baseline/nnUNet models
                f.write("| Split | Samples | Pixel Acc | Mean Dice | Patch Acc |\n")
                f.write("|-------|---------|-----------|-----------|----------|\n")

                for split, metrics in results.items():
                    f.write(f"| {split.capitalize():5s} | {metrics['num_samples']:7d} | "
                           f"{metrics['pixel_accuracy']:9.4f} | {metrics['dice_mean']:9.4f} | "
                           f"{metrics['patch_accuracy']:9.4f} |\n")

            f.write("\n## Per-Class Segmentation Performance\n\n")
            for split, metrics in results.items():
                f.write(f"### {split.capitalize()} Set\n")
                for i, class_name in enumerate(self.class_names):
                    dice_score = metrics[f'dice_class_{i}']
                    f.write(f"- **{class_name}:** Dice = {dice_score:.4f}\n")
                f.write("\n")

            f.write("## Key Findings\n\n")

            # Find best performing split
            best_split = max(results.keys(), key=lambda s: results[s]['dice_mean'])
            f.write(f"- **Best Overall Performance:** {best_split.capitalize()} set "
                   f"(Dice: {results[best_split]['dice_mean']:.4f})\n")

            # Find best class
            all_dice_scores = []
            for split in results:
                for i in range(len(self.class_names)):
                    all_dice_scores.append((split, i, results[split][f'dice_class_{i}']))

            best_class = max(all_dice_scores, key=lambda x: x[2])
            f.write(f"- **Best Class Performance:** {self.class_names[best_class[1]]} "
                   f"(Dice: {best_class[2]:.4f} on {best_class[0]} set)\n")

            # Overall assessment
            avg_dice = np.mean([results[split]['dice_mean'] for split in results])
            if avg_dice > 0.8:
                assessment = "Excellent"
            elif avg_dice > 0.7:
                assessment = "Good"
            elif avg_dice > 0.6:
                assessment = "Fair"
            else:
                assessment = "Needs Improvement"

            f.write(f"- **Overall Assessment:** {assessment} (Average Dice: {avg_dice:.4f})\n")

            # Add Teacher-Student specific configuration section
            if self.architecture == 'teacher_student_unet':
                f.write("\n## Teacher-Student Configuration\n\n")

                # Try to extract configuration from checkpoint
                try:
                    if hasattr(self, 'checkpoint') and self.checkpoint and 'config' in self.checkpoint:
                        config = self.checkpoint['config']
                        ts_config = config.get('teacher_student_config', {})

                        f.write("### Training Configuration\n")
                        f.write(f"- **EMA Decay:** {ts_config.get('ema_decay', 'N/A')}\n")
                        f.write(f"- **Teacher Init Epoch:** {ts_config.get('teacher_init_epoch', 'N/A')}\n")
                        f.write(f"- **Alpha Range:** {ts_config.get('max_alpha', 'N/A')} → {ts_config.get('min_alpha', 'N/A')}\n")
                        f.write(f"- **Consistency Loss Type:** {ts_config.get('consistency_loss_type', 'N/A')}\n")
                        f.write(f"- **Consistency Temperature:** {ts_config.get('consistency_temperature', 'N/A')}\n")
                        f.write(f"- **Gland Consistency Enabled:** {ts_config.get('enable_gland_consistency', 'N/A')}\n")

                        if 'consistency_loss_components' in ts_config:
                            f.write(f"- **Consistency Components:** {', '.join(ts_config['consistency_loss_components'])}\n")

                        f.write(f"\n### Network Architecture\n")
                        f.write(f"- **Depth:** {ts_config.get('depth', 'N/A')}\n")
                        f.write(f"- **Initial Channels:** {ts_config.get('initial_channels', 'N/A')}\n")
                        f.write(f"- **Total Parameters:** ~62M+ (Student + Teacher)\n")

                except Exception as e:
                    f.write("### Configuration\n")
                    f.write("- Configuration details not available in checkpoint\n")

                f.write(f"\n### Evaluation Mode\n")
                f.write(f"- **Network Evaluated:** {self.teacher_student_mode.capitalize()}\n")
                f.write(f"- **Pseudo-Mask Source:** Teacher Network\n")
                f.write(f"- **Consistency Learning:** Teacher → Student knowledge transfer\n")

        self.logger.info(f"📋 Evaluation report saved: {report_path}")

    def run_comprehensive_evaluation(self) -> Dict[str, Dict[str, Any]]:
        """
        Run complete comprehensive evaluation using memory-efficient streaming approach

        Returns:
            Dictionary containing results for all splits
        """
        self.logger.info("🚀 Starting comprehensive post-training evaluation (STREAMING MODE)...")

        # Load model and prepare data
        self.load_model()
        self.prepare_data_loaders()

        # Phase 1: Streaming evaluation on complete datasets
        self.logger.info("📊 Phase 1: Memory-efficient streaming evaluation...")
        results = {}
        all_visualization_data = {}

        for split in ['train', 'val', 'test']:
            # print("Spliting Evaluation....")
            # continue # Hikmat
            # Use streaming evaluation that returns both metrics and visualization data
            metrics, viz_data = self.evaluate_complete_split_streaming(
                split,
                max_viz_samples=self.visualization_samples
            )
            results[split] = metrics
            all_visualization_data[split] = viz_data

            # Clear any remaining memory after each split
            torch.cuda.empty_cache()

        # Phase 2: Generate visualizations from collected data
        self.logger.info("🎨 Phase 2: Creating visualization figures from streaming data...")
        for split in ['train', 'val', 'test']:
            if split in all_visualization_data and all_visualization_data[split]:
                self.create_visualization_figures_from_data(split, all_visualization_data[split])

        # Save comprehensive results
        self.logger.info("💾 Saving comprehensive evaluation results...")
        self.save_comprehensive_metrics(results)
        self.create_evaluation_report(results)

        self.logger.info("🎉 Memory-efficient streaming evaluation completed successfully!")
        self.logger.info(f"📁 All results saved to: {self.output_dir}")
        self.logger.info(f"💾 Memory usage was constant regardless of dataset size")

        return results

    def run_legacy_evaluation(self) -> Dict[str, Dict[str, Any]]:
        """
        Run legacy evaluation (original memory-intensive approach) - kept for compatibility
        WARNING: This method may cause OOM on large datasets
        """
        self.logger.warning("⚠️  Running LEGACY evaluation mode - may cause OOM on large datasets!")
        self.logger.info("🚀 Starting comprehensive post-training evaluation (LEGACY MODE)...")

        # Load model and prepare data
        self.load_model()
        self.prepare_data_loaders()

        # Phase 1: Evaluate on complete datasets (original method)
        self.logger.info("📊 Phase 1: Complete dataset evaluation (legacy)...")
        results = {}
        for split in ['train', 'val', 'test']:
            results[split] = self.evaluate_complete_split(split)

        # Phase 2: Generate visualizations for sampled images
        self.logger.info("🎨 Phase 2: Generating visualizations for sampled images...")

        # Generate visualization figures using legacy method
        for split in ['train', 'val', 'test']:
            # Sample random indices for visualization
            dataset_size = len(self.data_loaders[split].dataset)
            if dataset_size <= self.visualization_samples:
                sample_indices = list(range(dataset_size))
            else:
                sample_indices = random.sample(range(dataset_size), self.visualization_samples)

            self.generate_visualization_figures(split, sample_indices)

        # Save comprehensive results
        self.logger.info("💾 Saving comprehensive evaluation results...")
        self.save_comprehensive_metrics(results)
        self.create_evaluation_report(results)

        self.logger.info("🎉 Comprehensive post-training evaluation completed!")
        self.logger.info(f"📁 All results saved to: {self.output_dir}")

        return results


def save_teacher_student_comprehensive_metrics(
    results: Dict[str, Dict[str, Any]],
    output_dir: Path
):
    """
    Save comprehensive Teacher-Student evaluation metrics in table and markdown formats

    Args:
        results: Dictionary containing teacher and student evaluation results
        output_dir: Directory to save metrics files
    """
    comparison_dir = output_dir / "evaluations_comparison"
    comparison_dir.mkdir(parents=True, exist_ok=True)

    # Create comprehensive metrics data for both networks
    all_metrics_data = []

    # Process each network (teacher, student)
    for network_type in ['teacher', 'student']:
        if network_type not in results:
            continue

        network_results = results[network_type]

        # Process each split (train, val, test)
        for split, metrics in network_results.items():
            all_metrics_data.append({
                'Network': network_type.capitalize(),
                'Split': split.capitalize(),
                'Total_Samples': metrics.get('num_samples', 0),

                # Overall Segmentation Metrics
                'Dice_Score': f"{metrics.get('dice_mean', 0):.4f}",
                'IoU_Score': f"{metrics.get('iou_mean', 0):.4f}",
                'Pixel_Accuracy': f"{metrics.get('pixel_accuracy_overall', metrics.get('pixel_accuracy', 0)):.4f}",

                # Per-class Dice Scores
                'Background_Dice': f"{metrics.get('dice_background', metrics.get('dice_class_0', 0)):.4f}",
                'Benign_Dice': f"{metrics.get('dice_benign_glands', metrics.get('dice_class_1', 0)):.4f}",
                'Malignant_Dice': f"{metrics.get('dice_malignant_glands', metrics.get('dice_class_2', 0)):.4f}",
                'PDC_Dice': f"{metrics.get('dice_pdc', metrics.get('dice_class_3', 0)):.4f}",

                # Per-class IoU Scores
                'Background_IoU': f"{metrics.get('iou_background', metrics.get('iou_class_0', 0)):.4f}",
                'Benign_IoU': f"{metrics.get('iou_benign_glands', metrics.get('iou_class_1', 0)):.4f}",
                'Malignant_IoU': f"{metrics.get('iou_malignant_glands', metrics.get('iou_class_2', 0)):.4f}",
                'PDC_IoU': f"{metrics.get('iou_pdc', metrics.get('iou_class_3', 0)):.4f}",

                # Classification Metrics
                'Patch_Accuracy': f"{metrics.get('patch_accuracy', 0):.4f}",
                'Gland_Classification_Acc': f"{metrics.get('gland_classification_accuracy', 0):.4f}",
            })

    # Save as CSV table
    df = pd.DataFrame(all_metrics_data)
    csv_path = comparison_dir / "teacher_student_comprehensive_metrics.csv"
    df.to_csv(csv_path, index=False)

    # Save as formatted text table
    table_path = comparison_dir / "teacher_student_metrics_table.txt"
    with open(table_path, 'w') as f:
        f.write("=" * 140 + "\n")
        f.write("Teacher-Student UNet Comprehensive Evaluation - All Metrics\n")
        f.write("=" * 140 + "\n\n")

        # Create formatted table string
        table_str = df.to_string(index=False, justify='center')
        f.write(table_str)
        f.write("\n\n")

        # Add network-wise summary statistics
        f.write("NETWORK COMPARISON SUMMARY:\n")
        f.write("-" * 60 + "\n")

        for network_type in ['teacher', 'student']:
            if network_type in results:
                network_data = [row for row in all_metrics_data if row['Network'].lower() == network_type]
                if network_data:
                    avg_dice = np.mean([float(row['Dice_Score']) for row in network_data])
                    avg_patch_acc = np.mean([float(row['Patch_Accuracy']) for row in network_data])
                    avg_pixel_acc = np.mean([float(row['Pixel_Accuracy']) for row in network_data])

                    f.write(f"\n{network_type.upper()} NETWORK AVERAGES:\n")
                    f.write(f"  - Average Dice Score:     {avg_dice:.4f}\n")
                    f.write(f"  - Average Patch Accuracy: {avg_patch_acc:.4f}\n")
                    f.write(f"  - Average Pixel Accuracy: {avg_pixel_acc:.4f}\n")

        f.write("\n" + "=" * 140 + "\n")

    # Save as comprehensive markdown report
    md_path = comparison_dir / "teacher_student_comprehensive_report.md"
    with open(md_path, 'w') as f:
        f.write("# Teacher-Student UNet Comprehensive Evaluation Report\n\n")
        f.write(f"**Evaluation Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"**Architecture:** Teacher-Student UNet with Independent Networks\n")
        f.write(f"**Total Networks Evaluated:** {len(results)}\n\n")

        # Overall metrics table
        f.write("## Complete Metrics Overview\n\n")
        f.write("| Network | Split | Samples | Dice | IoU | Pixel Acc | Patch Acc | Benign Dice | Malignant Dice | PDC Dice |\n")
        f.write("|---------|-------|---------|------|-----|-----------|-----------|-------------|----------------|----------|\n")

        for row in all_metrics_data:
            f.write(f"| {row['Network']:7s} | {row['Split']:5s} | {row['Total_Samples']:7d} | "
                   f"{row['Dice_Score']:4s} | {row['IoU_Score']:3s} | {row['Pixel_Accuracy']:9s} | "
                   f"{row['Patch_Accuracy']:9s} | {row['Benign_Dice']:11s} | {row['Malignant_Dice']:14s} | "
                   f"{row['PDC_Dice']:8s} |\n")

        f.write("\n## Per-Class Segmentation Performance\n\n")
        f.write("| Network | Split | Background | Benign Glands | Malignant Glands | PDC |\n")
        f.write("|---------|-------|------------|---------------|------------------|-----|\n")

        for row in all_metrics_data:
            f.write(f"| {row['Network']:7s} | {row['Split']:5s} | {row['Background_Dice']:10s} | "
                   f"{row['Benign_Dice']:13s} | {row['Malignant_Dice']:16s} | {row['PDC_Dice']:3s} |\n")

        f.write("\n## IoU Scores by Class\n\n")
        f.write("| Network | Split | Background IoU | Benign IoU | Malignant IoU | PDC IoU |\n")
        f.write("|---------|-------|----------------|------------|---------------|----------|\n")

        for row in all_metrics_data:
            f.write(f"| {row['Network']:7s} | {row['Split']:5s} | {row['Background_IoU']:14s} | "
                   f"{row['Benign_IoU']:10s} | {row['Malignant_IoU']:13s} | {row['PDC_IoU']:7s} |\n")

        # Network comparison analysis
        f.write("\n## Network Performance Analysis\n\n")

        if 'teacher' in results and 'student' in results:
            # Calculate overall averages for comparison
            teacher_splits = list(results['teacher'].keys())
            student_splits = list(results['student'].keys())
            common_splits = set(teacher_splits) & set(student_splits)

            if common_splits:
                teacher_avg_dice = np.mean([results['teacher'][split]['dice_mean'] for split in common_splits])
                student_avg_dice = np.mean([results['student'][split]['dice_mean'] for split in common_splits])

                teacher_avg_patch = np.mean([results['teacher'][split]['patch_accuracy'] for split in common_splits])
                student_avg_patch = np.mean([results['student'][split]['patch_accuracy'] for split in common_splits])

                f.write("### Overall Performance Comparison\n\n")
                f.write(f"- **Teacher Network Average Dice:** {teacher_avg_dice:.4f}\n")
                f.write(f"- **Student Network Average Dice:** {student_avg_dice:.4f}\n")
                f.write(f"- **Teacher Network Average Patch Accuracy:** {teacher_avg_patch:.4f}\n")
                f.write(f"- **Student Network Average Patch Accuracy:** {student_avg_patch:.4f}\n\n")

                if teacher_avg_dice > student_avg_dice:
                    better_network = "Teacher"
                    dice_diff = teacher_avg_dice - student_avg_dice
                else:
                    better_network = "Student"
                    dice_diff = student_avg_dice - teacher_avg_dice

                f.write(f"### Performance Summary\n\n")
                f.write(f"- **Better Performing Network (Dice):** {better_network} (+{dice_diff:.4f})\n")

                if teacher_avg_patch > student_avg_patch:
                    better_patch_network = "Teacher"
                    patch_diff = teacher_avg_patch - student_avg_patch
                else:
                    better_patch_network = "Student"
                    patch_diff = student_avg_patch - teacher_avg_patch

                f.write(f"- **Better Performing Network (Patch Acc):** {better_patch_network} (+{patch_diff:.4f})\n\n")

        # Per-split detailed comparison
        f.write("### Split-wise Detailed Comparison\n\n")
        for split in ['train', 'val', 'test']:
            if 'teacher' in results and 'student' in results and split in results['teacher'] and split in results['student']:
                teacher_metrics = results['teacher'][split]
                student_metrics = results['student'][split]

                f.write(f"#### {split.capitalize()} Set\n\n")
                f.write(f"| Metric | Teacher | Student | Difference |\n")
                f.write(f"|--------|---------|---------|------------|\n")
                f.write(f"| Dice Score | {teacher_metrics['dice_mean']:.4f} | {student_metrics['dice_mean']:.4f} | {teacher_metrics['dice_mean'] - student_metrics['dice_mean']:+.4f} |\n")
                f.write(f"| Patch Accuracy | {teacher_metrics['patch_accuracy']:.4f} | {student_metrics['patch_accuracy']:.4f} | {teacher_metrics['patch_accuracy'] - student_metrics['patch_accuracy']:+.4f} |\n")
                f.write(f"| Pixel Accuracy | {teacher_metrics['pixel_accuracy']:.4f} | {student_metrics['pixel_accuracy']:.4f} | {teacher_metrics['pixel_accuracy'] - student_metrics['pixel_accuracy']:+.4f} |\n\n")

        f.write("---\n")
        f.write("*Generated by Teacher-Student UNet Evaluation Pipeline*\n")

    # Save detailed JSON with all metrics
    json_path = comparison_dir / "teacher_student_detailed_metrics.json"
    with open(json_path, 'w') as f:
        # Convert numpy types for JSON serialization
        json_results = {}
        for network_type, network_results in results.items():
            json_results[network_type] = {}
            for split, metrics in network_results.items():
                json_results[network_type][split] = {}
                for key, value in metrics.items():
                    if isinstance(value, (np.integer, np.floating)):
                        json_results[network_type][split][key] = float(value)
                    else:
                        json_results[network_type][split][key] = value

        json.dump(json_results, f, indent=2)

    # Log the saved files
    logger = logging.getLogger("TeacherStudentEvaluator")
    logger.info(f"💾 Teacher-Student comprehensive metrics saved:")
    logger.info(f"   📊 CSV table: {csv_path}")
    logger.info(f"   📋 Formatted table: {table_path}")
    logger.info(f"   📝 Markdown report: {md_path}")
    logger.info(f"   🔍 Detailed JSON: {json_path}")


def run_teacher_student_evaluation(
    experiment_dir: str,
    dataset_key: str = "mixed",
    visualization_samples: int = 100,
    device: Optional[torch.device] = None
) -> Dict[str, Dict[str, Any]]:
    """
    Run comprehensive evaluation for Teacher-Student models

    Evaluates both teacher and student networks separately using their best checkpoints

    Args:
        experiment_dir: Path to experiment directory containing checkpoints
        dataset_key: Dataset to evaluate on
        visualization_samples: Number of samples per split for visualization
        device: Device to run evaluation on

    Returns:
        Dictionary containing results for both teacher and student evaluations
    """
    experiment_path = Path(experiment_dir)
    results = {}

    # Setup logging
    logger = logging.getLogger("TeacherStudentEvaluator")
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    logger.info("🎓👨‍🏫 Starting Teacher-Student comprehensive evaluation...")
    logger.info(f"📂 Experiment directory: {experiment_path}")

    # Find checkpoint files
    models_dir = experiment_path / "models"

    teacher_checkpoint = models_dir / "best_teacher_model.pth"
    student_checkpoint = models_dir / "best_student_model.pth"

    # Check if checkpoints exist
    if not teacher_checkpoint.exists():
        logger.error(f"❌ Teacher checkpoint not found: {teacher_checkpoint}")
        raise FileNotFoundError(f"Teacher checkpoint not found: {teacher_checkpoint}")

    if not student_checkpoint.exists():
        logger.error(f"❌ Student checkpoint not found: {student_checkpoint}")
        raise FileNotFoundError(f"Student checkpoint not found: {student_checkpoint}")

    logger.info(f"✅ Found teacher checkpoint: {teacher_checkpoint}")
    logger.info(f"✅ Found student checkpoint: {student_checkpoint}")

    # Evaluate Teacher Network
    logger.info("=" * 80)
    logger.info("👨‍🏫 EVALUATING TEACHER NETWORK")
    logger.info("=" * 80)

    teacher_evaluator = PostTrainingEvaluator(
        model_path=str(teacher_checkpoint),
        dataset_key=dataset_key,
        output_dir=str(experiment_path),
        device=device,
        visualization_samples=visualization_samples,
        architecture="teacher_student_unet",
        teacher_student_mode="teacher"
    )

    results['teacher'] = teacher_evaluator.run_comprehensive_evaluation()

    # Evaluate Student Network
    logger.info("=" * 80)
    logger.info("🎓 EVALUATING STUDENT NETWORK")
    logger.info("=" * 80)

    student_evaluator = PostTrainingEvaluator(
        model_path=str(student_checkpoint),
        dataset_key=dataset_key,
        output_dir=str(experiment_path),
        device=device,
        visualization_samples=visualization_samples,
        architecture="teacher_student_unet",
        teacher_student_mode="student"
    )

    results['student'] = student_evaluator.run_comprehensive_evaluation()

    # Generate comparison report
    logger.info("=" * 80)
    logger.info("📊 TEACHER-STUDENT COMPARISON SUMMARY")
    logger.info("=" * 80)

    comparison_path = experiment_path / "evaluations_comparison"
    comparison_path.mkdir(parents=True, exist_ok=True)

    # Create comparison report
    report_path = comparison_path / "teacher_student_comparison.md"

    with open(report_path, 'w') as f:
        f.write("# Teacher-Student Model Comparison Report\n\n")
        f.write(f"**Experiment:** {experiment_path.name}\n")
        f.write(f"**Dataset:** {dataset_key}\n")
        f.write(f"**Evaluation Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write("## Performance Comparison\n\n")
        f.write("| Split | Network | Pixel Acc | Mean Dice | Patch Acc |\n")
        f.write("|-------|---------|-----------|-----------|----------|\n")

        for split in ['train', 'val', 'test']:
            if split in results['teacher'] and split in results['student']:
                teacher_metrics = results['teacher'][split]
                student_metrics = results['student'][split]

                f.write(f"| {split.capitalize():5s} | Teacher | {teacher_metrics['pixel_accuracy']:9.4f} | "
                       f"{teacher_metrics['dice_mean']:9.4f} | {teacher_metrics['patch_accuracy']:9.4f} |\n")
                f.write(f"| {split.capitalize():5s} | Student | {student_metrics['pixel_accuracy']:9.4f} | "
                       f"{student_metrics['dice_mean']:9.4f} | {student_metrics['patch_accuracy']:9.4f} |\n")

        f.write("\n## Analysis\n\n")

        # Calculate average performance
        teacher_avg_dice = np.mean([results['teacher'][split]['dice_mean'] for split in ['train', 'val', 'test'] if split in results['teacher']])
        student_avg_dice = np.mean([results['student'][split]['dice_mean'] for split in ['train', 'val', 'test'] if split in results['student']])

        if teacher_avg_dice > student_avg_dice:
            better_network = "Teacher"
            performance_diff = teacher_avg_dice - student_avg_dice
        else:
            better_network = "Student"
            performance_diff = student_avg_dice - teacher_avg_dice

        f.write(f"- **Better Performing Network:** {better_network} "
               f"(+{performance_diff:.4f} average Dice score improvement)\n")
        f.write(f"- **Teacher Average Dice:** {teacher_avg_dice:.4f}\n")
        f.write(f"- **Student Average Dice:** {student_avg_dice:.4f}\n")

    logger.info(f"📋 Comparison report saved: {report_path}")
    logger.info(f"👨‍🏫 Teacher avg dice: {teacher_avg_dice:.4f}")
    logger.info(f"🎓 Student avg dice: {student_avg_dice:.4f}")
    logger.info(f"🏆 Better network: {better_network} (+{performance_diff:.4f})")

    # Save comprehensive metrics in table and markdown formats
    logger.info("=" * 80)
    logger.info("💾 SAVING COMPREHENSIVE METRICS")
    logger.info("=" * 80)
    save_teacher_student_comprehensive_metrics(results, experiment_path)

    logger.info("🎉 Teacher-Student evaluation completed!")
    logger.info(f"📁 Results saved to:")
    logger.info(f"   👨‍🏫 Teacher: {experiment_path}/evaluations_teacher/")
    logger.info(f"   🎓 Student: {experiment_path}/evaluations_student/")
    logger.info(f"   📊 Comparison: {comparison_path}/")

    return results


def main():
    """Main entry point for standalone evaluation"""
    import argparse

    parser = argparse.ArgumentParser(description="Post-Training Evaluation")
    parser.add_argument("--model", type=str,
                       help="Path to best model checkpoint (for single model evaluation)")
    parser.add_argument("--experiment", type=str,
                       help="Path to experiment directory (for teacher-student evaluation)")
    parser.add_argument("--architecture", type=str, default="nnunet",
                       choices=["baseline_unet", "nnunet", "teacher_student_unet"],
                       help="Model architecture")
    parser.add_argument("--dataset", type=str, default="mixed",
                       help="Dataset key")
    parser.add_argument("--output", type=str,
                       help="Output directory (for single model evaluation)")
    parser.add_argument("--samples", type=int, default=100,
                       help="Number of visualization samples per split")

    args = parser.parse_args()

    if args.architecture == "teacher_student_unet":
        # Teacher-Student evaluation
        if not args.experiment:
            raise ValueError("--experiment directory is required for teacher_student_unet architecture")

        results = run_teacher_student_evaluation(
            experiment_dir=args.experiment,
            dataset_key=args.dataset,
            visualization_samples=args.samples
        )

        self.logger.info("🎉 Teacher-Student evaluation completed!")
        for network in ['teacher', 'student']:
            for split, metrics in results[network].items():
                self.logger.info(f"{network.capitalize()} {split.capitalize()}: "
                      f"Dice={metrics['dice_mean']:.4f}, "
                      f"Patch Acc={metrics['patch_accuracy']:.4f}")
    else:
        # Single model evaluation
        if not args.model or not args.output:
            raise ValueError("--model and --output are required for single model evaluation")

        evaluator = PostTrainingEvaluator(
            model_path=args.model,
            dataset_key=args.dataset,
            output_dir=args.output,
            visualization_samples=args.samples,
            architecture=args.architecture
        )

        results = evaluator.run_comprehensive_evaluation()

        evaluator.logger.info("🎉 Post-training evaluation completed!")
        for split, metrics in results.items():
            evaluator.logger.info(f"{split.capitalize()}: Dice={metrics['dice_mean']:.4f}, "
                  f"Patch Acc={metrics['patch_accuracy']:.4f}")


if __name__ == "__main__":
    main()