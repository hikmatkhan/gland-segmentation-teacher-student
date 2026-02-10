#!/usr/bin/env python3
"""
4-Class nnU-Net Multi-Task Evaluator for Combined Gland Segmentation
===================================================================

Comprehensive evaluation pipeline for 4-class gland segmentation with multi-label classification.
Generates detailed metrics, visualizations, and reports.

Features:
- 4-class segmentation evaluation with per-class metrics
- Multi-label patch classification evaluation
- Single-label gland classification evaluation
- Rich composite visualizations
- Detailed metric reports and confusion matrices
- Export capabilities for research presentations

Author: Claude Code - Generated for OSU CRC Research
Date: 2025-09-16
"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, roc_curve, precision_recall_curve,
    average_precision_score, multilabel_confusion_matrix
)
from sklearn.preprocessing import label_binarize
import cv2
from tqdm import tqdm
import pandas as pd

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from configs.paths_config import get_dataset_path, EVALUATION_CONFIG
from src.models.multi_task_wrapper import create_multitask_model
from src.models.metrics import SegmentationMetrics
from src.training.dataset import create_combined_data_loaders


class FourClassMultiTaskEvaluator:
    """
    Comprehensive evaluator for 4-class multi-task gland segmentation

    Evaluates:
    - 4-class segmentation: Background(0), Benign(1), Malignant(2), PDC(3)
    - Multi-label patch classification
    - Single-label gland classification
    - Generates rich visualizations and detailed reports
    """

    def __init__(
        self,
        model_path: str,
        dataset_key: str = "mixed",
        output_dir: Optional[str] = None,
        device: Optional[torch.device] = None
    ):
        """
        Initialize the evaluator

        Args:
            model_path: Path to trained model checkpoint
            dataset_key: Dataset to evaluate on
            output_dir: Output directory for results
            device: Device to run evaluation on
        """
        self.model_path = Path(model_path)
        self.dataset_key = dataset_key
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Setup output directory
        if output_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = f"outputs/evaluation_{dataset_key}_{timestamp}"

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Setup subdirectories
        self.metrics_dir = self.output_dir / "metrics"
        self.visualizations_dir = self.output_dir / "visualizations"
        self.reports_dir = self.output_dir / "reports"

        for dir_path in [self.metrics_dir, self.visualizations_dir, self.reports_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

        # Setup logging
        self.setup_logging()

        # Class information
        self.class_names = ['Background', 'Benign', 'Malignant', 'PDC']
        self.class_colors = [
            [0, 0, 0],        # Background - Black
            [0, 255, 0],      # Benign - Green
            [255, 0, 0],      # Malignant - Red
            [255, 165, 0]     # PDC - Orange
        ]

        # Initialize segmentation metrics calculator
        self.metrics_calculator = SegmentationMetrics(num_classes=4, ignore_background=True)

        # Initialize components
        self.model = None
        self.test_loader = None
        self.checkpoint = None

        self.logger.info(f"🔬 Evaluator initialized:")
        self.logger.info(f"   📂 Model: {self.model_path}")
        self.logger.info(f"   📊 Dataset: {self.dataset_key}")
        self.logger.info(f"   📱 Device: {self.device}")
        self.logger.info(f"   📁 Output: {self.output_dir}")

    def setup_logging(self):
        """Setup logging configuration"""
        log_file = self.output_dir / "evaluation.log"

        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def load_model(self):
        """Load trained model from checkpoint"""
        self.logger.info("📂 Loading trained model...")

        if not self.model_path.exists():
            raise FileNotFoundError(f"Model checkpoint not found: {self.model_path}")

        # Load checkpoint with PyTorch 2.6 compatibility
        try:
            self.checkpoint = torch.load(self.model_path, map_location='cpu', weights_only=True)
        except Exception as e:
            self.logger.warning(f"⚠️ Secure loading failed, using compatibility mode: {str(e)[:100]}...")
            self.checkpoint = torch.load(self.model_path, map_location='cpu', weights_only=False)

        # Create model
        self.model = create_multitask_model()
        self.model.load_state_dict(self.checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()

        self.logger.info("✅ Model loaded successfully")

        # Log training information
        if 'config' in self.checkpoint:
            config = self.checkpoint['config']
            self.logger.info(f"📋 Training config:")
            self.logger.info(f"   🔄 Epochs: {config.get('epochs', 'N/A')}")
            self.logger.info(f"   📦 Batch size: {config.get('batch_size', 'N/A')}")
            self.logger.info(f"   📈 Learning rate: {config.get('learning_rate', 'N/A')}")

        if 'best_metrics' in self.checkpoint:
            best_metrics = self.checkpoint['best_metrics']
            self.logger.info(f"🏆 Best training metrics:")
            for key, value in best_metrics.items():
                if isinstance(value, (int, float)):
                    self.logger.info(f"   {key}: {value:.4f}")

    def prepare_data(self):
        """Prepare test data loader"""
        self.logger.info("📊 Preparing test data...")

        dataset_path = get_dataset_path(self.dataset_key)

        # Create data loaders
        loaders = create_combined_data_loaders(
            dataset_key=self.dataset_key,
            batch_size=EVALUATION_CONFIG.get('batch_size', 1),  # Use batch size 1 for detailed evaluation
            num_workers=EVALUATION_CONFIG.get('num_workers', 4),
            use_multilabel_patch=True,
            disable_augmentation=True  # No augmentation for evaluation
        )

        self.test_loader = loaders['test']
        self.logger.info(f"✅ Test data prepared: {len(self.test_loader)} batches")

    def calculate_segmentation_metrics(
        self,
        predictions: np.ndarray,
        targets: np.ndarray
    ) -> Dict[str, float]:
        """
        Calculate comprehensive segmentation metrics using both legacy and new methods

        Args:
            predictions: Predicted segmentation masks [N, H, W]
            targets: Ground truth segmentation masks [N, H, W]

        Returns:
            Dictionary of metrics
        """
        # NEW: Use comprehensive metrics calculator
        # Convert to tensors for the metrics calculator (taking first sample for batch processing)
        if len(predictions.shape) == 3:  # [N, H, W]
            # Process each sample and average results
            comprehensive_metrics_list = []
            for i in range(predictions.shape[0]):
                pred_tensor = torch.from_numpy(predictions[i]).long()
                target_tensor = torch.from_numpy(targets[i]).long()
                sample_metrics = self.metrics_calculator.compute_all_metrics(pred_tensor, target_tensor)
                comprehensive_metrics_list.append(sample_metrics)

            # Average metrics across all samples
            comprehensive_metrics = {}
            for key in comprehensive_metrics_list[0].keys():
                comprehensive_metrics[key] = np.mean([m[key] for m in comprehensive_metrics_list])
        else:  # [H, W] single image
            pred_tensor = torch.from_numpy(predictions).long()
            target_tensor = torch.from_numpy(targets).long()
            comprehensive_metrics = self.metrics_calculator.compute_all_metrics(pred_tensor, target_tensor)

        # Start with comprehensive metrics
        metrics = comprehensive_metrics.copy()

        num_classes = len(self.class_names)

        # LEGACY: Keep original calculations for backward compatibility
        # Overall pixel accuracy
        pixel_accuracy = (predictions == targets).mean()
        metrics['pixel_accuracy_legacy'] = pixel_accuracy

        # Per-class metrics
        dice_scores = []
        iou_scores = []

        for class_idx in range(num_classes):
            pred_mask = (predictions == class_idx)
            true_mask = (targets == class_idx)

            # Dice coefficient
            intersection = (pred_mask & true_mask).sum()
            union = pred_mask.sum() + true_mask.sum()

            if union > 0:
                dice = (2.0 * intersection) / union
                dice_scores.append(dice)
            else:
                dice_scores.append(1.0)  # Perfect if no pixels of this class

            # IoU
            intersection = (pred_mask & true_mask).sum()
            union_iou = (pred_mask | true_mask).sum()

            if union_iou > 0:
                iou = intersection / union_iou
                iou_scores.append(iou)
            else:
                iou_scores.append(1.0)

            # Store per-class metrics
            metrics[f'dice_class_{class_idx}_{self.class_names[class_idx]}'] = dice_scores[-1]
            metrics[f'iou_class_{class_idx}_{self.class_names[class_idx]}'] = iou_scores[-1]

        # Mean metrics
        metrics['dice_mean'] = np.mean(dice_scores)
        metrics['iou_mean'] = np.mean(iou_scores)

        # Background vs glandular tissue
        bg_mask_pred = (predictions == 0)
        bg_mask_true = (targets == 0)
        gland_mask_pred = (predictions > 0)
        gland_mask_true = (targets > 0)

        # Background dice
        bg_intersection = (bg_mask_pred & bg_mask_true).sum()
        bg_union = bg_mask_pred.sum() + bg_mask_true.sum()
        if bg_union > 0:
            metrics['dice_background'] = (2.0 * bg_intersection) / bg_union
        else:
            metrics['dice_background'] = 1.0

        # Glandular tissue dice
        gland_intersection = (gland_mask_pred & gland_mask_true).sum()
        gland_union = gland_mask_pred.sum() + gland_mask_true.sum()
        if gland_union > 0:
            metrics['dice_glandular'] = (2.0 * gland_intersection) / gland_union
        else:
            metrics['dice_glandular'] = 1.0

        return metrics

    def calculate_classification_metrics(
        self,
        patch_predictions: np.ndarray,
        patch_targets: np.ndarray,
        is_multilabel: bool = True
    ) -> Dict[str, Any]:
        """
        Calculate classification metrics

        Args:
            patch_predictions: Patch predictions [N, num_classes] for multilabel or [N] for single-label
            patch_targets: Patch targets [N, num_classes] for multilabel or [N] for single-label
            is_multilabel: Whether this is multi-label classification

        Returns:
            Dictionary of metrics and reports
        """
        metrics = {}

        if is_multilabel:
            # Multi-label classification metrics
            # Convert predictions to binary
            patch_pred_binary = (patch_predictions > 0.5).astype(int)

            # Exact match accuracy (all labels must match)
            exact_match = (patch_pred_binary == patch_targets).all(axis=1).mean()
            metrics['exact_match_accuracy'] = exact_match

            # Per-class accuracy
            class_accuracies = []
            class_f1_scores = []
            class_auc_scores = []

            for class_idx in range(patch_targets.shape[1]):
                class_acc = (patch_pred_binary[:, class_idx] == patch_targets[:, class_idx]).mean()
                class_accuracies.append(class_acc)

                # F1 score for this class
                from sklearn.metrics import f1_score
                f1 = f1_score(patch_targets[:, class_idx], patch_pred_binary[:, class_idx])
                class_f1_scores.append(f1)

                # AUC for this class
                if len(np.unique(patch_targets[:, class_idx])) > 1:
                    auc = roc_auc_score(patch_targets[:, class_idx], patch_predictions[:, class_idx])
                    class_auc_scores.append(auc)
                else:
                    class_auc_scores.append(1.0)

                metrics[f'class_{class_idx}_{self.class_names[class_idx]}_accuracy'] = class_acc
                metrics[f'class_{class_idx}_{self.class_names[class_idx]}_f1'] = f1
                metrics[f'class_{class_idx}_{self.class_names[class_idx]}_auc'] = class_auc_scores[-1]

            metrics['mean_class_accuracy'] = np.mean(class_accuracies)
            metrics['mean_f1_score'] = np.mean(class_f1_scores)
            metrics['mean_auc_score'] = np.mean(class_auc_scores)

            # Hamming loss (fraction of wrong labels)
            from sklearn.metrics import hamming_loss
            metrics['hamming_loss'] = hamming_loss(patch_targets, patch_pred_binary)

            # Multi-label confusion matrices
            confusion_matrices = multilabel_confusion_matrix(patch_targets, patch_pred_binary)
            metrics['confusion_matrices'] = confusion_matrices

        else:
            # Single-label classification metrics
            pred_labels = np.argmax(patch_predictions, axis=1)

            accuracy = (pred_labels == patch_targets).mean()
            metrics['accuracy'] = accuracy

            # Classification report
            report = classification_report(
                patch_targets, pred_labels,
                target_names=self.class_names,
                output_dict=True,
                zero_division=0
            )
            metrics['classification_report'] = report

            # Confusion matrix
            cm = confusion_matrix(patch_targets, pred_labels)
            metrics['confusion_matrix'] = cm

            # Multi-class AUC (one-vs-rest)
            try:
                # Binarize targets for multi-class AUC
                patch_targets_bin = label_binarize(patch_targets, classes=range(len(self.class_names)))
                if patch_targets_bin.shape[1] == 1:  # Binary case
                    patch_targets_bin = np.hstack([1 - patch_targets_bin, patch_targets_bin])

                auc_scores = []
                for i in range(len(self.class_names)):
                    if len(np.unique(patch_targets_bin[:, i])) > 1:
                        auc = roc_auc_score(patch_targets_bin[:, i], patch_predictions[:, i])
                        auc_scores.append(auc)
                    else:
                        auc_scores.append(1.0)

                metrics['per_class_auc'] = auc_scores
                metrics['mean_auc'] = np.mean(auc_scores)
            except Exception as e:
                self.logger.warning(f"Could not calculate AUC: {e}")

        return metrics

    def create_segmentation_overlay(
        self,
        image: np.ndarray,
        prediction: np.ndarray,
        target: np.ndarray,
        alpha: float = 0.6
    ) -> np.ndarray:
        """
        Create overlay visualization for segmentation

        Args:
            image: Original image [H, W, 3]
            prediction: Predicted mask [H, W]
            target: Ground truth mask [H, W]
            alpha: Overlay transparency

        Returns:
            Composite overlay image [H, W, 3]
        """
        h, w = image.shape[:2]

        # Create colored masks
        pred_colored = np.zeros((h, w, 3), dtype=np.uint8)
        target_colored = np.zeros((h, w, 3), dtype=np.uint8)

        for class_idx, color in enumerate(self.class_colors):
            pred_colored[prediction == class_idx] = color
            target_colored[target == class_idx] = color

        # Create composite image
        # Left half: prediction overlay, Right half: ground truth overlay
        composite = np.zeros((h, w * 2, 3), dtype=np.uint8)

        # Prediction side
        img_left = (image * (1 - alpha) + pred_colored * alpha).astype(np.uint8)
        composite[:, :w] = img_left

        # Ground truth side
        img_right = (image * (1 - alpha) + target_colored * alpha).astype(np.uint8)
        composite[:, w:] = img_right

        return composite

    def create_comprehensive_visualization(
        self,
        image: np.ndarray,
        seg_pred: np.ndarray,
        seg_target: np.ndarray,
        patch_pred: np.ndarray,
        patch_target: np.ndarray,
        sample_idx: int
    ) -> np.ndarray:
        """
        Create comprehensive visualization showing all predictions

        Args:
            image: Original image [H, W, 3]
            seg_pred: Segmentation prediction [H, W]
            seg_target: Segmentation target [H, W]
            patch_pred: Patch prediction probabilities [num_classes]
            patch_target: Patch target labels [num_classes]
            sample_idx: Sample index for labeling

        Returns:
            Composite visualization image
        """
        h, w = image.shape[:2]

        # Create segmentation overlay
        seg_overlay = self.create_segmentation_overlay(image, seg_pred, seg_target)

        # Create info panel
        info_height = 200
        info_panel = np.ones((info_height, w * 2, 3), dtype=np.uint8) * 255

        # Add text information
        y_pos = 30
        cv2.putText(info_panel, f"Sample {sample_idx}", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

        y_pos += 40
        cv2.putText(info_panel, "Patch Classification:", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

        y_pos += 25
        for i, (class_name, pred_prob, target_val) in enumerate(
            zip(self.class_names, patch_pred, patch_target)
        ):
            color = (0, 150, 0) if (pred_prob > 0.5) == target_val else (0, 0, 150)
            text = f"{class_name}: {pred_prob:.3f} (GT: {int(target_val)})"
            cv2.putText(info_panel, text, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            y_pos += 20

        # Combine segmentation and info panel
        composite = np.vstack([seg_overlay, info_panel])

        return composite

    def plot_confusion_matrices(self, metrics: Dict[str, Any]):
        """Plot confusion matrices for segmentation and classification"""

        # Segmentation confusion matrix
        if 'segmentation_confusion_matrix' in metrics:
            plt.figure(figsize=(10, 8))
            cm = metrics['segmentation_confusion_matrix']
            sns.heatmap(
                cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=self.class_names,
                yticklabels=self.class_names
            )
            plt.title('4-Class Segmentation Confusion Matrix')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            plt.tight_layout()
            plt.savefig(self.visualizations_dir / 'segmentation_confusion_matrix.png', dpi=300)
            plt.close()

        # Multi-label classification confusion matrices
        if 'patch_classification_confusion_matrices' in metrics:
            confusion_matrices = metrics['patch_classification_confusion_matrices']
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            axes = axes.flatten()

            for i, (cm, class_name) in enumerate(zip(confusion_matrices, self.class_names)):
                sns.heatmap(
                    cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Absent', 'Present'],
                    yticklabels=['Absent', 'Present'],
                    ax=axes[i]
                )
                axes[i].set_title(f'{class_name} Classification')
                axes[i].set_ylabel('True')
                axes[i].set_xlabel('Predicted')

            plt.tight_layout()
            plt.savefig(self.visualizations_dir / 'patch_classification_confusion_matrices.png', dpi=300)
            plt.close()

    def plot_metrics_summary(self, metrics: Dict[str, float]):
        """Plot summary of key metrics"""
        # Extract key metrics for plotting
        seg_metrics = {
            'Mean Dice': metrics.get('dice_mean', 0),
            'Mean IoU': metrics.get('iou_mean', 0),
            'Pixel Accuracy': metrics.get('pixel_accuracy', 0)
        }

        patch_metrics = {
            'Exact Match': metrics.get('exact_match_accuracy', 0),
            'Mean F1': metrics.get('mean_f1_score', 0),
            'Mean AUC': metrics.get('mean_auc_score', 0)
        }

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Segmentation metrics
        bars1 = ax1.bar(seg_metrics.keys(), seg_metrics.values(), color=['skyblue', 'lightgreen', 'lightcoral'])
        ax1.set_title('Segmentation Metrics')
        ax1.set_ylabel('Score')
        ax1.set_ylim(0, 1)
        for bar, value in zip(bars1, seg_metrics.values()):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{value:.3f}', ha='center', va='bottom')

        # Classification metrics
        bars2 = ax2.bar(patch_metrics.keys(), patch_metrics.values(), color=['orange', 'pink', 'lightblue'])
        ax2.set_title('Multi-Label Patch Classification Metrics')
        ax2.set_ylabel('Score')
        ax2.set_ylim(0, 1)
        for bar, value in zip(bars2, patch_metrics.values()):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{value:.3f}', ha='center', va='bottom')

        plt.tight_layout()
        plt.savefig(self.visualizations_dir / 'metrics_summary.png', dpi=300)
        plt.close()

    def evaluate(self) -> Dict[str, Any]:
        """
        Run comprehensive evaluation

        Returns:
            Dictionary containing all evaluation results
        """
        self.logger.info("🔬 Starting comprehensive evaluation...")

        # Load model and data
        self.load_model()
        self.prepare_data()

        # Initialize result storage
        all_seg_predictions = []
        all_seg_targets = []
        all_patch_predictions = []
        all_patch_targets = []
        all_images = []

        # Evaluation loop
        self.logger.info("🧪 Running inference on test set...")

        with torch.no_grad():
            for batch_idx, batch_data in enumerate(tqdm(self.test_loader, desc="Evaluating")):
                # Move data to device
                images = batch_data['image'].to(self.device)
                seg_masks = batch_data['segmentation'].to(self.device)
                patch_labels = batch_data['patch_labels'].to(self.device)

                # Forward pass
                outputs = self.model(images)

                # Get predictions
                seg_logits = outputs['segmentation']
                patch_logits = outputs['patch_classification']

                # Convert to predictions
                seg_pred = torch.argmax(seg_logits, dim=1).cpu().numpy()
                patch_pred = torch.sigmoid(patch_logits).cpu().numpy()  # Multi-label probabilities

                # Store results
                all_seg_predictions.append(seg_pred)
                all_seg_targets.append(seg_masks.cpu().numpy())
                all_patch_predictions.append(patch_pred)
                all_patch_targets.append(patch_labels.cpu().numpy())
                all_images.append(images.cpu().numpy())

                # Limit visualization samples
                if batch_idx < 20:  # Save first 20 samples for visualization
                    for i in range(images.shape[0]):
                        # Create comprehensive visualization
                        img = (images[i].cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
                        vis = self.create_comprehensive_visualization(
                            img, seg_pred[i], seg_masks[i].cpu().numpy(),
                            patch_pred[i], patch_labels[i].cpu().numpy(), batch_idx * images.shape[0] + i
                        )

                        # Save visualization
                        vis_path = self.visualizations_dir / f"sample_{batch_idx:03d}_{i:02d}.png"
                        cv2.imwrite(str(vis_path), cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))

        # Concatenate all results
        all_seg_predictions = np.concatenate(all_seg_predictions, axis=0)
        all_seg_targets = np.concatenate(all_seg_targets, axis=0)
        all_patch_predictions = np.concatenate(all_patch_predictions, axis=0)
        all_patch_targets = np.concatenate(all_patch_targets, axis=0)

        self.logger.info(f"✅ Inference completed on {len(all_seg_predictions)} samples")

        # Calculate metrics
        self.logger.info("📊 Calculating comprehensive metrics...")

        # Segmentation metrics
        seg_metrics = self.calculate_segmentation_metrics(all_seg_predictions, all_seg_targets)

        # Segmentation confusion matrix
        seg_cm = confusion_matrix(
            all_seg_targets.flatten(),
            all_seg_predictions.flatten(),
            labels=list(range(len(self.class_names)))
        )
        seg_metrics['segmentation_confusion_matrix'] = seg_cm

        # Patch classification metrics (multi-label)
        patch_metrics = self.calculate_classification_metrics(
            all_patch_predictions, all_patch_targets, is_multilabel=True
        )

        # Combine all metrics
        all_metrics = {**seg_metrics, **patch_metrics}
        all_metrics['num_samples'] = len(all_seg_predictions)

        # Create visualizations
        self.logger.info("📈 Creating visualizations...")
        self.plot_confusion_matrices(all_metrics)
        self.plot_metrics_summary(all_metrics)

        # Save detailed metrics
        self.logger.info("💾 Saving results...")

        # Save metrics as JSON
        metrics_file = self.metrics_dir / "evaluation_metrics.json"
        with open(metrics_file, 'w') as f:
            # Convert numpy arrays to lists for JSON serialization
            json_metrics = {}
            for key, value in all_metrics.items():
                if isinstance(value, np.ndarray):
                    json_metrics[key] = value.tolist()
                elif isinstance(value, (np.integer, np.floating)):
                    json_metrics[key] = float(value)
                else:
                    json_metrics[key] = value

            json.dump(json_metrics, f, indent=2)

        # Create summary report
        self.create_summary_report(all_metrics)

        self.logger.info("🎉 Evaluation completed successfully!")
        self.logger.info(f"📁 Results saved to: {self.output_dir}")

        return all_metrics

    def create_summary_report(self, metrics: Dict[str, Any]):
        """Create a comprehensive summary report"""
        report_path = self.reports_dir / "evaluation_summary.md"

        with open(report_path, 'w') as f:
            f.write("# 4-Class nnU-Net Multi-Task Evaluation Report\n\n")
            f.write(f"**Dataset:** {self.dataset_key}\n")
            f.write(f"**Model:** {self.model_path.name}\n")
            f.write(f"**Evaluation Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"**Number of Samples:** {metrics.get('num_samples', 'N/A')}\n\n")

            f.write("## Segmentation Results\n\n")
            f.write("### Overall Metrics\n")
            f.write(f"- **Mean Dice Coefficient:** {metrics.get('dice_mean', 0):.4f}\n")
            f.write(f"- **Mean IoU:** {metrics.get('iou_mean', 0):.4f}\n")
            f.write(f"- **Pixel Accuracy:** {metrics.get('pixel_accuracy', 0):.4f}\n\n")

            f.write("### Per-Class Segmentation Metrics\n")
            for i, class_name in enumerate(self.class_names):
                dice_key = f'dice_class_{i}_{class_name}'
                iou_key = f'iou_class_{i}_{class_name}'
                f.write(f"- **{class_name}:**\n")
                f.write(f"  - Dice: {metrics.get(dice_key, 0):.4f}\n")
                f.write(f"  - IoU: {metrics.get(iou_key, 0):.4f}\n")

            f.write("\n## Multi-Label Patch Classification Results\n\n")
            f.write("### Overall Metrics\n")
            f.write(f"- **Exact Match Accuracy:** {metrics.get('exact_match_accuracy', 0):.4f}\n")
            f.write(f"- **Mean F1 Score:** {metrics.get('mean_f1_score', 0):.4f}\n")
            f.write(f"- **Mean AUC:** {metrics.get('mean_auc_score', 0):.4f}\n")
            f.write(f"- **Hamming Loss:** {metrics.get('hamming_loss', 0):.4f}\n\n")

            f.write("### Per-Class Classification Metrics\n")
            for i, class_name in enumerate(self.class_names):
                acc_key = f'class_{i}_{class_name}_accuracy'
                f1_key = f'class_{i}_{class_name}_f1'
                auc_key = f'class_{i}_{class_name}_auc'
                f.write(f"- **{class_name}:**\n")
                f.write(f"  - Accuracy: {metrics.get(acc_key, 0):.4f}\n")
                f.write(f"  - F1 Score: {metrics.get(f1_key, 0):.4f}\n")
                f.write(f"  - AUC: {metrics.get(auc_key, 0):.4f}\n")

            f.write("\n## Key Findings\n\n")

            # Analyze results
            best_seg_class = max(self.class_names, key=lambda c: metrics.get(f'dice_class_{self.class_names.index(c)}_{c}', 0))
            worst_seg_class = min(self.class_names, key=lambda c: metrics.get(f'dice_class_{self.class_names.index(c)}_{c}', 0))

            f.write(f"- **Best Segmentation Performance:** {best_seg_class}\n")
            f.write(f"- **Most Challenging Segmentation:** {worst_seg_class}\n")

            if metrics.get('dice_mean', 0) > 0.8:
                f.write("- **Overall Assessment:** Excellent segmentation performance\n")
            elif metrics.get('dice_mean', 0) > 0.7:
                f.write("- **Overall Assessment:** Good segmentation performance\n")
            else:
                f.write("- **Overall Assessment:** Room for improvement in segmentation\n")

            if metrics.get('exact_match_accuracy', 0) > 0.8:
                f.write("- **Multi-Label Classification:** Strong multi-label prediction capability\n")
            elif metrics.get('exact_match_accuracy', 0) > 0.6:
                f.write("- **Multi-Label Classification:** Moderate multi-label prediction capability\n")
            else:
                f.write("- **Multi-Label Classification:** Challenging multi-label scenarios need attention\n")

        self.logger.info(f"📋 Summary report saved: {report_path}")


def main():
    """Main entry point for evaluation"""
    import argparse

    parser = argparse.ArgumentParser(description="4-Class nnU-Net Multi-Task Evaluation")
    parser.add_argument("--model", type=str, required=True,
                       help="Path to trained model checkpoint")
    parser.add_argument("--dataset", type=str, default="mixed",
                       choices=["mixed", "mag5x", "mag10x", "mag20x", "mag40x", "warwick"],
                       help="Dataset to evaluate on")
    parser.add_argument("--output", type=str, default=None,
                       help="Output directory for evaluation results")

    args = parser.parse_args()

    # Create evaluator
    evaluator = FourClassMultiTaskEvaluator(
        model_path=args.model,
        dataset_key=args.dataset,
        output_dir=args.output
    )

    # Run evaluation
    results = evaluator.evaluate()

    print(f"\n🎉 Evaluation completed!")
    print(f"📊 Key Results:")
    print(f"   Segmentation Dice: {results.get('dice_mean', 0):.4f}")
    print(f"   Multi-Label Accuracy: {results.get('exact_match_accuracy', 0):.4f}")
    print(f"   Mean F1 Score: {results.get('mean_f1_score', 0):.4f}")


if __name__ == "__main__":
    main()