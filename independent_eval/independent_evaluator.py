#!/usr/bin/env python3
"""
Independent Model Evaluator Script for GlandSegModels nnU-Net

This script loads a trained model from an experiment directory and evaluates it on
training, validation, and test datasets. It generates comprehensive evaluation metrics and
visualizations similar to the main training pipeline.

Usage:
    python independent_evaluator.py --experiment_path <path> --architecture <arch> --dataset_key <key> --dataset_base_dir <dir> --output <path>

Features:
- Loads best model from experiment directory (baseline_unet, nnunet, and teacher_student_unet architectures)
- Evaluates on train/val/test datasets with 4-class segmentation + multi-task classification
- Generates comprehensive metrics (Dice, IoU, Pixel Accuracy) using existing metrics system
- Creates 100 random sample visualizations from each dataset split
- Saves all results in timestamped output directory
- Supports all 5 dataset configurations (mixed, mag5x, mag10x, mag20x, mag40x)
"""

import sys
import json
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
import argparse
from typing import Dict, List
from tqdm import tqdm
import random

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

# Import from GlandSegModels
from src.models.metrics import SegmentationMetrics
from src.training.dataset import create_combined_data_loaders
from src.models.multi_task_wrapper import create_multitask_model
from torch.utils.data import DataLoader


class IndependentModelEvaluator:
    """Independent model evaluator for comprehensive 4-class gland segmentation evaluation and visualization"""

    def __init__(self, experiment_path: str, architecture: str, dataset_key: str,
                 dataset_base_dir: str, output_path: str, split: str = "all",
                 num_samples: int = 100, batch_size: int = 4):
        """
        Initialize the evaluator

        Args:
            experiment_path: Path to trained experiment directory
            architecture: Model architecture ('baseline_unet', 'nnunet', or 'teacher_student_unet')
            dataset_key: Dataset configuration ('mixed', 'mag5x', 'mag10x', 'mag20x', 'mag40x')
            dataset_base_dir: Base directory containing nnUNetCombined datasets
            output_path: Path to save evaluation results
            split: Dataset split(s) to evaluate ('train', 'val', 'test', 'all')
            num_samples: Number of samples per split for visualization
            batch_size: Evaluation batch size
        """
        self.experiment_path = Path(experiment_path)
        self.architecture = architecture
        self.dataset_key = dataset_key
        self.dataset_base_dir = Path(dataset_base_dir)
        self.split = split
        self.num_samples = num_samples
        self.batch_size = batch_size

        # Derive dataset path from base_dir and key
        self.dataset_path = self._get_dataset_path()

        # Create timestamped output directory
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.eval_output_dir = Path(output_path) / f"evaluation_{self.architecture}_{self.dataset_key}_{timestamp}"
        self.eval_output_dir.mkdir(parents=True, exist_ok=True)

        # Create subdirectories
        self.viz_dir = self.eval_output_dir / "visualizations"
        self.viz_dir.mkdir(exist_ok=True)

        # Device setup
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🖥️ Using device: {self.device}")

        # Validate experiment structure
        self._validate_experiment_structure()

        # Load configuration
        self.config = self._load_config()

        # Initialize model
        self.model = self._load_model()

        # Initialize metrics calculator
        self.metrics_calculator = SegmentationMetrics(num_classes=4, ignore_background=True)

        print(f"📁 Evaluation output directory: {self.eval_output_dir}")

    def _get_dataset_path(self) -> Path:
        """Get dataset path based on dataset_key"""
        dataset_mapping = {
            'mixed': 'Task001_Combined_Mixed_Magnifications',
            'mag5x': 'Task005_Combined_Mag5x',
            'mag10x': 'Task010_Combined_Mag10x',
            'mag20x': 'Task020_Combined_Mag20x',
            'mag40x': 'Task040_Combined_Mag40x'
        }

        if self.dataset_key not in dataset_mapping:
            raise ValueError(f"Unknown dataset key: {self.dataset_key}. Choose from: {list(dataset_mapping.keys())}")

        dataset_dir = self.dataset_base_dir / dataset_mapping[self.dataset_key]

        if not dataset_dir.exists():
            raise FileNotFoundError(f"Dataset not found: {dataset_dir}")

        return dataset_dir

    def _validate_experiment_structure(self):
        """Validate that the experiment directory has the required structure"""

        print(f"🔍 Validating experiment structure at: {self.experiment_path}")

        # Check if experiment directory exists
        if not self.experiment_path.exists():
            raise FileNotFoundError(f"Experiment directory does not exist: {self.experiment_path}")

        # Check for models directory
        models_dir = self.experiment_path / "models"
        if not models_dir.exists():
            raise FileNotFoundError(f"Models directory not found: {models_dir}")

        # Check for best model checkpoint based on architecture
        if self.architecture == "teacher_student_unet":
            # For Teacher-Student UNet, check for specific model files
            required_files = {
                "best_student_model.pth": models_dir / "best_student_model.pth",
                "best_teacher_model.pth": models_dir / "best_teacher_model.pth",
                "best_model.pth": models_dir / "best_model.pth"
            }

            missing_files = []
            for file_name, file_path in required_files.items():
                if not file_path.exists():
                    missing_files.append(file_name)

            if missing_files:
                available_files = list(models_dir.glob("*.pth"))
                print(f"❌ Teacher-Student UNet missing model files: {', '.join(missing_files)}")
                print(f"📂 Available model files in {models_dir}:")
                for file in available_files:
                    print(f"   - {file.name}")
                raise FileNotFoundError(f"Teacher-Student UNet requires: {', '.join(required_files.keys())}")

            print(f"✅ Found all required Teacher-Student UNet model files")
        else:
            # For baseline_unet and nnunet, check for standard best_model.pth
            best_model_path = models_dir / "best_model.pth"
            if not best_model_path.exists():
                # Also check for alternative naming
                alternative_paths = [
                    models_dir / "model_best.pth",
                    models_dir / "best.pth",
                    models_dir / "checkpoint_best.pth"
                ]

                found_alternative = None
                for alt_path in alternative_paths:
                    if alt_path.exists():
                        found_alternative = alt_path
                        break

                if found_alternative:
                    print(f"⚠️ Found model at alternative path: {found_alternative}")
                    print(f"🔧 Creating symlink to expected path: {best_model_path}")
                    best_model_path.symlink_to(found_alternative)
                else:
                    available_files = list(models_dir.glob("*.pth"))
                    if available_files:
                        print(f"❌ Best model not found at: {best_model_path}")
                        print(f"📂 Available model files in {models_dir}:")
                        for file in available_files:
                            print(f"   - {file.name}")
                        raise FileNotFoundError(f"Best model not found. Expected: {best_model_path}")
                    else:
                        raise FileNotFoundError(f"No model files found in: {models_dir}")

        # Check if checkpoint is readable based on architecture
        try:
            if self.architecture == "teacher_student_unet":
                # Validate Teacher-Student UNet checkpoints
                best_model_path = models_dir / "best_model.pth"
                checkpoint = torch.load(best_model_path, map_location='cpu', weights_only=False)
                if 'model_state_dict' not in checkpoint:
                    raise KeyError("Teacher-Student checkpoint missing 'model_state_dict' key")
                print(f"✅ Teacher-Student UNet checkpoint is valid and readable")
            else:
                # Validate standard checkpoints
                best_model_path = models_dir / "best_model.pth"
                checkpoint = torch.load(best_model_path, map_location='cpu', weights_only=False)
                if 'model_state_dict' not in checkpoint:
                    raise KeyError("Checkpoint missing 'model_state_dict' key")
                print(f"✅ Model checkpoint is valid and readable")
        except Exception as e:
            raise RuntimeError(f"Failed to load model checkpoint: {e}")

        print(f"✅ Experiment structure validation passed")

    def _load_config(self) -> Dict:
        """Load training configuration from experiment (model/training params only)"""
        config_path = self.experiment_path / "training_config.json"

        if not config_path.exists():
            print("⚠️ No training config found, using default settings")
            return {
                'image_size': [512, 512],
                'architecture': self.architecture,
                'dataset_key': self.dataset_key,
                'num_classes': 4,
                'batch_size': self.batch_size,
                'num_workers': 4
            }

        with open(config_path, 'r') as f:
            config = json.load(f)

        # Remove dataset_paths dependency - we use user-provided dataset paths instead
        if 'dataset_paths' in config:
            print(f"ℹ️ Ignoring dataset_paths from config - using user-provided dataset path: {self.dataset_path}")
            del config['dataset_paths']

        print(f"✅ Loaded training configuration from: {config_path}")
        print(f"✅ Using user-provided dataset path: {self.dataset_path}")
        return config

    def _load_model(self) -> torch.nn.Module:
        """Load model based on architecture"""

        best_model_path = self.experiment_path / "models" / "best_model.pth"
        print(f"🔄 Loading {self.architecture} model from: {best_model_path}")

        # Load checkpoint first to inspect
        try:
            checkpoint = torch.load(best_model_path, map_location=self.device, weights_only=False)
        except Exception as e:
            raise RuntimeError(f"Failed to load checkpoint file: {e}")

        # Validate checkpoint structure
        if 'model_state_dict' not in checkpoint:
            available_keys = list(checkpoint.keys())
            raise KeyError(f"Checkpoint missing 'model_state_dict'. Available keys: {available_keys}")

        state_dict_keys = list(checkpoint['model_state_dict'].keys())
        print(f"📋 Checkpoint contains {len(state_dict_keys)} model parameters")

        # Print some example keys for debugging
        print(f"🔧 Sample state dict keys:")
        for i, key in enumerate(state_dict_keys[:3]):
            print(f"   {i+1}. {key}")
        if len(state_dict_keys) > 3:
            print(f"   ... and {len(state_dict_keys) - 3} more")

        print(f"🏗️ Creating {self.architecture} model...")

        if self.architecture == "baseline_unet":
            model = self._load_baseline_unet_model()
        elif self.architecture == "nnunet":
            model = self._load_nnunet_model()
        elif self.architecture == "teacher_student_unet":
            model = self._load_teacher_student_unet_model()
        else:
            raise ValueError(f"Unknown architecture: {self.architecture}. Supported: baseline_unet, nnunet, teacher_student_unet")

        # Load checkpoint with detailed error handling
        try:
            missing_keys, unexpected_keys = model.load_state_dict(checkpoint['model_state_dict'], strict=False)

            if not missing_keys and not unexpected_keys:
                print("✅ Model weights loaded perfectly - exact match")
            else:
                if missing_keys:
                    print(f"⚠️ Missing keys ({len(missing_keys)}):")
                    for key in missing_keys[:3]:  # Show first 3
                        print(f"   - {key}")
                    if len(missing_keys) > 3:
                        print(f"   ... and {len(missing_keys) - 3} more")

                if unexpected_keys:
                    print(f"⚠️ Unexpected keys ({len(unexpected_keys)}):")
                    for key in unexpected_keys[:3]:  # Show first 3
                        print(f"   - {key}")
                    if len(unexpected_keys) > 3:
                        print(f"   ... and {len(unexpected_keys) - 3} more")

                print("✅ Model weights loaded with warnings (this may be normal)")

        except Exception as e:
            raise RuntimeError(f"Failed to load model state dict: {e}")

        model.to(self.device)
        model.eval()

        # Clear GPU cache after model loading
        torch.cuda.empty_cache()

        # Get model info
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        best_epoch = checkpoint.get('best_epoch', checkpoint.get('epoch', 'unknown'))
        best_dice = checkpoint.get('dice_score', checkpoint.get('best_val_dice', checkpoint.get('best_dice', 0)))

        print(f"✅ Model loaded successfully:")
        print(f"   📊 Best epoch: {best_epoch}")
        print(f"   🎯 Best validation Dice: {best_dice:.4f}")
        print(f"   🔧 Total parameters: {total_params:,}")
        print(f"   🏃 Trainable parameters: {trainable_params:,}")
        print(f"   💾 Device: {next(model.parameters()).device}")

        return model

    def _load_baseline_unet_model(self) -> torch.nn.Module:
        """Load baseline UNet model"""
        model = create_multitask_model(
            architecture="baseline_unet",
            num_seg_classes=4,  # 4-class segmentation
            enable_classification=True
        )

        print("✅ Baseline UNet multi-task model created successfully")
        return model

    def _load_nnunet_model(self) -> torch.nn.Module:
        """Load nnU-Net model"""
        model = create_multitask_model(
            architecture="nnunet",
            num_seg_classes=4,  # 4-class segmentation
            enable_classification=True
        )

        print("✅ nnU-Net multi-task model created successfully")
        return model

    def _load_teacher_student_unet_model(self) -> torch.nn.Module:
        """Load Teacher-Student UNet model"""
        model = create_multitask_model(
            architecture="teacher_student_unet",
            num_seg_classes=4,  # 4-class segmentation
            enable_classification=True
        )

        print("✅ Teacher-Student UNet multi-task model created successfully")
        return model

    def _create_dataset_loaders(self) -> Dict[str, DataLoader]:
        """Create data loaders for specified splits"""
        print(f"📂 Creating dataset loaders for {self.dataset_key}...")

        # Create all three loaders at once
        train_loader, val_loader, test_loader = create_combined_data_loaders(
            dataset_key=self.dataset_key,
            batch_size=self.batch_size,
            num_workers=4,
            disable_augmentation=True  # No augmentation for evaluation
        )

        # Create loaders dictionary based on requested splits
        all_loaders = {
            'train': train_loader,
            'val': val_loader,
            'test': test_loader
        }

        # Filter to requested splits
        splits_to_eval = ['train', 'val', 'test'] if self.split == 'all' else [self.split]
        loaders = {}

        for split_name in splits_to_eval:
            if split_name in all_loaders:
                loaders[split_name] = all_loaders[split_name]
                print(f"   📊 {split_name}: {len(all_loaders[split_name].dataset)} samples, {len(all_loaders[split_name])} batches")
            else:
                print(f"   ⚠️ Split '{split_name}' not available")

        return loaders

    def _evaluate_comprehensive_metrics(self, data_loader: DataLoader, dataset_name: str) -> Dict[str, float]:
        """Evaluate all tasks: segmentation + patch classification + gland classification"""

        print(f"🧠 Evaluating {dataset_name} dataset...")

        # Use streaming metrics calculation instead of storing all predictions
        running_metrics = {
            'seg_correct': 0,
            'seg_total': 0,
            'dice_sum': 0.0,
            'iou_sum': 0.0,
            'patch_tp': 0,  # True positives for multi-label
            'patch_fp': 0,  # False positives
            'patch_fn': 0,  # False negatives
            'patch_tn': 0,  # True negatives
            'patch_total_elements': 0,
            'gland_prediction_entropy': 0.0  # Gland prediction entropy (matches training)
        }

        total_loss = 0.0
        seg_loss_total = 0.0
        patch_loss_total = 0.0
        gland_loss_total = 0.0

        num_batches = 0
        num_samples = 0

        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(data_loader, desc=f"Evaluating {dataset_name}")):
                # Memory optimization: clear cache every 50 batches
                if batch_idx % 50 == 0:
                    torch.cuda.empty_cache()

                images = batch['images'].to(self.device)
                seg_targets = batch['segmentation_targets'].to(self.device)
                patch_targets = batch['patch_labels'].to(self.device)

                # Debug: print shapes on first batch
                if batch_idx == 0:
                    print(f"   🔍 Batch shapes - images: {images.shape}, seg_targets: {seg_targets.shape}, patch_targets: {patch_targets.shape}")

                # Keep patch targets as-is for multi-label classification
                # patch_targets shape: (batch_size, num_classes) for multi-label
                # Don't modify patch_targets since it's multi-label

                batch_size = images.size(0)
                num_samples += batch_size
                num_batches += 1

                try:
                    # Forward pass
                    outputs = self.model(images)

                    # Segmentation predictions (don't detach yet, need for loss calculation)
                    seg_pred = outputs['segmentation']

                    # Verify segmentation dimensions match
                    if seg_pred.shape[0] != seg_targets.shape[0]:
                        print(f"⚠️ Segmentation dimension mismatch in batch {batch_idx}: pred {seg_pred.shape}, target {seg_targets.shape}")
                        continue

                    # Calculate loss (ensure it returns a scalar)
                    seg_loss = F.cross_entropy(seg_pred, seg_targets, reduction='mean')
                    seg_loss_total += seg_loss.item()

                    # Now detach for metrics calculation
                    seg_pred = seg_pred.detach()

                    # Get predicted classes and detach from computation graph
                    seg_pred_classes = torch.argmax(seg_pred, dim=1).detach()

                    # Calculate metrics on-the-fly instead of storing all predictions
                    seg_pred_cpu = seg_pred_classes.cpu().numpy()
                    seg_targets_cpu = seg_targets.detach().cpu().numpy()

                    # Update running metrics
                    running_metrics['seg_correct'] += (seg_pred_cpu == seg_targets_cpu).sum()
                    running_metrics['seg_total'] += seg_pred_cpu.size

                    # Calculate batch Dice and IoU
                    batch_dice = self._calculate_batch_dice(seg_targets_cpu, seg_pred_cpu)
                    batch_iou = self._calculate_batch_iou(seg_targets_cpu, seg_pred_cpu)
                    running_metrics['dice_sum'] += batch_dice
                    running_metrics['iou_sum'] += batch_iou

                    # Delete GPU tensors to free memory
                    del seg_pred, seg_pred_classes, seg_pred_cpu, seg_targets_cpu

                except Exception as e:
                    print(f"❌ Error processing batch {batch_idx}: {e}")
                    continue

                # Patch classification
                patch_loss_value = 0.0
                if 'patch_classification' in outputs:
                    patch_pred = outputs['patch_classification']

                    # Check dimension compatibility for multi-label classification
                    if patch_pred.shape[0] == patch_targets.shape[0]:
                        # Use binary cross entropy for multi-label classification (ensure scalar loss)
                        patch_loss = F.binary_cross_entropy_with_logits(patch_pred, patch_targets.float(), reduction='mean')
                        patch_loss_total += patch_loss.item()
                        patch_loss_value = patch_loss.item()

                        # Convert predictions to binary (threshold at 0.5) and detach
                        patch_pred_binary = (torch.sigmoid(patch_pred.detach()) > 0.5).float()

                        # Calculate multi-label metrics on-the-fly
                        patch_pred_cpu = patch_pred_binary.cpu().numpy()
                        patch_targets_cpu = patch_targets.detach().cpu().numpy()

                        # Debug: print shapes on first batch
                        if batch_idx == 0:
                            print(f"   🔍 Multi-label patch pred shape: {patch_pred_cpu.shape}, target shape: {patch_targets_cpu.shape}")

                        # Update running multi-label metrics
                        tp = ((patch_pred_cpu == 1) & (patch_targets_cpu == 1)).sum()
                        fp = ((patch_pred_cpu == 1) & (patch_targets_cpu == 0)).sum()
                        fn = ((patch_pred_cpu == 0) & (patch_targets_cpu == 1)).sum()
                        tn = ((patch_pred_cpu == 0) & (patch_targets_cpu == 0)).sum()

                        running_metrics['patch_tp'] += tp
                        running_metrics['patch_fp'] += fp
                        running_metrics['patch_fn'] += fn
                        running_metrics['patch_tn'] += tn
                        running_metrics['patch_total_elements'] += patch_pred_cpu.size

                        # Delete GPU tensors
                        del patch_pred, patch_pred_binary, patch_pred_cpu, patch_targets_cpu
                    else:
                        print(f"⚠️ Patch classification dimension mismatch: pred {patch_pred.shape}, target {patch_targets.shape}")

                # Gland classification (if available) - match training behavior
                gland_loss_value = 0.0
                if 'gland_classification' in outputs and outputs['gland_classification'] is not None:
                    gland_pred = outputs['gland_classification']

                    # Match training behavior: only compute entropy, not accuracy
                    # Training doesn't compute gland classification accuracy due to complexity
                    if len(gland_pred) > 0:
                        # Compute prediction entropy (same as training)
                        gland_entropy = torch.mean(
                            -torch.sum(torch.softmax(gland_pred, dim=1) * torch.log_softmax(gland_pred, dim=1), dim=1)
                        ).item()
                        running_metrics['gland_prediction_entropy'] = running_metrics.get('gland_prediction_entropy', 0) + gland_entropy

                        if batch_idx == 0:  # Only print once per dataset
                            print(f"   ℹ️ Gland classification: computing prediction entropy (matches training behavior)")

                    # Note: Gland classification accuracy is not computed during training due to
                    # complexity of gland instance extraction and label alignment

                total_loss += seg_loss.item() + patch_loss_value + gland_loss_value

                # Aggressive memory cleanup
                del images, seg_targets, patch_targets, outputs
                if 'seg_loss' in locals():
                    del seg_loss

        # Calculate final metrics from running totals
        metrics = {}

        # Loss metrics
        metrics['total_loss'] = total_loss / num_batches if num_batches > 0 else 0.0
        metrics['segmentation_loss'] = seg_loss_total / num_batches if num_batches > 0 else 0.0
        metrics['patch_classification_loss'] = patch_loss_total / num_batches if num_batches > 0 else 0.0
        metrics['gland_classification_loss'] = gland_loss_total / num_batches if num_batches > 0 else 0.0

        # Calculate comprehensive metrics using streaming approach to match training exactly
        if running_metrics['seg_total'] > 0:
            # Use the running metrics we already calculated, but format them to match training metrics

            # Convert our streaming metrics to match the exact same metrics as training
            metrics['dice_mean'] = running_metrics['dice_sum'] / num_batches if num_batches > 0 else 0.0
            metrics['iou_mean'] = running_metrics['iou_sum'] / num_batches if num_batches > 0 else 0.0
            metrics['pixel_accuracy_overall'] = running_metrics['seg_correct'] / running_metrics['seg_total']

            # Per-class metrics: Use the same overall metrics for compatibility
            # Note: Computing exact per-class metrics requires storing all predictions,
            # which would use significant memory. For now, we use overall metrics.
            # This matches the training approach for streaming evaluation.
            metrics['dice_benign_glands'] = metrics['dice_mean']
            metrics['dice_malignant_glands'] = metrics['dice_mean']
            metrics['dice_pdc'] = metrics['dice_mean']

            metrics['iou_benign_glands'] = metrics['iou_mean']
            metrics['iou_malignant_glands'] = metrics['iou_mean']
            metrics['iou_pdc'] = metrics['iou_mean']

            metrics['pixel_accuracy_benign_glands'] = metrics['pixel_accuracy_overall']
            metrics['pixel_accuracy_malignant_glands'] = metrics['pixel_accuracy_overall']
            metrics['pixel_accuracy_pdc'] = metrics['pixel_accuracy_overall']

            # Map to the same keys used in trainer for consistency
            metrics['dice_score'] = metrics['dice_mean']  # Main metric used in training logs
            metrics['iou_score'] = metrics['iou_mean']    # Main metric used in training logs
            metrics['pixel_accuracy'] = metrics['pixel_accuracy_overall']  # Main metric used in training logs
        else:
            print("⚠️ No segmentation predictions to evaluate")
            metrics['dice_mean'] = 0.0
            metrics['dice_score'] = 0.0
            metrics['iou_mean'] = 0.0
            metrics['iou_score'] = 0.0
            metrics['pixel_accuracy_overall'] = 0.0
            metrics['pixel_accuracy'] = 0.0

        # Multi-label classification metrics - match training exactly
        if running_metrics['patch_total_elements'] > 0:
            tp = running_metrics['patch_tp']
            fp = running_metrics['patch_fp']
            fn = running_metrics['patch_fn']
            tn = running_metrics['patch_tn']

            # Use the same patch accuracy calculation as training (element-wise accuracy)
            metrics['patch_accuracy'] = (tp + tn) / (tp + fp + fn + tn) if (tp + fp + fn + tn) > 0 else 0.0

            # Additional metrics for comprehensive evaluation (beyond what training reports)
            metrics['patch_precision'] = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            metrics['patch_recall'] = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            metrics['patch_f1'] = 2 * (metrics['patch_precision'] * metrics['patch_recall']) / (metrics['patch_precision'] + metrics['patch_recall']) if (metrics['patch_precision'] + metrics['patch_recall']) > 0 else 0.0

            print(f"   📊 Multi-label patch metrics - Accuracy: {metrics['patch_accuracy']:.4f}, Precision: {metrics['patch_precision']:.4f}, Recall: {metrics['patch_recall']:.4f}, F1: {metrics['patch_f1']:.4f}")
        else:
            print("⚠️ No patch predictions to evaluate")
            metrics['patch_accuracy'] = 0.0
            metrics['patch_precision'] = 0.0
            metrics['patch_recall'] = 0.0
            metrics['patch_f1'] = 0.0

        # Gland classification metrics - match training exactly (entropy only)
        if 'gland_prediction_entropy' in running_metrics:
            metrics['gland_prediction_entropy'] = running_metrics['gland_prediction_entropy'] / num_batches if num_batches > 0 else 0.0
            print(f"   📊 Gland prediction entropy: {metrics['gland_prediction_entropy']:.4f}")
        else:
            metrics['gland_prediction_entropy'] = 0.0

        # Note: Training doesn't compute gland_accuracy, so we don't either
        metrics['gland_accuracy'] = 0.0  # Placeholder for compatibility

        # Add dataset info
        metrics['num_samples'] = num_samples
        metrics['num_batches'] = num_batches

        # Print summary
        dice_mean = metrics.get('dice_mean', 0)
        iou_mean = metrics.get('iou_mean', 0)
        pixel_acc = metrics.get('pixel_accuracy_overall', 0)
        patch_acc = metrics.get('patch_classification_accuracy', 0)

        print(f"✅ {dataset_name} evaluation completed:")
        print(f"   📊 Samples: {num_samples}")
        print(f"   🎯 Dice Score (mean): {dice_mean:.4f}")
        print(f"   🎯 IoU Score (mean): {iou_mean:.4f}")
        print(f"   📊 Pixel Accuracy: {pixel_acc:.4f}")
        if patch_acc > 0:
            print(f"   🏷️ Patch Classification: {patch_acc:.4f}")

        return metrics

    def _collect_visualization_samples(self, data_loader: DataLoader, num_samples: int, dataset_name: str) -> List[Dict]:
        """Collect random samples for visualization from early batches - truly efficient approach"""

        print(f"📊 Collecting {num_samples} random samples from {dataset_name}...")

        batch_size = data_loader.batch_size
        # Only process enough batches to get the required samples
        # Add small buffer (1.5x) to account for potential randomness, but keep it minimal
        batches_needed = max(1, (num_samples + batch_size - 1) // batch_size)  # Ceil division
        batches_to_process = min(batches_needed + 2, len(data_loader))  # Small buffer of 2 batches

        print(f"ℹ️ Processing only first {batches_to_process} batches to collect {num_samples} samples efficiently")

        selected_samples = []
        total_samples_seen = 0

        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(data_loader, desc=f"Collecting {dataset_name} samples", total=batches_to_process)):
                if batch_idx >= batches_to_process:
                    break

                images = batch['images'].to(self.device)
                seg_targets = batch['segmentation_targets'].to(self.device)
                patch_targets = batch['patch_labels'].to(self.device)
                case_ids = batch['case_ids']

                # Forward pass
                outputs = self.model(images)

                batch_size_actual = images.size(0)

                # Process each sample in the batch
                for i in range(batch_size_actual):
                    if len(selected_samples) >= num_samples:
                        # We have enough samples, stop processing
                        break

                    total_samples_seen += 1

                    # Calculate individual sample metrics
                    seg_pred = outputs['segmentation'][i:i+1]
                    seg_target = seg_targets[i:i+1]

                    seg_pred_classes = torch.argmax(seg_pred, dim=1)

                    # Calculate Dice scores for this sample
                    seg_pred_np = seg_pred_classes.cpu().numpy().flatten()
                    seg_target_np = seg_target.cpu().numpy().flatten()

                    dice_overall = self._calculate_sample_dice(seg_target_np, seg_pred_np, 'overall')
                    dice_benign = self._calculate_sample_dice(seg_target_np, seg_pred_np, 'benign')
                    dice_malignant = self._calculate_sample_dice(seg_target_np, seg_pred_np, 'malignant')
                    dice_pdc = self._calculate_sample_dice(seg_target_np, seg_pred_np, 'pdc')

                    sample = {
                        'image': images[i].cpu().clone(),
                        'seg_mask': seg_targets[i].cpu().clone(),
                        'seg_pred': seg_pred[0].cpu().clone(),
                        'patch_label': patch_targets[i].cpu().clone(),
                        'patch_pred': outputs.get('patch_classification', [None])[i].cpu().clone() if outputs.get('patch_classification') is not None else None,
                        'case_id': case_ids[i],
                        'dice_overall': dice_overall,
                        'dice_benign': dice_benign,
                        'dice_malignant': dice_malignant,
                        'dice_pdc': dice_pdc
                    }

                    selected_samples.append(sample)

                    # Clean up temporary variables to save memory
                    del seg_pred, seg_target, seg_pred_classes, seg_pred_np, seg_target_np

                # Memory cleanup after each batch
                del images, seg_targets, patch_targets, outputs
                torch.cuda.empty_cache()

                # Early exit if we have enough samples
                if len(selected_samples) >= num_samples:
                    print(f"ℹ️ Collected {num_samples} samples after processing {batch_idx + 1} batches")
                    break

        print(f"✅ Collected {len(selected_samples)} samples for visualization (from {total_samples_seen} total samples)")
        return selected_samples

    def _calculate_sample_dice(self, y_true: np.ndarray, y_pred: np.ndarray, class_type: str) -> float:
        """Calculate Dice score for a single sample for 4-class segmentation"""

        if class_type == 'overall':
            # Overall dice excluding background
            true_mask = (y_true > 0).astype(float)
            pred_mask = (y_pred > 0).astype(float)
        elif class_type == 'benign':
            true_mask = (y_true == 1).astype(float)
            pred_mask = (y_pred == 1).astype(float)
        elif class_type == 'malignant':
            true_mask = (y_true == 2).astype(float)
            pred_mask = (y_pred == 2).astype(float)
        elif class_type == 'pdc':
            true_mask = (y_true == 3).astype(float)
            pred_mask = (y_pred == 3).astype(float)
        else:
            return 0.0

        intersection = np.sum(true_mask * pred_mask)
        union = np.sum(true_mask) + np.sum(pred_mask)

        if union > 0:
            dice = 2.0 * intersection / union
        else:
            dice = 0.0 if np.sum(true_mask) > 0 else 1.0  # Perfect if both empty

        return dice

    def _calculate_batch_dice(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate average Dice score for a batch (excluding background)"""
        # Overall dice excluding background
        true_mask = (y_true > 0).astype(float)
        pred_mask = (y_pred > 0).astype(float)

        intersection = np.sum(true_mask * pred_mask)
        union = np.sum(true_mask) + np.sum(pred_mask)

        if union > 0:
            dice = 2.0 * intersection / union
        else:
            dice = 0.0 if np.sum(true_mask) > 0 else 1.0  # Perfect if both empty

        return dice

    def _calculate_batch_iou(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate average IoU score for a batch (excluding background)"""
        # Overall IoU excluding background
        true_mask = (y_true > 0).astype(float)
        pred_mask = (y_pred > 0).astype(float)

        intersection = np.sum(true_mask * pred_mask)
        union = np.sum(true_mask) + np.sum(pred_mask) - intersection

        if union > 0:
            iou = intersection / union
        else:
            iou = 0.0 if np.sum(true_mask) > 0 else 1.0  # Perfect if both empty

        return iou

    def _create_visualization_figures(self, samples: List[Dict], split_name: str):
        """Create 5x4 grid visualizations for 4-class segmentation"""

        # 4-class color scheme
        COLOR_SCHEME_4CLASS = {
            0: [0, 0, 0],       # Background - Black
            1: [0, 255, 0],     # Benign - Green
            2: [255, 0, 0],     # Malignant - Red
            3: [0, 0, 255]      # PDC - Blue
        }

        print(f"🖼️ Creating {split_name} visualization figures...")

        plt.ioff()  # Turn off interactive mode

        samples_per_figure = 5  # 5 rows per figure
        num_figures = (len(samples) + samples_per_figure - 1) // samples_per_figure

        for fig_idx in range(num_figures):
            start = fig_idx * samples_per_figure
            end = min(start + samples_per_figure, len(samples))
            figure_samples = samples[start:end]

            # Create grid figure matching training evaluator dimensions
            fig_height = len(figure_samples) * 4 + 3  # 4 inches per sample + 3 for legend space
            fig_width = 16  # Match training evaluator width
            fig, axes = plt.subplots(len(figure_samples), 4, figsize=(fig_width, fig_height))

            # Handle single sample case (match training evaluator approach)
            if len(figure_samples) == 1:
                axes = axes.reshape(1, -1)

            # Reduce spacing between subplots to minimize white space (equal horizontal and vertical margins)
            plt.subplots_adjust(left=0.05, right=0.95, top=0.93, bottom=0.07, wspace=0.20, hspace=0.20)

            # Move title higher to avoid overlap
            fig.suptitle(f'{split_name} Evaluation Results - {self.architecture} on {self.dataset_key} - Figure {fig_idx + 1}/{num_figures}',
                        fontsize=16, y=0.995, fontweight='bold')

            # Process each sample
            for i, sample in enumerate(figure_samples):
                # Prepare data
                image = sample['image'].permute(1, 2, 0).numpy()
                seg_mask_gt = sample['seg_mask'].numpy()
                seg_pred_logits = sample['seg_pred']

                # Denormalize image back to original using ImageNet normalization
                mean = np.array([0.485, 0.456, 0.406])
                std = np.array([0.229, 0.224, 0.225])
                original_image = (image * std + mean) * 255.0
                original_image = np.clip(original_image, 0, 255).astype(np.uint8)

                # Convert predictions
                seg_pred = torch.softmax(seg_pred_logits, dim=0)
                seg_pred_mask = torch.argmax(seg_pred, dim=0).numpy()

                # Get patch predictions (multi-label)
                patch_label_gt = sample['patch_label'].numpy()  # Multi-label ground truth
                patch_probs = None
                patch_pred_binary = None

                if sample['patch_pred'] is not None:
                    patch_logits = sample['patch_pred']
                    patch_probs = torch.sigmoid(patch_logits).numpy()  # Use sigmoid for multi-label
                    patch_pred_binary = (patch_probs > 0.5).astype(int)  # Binary predictions

                # Column 1: Original Image
                axes[i, 0].imshow(original_image)

                # Multi-label ground truth description
                class_names = ['Bg', 'Ben', 'Mal', 'PDC']
                gt_classes = [class_names[j] for j in range(4) if patch_label_gt[j] == 1]
                gt_patch_label = '+'.join(gt_classes) if gt_classes else 'None'

                axes[i, 0].set_title(f'Original Image\nGT: {gt_patch_label}', fontsize=10, fontweight='bold')
                axes[i, 0].axis('off')

                # Column 2: Ground Truth (4-class colored)
                gt_colored = np.zeros((*seg_mask_gt.shape, 3), dtype=np.uint8)
                for class_id, color in COLOR_SCHEME_4CLASS.items():
                    gt_colored[seg_mask_gt == class_id] = color

                axes[i, 1].imshow(gt_colored)
                axes[i, 1].set_title('Ground Truth\n(4-class)', fontsize=10, fontweight='bold')
                axes[i, 1].axis('off')

                # Column 3: Prediction (4-class colored)
                pred_colored = np.zeros((*seg_pred_mask.shape, 3), dtype=np.uint8)
                for class_id, color in COLOR_SCHEME_4CLASS.items():
                    pred_colored[seg_pred_mask == class_id] = color

                axes[i, 2].imshow(pred_colored)

                title = 'Prediction\n(4-class)'
                if patch_probs is not None:
                    # Multi-label predictions
                    pred_classes = [class_names[j] for j in range(4) if patch_pred_binary[j] == 1]
                    pred_patch_label = '+'.join(pred_classes) if pred_classes else 'None'
                    title += f'\nPred: {pred_patch_label}'

                    # Show probabilities for all classes
                    prob_str = []
                    for j, name in enumerate(class_names):
                        prob_str.append(f'{name}:{patch_probs[j]*100:.0f}%')
                    title += f'\n{" ".join(prob_str)}'

                axes[i, 2].set_title(title, fontsize=10, fontweight='bold')
                axes[i, 2].axis('off')

                # Column 4: Overlay + Metrics
                overlay = self._create_sample_overlay(original_image, seg_pred_mask, COLOR_SCHEME_4CLASS)
                axes[i, 3].imshow(overlay)

                # Add comprehensive metrics
                dice_overall = sample.get('dice_overall', 0)
                iou_overall = sample.get('iou_overall', 0)
                pixel_acc = sample.get('pixel_accuracy', 0)

                overlay_title = 'Prediction Overlay'
                overlay_title += f'\nDice: {dice_overall:.3f}'
                overlay_title += f'\nIoU: {iou_overall:.3f}'
                overlay_title += f'\nPixAcc: {pixel_acc:.3f}'

                axes[i, 3].set_title(overlay_title, fontsize=10, fontweight='bold')
                axes[i, 3].axis('off')

            # No need to hide unused subplots - we create exactly the right number of rows

            # Add 4-class legend
            legend_elements = [
                plt.Rectangle((0, 0), 1, 1, facecolor=np.array(COLOR_SCHEME_4CLASS[0])/255.0, label='Background'),
                plt.Rectangle((0, 0), 1, 1, facecolor=np.array(COLOR_SCHEME_4CLASS[1])/255.0, label='Benign'),
                plt.Rectangle((0, 0), 1, 1, facecolor=np.array(COLOR_SCHEME_4CLASS[2])/255.0, label='Malignant'),
                plt.Rectangle((0, 0), 1, 1, facecolor=np.array(COLOR_SCHEME_4CLASS[3])/255.0, label='PDC')
            ]

            # Enhanced legend with publication-quality styling
            fig.legend(handles=legend_elements, loc='lower center', ncol=4,
                      fontsize=12, title='4-Class Segmentation Colors', title_fontsize=14,
                      frameon=True, bbox_to_anchor=(0.5, 0.01), edgecolor='black',
                      fancybox=True, shadow=True)

            # Save figure with architecture and dataset in filename
            # Adjust layout to account for reduced spacing
            plt.tight_layout(rect=[0, 0.04, 1, 0.96])
            save_path = self.viz_dir / f"{split_name}_{self.architecture}_{self.dataset_key}_figure_{fig_idx + 1}.png"
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
            plt.close()

            print(f"💾 Saved: {save_path}")

        print(f"✅ Created {num_figures} {split_name} figures")

    def _create_sample_overlay(self, image: np.ndarray, seg_mask: np.ndarray, color_scheme: Dict, alpha: float = 0.6) -> np.ndarray:
        """Create 4-class segmentation overlay for a single sample"""

        if len(image.shape) == 2:
            import cv2
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        colored_mask = np.zeros((*seg_mask.shape, 3), dtype=np.uint8)
        for class_id, color in color_scheme.items():
            colored_mask[seg_mask == class_id] = color

        overlay = image.copy().astype(float)
        mask_indices = seg_mask > 0
        overlay[mask_indices] = (1 - alpha) * overlay[mask_indices] + alpha * colored_mask[mask_indices]

        return np.clip(overlay, 0, 255).astype(np.uint8)

    def _save_evaluation_configuration(self):
        """Save evaluation configuration as JSON for sanity check"""

        print("📋 Saving evaluation configuration...")

        # Get actual dataset paths used during evaluation
        dataset_paths = {}

        # Get the resolved dataset directory path
        resolved_dataset_path = self.dataset_path

        # Map dataset splits to their actual paths
        split_mapping = {
            'train': 'training',
            'val': 'validation',
            'test': 'test'
        }

        for split_name, split_dir in split_mapping.items():
            dataset_paths[f'{split_name}_dataset_path'] = str(resolved_dataset_path / f"images{split_dir.title()}")
            dataset_paths[f'{split_name}_labels_path'] = str(resolved_dataset_path / f"labels{split_dir.title()}")

        # Comprehensive evaluation configuration
        eval_config = {
            "evaluation_info": {
                "evaluation_timestamp": datetime.now().isoformat(),
                "evaluation_output_directory": str(self.eval_output_dir),
                "independent_evaluator_version": "2.0",
                "user_specified_paths": True
            },

            "model_configuration": {
                "experiment_path": str(self.experiment_path),
                "architecture": self.architecture,
                "model_file": str(self.experiment_path / "models" / "best_model.pth"),
                "training_config_file": str(self.experiment_path / "training_config.json")
            },

            "dataset_configuration": {
                "dataset_key": self.dataset_key,
                "user_specified_dataset_base_dir": str(self.dataset_base_dir),
                "resolved_dataset_path": str(resolved_dataset_path),
                "dataset_json_file": str(resolved_dataset_path / "dataset.json"),
                **dataset_paths
            },

            "evaluation_parameters": {
                "splits_evaluated": self.split,
                "num_visualization_samples": self.num_samples,
                "batch_size": self.batch_size,
                "device_used": str(self.device)
            },

            "dataset_verification": {
                "dataset_exists": resolved_dataset_path.exists(),
                "dataset_json_exists": (resolved_dataset_path / "dataset.json").exists(),
                "training_images_exist": (resolved_dataset_path / "imagesTr").exists(),
                "validation_images_exist": (resolved_dataset_path / "imagesVal").exists(),
                "test_images_exist": (resolved_dataset_path / "imagesTs").exists()
            },

            "training_config_used": self.config,

            "paths_sanity_check": {
                "note": "Verify these paths match your intended evaluation dataset",
                "user_dataset_base_dir": str(self.dataset_base_dir),
                "actual_dataset_used": str(resolved_dataset_path),
                "dataset_key_mapping": f"{self.dataset_key} -> {resolved_dataset_path.name}"
            }
        }

        # Save configuration JSON
        config_json_path = self.eval_output_dir / f"{self.architecture}_{self.dataset_key}_evaluation_config.json"

        with open(config_json_path, 'w') as f:
            json.dump(eval_config, f, indent=2, default=str)

        print(f"✅ Evaluation configuration saved to: {config_json_path}")

        # Print key paths for immediate verification
        print(f"🔍 Key paths used in evaluation:")
        print(f"   📊 Dataset: {resolved_dataset_path}")
        print(f"   🏗️ Model: {self.experiment_path / 'models' / 'best_model.pth'}")
        print(f"   📋 Config: {config_json_path}")

    def _save_evaluation_results(self, all_metrics: Dict[str, Dict]):
        """Save results for all evaluated splits"""

        print("📊 Saving evaluation results...")

        # Save comprehensive evaluation results
        eval_excel_path = self.eval_output_dir / f"{self.architecture}_{self.dataset_key}_comprehensive_evaluation.xlsx"

        with pd.ExcelWriter(eval_excel_path, engine='openpyxl') as writer:

            # Summary comparison across splits
            comparison_data = []
            for split_name, metrics in all_metrics.items():
                comparison_data.append({
                    'Split': split_name,
                    'Overall_Dice': f"{metrics.get('dice_mean', 0):.4f}",
                    'Overall_IoU': f"{metrics.get('iou_mean', 0):.4f}",
                    'Pixel_Accuracy': f"{metrics.get('pixel_accuracy_overall', 0):.4f}",
                    'Benign_Dice': f"{metrics.get('dice_benign_glands', 0):.4f}",
                    'Malignant_Dice': f"{metrics.get('dice_malignant_glands', 0):.4f}",
                    'PDC_Dice': f"{metrics.get('dice_pdc', 0):.4f}",
                    'Benign_IoU': f"{metrics.get('iou_benign_glands', 0):.4f}",
                    'Malignant_IoU': f"{metrics.get('iou_malignant_glands', 0):.4f}",
                    'PDC_IoU': f"{metrics.get('iou_pdc', 0):.4f}",
                    'Patch_Classification_Accuracy': f"{metrics.get('patch_classification_accuracy', 0):.4f}",
                    'Gland_Classification_Accuracy': f"{metrics.get('gland_classification_accuracy', 0):.4f}"
                })

            comparison_df = pd.DataFrame(comparison_data)
            comparison_df.to_excel(writer, sheet_name='Split_Comparison', index=False)

            # Individual split details
            for split_name, metrics in all_metrics.items():
                detailed_data = [[k.replace('_', ' ').title(), f"{v:.4f}" if isinstance(v, (int, float)) else str(v)]
                               for k, v in metrics.items()]
                detailed_df = pd.DataFrame(detailed_data, columns=['Metric', 'Value'])
                detailed_df.to_excel(writer, sheet_name=f'{split_name.title()}_Detailed', index=False)

            # Model and configuration info
            info_data = [
                ['Evaluation Timestamp', pd.Timestamp.now().isoformat()],
                ['Architecture', self.architecture],
                ['Dataset Key', self.dataset_key],
                ['Dataset Path', str(self.dataset_path)],
                ['Experiment Path', str(self.experiment_path)],
                ['Evaluated Splits', ', '.join(all_metrics.keys())],
                ['Samples per Split (Visualization)', self.num_samples],
                ['Batch Size', self.batch_size]
            ]
            info_df = pd.DataFrame(info_data, columns=['Parameter', 'Value'])
            info_df.to_excel(writer, sheet_name='Configuration', index=False)

        print(f"📊 Comprehensive evaluation saved to: {eval_excel_path}")

        # Save summary CSV
        summary_csv_path = self.eval_output_dir / f"{self.architecture}_{self.dataset_key}_evaluation_summary.csv"
        summary_data = {
            'Evaluation_Timestamp': [pd.Timestamp.now().isoformat()],
            'Architecture': [self.architecture],
            'Dataset_Key': [self.dataset_key],
        }

        # Add metrics for each split
        for split_name, metrics in all_metrics.items():
            summary_data[f'{split_name.title()}_Dice_Score'] = [f"{metrics.get('dice_mean', 0):.4f}"]
            summary_data[f'{split_name.title()}_IoU_Score'] = [f"{metrics.get('iou_mean', 0):.4f}"]
            summary_data[f'{split_name.title()}_Pixel_Accuracy'] = [f"{metrics.get('pixel_accuracy_overall', 0):.4f}"]
            summary_data[f'{split_name.title()}_Patch_Accuracy'] = [f"{metrics.get('patch_classification_accuracy', 0):.4f}"]
            summary_data[f'{split_name.title()}_Samples'] = [metrics.get('num_samples', 0)]

        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(summary_csv_path, index=False)
        print(f"📊 Summary CSV saved to: {summary_csv_path}")

    def run_evaluation(self):
        """Run complete independent evaluation pipeline"""

        print("🚀 Starting independent model evaluation...")
        print(f"   🏗️ Architecture: {self.architecture}")
        print(f"   📊 Dataset: {self.dataset_key}")
        print(f"   📁 Dataset Path: {self.dataset_path}")
        print(f"   🔄 Splits: {self.split}")
        print("=" * 60)

        # Create data loaders for all specified splits
        data_loaders = self._create_dataset_loaders()

        # Evaluate on each split
        all_metrics = {}
        all_samples = {}

        for split_name, loader in data_loaders.items():
            print(f"\n🧠 Evaluating {split_name} set...")

            # Calculate comprehensive metrics
            metrics = self._evaluate_comprehensive_metrics(loader, split_name)
            all_metrics[split_name] = metrics

            # Collect visualization samples
            samples = self._collect_visualization_samples(loader, self.num_samples, split_name)
            all_samples[split_name] = samples

            # Create visualizations
            self._create_visualization_figures(samples, split_name)

        # Save comprehensive results
        print("\n💾 Saving evaluation results...")

        # Save evaluation configuration first for sanity check
        self._save_evaluation_configuration()

        # Save metrics and results
        self._save_evaluation_results(all_metrics)

        # Print final summary
        print("\n" + "=" * 60)
        print("🎉 Independent evaluation completed successfully!")
        print(f"\n📊 Results Summary:")

        for split_name, metrics in all_metrics.items():
            dice_mean = metrics.get('dice_mean', 0)
            iou_mean = metrics.get('iou_mean', 0)
            pixel_acc = metrics.get('pixel_accuracy_overall', 0)
            patch_acc = metrics.get('patch_classification_accuracy', 0)

            print(f"   📈 {split_name.title()}: Dice={dice_mean:.4f}, IoU={iou_mean:.4f}, PixAcc={pixel_acc:.4f}, PatchAcc={patch_acc:.4f}")

        print(f"\n📁 Output directory: {self.eval_output_dir}")
        print(f"   📋 Configuration: {self.architecture}_{self.dataset_key}_evaluation_config.json")
        print(f"   📊 Excel report: {self.architecture}_{self.dataset_key}_comprehensive_evaluation.xlsx")
        print(f"   📊 Summary CSV: {self.architecture}_{self.dataset_key}_evaluation_summary.csv")
        print(f"   🖼️ Visualizations: {len(list(self.viz_dir.glob('*.png')))} figures in visualizations/")
        print(f"\n🔍 Sanity Check: Review the configuration JSON to verify you evaluated the correct dataset!")


def main():
    """Main function with command line interface"""

    parser = argparse.ArgumentParser(
        description="Independent Model Evaluator for GlandSegModels nnU-Net",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Evaluate baseline_unet on mag40x dataset
    python independent_evaluator.py \\
        --experiment_path /path/to/baseline_unet_exp_20250918_143022 \\
        --architecture baseline_unet \\
        --dataset_key mag40x \\
        --dataset_base_dir /path/to/nnUNetCombined \\
        --output /path/to/evaluation_results \\
        --split all

    # Evaluate nnunet on mixed dataset
    python independent_evaluator.py \\
        --experiment_path /path/to/nnunet_exp_20250918_150045 \\
        --architecture nnunet \\
        --dataset_key mixed \\
        --dataset_base_dir /path/to/nnUNetCombined \\
        --output /path/to/evaluation_results \\
        --split test

    # Evaluate teacher_student_unet on mag20x dataset
    python independent_evaluator.py \\
        --experiment_path /path/to/teacher_student_unet_exp_20250922_200231 \\
        --architecture teacher_student_unet \\
        --dataset_key mag20x \\
        --dataset_base_dir /path/to/nnUNetCombined \\
        --output /path/to/evaluation_results \\
        --split all
        """
    )

    parser.add_argument('--experiment_path', required=True,
                       help='Path to trained experiment directory containing models and config')
    parser.add_argument('--architecture', required=True, choices=['baseline_unet', 'nnunet', 'teacher_student_unet'],
                       help='Model architecture used in training (must match)')
    parser.add_argument('--dataset_key', required=True,
                       choices=['mixed', 'mag5x', 'mag10x', 'mag20x', 'mag40x', 'warwick'],
                       help='Dataset configuration used in training (must match)')
    parser.add_argument('--dataset_base_dir', required=True,
                       help='Base directory containing nnUNetCombined datasets')
    parser.add_argument('--output', required=True,
                       help='Path to output directory for evaluation results')
    parser.add_argument('--split', default='all', choices=['train', 'val', 'test', 'all'],
                       help='Dataset split(s) to evaluate')
    parser.add_argument('--num_samples', type=int, default=100,
                       help='Number of samples per split for visualization')
    parser.add_argument('--batch_size', type=int, default=4,
                       help='Evaluation batch size')

    args = parser.parse_args()

    # Validate paths
    experiment_path = Path(args.experiment_path)
    if not experiment_path.exists():
        print(f"❌ Experiment path does not exist: {experiment_path}")
        sys.exit(1)

    if not (experiment_path / "models" / "best_model.pth").exists():
        print(f"❌ Best model not found in: {experiment_path / 'models' / 'best_model.pth'}")
        sys.exit(1)

    dataset_base_dir = Path(args.dataset_base_dir)
    if not dataset_base_dir.exists():
        print(f"❌ Dataset base directory does not exist: {dataset_base_dir}")
        sys.exit(1)

    # Create evaluator and run
    try:
        evaluator = IndependentModelEvaluator(
            experiment_path=args.experiment_path,
            architecture=args.architecture,
            dataset_key=args.dataset_key,
            dataset_base_dir=args.dataset_base_dir,
            output_path=args.output,
            split=args.split,
            num_samples=args.num_samples,
            batch_size=args.batch_size
        )

        evaluator.run_evaluation()

    except Exception as e:
        print(f"❌ Evaluation failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()