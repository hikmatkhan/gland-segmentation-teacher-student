#!/usr/bin/env python3
"""
4-Class nnU-Net Multi-Task Trainer for Combined Gland Segmentation
================================================================

Comprehensive training pipeline for 4-class gland segmentation with multi-label classification.
Supports Warwick GlaS + OSU Makoto combined datasets with any magnification strategy.

Features:
- 4-class segmentation: Background(0), Benign(1), Malignant(2), PDC(3)
- Multi-label patch classification (patches can contain multiple gland types)
- Single-label gland classification
- Adaptive loss weighting for multi-task learning
- Advanced metrics tracking and visualization
- Automatic checkpointing and early stopping
- Research-ready training (100+ epochs)

Author: Claude Code - Generated for OSU CRC Research
Date: 2025-09-16
"""

import os
import sys
import time
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm

# Configure matplotlib backend BEFORE importing pyplot to avoid NFS font issues
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend (no display needed)
matplotlib.rcParams['text.usetex'] = False  # Disable LaTeX rendering
matplotlib.rcParams['mathtext.fontset'] = 'dejavusans'  # Use built-in fonts instead of system fonts
import matplotlib.pyplot as plt

from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from configs.paths_config import get_dataset_path, DEFAULT_CONFIG, EVALUATION_CONFIG
from src.models.multi_task_wrapper import create_multitask_model
from src.models.loss_functions import MultiTaskLoss
from src.models.teacher_student_loss import TeacherStudentLoss
from src.models.metrics import SegmentationMetrics
from src.training.dataset import create_combined_data_loaders


class MultiTaskTrainer:
    """
    Comprehensive trainer for 4-class multi-task gland segmentation

    Handles:
    - Multi-task learning with segmentation + classification
    - Multi-label patch classification (realistic histopathology)
    - Advanced metrics tracking and visualization
    - Automatic checkpointing and early stopping
    - Research-ready training configurations
    """

    def __init__(
        self,
        dataset_key: str = "mixed",
        config: Optional[Dict] = None,
        output_base_dir: str = None,
        experiment_name: Optional[str] = None
    ):
        """
        Initialize the multi-task trainer

        Args:
            dataset_key: Dataset to use ('mixed', 'mag5x', 'mag10x', 'mag20x', 'mag40x')
            config: Training configuration (uses DEFAULT_CONFIG if None)
            output_base_dir: Base output directory for experiments (REQUIRED)
            experiment_name: Name for this experiment
        """
        self.dataset_key = dataset_key
        self.config = config or DEFAULT_CONFIG.copy()

        if output_base_dir is None:
            raise ValueError("output_base_dir is required. Please provide the base directory for experiments.")

        # Setup experiment directory following GlaS_MultiTask pattern
        if experiment_name is None:
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            # Add architecture prefix to experiment name
            architecture = self.config.get('architecture', 'nnunet')
            if architecture == 'baseline_unet':
                prefix = 'baseline_unet'
            elif architecture == 'teacher_student_unet':
                prefix = 'teacher_student_unet'
            else:
                prefix = 'nnunet'
            experiment_name = f"{prefix}_exp_{timestamp}"

        self.experiment_name = experiment_name
        self.output_base_dir = Path(output_base_dir)
        self.output_dir = self.output_base_dir / experiment_name
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Setup directories following GlaS_MultiTask structure
        self.models_dir = self.output_dir / "models"  # For best_model.pth
        self.logs_dir = self.output_dir / "logs"      # For training logs
        self.evaluations_dir = self.output_dir / "evaluations"  # For evaluation results
        self.visualizations_dir = self.output_dir / "visualizations"  # For plots/figures

        for dir_path in [self.models_dir, self.logs_dir, self.evaluations_dir, self.visualizations_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

        # Setup logging
        self.setup_logging()

        # Training state
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.loss_fn = None
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        self.writer = None

        # Metrics tracking
        self.train_history = {
            'epoch': [],
            'train_loss': [],
            'val_loss': [],

            # Individual loss components for detailed tracking
            'train_seg_loss': [],
            'train_patch_loss': [],
            'train_gland_loss': [],
            'val_seg_loss': [],
            'val_patch_loss': [],
            'val_gland_loss': [],

            # Legacy dice metric (from loss function)
            'seg_dice': [],

            # Classification metrics
            'patch_accuracy': [],
            'train_patch_accuracy': [],
            'gland_accuracy': [],

            # NEW: Comprehensive segmentation metrics
            'train_dice_score': [],
            'val_dice_score': [],
            'train_iou_score': [],
            'val_iou_score': [],
            'train_pixel_accuracy': [],
            'val_pixel_accuracy': [],

            # Teacher-Student specific metrics
            'train_consistency_loss': [],
            'val_consistency_loss': [],
            'alpha': [],

            # Pseudo-GT metrics (Student vs Pseudo-GT for monitoring alignment)
            'train_pseudo_dice': [],
            'train_pseudo_iou': [],
            'val_pseudo_dice': [],
            'val_pseudo_iou': [],

            # Teacher-Student EMA tracking
            'ema_decay': [],

            'learning_rate': []
        }

        self.best_metrics = {
            'best_val_loss': float('inf'),
            'best_dice': 0.0,
            'best_iou': 0.0,
            'best_patch_acc': 0.0,
            'best_epoch': 0
        }

        # Initialize segmentation metrics calculator
        self.metrics_calculator = SegmentationMetrics(num_classes=4, ignore_background=True)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.logger.info(f"🚀 Trainer initialized for dataset: {dataset_key}")
        self.logger.info(f"📱 Device: {self.device}")
        self.logger.info(f"📁 Output directory: {self.output_dir}")

    @classmethod
    def from_checkpoint(cls, experiment_dir: str, use_best: bool = False):
        """
        Create trainer instance from existing experiment checkpoint.
        Works with already interrupted experiments.

        Args:
            experiment_dir: Path to experiment directory (e.g., teacher_student_nnunet_mag5x_enhanced_20251026_150500)
            use_best: If True, resume from best checkpoint; otherwise use latest

        Returns:
            MultiTaskTrainer: Trainer instance ready to resume training
        """
        import json

        experiment_dir = Path(experiment_dir)

        # Load training_config.json
        config_path = experiment_dir / 'training_config.json'
        if not config_path.exists():
            raise FileNotFoundError(f"training_config.json not found in {experiment_dir}")

        with open(config_path, 'r') as f:
            config = json.load(f)

        # Find checkpoint file based on architecture
        models_dir = experiment_dir / 'models'
        if not models_dir.exists():
            raise FileNotFoundError(f"models directory not found in {experiment_dir}")

        architecture = config.get('architecture', 'baseline_unet')

        # Determine checkpoint name based on architecture
        if architecture == 'teacher_student_unet':
            checkpoint_name = 'best_student_model.pth' if use_best else 'latest_student_model.pth'
        else:
            checkpoint_name = 'best_model.pth' if use_best else 'latest_model.pth'

        checkpoint_path = models_dir / checkpoint_name

        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        # Load checkpoint to get epoch information
        print(f"Loading checkpoint from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        last_completed_epoch = checkpoint['epoch']

        print(f"{'='*80}")
        print(f"CHECKPOINT INFORMATION")
        print(f"{'='*80}")
        print(f"Experiment Directory: {experiment_dir}")
        print(f"Checkpoint File: {checkpoint_name}")
        print(f"Last Completed Epoch: {last_completed_epoch}")
        print(f"Will Resume From Epoch: {last_completed_epoch + 1}")
        print(f"Total Epochs: {config.get('epochs', 150)}")
        print(f"Remaining Epochs: {config.get('epochs', 150) - (last_completed_epoch + 1) + 1}")
        print(f"{'='*80}\n")

        # Extract dataset_key from config or infer from training_dataset path
        dataset_key = config.get('dataset_key')

        # If dataset_key not in config, infer it from the training_dataset path
        if not dataset_key:
            dataset_paths = config.get('dataset_paths', {})
            training_dataset_path = dataset_paths.get('training_dataset', '')

            # Infer dataset_key from path
            if 'Task001_Combined_Mixed_Magnifications' in training_dataset_path:
                dataset_key = 'mixed'
            elif 'Task005_Combined_Mag5x' in training_dataset_path:
                dataset_key = 'mag5x'
            elif 'Task010_Combined_Mag10x' in training_dataset_path:
                dataset_key = 'mag10x'
            elif 'Task020_Combined_Mag20x' in training_dataset_path:
                dataset_key = 'mag20x'
            elif 'Task040_Combined_Mag40x' in training_dataset_path:
                dataset_key = 'mag40x'
            elif 'Task002_WarwickGlaSTeacherStudent' in training_dataset_path or 'warwick' in training_dataset_path.lower():
                dataset_key = 'warwick'
            else:
                # Default fallback
                print(f"⚠️  Warning: Could not infer dataset_key from path: {training_dataset_path}")
                print(f"⚠️  Using default: 'mixed'")
                dataset_key = 'mixed'

            print(f"✓ Inferred dataset_key from training path: '{dataset_key}'")

        # Get output_base_dir from experiment_dir parent
        output_base_dir = experiment_dir.parent
        experiment_name = experiment_dir.name

        # Update config with resume information
        config['resume_mode'] = True
        config['resume_from_epoch'] = last_completed_epoch
        config['resume_checkpoint_path'] = str(checkpoint_path)
        config['resume_from_best'] = use_best  # Track whether using best or latest checkpoint

        # IMPORTANT: Mark that we should use saved dataset paths, not reconstruct them
        # This is crucial for experiments that used different base directories
        config['use_saved_dataset_paths'] = True

        # Create trainer instance with loaded config
        # We pass the existing experiment_name to ensure we use the same directory
        trainer = cls(
            dataset_key=dataset_key,
            config=config,
            output_base_dir=str(output_base_dir),
            experiment_name=experiment_name
        )

        # Store checkpoint for resume_training to use
        trainer._resume_checkpoint = checkpoint

        return trainer

    def resume_training(self):
        """
        Resume training from checkpoint.
        Restores model, optimizer, scheduler, and training history.
        """
        if not hasattr(self, '_resume_checkpoint'):
            raise RuntimeError("No checkpoint loaded. Use from_checkpoint() to create trainer for resumption.")

        checkpoint = self._resume_checkpoint
        start_epoch = self.config['resume_from_epoch'] + 1
        total_epochs = self.config['epochs']

        self.logger.info(f"\n{'='*80}")
        self.logger.info(f"RESUMING TRAINING")
        self.logger.info(f"{'='*80}")
        self.logger.info(f"Experiment: {self.output_dir}")
        self.logger.info(f"Checkpoint: {os.path.basename(self.config['resume_checkpoint_path'])}")
        self.logger.info(f"Last completed epoch: {self.config['resume_from_epoch']}")
        self.logger.info(f"Resuming from epoch: {start_epoch}")
        self.logger.info(f"Total epochs: {total_epochs}")
        self.logger.info(f"Remaining epochs: {total_epochs - start_epoch + 1}")
        self.logger.info(f"{'='*80}\n")

        # Prepare data and model (this will create fresh model, optimizer, etc.)
        self.prepare_data()
        self.setup_model()

        # Restore model state
        # For Teacher-Student models, the teacher might not be initialized yet at early epochs
        # Use strict=False to allow loading checkpoints before teacher initialization
        architecture = self.config.get('architecture', 'nnunet')
        teacher_initialized = checkpoint.get('teacher_initialized', False)

        # Check if teacher weights actually exist in checkpoint
        has_teacher_weights = any('teacher' in key.lower() for key in checkpoint['model_state_dict'].keys())

        if architecture == 'teacher_student_unet' and not has_teacher_weights and teacher_initialized:
            # Teacher-Student model with separate checkpoints - need to load both student and teacher
            self.logger.info(f"ℹ️  Loading Teacher-Student model from separate checkpoints (epoch {checkpoint['epoch']})")

            # Load student weights
            self.model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            self.logger.info("✓ Student model state restored")

            # Check if separate teacher checkpoint exists
            checkpoint_suffix = 'best_teacher_model.pth' if self.config.get('resume_from_best', False) else 'latest_teacher_model.pth'
            teacher_checkpoint_path = self.models_dir / checkpoint_suffix

            if teacher_checkpoint_path.exists():
                self.logger.info(f"📂 Loading teacher weights from: {checkpoint_suffix}")
                teacher_checkpoint = torch.load(teacher_checkpoint_path, map_location=self.device)
                self.model.load_state_dict(teacher_checkpoint['model_state_dict'], strict=False)
                self.logger.info("✓ Teacher model state restored")
            else:
                self.logger.warning(f"⚠️  Teacher checkpoint not found: {teacher_checkpoint_path}")
                self.logger.warning("    Teacher will be reinitialized from student during training")
        elif architecture == 'teacher_student_unet' and not has_teacher_weights:
            # Teacher not yet initialized - load student weights only
            self.logger.info(f"ℹ️  Teacher not yet initialized at epoch {checkpoint['epoch']}")
            self.logger.info("    Loading student weights only (strict=False)")
            self.model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            self.logger.info("✓ Student model state restored (teacher will be initialized during training)")
        else:
            # Normal case - combined checkpoint with all weights
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.logger.info("✓ Model state restored")

        # Restore optimizer state
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.logger.info("✓ Optimizer state restored")

        # Restore scheduler state
        if self.scheduler is not None and checkpoint.get('scheduler_state_dict') is not None:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            self.logger.info("✓ Scheduler state restored")

        # Restore training history
        if 'train_history' in checkpoint:
            self.train_history = checkpoint['train_history']
            self.logger.info(f"✓ Training history restored ({len(self.train_history.get('epoch', []))} epochs)")

        # Restore best metrics
        if 'metrics' in checkpoint:
            # Update best_metrics from checkpoint
            self.best_metrics['best_val_loss'] = checkpoint['metrics'].get('val_loss', float('inf'))
            self.best_metrics['best_dice'] = checkpoint['metrics'].get('val_dice_mean', 0.0)
            self.best_metrics['best_iou'] = checkpoint['metrics'].get('val_iou_mean', 0.0)
            self.best_metrics['best_patch_acc'] = checkpoint['metrics'].get('val_patch_accuracy', 0.0)
            self.best_metrics['best_epoch'] = checkpoint['epoch']
            self.logger.info(f"✓ Best metrics restored (best epoch: {self.best_metrics['best_epoch']})")

        # Restore teacher-student specific state
        architecture = self.config.get('architecture', 'nnunet')
        if architecture == 'teacher_student_unet':
            seg_model = self.model.segmentation_model

            if 'teacher_initialized' in checkpoint:
                seg_model.teacher_initialized = checkpoint['teacher_initialized']
                self.logger.info(f"✓ Teacher initialized: {seg_model.teacher_initialized}")

            if 'ema_decay' in checkpoint:
                seg_model.ema_decay = checkpoint['ema_decay']
                self.logger.info(f"✓ EMA decay restored: {seg_model.ema_decay:.6f}")

        self.logger.info(f"\n{'='*80}")
        self.logger.info("Starting training loop...")
        self.logger.info(f"{'='*80}\n")

        # Continue training using the refactored _train_loop
        self._train_loop(start_epoch=start_epoch, total_epochs=total_epochs)

    def _train_loop(self, start_epoch: int, total_epochs: int):
        """
        Core training loop that can be used for both new training and resumed training.

        Args:
            start_epoch: Epoch to start from (0 for new training, >0 for resume)
            total_epochs: Total number of epochs to train
        """
        # Training configuration
        self.num_epochs = total_epochs  # Store as instance variable for EMA scheduling
        early_stop_patience = self.config.get('early_stop_patience', 30)

        # Calculate early stop counter based on best epoch for resumed training
        if start_epoch > 0 and hasattr(self, 'best_metrics'):
            best_epoch = self.best_metrics.get('best_epoch', 0)
            early_stop_counter = start_epoch - best_epoch
            self.logger.info(f"📊 Resume: Early stop counter initialized to {early_stop_counter} (epochs since best: {best_epoch})")
        else:
            early_stop_counter = 0

        # Training loop
        for epoch in range(start_epoch, total_epochs):

            start_time = time.time()

            # Train epoch
            train_metrics = self.train_epoch(epoch)

            # Validate epoch
            val_metrics = self.validate_epoch(epoch)

            # Update learning rate
            if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(val_metrics['val_loss'])
            else:
                self.scheduler.step()

            # Get current learning rate
            current_lr = self.optimizer.param_groups[0]['lr']

            # Update history
            self.train_history['epoch'].append(epoch + 1)
            self.train_history['train_loss'].append(train_metrics['train_loss'])
            self.train_history['val_loss'].append(val_metrics['val_loss'])

            # Individual loss components for detailed tracking
            self.train_history['train_seg_loss'].append(train_metrics.get('train_seg_loss', 0))
            self.train_history['train_patch_loss'].append(train_metrics.get('train_patch_loss', 0))
            self.train_history['train_gland_loss'].append(train_metrics.get('train_gland_loss', 0))
            self.train_history['val_seg_loss'].append(val_metrics.get('val_seg_loss', 0))
            self.train_history['val_patch_loss'].append(val_metrics.get('val_patch_loss', 0))
            self.train_history['val_gland_loss'].append(val_metrics.get('val_gland_loss', 0))

            self.train_history['seg_dice'].append(val_metrics.get('val_dice_mean', 0))
            self.train_history['patch_accuracy'].append(val_metrics.get('val_patch_accuracy', 0))
            self.train_history['train_patch_accuracy'].append(train_metrics.get('train_patch_accuracy', 0))
            self.train_history['gland_accuracy'].append(val_metrics.get('val_gland_accuracy', 0))

            # NEW: Track comprehensive segmentation metrics
            self.train_history['train_dice_score'].append(train_metrics.get('train_dice_mean', 0))
            self.train_history['val_dice_score'].append(val_metrics.get('val_dice_mean', 0))
            self.train_history['train_iou_score'].append(train_metrics.get('train_iou_mean', 0))
            self.train_history['val_iou_score'].append(val_metrics.get('val_iou_mean', 0))
            self.train_history['train_pixel_accuracy'].append(train_metrics.get('train_pixel_accuracy_overall', 0))
            self.train_history['val_pixel_accuracy'].append(val_metrics.get('val_pixel_accuracy_overall', 0))

            # NEW: Track Pseudo-GT metrics (Student vs Teacher alignment monitoring)
            self.train_history['train_pseudo_dice'].append(train_metrics.get('pseudo_dice', 0))
            self.train_history['train_pseudo_iou'].append(train_metrics.get('pseudo_iou', 0))
            self.train_history['val_pseudo_dice'].append(val_metrics.get('val_pseudo_dice', 0))
            self.train_history['val_pseudo_iou'].append(val_metrics.get('val_pseudo_iou', 0))

            # Teacher-Student specific metrics
            self.train_history['train_consistency_loss'].append(train_metrics.get('train_consistency_loss', 0))
            self.train_history['val_consistency_loss'].append(val_metrics.get('val_consistency_loss', 0))
            # Alpha is epoch-level, so we need to get it from the last batch or compute it
            # For now, we'll compute it directly here
            if hasattr(self, 'loss_fn') and hasattr(self.loss_fn, 'alpha_scheduler'):
                current_alpha = self.loss_fn.alpha_scheduler.get_alpha(epoch)
                self.train_history['alpha'].append(current_alpha)
            else:
                self.train_history['alpha'].append(0.0)

            # Track EMA decay for Teacher-Student models
            architecture = self.config.get('architecture', 'nnunet')
            if architecture == 'teacher_student_unet':
                seg_model = self.model.segmentation_model
                if hasattr(seg_model, 'current_ema_decay'):
                    self.train_history['ema_decay'].append(seg_model.current_ema_decay)
                else:
                    self.train_history['ema_decay'].append(0.0)
            else:
                self.train_history['ema_decay'].append(0.0)

            self.train_history['learning_rate'].append(current_lr)

            # Log to tensorboard
            self.writer.add_scalar('Loss/Train', train_metrics['train_loss'], epoch)
            self.writer.add_scalar('Loss/Validation', val_metrics['val_loss'], epoch)
            self.writer.add_scalar('Metrics/Dice', val_metrics.get('val_dice_mean', 0), epoch)
            self.writer.add_scalar('Metrics/PatchAccuracy', val_metrics.get('val_patch_accuracy', 0), epoch)

            # Teacher-Student specific tensorboard logs
            if train_metrics.get('train_consistency_loss', 0) > 0 or val_metrics.get('val_consistency_loss', 0) > 0:
                self.writer.add_scalar('TeacherStudent/Train_Consistency_Loss', train_metrics.get('train_consistency_loss', 0), epoch)
                self.writer.add_scalar('TeacherStudent/Val_Consistency_Loss', val_metrics.get('val_consistency_loss', 0), epoch)
                if hasattr(self, 'loss_fn') and hasattr(self.loss_fn, 'alpha_scheduler'):
                    current_alpha = self.loss_fn.alpha_scheduler.get_alpha(epoch)
                    self.writer.add_scalar('TeacherStudent/Alpha', current_alpha, epoch)

            # NEW: Log comprehensive segmentation metrics
            self.writer.add_scalar('Segmentation/Train_Dice_Score', train_metrics.get('train_dice_mean', 0), epoch)
            self.writer.add_scalar('Segmentation/Val_Dice_Score', val_metrics.get('val_dice_mean', 0), epoch)
            self.writer.add_scalar('Segmentation/Train_IoU_Score', train_metrics.get('train_iou_mean', 0), epoch)
            self.writer.add_scalar('Segmentation/Val_IoU_Score', val_metrics.get('val_iou_mean', 0), epoch)
            self.writer.add_scalar('Segmentation/Train_Pixel_Accuracy', train_metrics.get('train_pixel_accuracy_overall', 0), epoch)
            self.writer.add_scalar('Segmentation/Val_Pixel_Accuracy', val_metrics.get('val_pixel_accuracy_overall', 0), epoch)

            self.writer.add_scalar('LearningRate', current_lr, epoch)

            # Log comprehensive metrics to console for user visibility
            self._log_epoch_metrics(epoch, train_metrics, val_metrics, current_lr, time.time() - start_time)

            # Check for best model
            is_best = val_metrics['val_loss'] < self.best_metrics['best_val_loss']
            if is_best:
                self.best_metrics['best_val_loss'] = val_metrics['val_loss']
                self.best_metrics['best_dice'] = val_metrics.get('val_dice_mean', 0)
                self.best_metrics['best_iou'] = val_metrics.get('val_iou_mean', 0)
                self.best_metrics['best_patch_acc'] = val_metrics.get('val_patch_accuracy', 0)
                self.best_metrics['best_epoch'] = epoch + 1
                early_stop_counter = 0
            else:
                early_stop_counter += 1

            # Save checkpoint
            self.save_checkpoint(epoch, {**train_metrics, **val_metrics}, is_best)

            # Print epoch summary
            epoch_time = time.time() - start_time
            self.logger.info(
                f"Epoch {epoch+1:3d}/{total_epochs} | "
                f"Train Loss: {train_metrics['train_loss']:.4f} | "
                f"Val Loss: {val_metrics['val_loss']:.4f} | "
                f"Dice: {val_metrics.get('val_dice_mean', 0):.4f} | "
                f"IoU: {val_metrics.get('val_iou_mean', 0):.4f} | "
                f"PatchAcc: {val_metrics.get('val_patch_accuracy', 0):.4f} | "
                f"LR: {current_lr:.2e} | "
                f"Time: {epoch_time:.1f}s"
            )

            # Early stopping
            if early_stop_counter >= early_stop_patience:
                self.logger.info(f"⏱️ Early stopping triggered after {early_stop_patience} epochs without improvement")
                break

            # Plot curves every 10 epochs
            if (epoch + 1) % 10 == 0:
                self.plot_training_curves()

        # Final plots and summary
        self.plot_training_curves()

        # Save training history as CSV files (following GlaS_MultiTask pattern)
        self.save_training_history()

        # Save final results summary
        self.save_final_summary()

        self.writer.close()

        self.logger.info("🎉 Training completed!")
        self.logger.info(f"📊 Best results:")
        self.logger.info(f"   🏆 Best epoch: {self.best_metrics['best_epoch']}")
        self.logger.info(f"   📉 Best validation loss: {self.best_metrics['best_val_loss']:.4f}")
        self.logger.info(f"   🎯 Best Dice score: {self.best_metrics['best_dice']:.4f}")
        self.logger.info(f"   📐 Best IoU score: {self.best_metrics['best_iou']:.4f}")
        self.logger.info(f"   🎪 Best patch accuracy: {self.best_metrics['best_patch_acc']:.4f}")

        # NEW: Post-training comprehensive evaluation on best model
        self.logger.info("🔬 Starting post-training evaluation on best model...")
        self.run_post_training_evaluation()

    def setup_logging(self):
        """Setup comprehensive logging"""
        log_file = self.logs_dir / "training.log"

        # Create additional specialized log files
        self.path_verification_log = self.logs_dir / "path_verification.log"
        self.dataset_summary_log = self.logs_dir / "dataset_summary.log"
        self.metrics_log = self.logs_dir / "training_metrics.log"

        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

        # Create dedicated loggers for specialized outputs
        self._setup_specialized_loggers()

    def _setup_specialized_loggers(self):
        """Setup specialized loggers for different types of output"""
        # Path verification logger
        self.path_logger = logging.getLogger('path_verification')
        self.path_logger.setLevel(logging.INFO)
        path_handler = logging.FileHandler(self.path_verification_log)
        path_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
        self.path_logger.addHandler(path_handler)

        # Dataset summary logger
        self.dataset_logger = logging.getLogger('dataset_summary')
        self.dataset_logger.setLevel(logging.INFO)
        dataset_handler = logging.FileHandler(self.dataset_summary_log)
        dataset_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
        self.dataset_logger.addHandler(dataset_handler)

        # Metrics logger
        self.metrics_logger = logging.getLogger('training_metrics')
        self.metrics_logger.setLevel(logging.INFO)
        metrics_handler = logging.FileHandler(self.metrics_log)
        metrics_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
        self.metrics_logger.addHandler(metrics_handler)

    def _save_comprehensive_logs_to_files(self):
        """Save comprehensive logs to dedicated files for user review"""
        from datetime import datetime

        # Create a comprehensive summary file
        summary_file = self.logs_dir / "comprehensive_verification_summary.md"

        with open(summary_file, 'w') as f:
            f.write("# Training Path and Dataset Verification Summary\n\n")
            f.write(f"**Generated:** {datetime.now().isoformat()}\n")
            f.write(f"**Experiment:** {self.experiment_name}\n")
            f.write(f"**Dataset:** {self.dataset_key}\n\n")

            # Get comprehensive paths
            paths_config = self._collect_all_paths_for_config()

            f.write("## Environment Variables\n\n")
            for var, value in paths_config["environment_variables"].items():
                status = "✅" if value else "❌"
                f.write(f"- **{var}**: {status} `{value or 'NOT SET'}`\n")

            f.write("\n## Dataset Configuration\n\n")
            for key, value in paths_config["dataset_config"].items():
                status = "✅" if value else "❌"
                f.write(f"- **{key}**: {status} `{value or 'NOT SET'}`\n")

            f.write("\n## Dataset Split Paths\n\n")
            for split_name, split_path in paths_config["dataset_split_paths"].items():
                exists = Path(split_path).exists()
                status = "✅" if exists else "❌"
                f.write(f"- **{split_name}**: {status} `{split_path}`\n")

            f.write("\n## Processing Paths\n\n")
            for path_name, path_value in paths_config["processing_paths"].items():
                if path_value:
                    exists = Path(path_value).exists()
                    status = "✅" if exists else "⚠️"
                    f.write(f"- **{path_name}**: {status} `{path_value}`\n")
                else:
                    f.write(f"- **{path_name}**: ❌ `NOT SET`\n")

            f.write("\n## Experiment Paths\n\n")
            for path_name, path_value in paths_config["experiment_paths"].items():
                if path_value:
                    exists = Path(path_value).exists()
                    status = "✅" if exists else "⚠️"
                    f.write(f"- **{path_name}**: {status} `{path_value}`\n")
                else:
                    f.write(f"- **{path_name}**: ❌ `NOT SET`\n")

            f.write("\n## Data Loader Verification\n\n")
            for loader_name, loader_path in paths_config["data_loader_sources"].items():
                status = "✅" if loader_path else "⚠️"
                f.write(f"- **{loader_name}**: {status} `{loader_path or 'NOT AVAILABLE'}`\n")

            # Add dataset statistics if available
            if hasattr(self, 'train_loader'):
                f.write("\n## Dataset Statistics\n\n")
                f.write(f"- **Training samples**: {len(self.train_loader.dataset):,}\n")
                f.write(f"- **Validation samples**: {len(self.val_loader.dataset):,}\n")
                f.write(f"- **Test samples**: {len(self.test_loader.dataset):,}\n")
                f.write(f"- **Total samples**: {len(self.train_loader.dataset) + len(self.val_loader.dataset) + len(self.test_loader.dataset):,}\n")
                f.write(f"- **Batch size**: {self.config.get('batch_size', 4)}\n")
                f.write(f"- **Training batches**: {len(self.train_loader):,}\n")
                f.write(f"- **Validation batches**: {len(self.val_loader):,}\n")
                f.write(f"- **Test batches**: {len(self.test_loader):,}\n")

            f.write("\n---\n")
            f.write("*This file was automatically generated for user verification and debugging purposes.*\n")

        self.logger.info(f"📄 Comprehensive verification summary saved to: {summary_file}")
        return summary_file

    def prepare_data(self):
        """Prepare data loaders for training"""
        self.logger.info("📊 Preparing data loaders...")

        # Get dataset path - use saved path if resuming, otherwise reconstruct
        if self.config.get('use_saved_dataset_paths', False):
            # RESUME MODE: Use the saved dataset paths from the original training config
            dataset_paths = self.config.get('dataset_paths', {})
            dataset_path = dataset_paths.get('training_dataset', '')

            if not dataset_path:
                self.logger.error("❌ Resume mode enabled but no saved dataset path found in config")
                raise ValueError("Cannot resume: dataset_paths not found in training_config.json")

            self.logger.info(f"📁 Using saved dataset path from config: {dataset_path}")
        else:
            # NORMAL MODE: Reconstruct path using get_dataset_path
            dataset_path = get_dataset_path(self.dataset_key)
            self.logger.info(f"📁 Dataset path resolved: {dataset_path}")

        # Verify dataset path exists
        dataset_path_obj = Path(dataset_path)
        if not dataset_path_obj.exists():
            self.logger.error(f"❌ Dataset path does not exist: {dataset_path}")
            raise FileNotFoundError(f"Dataset path not found: {dataset_path}")
        else:
            self.logger.info(f"✅ Dataset path verified and exists")

        # Create data loaders
        train_loader, val_loader, test_loader = create_combined_data_loaders(
            dataset_key=self.dataset_key,
            batch_size=self.config.get('batch_size', 4),
            num_workers=self.config.get('num_workers', 4),
            image_size=tuple(self.config.get('image_size', [512, 512])),
            use_multilabel_patch=self.config.get('use_multilabel_patch', True)
        )

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader

        # Log comprehensive dataset summary for user verification
        self._log_dataset_summary(dataset_path)

        # Log comprehensive paths for user verification
        self._log_comprehensive_paths()

        # Save all logs to dedicated files for user review
        summary_file = self._save_comprehensive_logs_to_files()
        self.logger.info(f"📋 All verification logs saved to output directory for user review")

    def _log_dataset_summary(self, dataset_path: str):
        """Log comprehensive dataset summary for user verification"""
        self.logger.info("=" * 80)
        self.logger.info("📊 DATASET SUMMARY FOR USER VERIFICATION")
        self.logger.info("=" * 80)

        # Dataset configuration
        self.logger.info(f"🎯 Selected Dataset: {self.dataset_key}")
        self.logger.info(f"📁 Dataset Path: {dataset_path}")

        # Environment verification
        from configs.paths_config import get_combined_data_base
        try:
            base_path = get_combined_data_base()
            self.logger.info(f"🔧 Environment Base: {base_path}")
        except ValueError as e:
            self.logger.warning(f"⚠️ Environment base: {e}")

        # Log specific dataset split paths for user verification
        self.logger.info(f"")
        self.logger.info(f"📂 DATASET SPLIT PATHS FOR USER VERIFICATION:")
        dataset_path_obj = Path(dataset_path)
        split_paths = {
            "Training Images": dataset_path_obj / "imagesTr",
            "Training Labels": dataset_path_obj / "labelsTr",
            "Validation Images": dataset_path_obj / "imagesVal",
            "Validation Labels": dataset_path_obj / "labelsVal",
            "Test Images": dataset_path_obj / "imagesTs",
            "Test Labels": dataset_path_obj / "labelsTs"
        }

        for split_name, split_path in split_paths.items():
            if split_path.exists():
                self.logger.info(f"   ✅ {split_name}: {split_path}")
            else:
                self.logger.error(f"   ❌ {split_name}: {split_path} (NOT FOUND)")

        # Log dataset.json path
        dataset_json_path = dataset_path_obj / "dataset.json"
        if dataset_json_path.exists():
            self.logger.info(f"   ✅ Dataset Config: {dataset_json_path}")
        else:
            self.logger.error(f"   ❌ Dataset Config: {dataset_json_path} (NOT FOUND)")

        # Data loader statistics
        self.logger.info(f"")
        self.logger.info(f"📈 DATASET STATISTICS:")
        self.logger.info(f"   📚 Training samples: {len(self.train_loader.dataset):,}")
        self.logger.info(f"   ✅ Validation samples: {len(self.val_loader.dataset):,}")
        self.logger.info(f"   🧪 Test samples: {len(self.test_loader.dataset):,}")
        self.logger.info(f"   📦 Total samples: {len(self.train_loader.dataset) + len(self.val_loader.dataset) + len(self.test_loader.dataset):,}")

        self.logger.info(f"")
        self.logger.info(f"🔄 BATCH CONFIGURATION:")
        self.logger.info(f"   📦 Batch size: {self.config.get('batch_size', 4)}")
        self.logger.info(f"   📚 Training batches: {len(self.train_loader):,}")
        self.logger.info(f"   ✅ Validation batches: {len(self.val_loader):,}")
        self.logger.info(f"   🧪 Test batches: {len(self.test_loader):,}")

        # Dataset directory structure verification
        self.logger.info(f"")
        self.logger.info(f"📂 DATASET DIRECTORY STRUCTURE:")
        dataset_path_obj = Path(dataset_path)
        required_dirs = ["imagesTr", "labelsTr", "imagesVal", "labelsVal", "imagesTs", "labelsTs"]
        for dir_name in required_dirs:
            dir_path = dataset_path_obj / dir_name
            if dir_path.exists():
                file_count = len(list(dir_path.glob("*")))
                self.logger.info(f"   ✅ {dir_name}: {file_count:,} files")
            else:
                self.logger.warning(f"   ❌ {dir_name}: MISSING")

        # Check for dataset.json
        dataset_json = dataset_path_obj / "dataset.json"
        if dataset_json.exists():
            try:
                import json
                with open(dataset_json, 'r') as f:
                    dataset_config = json.load(f)
                self.logger.info(f"   ✅ dataset.json: Found")
                if 'description' in dataset_config:
                    self.logger.info(f"   📝 Description: {dataset_config['description']}")
                if 'labels' in dataset_config:
                    self.logger.info(f"   🏷️ Class labels: {dataset_config['labels']}")
            except Exception as e:
                self.logger.warning(f"   ⚠️ dataset.json: Error reading - {e}")
        else:
            self.logger.warning(f"   ❌ dataset.json: MISSING")

        # Test data loading with samples from each split
        self.logger.info(f"")
        self.logger.info(f"🧪 TESTING DATA LOADING:")

        # Test training data
        try:
            train_sample = next(iter(self.train_loader))
            self.logger.info(f"   ✅ Training batch loaded:")
            self.logger.info(f"      🖼️ Images shape: {train_sample['images'].shape}")
            self.logger.info(f"      🎯 Segmentation targets shape: {train_sample['segmentation_targets'].shape}")
            self.logger.info(f"      🏷️ Patch labels shape: {train_sample['patch_labels'].shape}")
            self.logger.info(f"      🔍 Gland labels shape: {train_sample['gland_labels'].shape}")

            # Log unique classes in segmentation masks
            unique_classes = torch.unique(train_sample['segmentation_targets'])
            self.logger.info(f"      📊 Segmentation classes present: {unique_classes.tolist()}")

        except Exception as e:
            self.logger.error(f"   ❌ Training data loading failed: {e}")

        # Test validation data
        try:
            val_sample = next(iter(self.val_loader))
            self.logger.info(f"   ✅ Validation batch loaded:")
            self.logger.info(f"      🖼️ Images shape: {val_sample['images'].shape}")
            self.logger.info(f"      🎯 Segmentation targets shape: {val_sample['segmentation_targets'].shape}")

        except Exception as e:
            self.logger.error(f"   ❌ Validation data loading failed: {e}")

        # Test test data
        try:
            test_sample = next(iter(self.test_loader))
            self.logger.info(f"   ✅ Test batch loaded:")
            self.logger.info(f"      🖼️ Images shape: {test_sample['images'].shape}")
            self.logger.info(f"      🎯 Segmentation targets shape: {test_sample['segmentation_targets'].shape}")

        except Exception as e:
            self.logger.error(f"   ❌ Test data loading failed: {e}")

        self.logger.info("=" * 80)
        self.logger.info("✅ Data loaders prepared and verified - Ready for training!")

    def _collect_all_paths_for_config(self) -> dict:
        """Collect all important paths for saving to training configuration"""
        from configs.paths_config import get_combined_data_base, DATA_PATHS
        import os

        dataset_path_obj = Path(get_dataset_path(self.dataset_key))

        # Collect all important paths
        paths_config = {
            # Environment Variables
            "environment_variables": {
                "GLAND_DATASET_BASE": os.getenv('GLAND_DATASET_BASE'),
                "GLAND_OUTPUT_DIR": os.getenv('GLAND_OUTPUT_DIR'),
                "NNUNET_PREPROCESSED": os.getenv('NNUNET_PREPROCESSED'),
                "NNUNET_RESULTS": os.getenv('NNUNET_RESULTS'),
                "GLAND_TEMP_DIR": os.getenv('GLAND_TEMP_DIR')
            },

            # Dataset Configuration
            "dataset_config": {
                "dataset_key": self.dataset_key,
                "resolved_dataset_path": str(dataset_path_obj),
                "dataset_base_from_env": get_combined_data_base() if get_combined_data_base() else None
            },

            # Dataset Split Paths
            "dataset_split_paths": {
                "training_images": str(dataset_path_obj / "imagesTr"),
                "training_labels": str(dataset_path_obj / "labelsTr"),
                "validation_images": str(dataset_path_obj / "imagesVal"),
                "validation_labels": str(dataset_path_obj / "labelsVal"),
                "test_images": str(dataset_path_obj / "imagesTs"),
                "test_labels": str(dataset_path_obj / "labelsTs"),
                "dataset_json": str(dataset_path_obj / "dataset.json")
            },

            # Processing Paths
            "processing_paths": {
                "nnunet_preprocessed": DATA_PATHS.get("nnunet_preprocessed"),
                "nnunet_results": DATA_PATHS.get("nnunet_results"),
                "temp_dir": DATA_PATHS.get("temp_dir")
            },

            # Experiment Paths
            "experiment_paths": {
                "output_base_dir": str(self.output_base_dir),
                "experiment_dir": str(self.output_dir),
                "models_dir": str(self.output_dir / "models"),
                "logs_dir": str(self.output_dir / "logs"),
                "visualizations_dir": str(self.output_dir / "visualizations")
            },

            # Data Loader Sources (for verification)
            "data_loader_sources": {
                "training_dataset_root": str(self.train_loader.dataset.data_root) if hasattr(self, 'train_loader') else None,
                "validation_dataset_root": str(self.val_loader.dataset.data_root) if hasattr(self, 'val_loader') else None,
                "test_dataset_root": str(self.test_loader.dataset.data_root) if hasattr(self, 'test_loader') else None
            }
        }

        return paths_config

    def _log_comprehensive_paths(self):
        """Log all important paths comprehensively for user verification"""
        separator = "=" * 100
        header = "🗂️  COMPREHENSIVE PATH CONFIGURATION FOR USER VERIFICATION"

        # Log to main logger and path verification logger
        self.logger.info(separator)
        self.logger.info(header)
        self.logger.info(separator)

        self.path_logger.info(separator)
        self.path_logger.info(header)
        self.path_logger.info(separator)

        paths_config = self._collect_all_paths_for_config()

        # Log environment variables
        env_header = "🌍 ENVIRONMENT VARIABLES:"
        self.logger.info(env_header)
        self.path_logger.info(env_header)

        for var, value in paths_config["environment_variables"].items():
            if value:
                msg = f"   ✅ {var}: {value}"
                self.logger.info(msg)
                self.path_logger.info(msg)
            else:
                msg = f"   ❌ {var}: NOT SET"
                self.logger.error(msg)
                self.path_logger.error(msg)

        # Log dataset configuration
        self.logger.info("")
        self.path_logger.info("")
        dataset_header = "📊 DATASET CONFIGURATION:"
        self.logger.info(dataset_header)
        self.path_logger.info(dataset_header)

        for key, value in paths_config["dataset_config"].items():
            if value:
                msg = f"   ✅ {key}: {value}"
                self.logger.info(msg)
                self.path_logger.info(msg)
            else:
                msg = f"   ❌ {key}: NOT SET"
                self.logger.error(msg)
                self.path_logger.error(msg)

        # Log dataset split paths
        self.logger.info("")
        self.path_logger.info("")
        split_header = "📂 DATASET SPLIT PATHS:"
        self.logger.info(split_header)
        self.path_logger.info(split_header)

        for split_name, split_path in paths_config["dataset_split_paths"].items():
            if Path(split_path).exists():
                msg = f"   ✅ {split_name}: {split_path}"
                self.logger.info(msg)
                self.path_logger.info(msg)
            else:
                msg = f"   ❌ {split_name}: {split_path} (NOT FOUND)"
                self.logger.error(msg)
                self.path_logger.error(msg)

        # Log processing paths
        self.logger.info("")
        self.path_logger.info("")
        proc_header = "⚙️ PROCESSING PATHS:"
        self.logger.info(proc_header)
        self.path_logger.info(proc_header)

        for path_name, path_value in paths_config["processing_paths"].items():
            if path_value:
                exists = "✅" if Path(path_value).exists() else "⚠️"
                msg = f"   {exists} {path_name}: {path_value}"
                self.logger.info(msg)
                self.path_logger.info(msg)
            else:
                msg = f"   ❌ {path_name}: NOT SET"
                self.logger.error(msg)
                self.path_logger.error(msg)

        # Log experiment paths
        self.logger.info("")
        self.path_logger.info("")
        exp_header = "🧪 EXPERIMENT PATHS:"
        self.logger.info(exp_header)
        self.path_logger.info(exp_header)

        for path_name, path_value in paths_config["experiment_paths"].items():
            if path_value:
                exists = "✅" if Path(path_value).exists() else "⚠️"
                msg = f"   {exists} {path_name}: {path_value}"
                self.logger.info(msg)
                self.path_logger.info(msg)
            else:
                msg = f"   ❌ {path_name}: NOT SET"
                self.logger.error(msg)
                self.path_logger.error(msg)

        # Log data loader sources
        self.logger.info("")
        self.path_logger.info("")
        loader_header = "🔄 DATA LOADER VERIFICATION:"
        self.logger.info(loader_header)
        self.path_logger.info(loader_header)

        for loader_name, loader_path in paths_config["data_loader_sources"].items():
            if loader_path:
                msg = f"   ✅ {loader_name}: {loader_path}"
                self.logger.info(msg)
                self.path_logger.info(msg)
            else:
                msg = f"   ⚠️ {loader_name}: NOT AVAILABLE"
                self.logger.warning(msg)
                self.path_logger.warning(msg)

        footer = "✅ All paths logged and ready for training!"
        self.logger.info(separator)
        self.logger.info(footer)

        self.path_logger.info(separator)
        self.path_logger.info(footer)

    def _log_epoch_metrics(self, epoch: int, train_metrics: dict, val_metrics: dict, current_lr: float, epoch_time: float):
        """Log comprehensive metrics for each epoch"""
        # Create a nicely formatted metrics log
        epoch_num = epoch + 1

        # Header for the epoch
        self.logger.info("=" * 100)
        self.logger.info(f"📊 EPOCH {epoch_num} METRICS SUMMARY")
        self.logger.info("=" * 100)

        # Time and learning rate info
        self.logger.info(f"⏱️  Epoch Time: {epoch_time:.2f}s | 📈 Learning Rate: {current_lr:.2e}")

        # Loss metrics
        self.logger.info("")
        self.logger.info("🔢 LOSS METRICS:")
        self.logger.info(f"   📉 Total Loss    - Train: {train_metrics.get('train_loss', 0):.4f} | Val: {val_metrics.get('val_loss', 0):.4f}")
        self.logger.info(f"   🎯 Seg Loss     - Train: {train_metrics.get('train_seg_loss', 0):.4f} | Val: {val_metrics.get('val_seg_loss', 0):.4f}")
        self.logger.info(f"   🏷️  Patch Loss   - Train: {train_metrics.get('train_patch_loss', 0):.4f} | Val: {val_metrics.get('val_patch_loss', 0):.4f}")
        self.logger.info(f"   🔍 Gland Loss   - Train: {train_metrics.get('train_gland_loss', 0):.4f} | Val: {val_metrics.get('val_gland_loss', 0):.4f}")

        # Segmentation metrics
        self.logger.info("")
        self.logger.info("🖼️  SEGMENTATION METRICS:")
        self.logger.info(f"   🎯 Dice Score   - Train: {train_metrics.get('train_dice_mean', 0):.4f} | Val: {val_metrics.get('val_dice_mean', 0):.4f}")
        self.logger.info(f"   📐 IoU Score    - Train: {train_metrics.get('train_iou_mean', 0):.4f} | Val: {val_metrics.get('val_iou_mean', 0):.4f}")
        self.logger.info(f"   🎨 Pixel Acc    - Train: {train_metrics.get('train_pixel_accuracy_overall', 0):.4f} | Val: {val_metrics.get('val_pixel_accuracy_overall', 0):.4f}")

        # Per-class segmentation metrics (if available)
        if 'train_dice_per_class' in train_metrics:
            self.logger.info("")
            self.logger.info("📋 PER-CLASS SEGMENTATION METRICS:")
            train_dice_per_class = train_metrics.get('train_dice_per_class', [])
            val_dice_per_class = val_metrics.get('val_dice_per_class', [])
            class_names = ["Background", "Benign", "Malignant", "PDC"]

            for i, class_name in enumerate(class_names):
                if i < len(train_dice_per_class) and i < len(val_dice_per_class):
                    self.logger.info(f"   🏷️  {class_name:<10} - Train: {train_dice_per_class[i]:.4f} | Val: {val_dice_per_class[i]:.4f}")

        # Classification metrics
        self.logger.info("")
        self.logger.info("🎪 CLASSIFICATION METRICS:")
        self.logger.info(f"   🏷️  Patch Acc    - Train: {train_metrics.get('train_patch_accuracy', 0):.4f} | Val: {val_metrics.get('val_patch_accuracy', 0):.4f}")
        self.logger.info(f"   🔍 Gland Acc    - Train: {train_metrics.get('train_gland_accuracy', 0):.4f} | Val: {val_metrics.get('val_gland_accuracy', 0):.4f}")

        # Best metrics tracking
        current_val_loss = val_metrics.get('val_loss', float('inf'))
        current_dice = val_metrics.get('val_dice_mean', 0)
        current_iou = val_metrics.get('val_iou_mean', 0)
        current_patch_acc = val_metrics.get('val_patch_accuracy', 0)

        is_best_loss = current_val_loss < self.best_metrics['best_val_loss']
        is_best_dice = current_dice > self.best_metrics['best_dice']
        is_best_iou = current_iou > self.best_metrics['best_iou']
        is_best_patch = current_patch_acc > self.best_metrics['best_patch_acc']

        self.logger.info("")
        self.logger.info("🏆 BEST METRICS TRACKING:")
        best_loss_marker = "🆕" if is_best_loss else "  "
        best_dice_marker = "🆕" if is_best_dice else "  "
        best_iou_marker = "🆕" if is_best_iou else "  "
        best_patch_marker = "🆕" if is_best_patch else "  "

        self.logger.info(f"   {best_loss_marker} Best Val Loss: {min(current_val_loss, self.best_metrics['best_val_loss']):.4f}")
        self.logger.info(f"   {best_dice_marker} Best Dice:     {max(current_dice, self.best_metrics['best_dice']):.4f}")
        self.logger.info(f"   {best_iou_marker} Best IoU:      {max(current_iou, self.best_metrics['best_iou']):.4f}")
        self.logger.info(f"   {best_patch_marker} Best Patch Acc: {max(current_patch_acc, self.best_metrics['best_patch_acc']):.4f}")

        # Progress indicator
        total_epochs = self.config.get('epochs', 150)
        progress = (epoch_num / total_epochs) * 100
        progress_bar = "█" * int(progress // 5) + "░" * (20 - int(progress // 5))

        self.logger.info("")
        self.logger.info(f"📈 TRAINING PROGRESS: [{progress_bar}] {progress:.1f}% ({epoch_num}/{total_epochs})")

        self.logger.info("=" * 100)

        # Also log to dedicated metrics file
        self._log_metrics_to_file(epoch_num, train_metrics, val_metrics, current_lr, epoch_time)

    def _log_metrics_to_file(self, epoch_num: int, train_metrics: dict, val_metrics: dict, current_lr: float, epoch_time: float):
        """Log metrics to dedicated file in CSV format for easy analysis"""
        # Create header if this is the first epoch
        if epoch_num == 1:
            header = (
                "epoch,epoch_time,learning_rate,"
                "train_total_loss,val_total_loss,"
                "train_seg_loss,val_seg_loss,"
                "train_patch_loss,val_patch_loss,"
                "train_gland_loss,val_gland_loss,"
                "train_dice_mean,val_dice_mean,"
                "train_iou_mean,val_iou_mean,"
                "train_pixel_accuracy,val_pixel_accuracy,"
                "train_patch_accuracy,val_patch_accuracy,"
                "train_gland_accuracy,val_gland_accuracy"
            )
            self.metrics_logger.info(header)

        # Log current epoch metrics
        metrics_row = (
            f"{epoch_num},{epoch_time:.2f},{current_lr:.2e},"
            f"{train_metrics.get('train_loss', 0):.6f},{val_metrics.get('val_loss', 0):.6f},"
            f"{train_metrics.get('train_seg_loss', 0):.6f},{val_metrics.get('val_seg_loss', 0):.6f},"
            f"{train_metrics.get('train_patch_loss', 0):.6f},{val_metrics.get('val_patch_loss', 0):.6f},"
            f"{train_metrics.get('train_gland_loss', 0):.6f},{val_metrics.get('val_gland_loss', 0):.6f},"
            f"{train_metrics.get('train_dice_mean', 0):.6f},{val_metrics.get('val_dice_mean', 0):.6f},"
            f"{train_metrics.get('train_iou_mean', 0):.6f},{val_metrics.get('val_iou_mean', 0):.6f},"
            f"{train_metrics.get('train_pixel_accuracy_overall', 0):.6f},{val_metrics.get('val_pixel_accuracy_overall', 0):.6f},"
            f"{train_metrics.get('train_patch_accuracy', 0):.6f},{val_metrics.get('val_patch_accuracy', 0):.6f},"
            f"{train_metrics.get('train_gland_accuracy', 0):.6f},{val_metrics.get('val_gland_accuracy', 0):.6f}"
        )
        self.metrics_logger.info(metrics_row)

    def setup_model(self):
        """Setup model, optimizer, scheduler, and loss function"""
        self.logger.info("🏗️ Setting up model and training components...")

        # Create model with architecture-specific parameters
        architecture = self.config.get('architecture', 'nnunet')
        model_kwargs = {}

        # Add Teacher-Student specific parameters if using teacher_student_unet
        if architecture == 'teacher_student_unet' and 'teacher_student_unet' in self.config:
            ts_config = self.config['teacher_student_unet']
            model_kwargs.update({
                'ema_decay': ts_config.get('ema_decay', 0.999),
                'teacher_init_epoch': ts_config.get('teacher_init_epoch', 20),
                'min_alpha': ts_config.get('min_alpha', 0.1),
                'max_alpha': ts_config.get('max_alpha', 1.0),
                'consistency_loss_type': ts_config.get('consistency_loss_type', 'mse'),
                'consistency_temperature': ts_config.get('consistency_temperature', 1.0),
                'depth': ts_config.get('depth', 4),
                'initial_channels': ts_config.get('initial_channels', 64)
            })
            self.logger.info(f"🎓 Teacher-Student config: {model_kwargs}")

        self.model = create_multitask_model(
            architecture=architecture,
            input_channels=3,
            num_seg_classes=self.config.get('num_seg_classes', 4),
            enable_classification=True,
            **model_kwargs
        )
        self.model.to(self.device)

        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        self.logger.info(f"📊 Model parameters: {total_params:,} total, {trainable_params:,} trainable")

        # Setup loss function
        architecture = self.config.get('architecture', 'nnunet')

        if architecture == 'teacher_student_unet':
            # Use Teacher-Student loss function
            ts_config = self.config.get('teacher_student_unet', {})
            self.loss_fn = TeacherStudentLoss(
                total_epochs=self.config.get('epochs', 100),
                warmup_epochs=ts_config.get('teacher_init_epoch', 20),
                min_alpha=ts_config.get('min_alpha', 0.01),
                max_alpha=ts_config.get('max_alpha', 1.0),
                consistency_loss_config={
                    'loss_type': ts_config.get('consistency_loss_type', 'kl_div'),
                    'temperature': ts_config.get('consistency_temperature', 1.0),
                    'enable_gland_consistency': ts_config.get('enable_gland_consistency', False),
                    'pseudo_mask_filtering': ts_config.get('pseudo_mask_filtering', 'none'),
                    'confidence_threshold': ts_config.get('confidence_threshold', 0.8),
                    'entropy_threshold': ts_config.get('entropy_threshold', 1.0),
                    'confidence_annealing': ts_config.get('confidence_annealing', 'none'),
                    'confidence_max_threshold': ts_config.get('confidence_max_threshold', 0.9),
                    'confidence_min_threshold': ts_config.get('confidence_min_threshold', 0.6),
                    'confidence_annealing_start_epoch': ts_config.get('confidence_annealing_start_epoch', 5),
                    'total_epochs': self.config.get('epochs', 100),
                    'gt_teacher_incorporate_enabled': ts_config.get('gt_teacher_incorporate_enabled', False),
                    'gt_incorporate_start_epoch': ts_config.get('gt_incorporate_start_epoch', 0),
                    'gt_incorporate_segmentation_only': ts_config.get('gt_incorporate_segmentation_only', True)
                },
                supervised_loss_config={
                    'use_multilabel_patch': self.config.get('use_multilabel_patch', True),
                    'use_focal_loss': self.config.get('use_focal_loss', False),
                    'use_adaptive_weighting': self.config.get('adaptive_weighting', True),
                    'dice_weight': self.config.get('dice_weight', 0.5),
                    'ce_weight': self.config.get('ce_weight', 0.5)
                }
            )
            self.logger.info(f"🔄 Using Teacher-Student loss function with {ts_config.get('consistency_loss_type', 'kl_div')} consistency loss")
        else:
            # Use standard multi-task loss function
            self.loss_fn = MultiTaskLoss(
                use_multilabel_patch=self.config.get('use_multilabel_patch', True),
                use_focal_loss=self.config.get('use_focal_loss', False),
                use_adaptive_weighting=self.config.get('adaptive_weighting', True),
                dice_weight=self.config.get('dice_weight', 0.5),
                ce_weight=self.config.get('ce_weight', 0.5)
            )
            self.logger.info(f"📊 Using standard multi-task loss function")

        # Setup optimizer
        optimizer_name = self.config.get('optimizer', 'AdamW')
        learning_rate = self.config.get('learning_rate', 1e-4)
        weight_decay = self.config.get('weight_decay', 1e-5)

        if optimizer_name.lower() == 'adamw':
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay,
                betas=(0.9, 0.999)
            )
        elif optimizer_name.lower() == 'sgd':
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=learning_rate,
                momentum=0.9,
                weight_decay=weight_decay
            )
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_name}")

        # Setup scheduler
        scheduler_name = self.config.get('scheduler', 'CosineAnnealingLR')
        if scheduler_name == 'CosineAnnealingLR':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.get('cosine_t_max', self.config.get('epochs', 150)),
                eta_min=self.config.get('cosine_eta_min', self.config.get('min_lr', 1e-7))
            )
        elif scheduler_name == 'ReduceLROnPlateau':
            lr_patience = self.config.get('lr_scheduler_patience', 15)
            min_lr = self.config.get('min_lr', 1e-7)
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=0.5,
                patience=lr_patience,
                threshold=0.001,
                min_lr=min_lr,
                verbose=True
            )
        elif scheduler_name == 'poly':
            # Polynomial learning rate decay
            self.scheduler = optim.lr_scheduler.PolynomialLR(
                self.optimizer,
                total_iters=self.config.get('epochs', 150),
                power=0.9
            )
        else:
            raise ValueError(f"Unsupported scheduler: {scheduler_name}. Supported schedulers: CosineAnnealingLR, ReduceLROnPlateau, poly")

        # Setup tensorboard
        self.writer = SummaryWriter(self.logs_dir / "tensorboard")

        self.logger.info(f"✅ Model setup complete:")
        self.logger.info(f"   🔧 Optimizer: {optimizer_name} (lr={learning_rate})")
        self.logger.info(f"   📈 Scheduler: {scheduler_name}")
        self.logger.info(f"   💾 Multi-label patches: {self.config.get('use_multilabel_patch', True)}")

    def calculate_metrics(self, outputs: Dict, targets: Dict, prefix: str = "", teacher_outputs: Dict = None) -> Dict[str, float]:
        """Calculate comprehensive metrics for multi-task learning"""
        metrics = {}

        # Segmentation metrics
        seg_pred = outputs['segmentation']

        seg_target = targets['segmentation']

        # Dice coefficient per class
        seg_pred_labels = torch.argmax(seg_pred, dim=1)
        dice_scores = []

        for class_idx in range(self.config.get('num_seg_classes', 4)):
            pred_mask = (seg_pred_labels == class_idx).float()
            true_mask = (seg_target == class_idx).float()

            intersection = (pred_mask * true_mask).sum()
            union = pred_mask.sum() + true_mask.sum()

            if union > 0:
                dice = (2.0 * intersection) / (union + 1e-8)
                dice_scores.append(dice.item())
            else:
                dice_scores.append(1.0)  # Perfect score if no pixels of this class

        metrics[f'{prefix}dice_mean'] = np.mean(dice_scores)
        for i, dice in enumerate(dice_scores):
            metrics[f'{prefix}dice_class_{i}'] = dice

        # NEW: Comprehensive segmentation metrics using the metrics calculator
        comprehensive_metrics = self.metrics_calculator.compute_all_metrics(seg_pred, seg_target)

        # Add comprehensive metrics to the main metrics dict with appropriate prefix
        for key, value in comprehensive_metrics.items():
            metrics[f'{prefix}{key}'] = value

        # NEW: Pseudo-GT metrics for Teacher-Student models (Student vs Pseudo-GT)
        # MONITORING ONLY - Complete GPU memory isolation with immediate cleanup
        if teacher_outputs is not None and 'segmentation' in teacher_outputs:
            with torch.no_grad():  # Ensure no gradients are tracked
                try:
                    # CRITICAL: Complete detachment and immediate CPU transfer to prevent GPU memory retention
                    teacher_seg_pred = teacher_outputs['segmentation'].detach().cpu()  # Move to CPU immediately
                    teacher_pseudo_masks = torch.argmax(teacher_seg_pred, dim=1).detach()  # Detached CPU tensor

                    # Student predictions - already detached and moved to CPU
                    student_labels_cpu = seg_pred_labels.detach().cpu()

                    # Force GPU memory cleanup immediately after tensor extraction
                    torch.cuda.empty_cache()

                    # Compute Student vs Pseudo-GT metrics entirely on CPU (zero GPU impact)
                    pseudo_dice_scores = []
                    pseudo_iou_scores = []

                    for class_idx in range(self.config.get('num_seg_classes', 4)):
                        # Create masks for current class (CPU only)
                        student_mask = (student_labels_cpu == class_idx).float()
                        pseudo_gt_mask = (teacher_pseudo_masks == class_idx).float()

                        # Pseudo-Dice computation (CPU only)
                        intersection = (student_mask * pseudo_gt_mask).sum()
                        union = student_mask.sum() + pseudo_gt_mask.sum()

                        if union > 0:
                            pseudo_dice = (2.0 * intersection) / (union + 1e-8)
                            pseudo_dice_scores.append(pseudo_dice.item())

                            # Pseudo-IoU computation (CPU only)
                            union_iou = student_mask.sum() + pseudo_gt_mask.sum() - intersection
                            if union_iou > 0:
                                pseudo_iou = intersection / (union_iou + 1e-8)
                                pseudo_iou_scores.append(pseudo_iou.item())
                            else:
                                pseudo_iou_scores.append(1.0)
                        else:
                            pseudo_dice_scores.append(1.0)  # Perfect score if no pixels
                            pseudo_iou_scores.append(1.0)

                        # Immediate cleanup of CPU tensors to prevent accumulation
                        del student_mask, pseudo_gt_mask, intersection, union

                    # Add pseudo-GT metrics to main metrics (monitoring only)
                    if pseudo_dice_scores:
                        metrics[f'{prefix}pseudo_dice'] = np.mean(pseudo_dice_scores)
                    if pseudo_iou_scores:
                        metrics[f'{prefix}pseudo_iou'] = np.mean(pseudo_iou_scores)

                    # Complete cleanup of all CPU tensors
                    del teacher_seg_pred, teacher_pseudo_masks, student_labels_cpu

                except (torch.cuda.OutOfMemoryError, RuntimeError):
                    # If still running out of memory, skip pseudo-GT metrics entirely
                    pass

                # Final GPU memory cleanup to ensure no leaks
                torch.cuda.empty_cache()

        # Classification metrics
        if 'patch_classification' in outputs and 'patch_labels' in targets:
            patch_pred = outputs['patch_classification']
            patch_target = targets['patch_labels']

            if self.config.get('use_multilabel_patch', True):
                # Multi-label accuracy
                patch_pred_binary = (torch.sigmoid(patch_pred) > 0.5).float()
                patch_accuracy = (patch_pred_binary == patch_target).float().mean()
            else:
                # Single-label accuracy
                patch_pred_labels = torch.argmax(patch_pred, dim=1)
                patch_accuracy = (patch_pred_labels == patch_target).float().mean()

            metrics[f'{prefix}patch_accuracy'] = patch_accuracy.item()

        if 'gland_classification' in outputs and len(outputs['gland_classification']) > 0:
            gland_pred = outputs['gland_classification']

            # For gland classification, we need to extract actual gland labels
            # This is more complex and would require gland instance extraction
            # For now, we'll compute a simple metric
            if len(gland_pred) > 0:
                gland_pred_labels = torch.argmax(gland_pred, dim=1)
                # Simplified gland accuracy (in practice, would need ground truth gland labels)
                metrics[f'{prefix}gland_prediction_entropy'] = torch.mean(
                    -torch.sum(torch.softmax(gland_pred, dim=1) * torch.log_softmax(gland_pred, dim=1), dim=1)
                ).item()

        return metrics

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        seg_loss = 0.0
        patch_loss = 0.0
        gland_loss = 0.0
        consistency_loss = 0.0
        all_metrics = {}

        progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}")

        for batch_idx, batch_data in enumerate(progress_bar):
            # Move data to device
            images = batch_data['images'].to(self.device)
            seg_masks = batch_data['segmentation_targets'].to(self.device)
            patch_labels = batch_data['patch_labels'].to(self.device)

            # Forward pass
            self.optimizer.zero_grad()

            # Check if this is a Teacher-Student model
            architecture = self.config.get('architecture', 'nnunet')
            if architecture == 'teacher_student_unet':
                # For Teacher-Student models, we need to handle different modes
                seg_model = self.model.segmentation_model  # Get the actual Teacher-Student model

                # Check if teacher should be initialized
                if hasattr(seg_model, 'should_initialize_teacher'):
                    # Get current validation loss (approximate with current loss for initialization)
                    current_val_loss = total_loss / max(1, batch_idx) if batch_idx > 0 else float('inf')
                    if seg_model.should_initialize_teacher(epoch, current_val_loss):
                        seg_model.initialize_teacher()

                # Choose forward mode based on teacher initialization and update EMA
                if hasattr(seg_model, 'teacher_initialized') and seg_model.teacher_initialized:
                    ts_mode = "teacher_student"
                    # Update teacher EMA weights before forward pass (with dynamic scheduling)
                    seg_model.update_teacher_ema(current_epoch=epoch, total_epochs=self.num_epochs)

                    # Log EMA decay value and annealing progress for monitoring (once per epoch)
                    if batch_idx == 0:
                        current_ema_decay = seg_model.get_dynamic_ema_decay(current_epoch=epoch, total_epochs=self.num_epochs)
                        if hasattr(seg_model, 'ema_schedule') and seg_model.ema_schedule != "fixed":
                            # Calculate annealing progress percentage
                            if epoch >= seg_model.ema_annealing_start_epoch:
                                progress = (epoch - seg_model.ema_annealing_start_epoch) / (self.num_epochs - seg_model.ema_annealing_start_epoch)
                                progress_pct = min(progress * 100, 100.0)
                                self.logger.info(f"📊 EMA Decay: {current_ema_decay:.4f} ({seg_model.ema_schedule} schedule, {progress_pct:.1f}% progress)")
                            else:
                                self.logger.info(f"📊 EMA Decay: {current_ema_decay:.4f} ({seg_model.ema_schedule} schedule, pre-annealing)")
                        else:
                            self.logger.info(f"📊 EMA Decay: {current_ema_decay:.4f} (fixed schedule)")
                else:
                    ts_mode = "student_only"

                # Forward through MultiTaskWrapper (which handles Teacher-Student correctly)
                outputs = self.model(images, mode=ts_mode)

                # For Teacher-Student loss, extract teacher outputs from existing computation
                if ts_mode == "teacher_student":
                    # Teacher outputs are already computed in teacher_student mode - extract them
                    # No additional forward pass needed - this saves GPU memory!
                    if 'teacher' in outputs:
                        teacher_outputs = {'segmentation': outputs['teacher']}
                        # Add teacher patch classification if available
                        if 'teacher_patch_classification' in outputs:
                            teacher_outputs['patch_classification'] = outputs['teacher_patch_classification']
                        if 'teacher_gland_classification' in outputs:
                            teacher_outputs['gland_classification'] = outputs['teacher_gland_classification']
                    else:
                        teacher_outputs = None
                else:
                    teacher_outputs = None
            else:
                # Standard model forward pass
                outputs = self.model(images)
                teacher_outputs = None

            # Prepare targets
            targets = {
                'segmentation': seg_masks,
                'patch_labels': patch_labels
            }

            # Calculate loss - pass epoch and teacher outputs for Teacher-Student loss
            if architecture == 'teacher_student_unet':
                loss_dict = self.loss_fn(outputs, teacher_outputs, targets, epoch)
            else:
                loss_dict = self.loss_fn(outputs, targets)

            total_loss_value = loss_dict['total_loss'] if 'total_loss' in loss_dict else loss_dict['total']

            # Ensure total_loss_value is a scalar for Teacher-Student models
            if architecture == 'teacher_student_unet' and hasattr(total_loss_value, 'dim') and total_loss_value.dim() > 0:
                total_loss_value = total_loss_value.mean()

            # Backward pass
            total_loss_value.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()

            # Update metrics
            total_loss += total_loss_value.item()

            # Track individual loss components (handle both standard and Teacher-Student prefixed keys)

            # For Teacher-Student models, the keys are prefixed with 'supervised_'
            seg_key = 'supervised_segmentation' if 'supervised_segmentation' in loss_dict else 'segmentation'
            patch_key = 'supervised_patch_classification' if 'supervised_patch_classification' in loss_dict else 'patch_classification'
            gland_key = 'supervised_gland_classification' if 'supervised_gland_classification' in loss_dict else 'gland_classification'

            if seg_key in loss_dict and loss_dict[seg_key].numel() == 1:
                seg_loss += loss_dict[seg_key].item()
            if patch_key in loss_dict and loss_dict[patch_key].numel() == 1:
                patch_loss += loss_dict[patch_key].item()
            if gland_key in loss_dict and loss_dict[gland_key].numel() == 1:
                gland_loss += loss_dict[gland_key].item()
            if 'consistency_loss' in loss_dict and loss_dict['consistency_loss'].numel() == 1:
                consistency_loss += loss_dict['consistency_loss'].item()

            # Calculate batch metrics
            with torch.no_grad():
                # Pass teacher outputs for pseudo-GT metrics computation when available
                batch_metrics = self.calculate_metrics(outputs, targets, teacher_outputs=teacher_outputs)
                for key, value in batch_metrics.items():
                    if key not in all_metrics:
                        all_metrics[key] = []
                    all_metrics[key].append(value)

            # Update progress bar with comprehensive metrics
            current_lr = self.optimizer.param_groups[0]['lr']
            postfix_dict = {
                'Loss': f"{total_loss_value.item():.4f}",
                'Dice': f"{batch_metrics.get('dice_mean', 0):.4f}",
                'IoU': f"{batch_metrics.get('iou_mean', 0):.4f}",
                'PatchAcc': f"{batch_metrics.get('patch_accuracy', 0):.4f}"
            }

            # Add consistency loss and alpha for Teacher-Student models
            if 'consistency_loss' in loss_dict:
                postfix_dict['Teacher_Consist_Loss'] = f"{loss_dict['consistency_loss'].item():.4f}"
            if 'alpha' in loss_dict:
                postfix_dict['Supervision_Weightage'] = f"{loss_dict['alpha']:.3f}"

            # Add EMA Decay for Teacher-Student models
            architecture = self.config.get('architecture', 'nnunet')
            if architecture == 'teacher_student_unet':
                seg_model = self.model.segmentation_model
                if hasattr(seg_model, 'current_ema_decay'):
                    postfix_dict['EMA'] = f"{seg_model.current_ema_decay:.4f}"

            # Add Pseudo-GT metrics for Teacher-Student models (monitoring only)
            if 'pseudo_dice' in batch_metrics:
                postfix_dict['Pseudo-Dice'] = f"{batch_metrics['pseudo_dice']:.4f}"
            if 'pseudo_iou' in batch_metrics:
                postfix_dict['Pseudo-IoU'] = f"{batch_metrics['pseudo_iou']:.4f}"

            postfix_dict['LR'] = f"{current_lr:.2e}"
            progress_bar.set_postfix(postfix_dict)

            # if batch_idx >  25:
            #     break # Hikmat

        # Average metrics
        avg_metrics = {}
        for key, values in all_metrics.items():
            avg_metrics[key] = np.mean(values)

        avg_metrics['train_loss'] = total_loss / len(self.train_loader)
        avg_metrics['train_seg_loss'] = seg_loss / len(self.train_loader)
        avg_metrics['train_patch_loss'] = patch_loss / len(self.train_loader)
        avg_metrics['train_gland_loss'] = gland_loss / len(self.train_loader)
        avg_metrics['train_consistency_loss'] = consistency_loss / len(self.train_loader)

        # Map metrics for consistency with training history keys
        if 'patch_accuracy' in avg_metrics:
            avg_metrics['train_patch_accuracy'] = avg_metrics['patch_accuracy']
        if 'dice_mean' in avg_metrics:
            avg_metrics['train_dice_mean'] = avg_metrics['dice_mean']
        if 'iou_mean' in avg_metrics:
            avg_metrics['train_iou_mean'] = avg_metrics['iou_mean']
        if 'pixel_accuracy' in avg_metrics:
            avg_metrics['train_pixel_accuracy'] = avg_metrics['pixel_accuracy']
        if 'pixel_accuracy_overall' in avg_metrics:
            avg_metrics['train_pixel_accuracy_overall'] = avg_metrics['pixel_accuracy_overall']

        return avg_metrics

    def validate_epoch(self, epoch: int) -> Dict[str, float]:
        """Validate for one epoch"""
        self.model.eval()
        total_loss = 0.0
        seg_loss = 0.0
        patch_loss = 0.0
        gland_loss = 0.0
        consistency_loss = 0.0
        all_metrics = {}

        with torch.no_grad():
            progress_bar = tqdm(self.val_loader, desc="Validation")
            for batch_data in progress_bar:
                # Move data to device
                images = batch_data['images'].to(self.device)
                seg_masks = batch_data['segmentation_targets'].to(self.device)
                patch_labels = batch_data['patch_labels'].to(self.device)

                # Forward pass
                architecture = self.config.get('architecture', 'nnunet')
                if architecture == 'teacher_student_unet':
                    # For Teacher-Student models during validation
                    seg_model = self.model.segmentation_model

                    # Use teacher for validation if initialized, otherwise student
                    if hasattr(seg_model, 'teacher_initialized') and seg_model.teacher_initialized:
                        ts_mode = "teacher_student"  # Use both for validation to compute consistency loss
                    else:
                        ts_mode = "student_only"

                    # Forward through MultiTaskWrapper
                    outputs = self.model(images, mode=ts_mode)

                    # Extract teacher outputs from existing computation for validation
                    if ts_mode == "teacher_student":
                        # Teacher outputs are already computed in teacher_student mode - extract them
                        # No additional forward pass needed - this saves GPU memory!
                        if 'teacher' in outputs:
                            teacher_outputs = {'segmentation': outputs['teacher']}
                            # Add teacher patch classification if available
                            if 'teacher_patch_classification' in outputs:
                                teacher_outputs['patch_classification'] = outputs['teacher_patch_classification']
                            if 'teacher_gland_classification' in outputs:
                                teacher_outputs['gland_classification'] = outputs['teacher_gland_classification']
                        else:
                            teacher_outputs = None
                    else:
                        teacher_outputs = None
                else:
                    outputs = self.model(images)
                    teacher_outputs = None

                # Prepare targets
                targets = {
                    'segmentation': seg_masks,
                    'patch_labels': patch_labels
                }

                # Calculate loss
                if architecture == 'teacher_student_unet':
                    # For Teacher-Student validation, teacher_outputs should already be in correct format
                    # or None during warmup
                    loss_dict = self.loss_fn(outputs, teacher_outputs, targets, epoch)
                else:
                    loss_dict = self.loss_fn(outputs, targets)

                total_loss_key = 'total_loss' if 'total_loss' in loss_dict else 'total'
                total_loss += loss_dict[total_loss_key].item()

                # Track individual loss components (handle both standard and Teacher-Student prefixed keys)
                # For Teacher-Student models, the keys are prefixed with 'supervised_'
                seg_key = 'supervised_segmentation' if 'supervised_segmentation' in loss_dict else 'segmentation'
                patch_key = 'supervised_patch_classification' if 'supervised_patch_classification' in loss_dict else 'patch_classification'
                gland_key = 'supervised_gland_classification' if 'supervised_gland_classification' in loss_dict else 'gland_classification'

                if seg_key in loss_dict and loss_dict[seg_key].numel() == 1:
                    seg_loss += loss_dict[seg_key].item()
                if patch_key in loss_dict and loss_dict[patch_key].numel() == 1:
                    patch_loss += loss_dict[patch_key].item()
                if gland_key in loss_dict and loss_dict[gland_key].numel() == 1:
                    gland_loss += loss_dict[gland_key].item()
                if 'consistency_loss' in loss_dict and loss_dict['consistency_loss'].numel() == 1:
                    consistency_loss += loss_dict['consistency_loss'].item()

                # Calculate metrics
                # Pass teacher outputs for pseudo-GT metrics computation when available
                batch_metrics = self.calculate_metrics(outputs, targets, prefix="val_", teacher_outputs=teacher_outputs)
                for key, value in batch_metrics.items():
                    if key not in all_metrics:
                        all_metrics[key] = []
                    all_metrics[key].append(value)

                # Update validation progress bar with current metrics
                current_lr = self.optimizer.param_groups[0]['lr']
                val_postfix_dict = {
                    'Loss': f"{loss_dict[total_loss_key].item():.4f}",
                    'Dice': f"{batch_metrics.get('val_dice_mean', 0):.4f}",
                    'IoU': f"{batch_metrics.get('val_iou_mean', 0):.4f}",
                    'PatchAcc': f"{batch_metrics.get('val_patch_accuracy', 0):.4f}"
                }

                # Add consistency loss and alpha for Teacher-Student models
                if 'consistency_loss' in loss_dict:
                    val_postfix_dict['Teacher_Consist_Loss'] = f"{loss_dict['consistency_loss'].item():.4f}"
                if 'alpha' in loss_dict:
                    val_postfix_dict['Supervision_Weightage'] = f"{loss_dict['alpha']:.3f}"

                # Add EMA Decay for Teacher-Student models
                architecture = self.config.get('architecture', 'nnunet')
                if architecture == 'teacher_student_unet':
                    seg_model = self.model.segmentation_model
                    if hasattr(seg_model, 'current_ema_decay'):
                        val_postfix_dict['EMA'] = f"{seg_model.current_ema_decay:.4f}"

                # Add Pseudo-GT metrics for Teacher-Student models (monitoring only)
                if 'val_pseudo_dice' in batch_metrics:
                    val_postfix_dict['Pseudo-Dice'] = f"{batch_metrics['val_pseudo_dice']:.4f}"
                if 'val_pseudo_iou' in batch_metrics:
                    val_postfix_dict['Pseudo-IoU'] = f"{batch_metrics['val_pseudo_iou']:.4f}"

                val_postfix_dict['LR'] = f"{current_lr:.2e}"
                progress_bar.set_postfix(val_postfix_dict)

        # Average metrics
        avg_metrics = {}
        for key, values in all_metrics.items():
            avg_metrics[key] = np.mean(values)

        avg_metrics['val_loss'] = total_loss / len(self.val_loader)
        avg_metrics['val_seg_loss'] = seg_loss / len(self.val_loader)
        avg_metrics['val_patch_loss'] = patch_loss / len(self.val_loader)
        avg_metrics['val_gland_loss'] = gland_loss / len(self.val_loader)
        avg_metrics['val_consistency_loss'] = consistency_loss / len(self.val_loader)

        return avg_metrics

    def save_checkpoint(self, epoch: int, metrics: Dict[str, float], is_best: bool = False):
        """
        Save model checkpoint following GlaS_MultiTask pattern

        For Teacher-Student UNet models, saves additional separate checkpoints:
        - best_model.pth: Complete model (default behavior)
        - latest_model.pth: Latest complete model
        - best_student_model.pth: Best student network only
        - best_teacher_model.pth: Best teacher network only (if initialized)
        - latest_student_model.pth: Latest student network
        - latest_teacher_model.pth: Latest teacher network (if initialized)
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'metrics': metrics,
            'config': self.config,
            'train_history': self.train_history
        }

        # Check if this is a Teacher-Student UNet model
        is_teacher_student = (hasattr(self.model, 'segmentation_model') and
                             hasattr(self.model.segmentation_model, 'teacher') and
                             hasattr(self.model.segmentation_model, 'student'))

        # Ensure models directory exists (defensive check)
        self.models_dir.mkdir(parents=True, exist_ok=True)

        # Save best checkpoint (main model file following GlaS pattern)
        if is_best:
            if not is_teacher_student:
                # For non-Teacher-Student models, save combined checkpoint
                best_path = self.models_dir / "best_model.pth"
                torch.save(checkpoint, best_path)
                self.logger.info(f"💾 Best model saved at epoch {epoch+1}")
            else:
                # For Teacher-Student models, only save separate student and teacher weights
                self.logger.info(f"💾 Saving separate best student and teacher models at epoch {epoch+1}")

            # For Teacher-Student models, save separate best student and teacher weights
            if is_teacher_student:
                seg_model = self.model.segmentation_model

                # Save best student weights (including classification heads)
                student_state_dict = {}
                # Add segmentation model weights
                for key, value in seg_model.student.state_dict().items():
                    student_state_dict[f'segmentation_model.student.{key}'] = value
                # Add student classification head (separate for teacher-student architecture)
                if hasattr(self.model, 'student_classification_head'):
                    for key, value in self.model.student_classification_head.state_dict().items():
                        student_state_dict[f'student_classification_head.{key}'] = value
                elif hasattr(self.model, 'classification_head'):
                    # Fallback for non-teacher-student architectures
                    for key, value in self.model.classification_head.state_dict().items():
                        student_state_dict[f'classification_head.{key}'] = value

                best_student_checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': student_state_dict,
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
                    'metrics': metrics,
                    'config': self.config,
                    'train_history': self.train_history,
                    'teacher_initialized': seg_model.teacher_initialized,
                    'ema_decay': seg_model.ema_decay
                }
                best_student_path = self.models_dir / "best_student_model.pth"
                torch.save(best_student_checkpoint, best_student_path)

                # Save best teacher weights (if teacher is initialized, including classification heads)
                if seg_model.teacher_initialized:
                    teacher_state_dict = {}
                    # Add segmentation model weights
                    for key, value in seg_model.teacher.state_dict().items():
                        teacher_state_dict[f'segmentation_model.teacher.{key}'] = value
                    # Add teacher classification head (separate for teacher-student architecture)
                    if hasattr(self.model, 'teacher_classification_head'):
                        for key, value in self.model.teacher_classification_head.state_dict().items():
                            teacher_state_dict[f'teacher_classification_head.{key}'] = value
                    elif hasattr(self.model, 'classification_head'):
                        # Fallback for non-teacher-student architectures
                        for key, value in self.model.classification_head.state_dict().items():
                            teacher_state_dict[f'classification_head.{key}'] = value

                    best_teacher_checkpoint = {
                        'epoch': epoch,
                        'model_state_dict': teacher_state_dict,
                        'metrics': metrics,
                        'config': self.config,
                        'train_history': self.train_history,
                        'teacher_initialized': seg_model.teacher_initialized,
                        'ema_decay': seg_model.ema_decay
                    }
                    best_teacher_path = self.models_dir / "best_teacher_model.pth"
                    torch.save(best_teacher_checkpoint, best_teacher_path)
                    self.logger.info(f"💾 Best Teacher-Student models saved at epoch {epoch+1}")
                else:
                    self.logger.info(f"💾 Best Student model saved at epoch {epoch+1} (Teacher not yet initialized)")

        # Save latest checkpoint for resuming
        if not is_teacher_student:
            # For non-Teacher-Student models, save combined checkpoint
            latest_path = self.models_dir / "latest_model.pth"
            torch.save(checkpoint, latest_path)
        else:
            # For Teacher-Student models, only save separate individual model weights
            self.logger.info(f"💾 Saving separate latest student and teacher models at epoch {epoch+1}")

        # For Teacher-Student models, save latest individual model weights
        if is_teacher_student:
            seg_model = self.model.segmentation_model

            # Save latest student weights (including classification heads)
            student_state_dict = {}
            # Add segmentation model weights
            for key, value in seg_model.student.state_dict().items():
                student_state_dict[f'segmentation_model.student.{key}'] = value
            # Add student classification head (separate for teacher-student architecture)
            if hasattr(self.model, 'student_classification_head'):
                for key, value in self.model.student_classification_head.state_dict().items():
                    student_state_dict[f'student_classification_head.{key}'] = value
            elif hasattr(self.model, 'classification_head'):
                # Fallback for non-teacher-student architectures
                for key, value in self.model.classification_head.state_dict().items():
                    student_state_dict[f'classification_head.{key}'] = value

            latest_student_checkpoint = {
                'epoch': epoch,
                'model_state_dict': student_state_dict,
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
                'metrics': metrics,
                'config': self.config,
                'train_history': self.train_history,
                'teacher_initialized': seg_model.teacher_initialized,
                'ema_decay': seg_model.ema_decay
            }
            latest_student_path = self.models_dir / "latest_student_model.pth"
            torch.save(latest_student_checkpoint, latest_student_path)

            # Save latest teacher weights (if teacher is initialized, including classification heads)
            if seg_model.teacher_initialized:
                teacher_state_dict = {}
                # Add segmentation model weights
                for key, value in seg_model.teacher.state_dict().items():
                    teacher_state_dict[f'segmentation_model.teacher.{key}'] = value
                # Add teacher classification head (separate for teacher-student architecture)
                if hasattr(self.model, 'teacher_classification_head'):
                    for key, value in self.model.teacher_classification_head.state_dict().items():
                        teacher_state_dict[f'teacher_classification_head.{key}'] = value
                elif hasattr(self.model, 'classification_head'):
                    # Fallback for non-teacher-student architectures
                    for key, value in self.model.classification_head.state_dict().items():
                        teacher_state_dict[f'classification_head.{key}'] = value

                latest_teacher_checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': teacher_state_dict,
                    'metrics': metrics,
                    'config': self.config,
                    'train_history': self.train_history,
                    'teacher_initialized': seg_model.teacher_initialized,
                    'ema_decay': seg_model.ema_decay
                }
                latest_teacher_path = self.models_dir / "latest_teacher_model.pth"
                torch.save(latest_teacher_checkpoint, latest_teacher_path)

    @staticmethod
    def load_teacher_student_checkpoint(checkpoint_path: str, model_type: str = "best"):
        """
        Load Teacher-Student model checkpoint

        Args:
            checkpoint_path: Path to the model directory or specific checkpoint file
            model_type: Type of model to load ("best", "latest", "teacher", "student")
                      - "best": Load best_model.pth (complete model)
                      - "latest": Load latest_model.pth (complete model)
                      - "teacher": Load best_teacher_model.pth (teacher only)
                      - "student": Load best_student_model.pth (student only)

        Returns:
            Loaded checkpoint dictionary
        """
        import os
        from pathlib import Path

        checkpoint_path = Path(checkpoint_path)

        # If path is a directory, construct the specific model file path
        if checkpoint_path.is_dir():
            if model_type == "best":
                checkpoint_path = checkpoint_path / "best_model.pth"
            elif model_type == "latest":
                checkpoint_path = checkpoint_path / "latest_model.pth"
            elif model_type == "teacher":
                checkpoint_path = checkpoint_path / "best_teacher_model.pth"
            elif model_type == "student":
                checkpoint_path = checkpoint_path / "best_student_model.pth"
            else:
                raise ValueError(f"Unknown model_type: {model_type}. Use 'best', 'latest', 'teacher', or 'student'")

        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        return checkpoint

    def plot_training_curves(self):
        """
        Plot comprehensive training curves with 9 specific subplots:
        1. Total Loss (Training vs Val)
        2. Dice Loss (Training vs Val)
        3. Regular Dice Score (Student vs GT)
        4. Regular IoU Score (Student vs GT)
        5. PSEUDO-Dice Score (Student vs Teacher)
        6. PSEUDO-IoU Score (Student vs Teacher)
        7. Pixel Accuracy (Training vs Val)
        8. Learning Rate Decay
        9. EMA Decay Evolution (Teacher-Student only)

        Includes robust error handling for NFS filesystem and font issues.
        """
        try:
            fig, axes = plt.subplots(3, 3, figsize=(30, 18))
            epochs = self.train_history['epoch']

            # Subplot 1: Total Loss (Training vs Validation)
            axes[0, 0].plot(epochs, self.train_history['train_loss'], 'b-', label='Training', linewidth=2)
            axes[0, 0].plot(epochs, self.train_history['val_loss'], 'r-', label='Validation', linewidth=2)
            axes[0, 0].set_title('Total Loss', fontsize=14, fontweight='bold')
            axes[0, 0].set_xlabel('Epoch')
            axes[0, 0].set_ylabel('Total Loss')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)

            # Subplot 2: Dice Loss (Training vs Val)
            axes[0, 1].plot(epochs, self.train_history['train_seg_loss'], 'g-', label='Training', linewidth=2)
            axes[0, 1].plot(epochs, self.train_history['val_seg_loss'], 'orange', label='Validation', linewidth=2)
            axes[0, 1].set_title('Dice Loss (should decrease)', fontsize=14, fontweight='bold')
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('Dice Loss')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)

            # Subplot 3: Dice Score (Training vs Val) - NEW!
            train_dice_score = [score * 100 for score in self.train_history.get('train_dice_score', [0] * len(epochs))]
            val_dice_score = [score * 100 for score in self.train_history.get('val_dice_score', [0] * len(epochs))]

            axes[0, 2].plot(epochs, train_dice_score, 'forestgreen', label='Training', linewidth=2)
            axes[0, 2].plot(epochs, val_dice_score, 'darkgreen', label='Validation', linewidth=2)
            axes[0, 2].set_title('Regular Dice Score (Student vs GT)', fontsize=14, fontweight='bold')
            axes[0, 2].set_xlabel('Epoch')
            axes[0, 2].set_ylabel('Dice Score (%)')
            axes[0, 2].legend()
            axes[0, 2].grid(True, alpha=0.3)

            # Subplot 4: IoU Score (Training vs Val) - NEW!
            train_iou_score = [score * 100 for score in self.train_history.get('train_iou_score', [0] * len(epochs))]
            val_iou_score = [score * 100 for score in self.train_history.get('val_iou_score', [0] * len(epochs))]

            axes[1, 0].plot(epochs, train_iou_score, 'purple', label='Training', linewidth=2)
            axes[1, 0].plot(epochs, val_iou_score, 'darkorchid', label='Validation', linewidth=2)
            axes[1, 0].set_title('Regular IoU Score (Student vs GT)', fontsize=14, fontweight='bold')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('IoU Score (%)')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)

            # Subplot 5: Pseudo-Dice Score - FIXED
            pseudo_dice_data = self.train_history.get('train_pseudo_dice', [])

            if len(pseudo_dice_data) > 0 and len(pseudo_dice_data) == len(epochs):
                # Teacher-Student model: plot actual pseudo-dice values
                train_pseudo_dice = [score * 100 for score in pseudo_dice_data]
                axes[1, 1].plot(epochs, train_pseudo_dice, 'goldenrod', label='Student vs Teacher Pseudo-GT',
                               linewidth=2, linestyle='--')
                axes[1, 1].plot(epochs, train_dice_score, 'forestgreen', label='Student vs GT',
                               linewidth=2, alpha=0.7)
                axes[1, 1].set_title('PSEUDO-Dice Score (Student vs Teacher Pseudo-GT)', fontsize=14, fontweight='bold')
            else:
                # Teacher NOT initialized OR Standard model: plot zeros
                axes[1, 1].plot(epochs, [0] * len(epochs), 'gray', label='Teacher Not Initialized',
                               linewidth=2, linestyle=':')
                axes[1, 1].set_title('PSEUDO-Dice Score (Teacher Not Initialized)', fontsize=14, fontweight='bold')

            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Dice Score (%)')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
            axes[1, 1].set_ylim(0, 100)

            # Subplot 6: Pseudo-IoU Score - FIXED
            pseudo_iou_data = self.train_history.get('train_pseudo_iou', [])

            if len(pseudo_iou_data) > 0 and len(pseudo_iou_data) == len(epochs):
                # Teacher-Student model: plot actual pseudo-IoU values
                train_pseudo_iou = [score * 100 for score in pseudo_iou_data]
                axes[1, 2].plot(epochs, train_pseudo_iou, 'darkorange', label='Student vs Teacher Pseudo-GT',
                               linewidth=2, linestyle='--')
                axes[1, 2].plot(epochs, train_iou_score, 'purple', label='Student vs GT',
                               linewidth=2, alpha=0.7)
                axes[1, 2].set_title('PSEUDO-IoU Score (Student vs Teacher Pseudo-GT)', fontsize=14, fontweight='bold')
            else:
                # Teacher NOT initialized OR Standard model: plot zeros
                axes[1, 2].plot(epochs, [0] * len(epochs), 'gray', label='Teacher Not Initialized',
                               linewidth=2, linestyle=':')
                axes[1, 2].set_title('PSEUDO-IoU Score (Teacher Not Initialized)', fontsize=14, fontweight='bold')

            axes[1, 2].set_xlabel('Epoch')
            axes[1, 2].set_ylabel('IoU Score (%)')
            axes[1, 2].legend()
            axes[1, 2].grid(True, alpha=0.3)
            axes[1, 2].set_ylim(0, 100)

            # Subplot 7: Pixel Accuracy (Training vs Val)
            train_pixel_acc = [acc * 100 for acc in self.train_history.get('train_pixel_accuracy', [0] * len(epochs))]
            val_pixel_acc = [acc * 100 for acc in self.train_history.get('val_pixel_accuracy', [0] * len(epochs))]

            axes[2, 0].plot(epochs, train_pixel_acc, 'cyan', label='Training', linewidth=2)
            axes[2, 0].plot(epochs, val_pixel_acc, 'navy', label='Validation', linewidth=2)
            axes[2, 0].set_title('Pixel Accuracy (should increase)', fontsize=14, fontweight='bold')
            axes[2, 0].set_xlabel('Epoch')
            axes[2, 0].set_ylabel('Pixel Accuracy (%)')
            axes[2, 0].legend()
            axes[2, 0].grid(True, alpha=0.3)

            # Subplot 8: Learning Rate Decay
            axes[2, 1].plot(epochs, self.train_history['learning_rate'], 'darkred', label='Learning Rate', linewidth=2, marker='o', markersize=3)
            axes[2, 1].set_title('Learning Rate Decay', fontsize=14, fontweight='bold')
            axes[2, 1].set_xlabel('Epoch')
            axes[2, 1].set_ylabel('Learning Rate (log scale)')
            axes[2, 1].set_yscale('log')
            axes[2, 1].legend()
            axes[2, 1].grid(True, alpha=0.3)

            # Subplot 9: EMA Decay Evolution (Teacher-Student only)
            ema_decay_values = self.train_history.get('ema_decay', [0] * len(epochs))
            architecture = self.config.get('architecture', 'nnunet')

            if architecture == 'teacher_student_unet' and any(val > 0 for val in ema_decay_values):
                axes[2, 2].plot(epochs, ema_decay_values, 'purple', label='EMA Decay', linewidth=3, marker='o', markersize=3)
                axes[2, 2].set_title('EMA Decay Evolution', fontsize=14, fontweight='bold')
                axes[2, 2].set_xlabel('Epoch')
                axes[2, 2].set_ylabel('EMA Decay Value')
                axes[2, 2].grid(True, alpha=0.3)
                axes[2, 2].set_ylim(0, 1.0)  # EMA decay is between 0 and 1

                # Add configuration details to legend
                ts_config = self.config.get('teacher_student_config', {})
                config_text = (
                    f"Schedule: {ts_config.get('ema_schedule', 'fixed')}\n"
                    f"Initial: {ts_config.get('ema_decay_initial', 0.999)}\n"
                    f"Final: {ts_config.get('ema_decay_final', 0.1)}\n"
                    f"Annealing Start: {ts_config.get('ema_annealing_start_epoch', 'N/A')}"
                )
                axes[2, 2].text(0.02, 0.98, config_text, transform=axes[2, 2].transAxes,
                               verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                               fontsize=10)

                # Highlight annealing start epoch if available
                annealing_start = ts_config.get('ema_annealing_start_epoch', None)
                if annealing_start and annealing_start < len(epochs):
                    axes[2, 2].axvline(x=annealing_start, color='red', linestyle='--', alpha=0.7,
                                      label=f'Annealing Start (Epoch {annealing_start})')

                axes[2, 2].legend()
            else:
                # Hide subplot for non-Teacher-Student models
                axes[2, 2].axis('off')
                axes[2, 2].text(0.5, 0.5, 'EMA Decay\n(Teacher-Student models only)',
                               ha='center', va='center', transform=axes[2, 2].transAxes,
                               fontsize=12, style='italic', alpha=0.6)

            # Overall figure title
            fig.suptitle('4-Class nnU-Net Multi-Task Training Curves with Pseudo-GT Monitoring', fontsize=18, fontweight='bold', y=0.98)

            plt.tight_layout()
            plt.subplots_adjust(top=0.93)  # Make room for main title

            # Save in visualizations directory
            plt.savefig(self.visualizations_dir / 'training_curves.png', dpi=300, bbox_inches='tight')
            plt.savefig(self.visualizations_dir / 'training_curves.pdf', dpi=300, bbox_inches='tight')
            plt.close()

            self.logger.info(f"📊 Training curves saved: {self.visualizations_dir / 'training_curves.png'}")

            # Create additional figure for patch classification accuracy (separate for clarity)
            fig2, ax = plt.subplots(1, 1, figsize=(10, 6))
            train_patch_acc = [acc * 100 for acc in self.train_history.get('train_patch_accuracy', [0] * len(epochs))]
            val_patch_acc = [acc * 100 for acc in self.train_history.get('patch_accuracy', [0] * len(epochs))]

            ax.plot(epochs, train_patch_acc, 'cyan', label='Training', linewidth=2)
            ax.plot(epochs, val_patch_acc, 'magenta', label='Validation', linewidth=2)
            ax.set_title('Patch Classification Accuracy', fontsize=14, fontweight='bold')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Patch Classification Accuracy (%)')
            ax.legend()
            ax.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig(self.visualizations_dir / 'patch_classification_curves.png', dpi=300, bbox_inches='tight')
            plt.close()

            # Teacher-Student specific plots (if applicable)
            self.plot_teacher_student_curves()

        except (OSError, RecursionError, RuntimeError) as e:
            # Handle stale file handles, matplotlib recursion errors, and other plotting issues
            self.logger.warning(f"⚠️  Failed to generate training plots: {type(e).__name__}: {str(e)[:200]}")
            self.logger.warning("This is likely due to NFS filesystem issues - training data is safe")
            plt.close('all')  # Clean up any partial figures

            # Optionally save metrics data for later plotting
            try:
                import pickle
                metrics_path = self.visualizations_dir / 'training_metrics_data.pkl'
                with open(metrics_path, 'wb') as f:
                    pickle.dump(self.train_history, f)
                self.logger.info(f"💾 Saved metrics data to {metrics_path} for later plotting")
            except Exception as save_error:
                self.logger.warning(f"Could not save metrics data: {save_error}")

    def plot_teacher_student_curves(self):
        """
        Plot Teacher-Student specific metrics if consistency loss is present.
        Includes robust error handling for NFS filesystem and font issues.
        """
        # Check if we have Teacher-Student metrics
        has_consistency = any(val > 0 for val in self.train_history.get('train_consistency_loss', []))
        has_alpha = len(self.train_history.get('alpha', [])) > 0

        if not (has_consistency or has_alpha):
            # No Teacher-Student metrics to plot
            return

        try:
            self.logger.info("📊 Creating Teacher-Student specific plots...")

            epochs = self.train_history['epoch']

            # Create Teacher-Student figure
            fig, axes = plt.subplots(1, 2, figsize=(16, 6))

            # Subplot 1: Consistency Loss (Training vs Validation)
            if has_consistency:
                train_cons_loss = self.train_history.get('train_consistency_loss', [0] * len(epochs))
                val_cons_loss = self.train_history.get('val_consistency_loss', [0] * len(epochs))

                axes[0].plot(epochs, train_cons_loss, 'darkblue', label='Training', linewidth=2)
                axes[0].plot(epochs, val_cons_loss, 'darkred', label='Validation', linewidth=2)
                axes[0].set_title('Consistency Loss (Teacher-Student)', fontsize=14, fontweight='bold')
                axes[0].set_xlabel('Epoch')
                axes[0].set_ylabel('Consistency Loss')
                axes[0].legend()
                axes[0].grid(True, alpha=0.3)
            else:
                axes[0].text(0.5, 0.5, 'No Consistency Loss Data', ha='center', va='center',
                            transform=axes[0].transAxes, fontsize=12)
                axes[0].set_title('Consistency Loss (Teacher-Student)', fontsize=14, fontweight='bold')

            # Subplot 2: Alpha Decay Schedule
            if has_alpha:
                alpha_values = self.train_history.get('alpha', [])
                axes[1].plot(epochs[:len(alpha_values)], alpha_values, 'darkgreen',
                            label='Alpha (Supervised ↔ Consistency)', linewidth=2, marker='o', markersize=3)
                axes[1].set_title('Alpha Decay Schedule (Teacher-Student)', fontsize=14, fontweight='bold')
                axes[1].set_xlabel('Epoch')
                axes[1].set_ylabel('Alpha Value')
                axes[1].set_ylim(0, 1.05)
                axes[1].legend()
                axes[1].grid(True, alpha=0.3)

                # Add text annotations
                axes[1].text(0.02, 0.95, 'High α: Supervised Loss', transform=axes[1].transAxes,
                            fontsize=10, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue'))
                axes[1].text(0.02, 0.05, 'Low α: Consistency Loss', transform=axes[1].transAxes,
                            fontsize=10, verticalalignment='bottom', bbox=dict(boxstyle='round', facecolor='lightgreen'))
            else:
                axes[1].text(0.5, 0.5, 'No Alpha Data', ha='center', va='center',
                            transform=axes[1].transAxes, fontsize=12)
                axes[1].set_title('Alpha Decay Schedule (Teacher-Student)', fontsize=14, fontweight='bold')

            # Main title
            fig.suptitle(f'Teacher-Student Training Metrics - {self.experiment_name}',
                        fontsize=16, fontweight='bold')

            plt.tight_layout()
            plt.subplots_adjust(top=0.9)  # Make room for main title

            # Save Teacher-Student plots
            plt.savefig(self.visualizations_dir / 'teacher_student_curves.png', dpi=300, bbox_inches='tight')
            plt.savefig(self.visualizations_dir / 'teacher_student_curves.pdf', dpi=300, bbox_inches='tight')
            plt.close()

            self.logger.info(f"📊 Teacher-Student curves saved: {self.visualizations_dir / 'teacher_student_curves.png'}")

        except (OSError, RecursionError, RuntimeError) as e:
            # Handle stale file handles, matplotlib recursion errors, and other plotting issues
            self.logger.warning(f"⚠️  Failed to generate Teacher-Student plots: {type(e).__name__}: {str(e)[:200]}")
            self.logger.warning("This is likely due to NFS filesystem issues - training data is safe")
            plt.close('all')  # Clean up any partial figures

    def train(self):
        """Main training loop"""
        self.logger.info("🚀 Starting training...")

        # Prepare data and model
        self.prepare_data()
        self.setup_model()

        # Training configuration
        epochs = self.config.get('epochs', 150)
        self.num_epochs = epochs  # Store as instance variable for EMA scheduling
        early_stop_patience = self.config.get('early_stop_patience', 30)
        early_stop_counter = 0

        # Save comprehensive configuration including all paths for user verification
        comprehensive_paths = self._collect_all_paths_for_config()

        # Add comprehensive path information to config
        self.config.update({
            'comprehensive_paths': comprehensive_paths,
            'dataset_paths': {  # Legacy format for compatibility
                'training_dataset': str(self.train_loader.dataset.data_root),
                'validation_dataset': str(self.val_loader.dataset.data_root),
                'test_dataset': str(self.test_loader.dataset.data_root)
            },
            'config_generation_info': {
                'generated_at': datetime.now().isoformat(),
                'config_version': '2.0_comprehensive_paths',
                'note': 'This configuration includes comprehensive path verification for user debugging'
            }
        })

        config_path = self.output_dir / "training_config.json"
        with open(config_path, 'w') as f:
            json.dump(self.config, f, indent=2)

        self.logger.info(f"💾 Comprehensive training configuration saved to: {config_path}")
        self.logger.info(f"📋 Configuration includes all paths for user verification and debugging")

        self.logger.info(f"📋 Training configuration:")
        self.logger.info(f"   📊 Dataset: {self.dataset_key}")
        self.logger.info(f"   📁 Training dataset: {self.train_loader.dataset.data_root}")
        self.logger.info(f"   🔄 Epochs: {epochs}")
        self.logger.info(f"   📦 Batch size: {self.config.get('batch_size', 4)}")
        self.logger.info(f"   📈 Learning rate: {self.config.get('learning_rate', 1e-4)}")
        self.logger.info(f"   ⚖️ Loss weights - Dice: {self.config.get('dice_weight', 0.5)}, CE: {self.config.get('ce_weight', 0.5)}")
        self.logger.info(f"   ⏱️ Early stopping patience: {early_stop_patience}")

        # Start training loop
        self._train_loop(start_epoch=0, total_epochs=epochs)

    def run_post_training_evaluation(self):
        """
        Run comprehensive post-training evaluation on best model

        Phase 1: Evaluate on COMPLETE train/val/test datasets for comprehensive metrics
        Phase 2: Generate 4-column visualizations for 100 randomly sampled images per split

        For Teacher-Student models:
        - "student": Evaluate only student network (default)
        - "teacher": Evaluate only teacher network
        - "both": Evaluate both teacher and student networks separately
        """
        try:
            from src.evaluation.post_training_evaluator import PostTrainingEvaluator

            # Initialize post-training evaluator
            best_model_path = self.models_dir / "best_model.pth"

            # Get architecture and post_eval_mode configuration
            architecture = self.config.get('architecture', 'nnunet')
            post_eval_mode = self.config.get('post_eval_mode', 'student')

            if architecture == 'teacher_student_unet':
                self.logger.info(f"🎯 Teacher-Student post-evaluation mode: {post_eval_mode}")

                # Determine which models to evaluate
                if post_eval_mode == 'both':
                    evaluation_modes = ['student', 'teacher']
                elif post_eval_mode in ['student', 'teacher']:
                    evaluation_modes = [post_eval_mode]
                else:
                    self.logger.warning(f"⚠️ Unknown post_eval_mode '{post_eval_mode}', defaulting to 'student'")
                    evaluation_modes = ['student']

                # Store results for both models if evaluating both
                all_results = {}

                # Log which checkpoint type is being used
                evaluator_type = os.getenv('GLAND_TEACHER_STUDENT_EVALUATOR', 'latest').lower()
                self.logger.info(f"🎯 Using '{evaluator_type}' checkpoints for Teacher-Student evaluation")

                for mode in evaluation_modes:
                    self.logger.info(f"🔄 Evaluating {mode} network...")

                    # Use individual model files for Teacher-Student evaluation (latest or best based on config)
                    if mode == 'student':
                        if evaluator_type == 'best':
                            model_path = self.models_dir / "best_student_model.pth"
                        else:  # default to 'latest'
                            model_path = self.models_dir / "latest_student_model.pth"
                    else:  # mode == 'teacher'
                        if evaluator_type == 'best':
                            model_path = self.models_dir / "best_teacher_model.pth"
                        else:  # default to 'latest'
                            model_path = self.models_dir / "latest_teacher_model.pth"

                    # Verify the specific model file exists
                    if not model_path.exists():
                        self.logger.error(f"❌ {mode.capitalize()} model file not found: {model_path}")
                        self.logger.warning(f"⚠️ Skipping {mode} evaluation")
                        continue

                    self.logger.info(f"📂 Using {mode} model: {model_path}")

                    # Create separate output directory for each model
                    if len(evaluation_modes) > 1:
                        mode_output_dir = self.output_dir / f"evaluation_{mode}"
                        mode_output_dir.mkdir(exist_ok=True)
                    else:
                        mode_output_dir = self.output_dir

                    evaluator = PostTrainingEvaluator(
                        model_path=str(model_path),
                        dataset_key=self.dataset_key,
                        output_dir=str(mode_output_dir),
                        device=self.device,
                        architecture=architecture,
                        teacher_student_mode=mode
                    )

                    # Run comprehensive evaluation for this mode
                    results = evaluator.run_comprehensive_evaluation()
                    all_results[mode] = results

                    self.logger.info(f"✅ {mode.capitalize()} network evaluation completed!")
                    self.logger.info(f"📊 {mode.capitalize()} evaluation metrics:")
                    self.logger.info(f"   📈 Train - Dice: {results['train'].get('dice_mean', 0):.4f}, IoU: {results['train'].get('iou_mean', 0):.4f}, Pixel Acc: {results['train'].get('pixel_accuracy_overall', 0):.4f}, Patch Acc: {results['train'].get('patch_accuracy', 0):.4f}")
                    self.logger.info(f"   ✅ Val   - Dice: {results['val'].get('dice_mean', 0):.4f}, IoU: {results['val'].get('iou_mean', 0):.4f}, Pixel Acc: {results['val'].get('pixel_accuracy_overall', 0):.4f}, Patch Acc: {results['val'].get('patch_accuracy', 0):.4f}")
                    self.logger.info(f"   🧪 Test  - Dice: {results['test'].get('dice_mean', 0):.4f}, IoU: {results['test'].get('iou_mean', 0):.4f}, Pixel Acc: {results['test'].get('pixel_accuracy_overall', 0):.4f}, Patch Acc: {results['test'].get('patch_accuracy', 0):.4f}")
                    self.logger.info(f"   📁 Results saved to: {mode_output_dir}")

                # If evaluating both models, create comparison summary
                if len(evaluation_modes) > 1:
                    self._create_teacher_student_comparison(all_results)

            else:
                # Standard evaluation for non-Teacher-Student models
                evaluator = PostTrainingEvaluator(
                    model_path=str(best_model_path),
                    dataset_key=self.dataset_key,
                    output_dir=str(self.output_dir),
                    device=self.device,
                    architecture=architecture,
                    teacher_student_mode=None
                )

                # Run comprehensive evaluation
                results = evaluator.run_comprehensive_evaluation()

                self.logger.info("✅ Post-training evaluation completed successfully!")
                self.logger.info(f"📊 Final evaluation metrics:")
                self.logger.info(f"   📈 Train - Dice: {results['train'].get('dice_mean', 0):.4f}, IoU: {results['train'].get('iou_mean', 0):.4f}, Pixel Acc: {results['train'].get('pixel_accuracy_overall', 0):.4f}, Patch Acc: {results['train'].get('patch_accuracy', 0):.4f}")
                self.logger.info(f"   ✅ Val   - Dice: {results['val'].get('dice_mean', 0):.4f}, IoU: {results['val'].get('iou_mean', 0):.4f}, Pixel Acc: {results['val'].get('pixel_accuracy_overall', 0):.4f}, Patch Acc: {results['val'].get('patch_accuracy', 0):.4f}")
                self.logger.info(f"   🧪 Test  - Dice: {results['test'].get('dice_mean', 0):.4f}, IoU: {results['test'].get('iou_mean', 0):.4f}, Pixel Acc: {results['test'].get('pixel_accuracy_overall', 0):.4f}, Patch Acc: {results['test'].get('patch_accuracy', 0):.4f}")
                self.logger.info(f"   📁 Results saved to: {self.output_dir}")

        except Exception as e:
            self.logger.error(f"❌ Post-training evaluation failed: {e}")
            self.logger.warning("⚠️ Training completed successfully, but evaluation failed")
            import traceback
            traceback.print_exc()

    def _create_teacher_student_comparison(self, all_results):
        """
        Create comparison summary and visualizations for teacher vs student evaluation results

        Args:
            all_results: Dict with 'teacher' and 'student' evaluation results
        """
        try:
            import pandas as pd
            import matplotlib.pyplot as plt
            import json

            self.logger.info("📊 Creating Teacher vs Student comparison analysis...")

            # Create comparison output directory
            comparison_dir = self.output_dir / "teacher_student_comparison"
            comparison_dir.mkdir(exist_ok=True)

            # Extract metrics for comparison
            comparison_data = {}
            splits = ['train', 'val', 'test']
            metrics = ['dice_mean', 'iou_mean', 'pixel_accuracy_overall', 'patch_accuracy']

            for split in splits:
                comparison_data[split] = {}
                for metric in metrics:
                    comparison_data[split][metric] = {
                        'student': all_results['student'][split].get(metric, 0),
                        'teacher': all_results['teacher'][split].get(metric, 0)
                    }

            # Save detailed comparison as JSON
            comparison_json_path = comparison_dir / "detailed_comparison.json"
            with open(comparison_json_path, 'w') as f:
                json.dump(comparison_data, f, indent=2)

            # Create comparison CSV
            comparison_rows = []
            for split in splits:
                for metric in metrics:
                    student_val = comparison_data[split][metric]['student']
                    teacher_val = comparison_data[split][metric]['teacher']
                    difference = student_val - teacher_val
                    relative_diff = (difference / teacher_val * 100) if teacher_val != 0 else 0

                    comparison_rows.append({
                        'Split': split.capitalize(),
                        'Metric': metric.replace('_', ' ').title(),
                        'Student': f"{student_val:.4f}",
                        'Teacher': f"{teacher_val:.4f}",
                        'Difference (S-T)': f"{difference:.4f}",
                        'Relative Diff (%)': f"{relative_diff:.2f}"
                    })

            comparison_df = pd.DataFrame(comparison_rows)
            comparison_csv_path = comparison_dir / "teacher_student_comparison.csv"
            comparison_df.to_csv(comparison_csv_path, index=False)

            # Create comparison visualization
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle('Teacher vs Student Model Comparison', fontsize=16, fontweight='bold')

            # Plot 1: Dice Score comparison
            splits_labels = [s.capitalize() for s in splits]
            student_dice = [comparison_data[s]['dice_mean']['student'] for s in splits]
            teacher_dice = [comparison_data[s]['dice_mean']['teacher'] for s in splits]

            x = range(len(splits))
            width = 0.35
            axes[0,0].bar([i - width/2 for i in x], student_dice, width, label='Student', alpha=0.8)
            axes[0,0].bar([i + width/2 for i in x], teacher_dice, width, label='Teacher', alpha=0.8)
            axes[0,0].set_xlabel('Dataset Split')
            axes[0,0].set_ylabel('Dice Score')
            axes[0,0].set_title('Dice Score Comparison')
            axes[0,0].set_xticks(x)
            axes[0,0].set_xticklabels(splits_labels)
            axes[0,0].legend()
            axes[0,0].grid(True, alpha=0.3)

            # Plot 2: IoU Score comparison
            student_iou = [comparison_data[s]['iou_mean']['student'] for s in splits]
            teacher_iou = [comparison_data[s]['iou_mean']['teacher'] for s in splits]

            axes[0,1].bar([i - width/2 for i in x], student_iou, width, label='Student', alpha=0.8)
            axes[0,1].bar([i + width/2 for i in x], teacher_iou, width, label='Teacher', alpha=0.8)
            axes[0,1].set_xlabel('Dataset Split')
            axes[0,1].set_ylabel('IoU Score')
            axes[0,1].set_title('IoU Score Comparison')
            axes[0,1].set_xticks(x)
            axes[0,1].set_xticklabels(splits_labels)
            axes[0,1].legend()
            axes[0,1].grid(True, alpha=0.3)

            # Plot 3: Pixel Accuracy comparison
            student_pixel = [comparison_data[s]['pixel_accuracy_overall']['student'] for s in splits]
            teacher_pixel = [comparison_data[s]['pixel_accuracy_overall']['teacher'] for s in splits]

            axes[1,0].bar([i - width/2 for i in x], student_pixel, width, label='Student', alpha=0.8)
            axes[1,0].bar([i + width/2 for i in x], teacher_pixel, width, label='Teacher', alpha=0.8)
            axes[1,0].set_xlabel('Dataset Split')
            axes[1,0].set_ylabel('Pixel Accuracy')
            axes[1,0].set_title('Pixel Accuracy Comparison')
            axes[1,0].set_xticks(x)
            axes[1,0].set_xticklabels(splits_labels)
            axes[1,0].legend()
            axes[1,0].grid(True, alpha=0.3)

            # Plot 4: Patch Accuracy comparison
            student_patch = [comparison_data[s]['patch_accuracy']['student'] for s in splits]
            teacher_patch = [comparison_data[s]['patch_accuracy']['teacher'] for s in splits]

            axes[1,1].bar([i - width/2 for i in x], student_patch, width, label='Student', alpha=0.8)
            axes[1,1].bar([i + width/2 for i in x], teacher_patch, width, label='Teacher', alpha=0.8)
            axes[1,1].set_xlabel('Dataset Split')
            axes[1,1].set_ylabel('Patch Accuracy')
            axes[1,1].set_title('Patch Accuracy Comparison')
            axes[1,1].set_xticks(x)
            axes[1,1].set_xticklabels(splits_labels)
            axes[1,1].legend()
            axes[1,1].grid(True, alpha=0.3)

            plt.tight_layout()
            comparison_plot_path = comparison_dir / "teacher_student_comparison.png"
            plt.savefig(comparison_plot_path, dpi=300, bbox_inches='tight')
            plt.close()

            # Log summary
            self.logger.info("📊 Teacher vs Student Comparison Summary:")
            for split in splits:
                self.logger.info(f"   {split.capitalize()} Set:")
                dice_diff = comparison_data[split]['dice_mean']['student'] - comparison_data[split]['dice_mean']['teacher']
                iou_diff = comparison_data[split]['iou_mean']['student'] - comparison_data[split]['iou_mean']['teacher']
                self.logger.info(f"      Dice: Student={comparison_data[split]['dice_mean']['student']:.4f}, Teacher={comparison_data[split]['dice_mean']['teacher']:.4f}, Diff={dice_diff:+.4f}")
                self.logger.info(f"      IoU:  Student={comparison_data[split]['iou_mean']['student']:.4f}, Teacher={comparison_data[split]['iou_mean']['teacher']:.4f}, Diff={iou_diff:+.4f}")

            self.logger.info(f"📁 Comparison analysis saved to: {comparison_dir}")

            # Create comprehensive comparison report
            self._create_comprehensive_comparison_report(all_results, comparison_dir)

        except Exception as e:
            self.logger.error(f"❌ Failed to create teacher-student comparison: {e}")
            import traceback
            traceback.print_exc()

    def _create_comprehensive_comparison_report(self, all_results, comparison_dir):
        """Create comprehensive Teacher vs Student comparison report in markdown format"""
        try:
            report_path = comparison_dir / "teacher_student_comparison_report.md"

            with open(report_path, 'w') as f:
                f.write("# Teacher vs Student Comprehensive Comparison Report\n\n")
                f.write(f"**Model:** best_model.pth\n")
                f.write(f"**Architecture:** teacher_student_unet\n")
                f.write(f"**Dataset:** {self.dataset_key}\n")
                f.write(f"**Evaluation Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"**Comparison Mode:** Both networks evaluated separately\n\n")

                # Summary comparison table
                f.write("## Performance Comparison Summary\n\n")
                f.write("| Network | Split | Samples | Pixel Acc | Mean Dice | Patch Acc | Total Consistency |\n")
                f.write("|---------|-------|---------|-----------|-----------|-----------|-------------------|\n")

                for network in ['student', 'teacher']:
                    for split in ['train', 'val', 'test']:
                        metrics = all_results[network][split]
                        consistency_loss = metrics.get('consistency_loss_mean', 'N/A')
                        if isinstance(consistency_loss, (int, float)):
                            consistency_str = f"{consistency_loss:17.4f}"
                        else:
                            consistency_str = f"{'N/A':>17}"

                        f.write(f"| {network.capitalize():7s} | {split.capitalize():5s} | {metrics['num_samples']:7d} | "
                               f"{metrics.get('pixel_accuracy', 0):9.4f} | {metrics.get('dice_mean', 0):9.4f} | "
                               f"{metrics.get('patch_accuracy', 0):9.4f} | {consistency_str} |\n")

                # Add detailed consistency loss component comparison
                f.write("\n### Consistency Loss Components Comparison (Lower is Better)\n\n")
                f.write("| Network | Split | Seg Consistency Loss | Patch Consistency Loss | Gland Consistency Loss |\n")
                f.write("|---------|-------|---------------------|------------------------|------------------------|\n")

                for network in ['student', 'teacher']:
                    for split in ['train', 'val', 'test']:
                        metrics = all_results[network][split]
                        seg_consistency = metrics.get('seg_consistency_mean', 'N/A')
                        patch_consistency = metrics.get('patch_consistency_mean', 'N/A')
                        gland_consistency = metrics.get('gland_consistency_mean', 'N/A')

                        seg_str = f"{seg_consistency:19.4f}" if isinstance(seg_consistency, (int, float)) else f"{'N/A':>19}"
                        patch_str = f"{patch_consistency:22.4f}" if isinstance(patch_consistency, (int, float)) else f"{'N/A':>22}"
                        gland_str = f"{gland_consistency:22.4f}" if isinstance(gland_consistency, (int, float)) else f"{'N/A':>22}"

                        f.write(f"| {network.capitalize():7s} | {split.capitalize():5s} | {seg_str} | {patch_str} | {gland_str} |\n")

                # Add mean consistency loss calculations across splits
                f.write("|---------|-------|---------------------|------------------------|------------------------|\n")

                for network in ['student', 'teacher']:
                    # Collect all consistency loss values for this network
                    seg_values = []
                    patch_values = []
                    gland_values = []

                    for split in ['train', 'val', 'test']:
                        metrics = all_results[network][split]

                        seg_consistency = metrics.get('seg_consistency_mean', None)
                        if isinstance(seg_consistency, (int, float)):
                            seg_values.append(seg_consistency)

                        patch_consistency = metrics.get('patch_consistency_mean', None)
                        if isinstance(patch_consistency, (int, float)):
                            patch_values.append(patch_consistency)

                        gland_consistency = metrics.get('gland_consistency_mean', None)
                        if isinstance(gland_consistency, (int, float)):
                            gland_values.append(gland_consistency)

                    # Calculate means
                    seg_mean = sum(seg_values) / len(seg_values) if seg_values else None
                    patch_mean = sum(patch_values) / len(patch_values) if patch_values else None
                    gland_mean = sum(gland_values) / len(gland_values) if gland_values else None

                    seg_mean_str = f"{seg_mean:19.4f}" if seg_mean is not None else f"{'N/A':>19}"
                    patch_mean_str = f"{patch_mean:22.4f}" if patch_mean is not None else f"{'N/A':>22}"
                    gland_mean_str = f"{gland_mean:22.4f}" if gland_mean is not None else f"{'N/A':>22}"

                    f.write(f"| {network.capitalize():7s} | **Mean** | {seg_mean_str} | {patch_mean_str} | {gland_mean_str} |\n")

                # Performance differences
                f.write("\n## Performance Differences (Student - Teacher)\n\n")
                f.write("| Split | Pixel Acc Δ | Mean Dice Δ | Patch Acc Δ |\n")
                f.write("|-------|-------------|-------------|-------------|\n")

                for split in ['train', 'val', 'test']:
                    student_metrics = all_results['student'][split]
                    teacher_metrics = all_results['teacher'][split]

                    pixel_diff = student_metrics.get('pixel_accuracy', 0) - teacher_metrics.get('pixel_accuracy', 0)
                    dice_diff = student_metrics.get('dice_mean', 0) - teacher_metrics.get('dice_mean', 0)
                    patch_diff = student_metrics.get('patch_accuracy', 0) - teacher_metrics.get('patch_accuracy', 0)

                    f.write(f"| {split.capitalize():5s} | {pixel_diff:+10.4f} | {dice_diff:+10.4f} | {patch_diff:+10.4f} |\n")

                # Per-class comparison
                f.write("\n## Per-Class Segmentation Comparison\n\n")
                class_names = ['Background', 'Benign', 'Malignant', 'PDC']

                for split in ['train', 'val', 'test']:
                    f.write(f"### {split.capitalize()} Set\n\n")
                    f.write("| Class | Student Dice | Teacher Dice | Difference |\n")
                    f.write("|-------|-------------|-------------|------------|\n")

                    for i, class_name in enumerate(class_names):
                        student_dice = all_results['student'][split].get(f'dice_class_{i}', 0)
                        teacher_dice = all_results['teacher'][split].get(f'dice_class_{i}', 0)
                        diff = student_dice - teacher_dice

                        f.write(f"| {class_name:10s} | {student_dice:11.4f} | {teacher_dice:11.4f} | {diff:+10.4f} |\n")
                    f.write("\n")

                # Key findings
                f.write("## Key Findings\n\n")

                # Overall winner
                student_avg = np.mean([all_results['student'][s]['dice_mean'] for s in ['train', 'val', 'test']])
                teacher_avg = np.mean([all_results['teacher'][s]['dice_mean'] for s in ['train', 'val', 'test']])

                if student_avg > teacher_avg:
                    f.write(f"- **Overall Winner:** Student network (Avg Dice: {student_avg:.4f} vs {teacher_avg:.4f})\n")
                elif teacher_avg > student_avg:
                    f.write(f"- **Overall Winner:** Teacher network (Avg Dice: {teacher_avg:.4f} vs {student_avg:.4f})\n")
                else:
                    f.write(f"- **Overall Performance:** Tied (Avg Dice: {student_avg:.4f})\n")

                # Best individual performance
                best_overall = ('student', 'train', all_results['student']['train']['dice_mean'])
                for network in ['student', 'teacher']:
                    for split in ['train', 'val', 'test']:
                        dice = all_results[network][split]['dice_mean']
                        if dice > best_overall[2]:
                            best_overall = (network, split, dice)

                f.write(f"- **Best Individual Performance:** {best_overall[0].capitalize()} on {best_overall[1]} set (Dice: {best_overall[2]:.4f})\n")

                # Knowledge transfer assessment
                diff_avg = student_avg - teacher_avg
                if diff_avg > 0.01:
                    transfer_assessment = "Successful - Student outperforms Teacher"
                elif diff_avg > -0.01:
                    transfer_assessment = "Balanced - Similar performance"
                else:
                    transfer_assessment = "Limited - Teacher outperforms Student"

                f.write(f"- **Knowledge Transfer:** {transfer_assessment} (Δ: {diff_avg:+.4f})\n")

            self.logger.info(f"📋 Comprehensive comparison report saved: {report_path}")

        except Exception as e:
            self.logger.error(f"❌ Failed to create comprehensive comparison report: {e}")
            import traceback
            traceback.print_exc()

    def save_training_history(self):
        """Save training history as CSV files following GlaS_MultiTask pattern"""
        import pandas as pd

        # Check for length consistency and fix if needed
        max_length = max(len(v) for v in self.train_history.values())
        for key, values in self.train_history.items():
            if len(values) < max_length:
                # Pad with zeros or last value to match length
                if len(values) > 0:
                    padding_value = values[-1] if key != 'epoch' else max_length
                else:
                    padding_value = 0
                while len(values) < max_length:
                    values.append(padding_value)
                self.logger.warning(f"⚠️ Padded {key} history to match length: {len(values)}")

        # Create DataFrame from training history
        df = pd.DataFrame(self.train_history)

        # Save complete loss history
        loss_history_path = self.output_dir / "loss_history.csv"
        df.to_csv(loss_history_path, index=False)

        # Save train losses separately
        train_df = df[['epoch', 'train_loss']].copy()
        train_losses_path = self.output_dir / "train_losses.csv"
        train_df.to_csv(train_losses_path, index=False)

        # Save validation losses separately
        val_df = df[['epoch', 'val_loss']].copy()
        val_losses_path = self.output_dir / "val_losses.csv"
        val_df.to_csv(val_losses_path, index=False)

        self.logger.info(f"📊 Training history saved as CSV files")

    def save_final_summary(self):
        """Save final summary following GlaS_MultiTask pattern"""
        import pandas as pd

        # Create comprehensive summary with new metrics
        summary = {
            'experiment_name': self.experiment_name,
            'dataset_key': self.dataset_key,
            'total_epochs': len(self.train_history['epoch']),
            'best_epoch': self.best_metrics['best_epoch'],
            'best_val_loss': self.best_metrics['best_val_loss'],
            'best_dice_score': self.best_metrics['best_dice'],
            'best_patch_accuracy': self.best_metrics['best_patch_acc'],

            # Final loss values
            'final_train_loss': self.train_history['train_loss'][-1] if self.train_history['train_loss'] else 0,
            'final_val_loss': self.train_history['val_loss'][-1] if self.train_history['val_loss'] else 0,

            # NEW: Final comprehensive segmentation metrics
            'final_train_dice_score': self.train_history['train_dice_score'][-1] if self.train_history['train_dice_score'] else 0,
            'final_val_dice_score': self.train_history['val_dice_score'][-1] if self.train_history['val_dice_score'] else 0,
            'final_train_iou_score': self.train_history['train_iou_score'][-1] if self.train_history['train_iou_score'] else 0,
            'final_val_iou_score': self.train_history['val_iou_score'][-1] if self.train_history['val_iou_score'] else 0,
            'final_train_pixel_accuracy': self.train_history['train_pixel_accuracy'][-1] if self.train_history['train_pixel_accuracy'] else 0,
            'final_val_pixel_accuracy': self.train_history['val_pixel_accuracy'][-1] if self.train_history['val_pixel_accuracy'] else 0,

            # Best comprehensive metrics (computed from final values for now)
            'best_train_dice_score': max(self.train_history['train_dice_score']) if self.train_history['train_dice_score'] else 0,
            'best_val_dice_score': max(self.train_history['val_dice_score']) if self.train_history['val_dice_score'] else 0,
            'best_train_iou_score': max(self.train_history['train_iou_score']) if self.train_history['train_iou_score'] else 0,
            'best_val_iou_score': max(self.train_history['val_iou_score']) if self.train_history['val_iou_score'] else 0,
            'best_train_pixel_accuracy': max(self.train_history['train_pixel_accuracy']) if self.train_history['train_pixel_accuracy'] else 0,
            'best_val_pixel_accuracy': max(self.train_history['val_pixel_accuracy']) if self.train_history['val_pixel_accuracy'] else 0,

            # Model and configuration info
            'num_parameters': sum(p.numel() for p in self.model.parameters()),
            'use_multilabel_patch': self.config.get('use_multilabel_patch', True)
        }

        # Add Teacher-Student specific metrics if available
        if self.train_history.get('train_consistency_loss'):
            summary.update({
                'final_train_consistency_loss': self.train_history['train_consistency_loss'][-1] if self.train_history['train_consistency_loss'] else 0,
                'final_val_consistency_loss': self.train_history['val_consistency_loss'][-1] if self.train_history['val_consistency_loss'] else 0,
                'final_alpha': self.train_history['alpha'][-1] if self.train_history['alpha'] else 0,
                'min_train_consistency_loss': min(self.train_history['train_consistency_loss']) if self.train_history['train_consistency_loss'] else 0,
                'min_val_consistency_loss': min(self.train_history['val_consistency_loss']) if self.train_history['val_consistency_loss'] else 0,
                'teacher_student_architecture': True
            })
        else:
            summary['teacher_student_architecture'] = False

        # Save quick summary CSV
        quick_summary_df = pd.DataFrame([summary])
        quick_summary_path = self.output_dir / "quick_summary.csv"
        quick_summary_df.to_csv(quick_summary_path, index=False)

        # Save detailed training summary as JSON
        results_path = self.output_dir / "training_results.json"
        with open(results_path, 'w') as f:
            json.dump({
                **summary,
                'config': self.config,
                'train_history': self.train_history
            }, f, indent=2)

        self.logger.info(f"📋 Final summary saved")


def main():
    """Main entry point for training"""
    import argparse

    parser = argparse.ArgumentParser(description="4-Class nnU-Net Multi-Task Training")
    parser.add_argument("--dataset", type=str, default="mixed",
                       choices=["mixed", "mag5x", "mag10x", "mag20x", "mag40x", "warwick"],
                       help="Dataset to use for training")
    parser.add_argument("--epochs", type=int, default=150,
                       help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4,
                       help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=1e-4,
                       help="Learning rate")
    parser.add_argument("--output_dir", type=str, default=None,
                       help="Output directory for results")
    parser.add_argument("--experiment_name", type=str, default=None,
                       help="Name for this experiment")
    parser.add_argument("--resume", type=str, default=None,
                       help="Path to checkpoint to resume from")

    args = parser.parse_args()

    # Create training configuration
    config = DEFAULT_CONFIG.copy()
    config.update({
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
    })

    # Create trainer
    trainer = MultiTaskTrainer(
        dataset_key=args.dataset,
        config=config,
        output_dir=args.output_dir,
        experiment_name=args.experiment_name
    )

    # Resume from checkpoint if specified
    if args.resume:
        print(f"⚠️  Note: Checkpoint resuming from command line not yet supported.")
        print(f"    For resuming training, use the resume_nnunet_training.sh script.")

    # Start training
    trainer.train()


if __name__ == "__main__":
    main()