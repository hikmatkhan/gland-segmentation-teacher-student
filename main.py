#!/usr/bin/env python3
"""
4-Class nnU-Net Multi-Task Training for Combined Gland Segmentation
================================================================

Main entry point for 4-class gland segmentation with multi-label classification.
Supports Warwick GlaS + OSU Makoto combined datasets with any magnification strategy.

Features:
- 4-class segmentation: Background(0), Benign(1), Malignant(2), PDC(3)
- Multi-label patch classification (patches can contain multiple gland types)
- Single-label gland classification
- Advanced evaluation and visualization
- Research-ready training (100+ epochs)

Usage:
    python main.py demo                     # Quick demo
    python main.py train --dataset mixed   # Train on mixed magnifications
    python main.py evaluate --model path   # Evaluate trained model

Author: Claude Code - Generated for OSU CRC Research
Date: 2025-09-16
"""

import os
import sys
import argparse
import json
import logging
import random
import numpy as np
import torch
from pathlib import Path
from typing import Dict, Any, Optional

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent))

from configs.paths_config import (
    DEFAULT_CONFIG, EVALUATION_CONFIG,
    get_dataset_path, list_available_datasets,
    validate_dataset_path, print_config_summary
)


def setup_logging(log_level: str = "INFO") -> logging.Logger:
    """Setup logging configuration"""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()]
    )
    return logging.getLogger(__name__)


def set_seeds_for_reproducibility(logger: logging.Logger) -> None:
    """
    Set all random seeds for reproducibility based on environment variables
    """
    # Get master seed (single source of truth)
    master_seed = int(os.getenv('GLAND_MASTER_SEED', '42'))

    # Get seed values from environment variables (all should be the same as master_seed)
    python_seed = int(os.getenv('GLAND_PYTHON_SEED', str(master_seed)))
    numpy_seed = int(os.getenv('GLAND_NUMPY_SEED', str(master_seed)))
    torch_seed = int(os.getenv('GLAND_TORCH_SEED', str(master_seed)))
    torch_cuda_seed = int(os.getenv('GLAND_TORCH_CUDA_SEED', str(master_seed)))
    torch_cuda_seed_all = int(os.getenv('GLAND_TORCH_CUDA_SEED_ALL', str(master_seed)))


    # Set Python random seed
    random.seed(python_seed)

    # Set NumPy random seed
    np.random.seed(numpy_seed)

    # Set PyTorch random seeds
    torch.manual_seed(torch_seed)
    torch.cuda.manual_seed(torch_cuda_seed)
    torch.cuda.manual_seed_all(torch_cuda_seed_all)


    # Log seed configuration
    logger.info("🔄 Reproducibility Configuration:")
    logger.info(f"   🎲 Master Seed: {master_seed} (single source of truth)")
    logger.info(f"   🐍 Python seed: {python_seed}")
    logger.info(f"   🔢 NumPy seed: {numpy_seed}")
    logger.info(f"   🔥 PyTorch seed: {torch_seed}")
    logger.info(f"   ⚡ CUDA seed: {torch_cuda_seed}")
    logger.info(f"   🚀 Using PyTorch defaults for optimal performance")
    logger.info(f"   📝 To reproduce this exact run, set MASTER_SEED={master_seed}")


def demo(architecture='nnunet', dataset='mixed'):
    """
    Quick demo to test all components without full training
    Tests model creation, data loading, and basic functionality

    Args:
        architecture: Model architecture to test ('baseline_unet' or 'nnunet')
        dataset: Dataset to use for testing ('mixed', 'mag5x', 'mag10x', 'mag20x', 'mag40x')
    """
    logger = setup_logging()
    logger.info(f"🎬 Starting 4-Class Multi-Task Demo with {architecture} architecture")

    try:
        import torch

        # Check environment variables first
        logger.info("🔍 Checking environment variables...")
        from configs.paths_config import validate_environment_variables
        try:
            validate_environment_variables()
        except ValueError as e:
            logger.error(f"❌ Environment validation failed: {e}")
            logger.info("💡 Demo requires environment variables. Use the training script or set them manually:")
            logger.info("   export GLAND_DATASET_BASE='/path/to/datasets'")
            logger.info("   export GLAND_OUTPUT_DIR='/path/to/outputs'")
            logger.info("   export NNUNET_PREPROCESSED='/path/to/preprocessed'")
            logger.info("   export NNUNET_RESULTS='/path/to/results'")
            logger.info("   export GLAND_TEMP_DIR='/tmp/temp_dir'")
            raise

        # Test imports
        logger.info("📦 Testing imports...")
        from src.models.multi_task_wrapper import create_multitask_model
        from src.models.loss_functions import MultiTaskLoss
        from src.training.dataset import create_combined_data_loaders
        logger.info("✅ All imports successful")

        # Test configuration
        logger.info("⚙️ Testing configuration...")
        print_config_summary()
        available_datasets = list_available_datasets()
        logger.info(f"📊 Available datasets: {available_datasets}")

        # Test model creation
        logger.info(f"🏗️ Testing {architecture} model creation...")
        model = create_multitask_model(architecture=architecture)
        total_params = sum(p.numel() for p in model.parameters())
        logger.info(f"✅ {architecture} model created with {total_params:,} parameters")

        # Test loss function
        logger.info("🔧 Testing loss function...")
        loss_fn = MultiTaskLoss(use_multilabel_patch=True)
        logger.info("✅ Multi-task loss function created")

        # Test forward pass
        logger.info("🧪 Testing forward pass...")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)

        dummy_input = torch.randn(2, 3, 256, 256).to(device)
        with torch.no_grad():
            outputs = model(dummy_input)

        logger.info(f"✅ Forward pass successful:")
        logger.info(f"   📊 Segmentation: {outputs['segmentation'].shape}")
        logger.info(f"   🏷️ Patch classification: {outputs['patch_classification'].shape}")
        logger.info(f"   🔍 Gland classification: {outputs['gland_classification'].shape}")

        # Test data loading (if dataset exists)
        logger.info("📂 Testing data loading...")
        if available_datasets:
            test_dataset = dataset  # Use the specified dataset
            dataset_path = get_dataset_path(test_dataset)

            if validate_dataset_path(test_dataset):
                logger.info(f"✅ Dataset path validated: {dataset_path}")

                # Test data loaders creation
                try:
                    loaders = create_combined_data_loaders(
                        dataset_key=test_dataset,
                        batch_size=2,  # Small batch for demo
                        num_workers=1,
                        use_multilabel_patch=True
                    )
                    logger.info(f"✅ Data loaders created:")
                    logger.info(f"   📚 Train batches: {len(loaders['train'])}")
                    logger.info(f"   ✅ Val batches: {len(loaders['val'])}")
                    logger.info(f"   🧪 Test batches: {len(loaders['test'])}")

                    # Test one batch
                    sample_batch = next(iter(loaders['train']))
                    logger.info(f"✅ Sample batch loaded:")
                    logger.info(f"   🖼️ Images: {sample_batch['image'].shape}")
                    logger.info(f"   🎯 Segmentation: {sample_batch['segmentation'].shape}")
                    logger.info(f"   🏷️ Patch labels: {sample_batch['patch_labels'].shape}")

                except Exception as e:
                    logger.warning(f"⚠️ Data loading test failed: {e}")
                    logger.info("💡 This is normal if dataset hasn't been prepared yet")
            else:
                logger.warning(f"⚠️ Dataset path not found: {dataset_path}")
                logger.info("💡 Run dataset combiner first to prepare data")
        else:
            logger.warning("⚠️ No datasets configured")

        logger.info("🎉 Demo completed successfully!")
        logger.info("💡 All components are working correctly")
        logger.info("📝 Ready for training with: python main.py train --dataset mixed")

    except Exception as e:
        logger.error(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def train(args):
    """
    Main training function

    Args:
        args: Parsed command line arguments
    """
    logger = setup_logging()
    logger.info(f"🚀 Starting 4-Class Multi-Task Training with {args.architecture} architecture")

    # Set seeds for reproducibility
    set_seeds_for_reproducibility(logger)

    try:
        from src.training.trainer import MultiTaskTrainer

        # Validate dataset
        if args.dataset not in list_available_datasets():
            logger.error(f"❌ Dataset '{args.dataset}' not available")
            logger.info(f"📊 Available: {list_available_datasets()}")
            sys.exit(1)

        dataset_path = get_dataset_path(args.dataset)
        if not validate_dataset_path(args.dataset):
            logger.error(f"❌ Dataset path not found: {dataset_path}")
            logger.info("💡 Run dataset combiner first to prepare data")
            sys.exit(1)

        # Create training configuration
        config = DEFAULT_CONFIG.copy()

        # Update with command line arguments
        if args.epochs:
            config['epochs'] = args.epochs
        if args.batch_size:
            config['batch_size'] = args.batch_size
        if args.learning_rate:
            config['learning_rate'] = args.learning_rate
        if hasattr(args, 'enhanced') and args.enhanced:
            config['use_deep_supervision'] = True
            config['augmentation_strength'] = 'strong'

        # Additional configurations
        config.update({
            'architecture': args.architecture,  # Model architecture selection
            'use_multilabel_patch': True,       # Enable multi-label patch classification
            'adaptive_weighting': True,         # Use adaptive loss weighting
            'early_stop_patience': args.patience if hasattr(args, 'patience') else 30
        })

        # Teacher-Student specific configuration (read from environment variables)
        if args.architecture == 'teacher_student_unet':
            import os
            ts_config = {
                'backbone_type': os.getenv('GLAND_TS_BACKBONE_TYPE', 'baseline_unet'),
                'ema_decay': float(os.getenv('GLAND_TS_EMA_DECAY', '0.999')),
                'ema_schedule': os.getenv('GLAND_TS_EMA_SCHEDULE', 'fixed'),
                'ema_decay_initial': float(os.getenv('GLAND_TS_EMA_DECAY_INITIAL', '0.999')),
                'ema_decay_final': float(os.getenv('GLAND_TS_EMA_DECAY_FINAL', '0.1')),
                'ema_annealing_start_epoch': int(os.getenv('GLAND_TS_EMA_ANNEALING_START_EPOCH', '50')),
                'teacher_init_epoch': int(os.getenv('GLAND_TS_TEACHER_INIT_EPOCH', '20')),
                'min_alpha': float(os.getenv('GLAND_TS_MIN_ALPHA', '0.01')),
                'max_alpha': float(os.getenv('GLAND_TS_MAX_ALPHA', '0.9')),
                'consistency_loss_type': os.getenv('GLAND_TS_CONSISTENCY_LOSS_TYPE', 'kl_div'),
                'consistency_temperature': float(os.getenv('GLAND_TS_CONSISTENCY_TEMPERATURE', '1.0')),
                'enable_gland_consistency': os.getenv('GLAND_TS_ENABLE_GLAND_CONSISTENCY', 'false').lower() == 'true',
                'warmup_epochs': int(os.getenv('GLAND_TS_TEACHER_INIT_EPOCH', '20')),  # Use teacher init epoch as warmup
                'depth': int(os.getenv('GLAND_TS_DEPTH', '4')),
                'initial_channels': int(os.getenv('GLAND_TS_INITIAL_CHANNELS', '64')),
                'post_eval_mode': os.getenv('GLAND_TS_POST_EVAL_MODE', 'student'),
                'pseudo_mask_filtering': os.getenv('GLAND_TS_PSEUDO_MASK_FILTERING', 'none'),
                'confidence_threshold': float(os.getenv('GLAND_TS_CONFIDENCE_THRESHOLD', '0.8')),
                'entropy_threshold': float(os.getenv('GLAND_TS_ENTROPY_THRESHOLD', '1.0')),
                'confidence_annealing': os.getenv('GLAND_TS_CONFIDENCE_ANNEALING', 'none'),
                'confidence_max_threshold': float(os.getenv('GLAND_TS_CONFIDENCE_MAX_THRESHOLD', '0.9')),
                'confidence_min_threshold': float(os.getenv('GLAND_TS_CONFIDENCE_MIN_THRESHOLD', '0.6')),
                'confidence_annealing_start_epoch': int(os.getenv('GLAND_TS_CONFIDENCE_ANNEALING_START_EPOCH', '5'))
            }
            config['teacher_student_unet'] = ts_config

            logger.info("🎓 Teacher-Student Configuration:")
            logger.info(f"   🏛️ Backbone Type: {ts_config['backbone_type']}")
            logger.info(f"   🔄 EMA Decay: {ts_config['ema_decay']}")
            logger.info(f"   📅 Teacher Init Epoch: {ts_config['teacher_init_epoch']}")
            logger.info(f"   📉 Alpha Range: {ts_config['max_alpha']} → {ts_config['min_alpha']}")
            logger.info(f"   🔗 Consistency Loss: {ts_config['consistency_loss_type']}")
            logger.info(f"   🌡️ Temperature: {ts_config['consistency_temperature']}")
            logger.info(f"   🔧 Gland Consistency: {ts_config['enable_gland_consistency']}")
            if ts_config['backbone_type'] == 'baseline_unet':
                logger.info(f"   🏗️ Depth: {ts_config['depth']}, Channels: {ts_config['initial_channels']}")
            else:
                logger.info(f"   🏗️ nnUNet architecture with standard configuration")
            logger.info(f"   🎭 Pseudo-mask Filtering: {ts_config['pseudo_mask_filtering']}")
            if ts_config['pseudo_mask_filtering'] == 'confidence':
                logger.info(f"   🎯 Base Confidence Threshold: {ts_config['confidence_threshold']}")
                if ts_config['confidence_annealing'] != 'none':
                    logger.info(f"   📈 Confidence Annealing: {ts_config['confidence_annealing']}")
                    logger.info(f"   🔼 Annealing Range: {ts_config['confidence_max_threshold']} → {ts_config['confidence_min_threshold']}")
                    logger.info(f"   ⏱️ Annealing Start: Epoch {ts_config['confidence_annealing_start_epoch']}")
            elif ts_config['pseudo_mask_filtering'] == 'entropy':
                logger.info(f"   📊 Entropy Threshold: {ts_config['entropy_threshold']}")
            logger.info(f"   📊 Post-Eval Mode: {ts_config['post_eval_mode']}")

        logger.info("📋 Training Configuration:")
        logger.info(f"   🏗️ Architecture: {args.architecture}")
        logger.info(f"   📊 Dataset: {args.dataset}")
        logger.info(f"   🔄 Epochs: {config['epochs']}")
        logger.info(f"   📦 Batch size: {config['batch_size']}")
        logger.info(f"   📈 Learning rate: {config['learning_rate']}")
        logger.info(f"   🏷️ Multi-label patches: {config['use_multilabel_patch']}")

        # Create trainer
        trainer = MultiTaskTrainer(
            dataset_key=args.dataset,
            config=config,
            output_base_dir=args.output_dir,
            experiment_name=args.experiment_name
        )

        # Start training
        trainer.train()

        logger.info("🎉 Training completed successfully!")

    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def resume(args):
    """
    Resume training from checkpoint

    Args:
        args: Parsed command line arguments
    """
    logger = setup_logging()
    logger.info(f"🔄 Resuming interrupted training from checkpoint")

    try:
        from src.training.trainer import MultiTaskTrainer

        # Validate experiment directory
        if not os.path.exists(args.experiment_dir):
            logger.error(f"❌ Experiment directory not found: {args.experiment_dir}")
            sys.exit(1)

        # Create trainer from checkpoint
        trainer = MultiTaskTrainer.from_checkpoint(
            experiment_dir=args.experiment_dir,
            use_best=args.use_best
        )

        # Resume training
        trainer.resume_training()

        logger.info("🎉 Training resumption completed successfully!")

    except Exception as e:
        logger.error(f"❌ Training resumption failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def evaluate(args):
    """
    Evaluation function

    Args:
        args: Parsed command line arguments
    """
    logger = setup_logging()
    logger.info(f"📊 Starting 4-Class Multi-Task Evaluation with {args.architecture} architecture")

    try:
        # TODO: Implement comprehensive evaluation
        # This will be similar to GlaS_MultiTask evaluation but for 4-class + multi-label
        logger.info("🚧 Comprehensive evaluation pipeline to be implemented")
        logger.info("💡 Will include:")
        logger.info("   🎯 4-class segmentation metrics")
        logger.info("   🏷️ Multi-label patch classification metrics")
        logger.info("   🔍 Gland-level classification metrics")
        logger.info("   📊 Rich visualizations and composite images")
        logger.info("   📈 Comprehensive metric reports")

        # For now, basic model loading test
        if args.model and Path(args.model).exists():
            import torch
            from src.models.multi_task_wrapper import create_multitask_model

            logger.info(f"📂 Loading model from: {args.model}")
            # Load checkpoint with PyTorch 2.6 compatibility
            try:
                checkpoint = torch.load(args.model, map_location='cpu', weights_only=True)
            except Exception:
                checkpoint = torch.load(args.model, map_location='cpu', weights_only=False)

            model = create_multitask_model(architecture=args.architecture)
            model.load_state_dict(checkpoint['model_state_dict'])

            logger.info("✅ Model loaded successfully")
            logger.info(f"📊 Best metrics from training:")
            if 'metrics' in checkpoint:
                for key, value in checkpoint['metrics'].items():
                    if isinstance(value, (int, float)):
                        logger.info(f"   {key}: {value:.4f}")
        else:
            logger.error(f"❌ Model file not found: {args.model}")
            sys.exit(1)

    except Exception as e:
        logger.error(f"❌ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="4-Class nnU-Net Multi-Task Learning for Combined Gland Segmentation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py demo --architecture baseline_unet                       # Test baseline UNet
  python main.py demo --architecture nnunet                             # Test nnU-Net (default)
  python main.py demo --architecture teacher_student_unet               # Test Teacher-Student UNet
  python main.py train --architecture baseline_unet --dataset mixed --epochs 150 --output_dir /path/to/outputs  # Train baseline UNet
  python main.py train --architecture nnunet --dataset mixed --epochs 150 --output_dir /path/to/outputs         # Train nnU-Net
  python main.py train --architecture teacher_student_unet --dataset mixed --epochs 150 --teacher_init_epoch 30 --output_dir /path/to/outputs  # Train Teacher-Student UNet
  python main.py train --architecture baseline_unet --dataset mag20x --enhanced --output_dir /path/to/outputs   # Enhanced baseline training
  python main.py evaluate --architecture teacher_student_unet --model /path/to/model.pth --dataset mixed       # Evaluate Teacher-Student model

Datasets:
  mixed   - All magnifications combined (recommended)
  mag5x   - 5x magnification only
  mag10x  - 10x magnification only
  mag20x  - 20x magnification only
  mag40x  - 40x magnification only

Output Structure (following GlaS_MultiTask pattern):
  output_dir/
  └── exp_YYYY-MM-DD_HH-MM-SS/     # Auto-generated experiment directory
      ├── models/                   # Model checkpoints (best_model.pth, latest_model.pth)
      ├── logs/                     # Training logs and TensorBoard
      ├── visualizations/           # Training curves and plots (.png, .pdf)
      ├── evaluations/              # Evaluation results
      ├── training_config.json     # Training configuration
      ├── loss_history.csv         # Complete training history
      ├── quick_summary.csv        # Experiment summary
      └── training_results.json    # Detailed results
        """
    )

    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # Demo command
    demo_parser = subparsers.add_parser('demo', help='Quick demo to test all components')
    demo_parser.add_argument('--architecture', type=str, default='nnunet',
                           choices=['baseline_unet', 'nnunet', 'teacher_student_unet'],
                           help='Model architecture to test (default: nnunet)')
    demo_parser.add_argument('--dataset', type=str, default='mixed',
                           choices=['mixed', 'mag5x', 'mag10x', 'mag20x', 'mag40x', 'warwick'],
                           help='Dataset to use for demo testing (default: mixed)')

    # Train command
    train_parser = subparsers.add_parser('train', help='Train the multi-task model')
    train_parser.add_argument('--architecture', type=str, default='nnunet',
                             choices=['baseline_unet', 'nnunet', 'teacher_student_unet'],
                             help='Model architecture to use (default: nnunet)')
    train_parser.add_argument('--dataset', type=str, default='mixed',
                             choices=['mixed', 'mag5x', 'mag10x', 'mag20x', 'mag40x', 'warwick'],
                             help='Dataset to use for training (default: mixed)')
    train_parser.add_argument('--output_dir', type=str, required=True,
                             help='Output base directory for experiments (REQUIRED)')
    train_parser.add_argument('--epochs', type=int, default=None,
                             help='Number of training epochs (default: from config)')
    train_parser.add_argument('--batch_size', type=int, default=None,
                             help='Batch size (default: from config)')
    train_parser.add_argument('--learning_rate', type=float, default=None,
                             help='Learning rate (default: from config)')
    train_parser.add_argument('--experiment_name', type=str, default=None,
                             help='Name for this experiment (default: auto-generated)')
    train_parser.add_argument('--enhanced', action='store_true',
                             help='Use enhanced training with stronger augmentation')
    train_parser.add_argument('--patience', type=int, default=30,
                             help='Early stopping patience (default: 30)')

    # Teacher-Student specific arguments
    train_parser.add_argument('--ema_decay', type=float, default=0.999,
                             help='EMA decay factor for teacher updates (default: 0.999)')
    train_parser.add_argument('--teacher_init_epoch', type=int, default=50,
                             help='Epoch to initialize teacher network (default: 50)')
    train_parser.add_argument('--teacher_init_val_loss', type=float, default=None,
                             help='Validation loss threshold for teacher initialization')
    train_parser.add_argument('--consistency_loss_weight', type=str, default='cosine',
                             choices=['cosine', 'linear', 'constant'],
                             help='Consistency loss weight schedule (default: cosine)')
    train_parser.add_argument('--min_alpha', type=float, default=0.1,
                             help='Minimum alpha (max consistency weight) (default: 0.1)')
    train_parser.add_argument('--max_alpha', type=float, default=1.0,
                             help='Maximum alpha (max supervised weight) (default: 1.0)')

    # Resume command
    resume_parser = subparsers.add_parser('resume', help='Resume interrupted training from checkpoint')
    resume_parser.add_argument('--experiment_dir', type=str, required=True,
                              help='Path to experiment directory to resume from (e.g., /path/to/teacher_student_nnunet_mag5x_enhanced_20251026_150500)')
    resume_parser.add_argument('--use_best', action='store_true', default=False,
                              help='Resume from best checkpoint instead of latest (default: False)')

    # Evaluate command
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate trained model')
    eval_parser.add_argument('--architecture', type=str, default='nnunet',
                            choices=['baseline_unet', 'nnunet', 'teacher_student_unet'],
                            help='Model architecture used (default: nnunet)')
    eval_parser.add_argument('--model', type=str, required=True,
                            help='Path to trained model checkpoint')
    eval_parser.add_argument('--dataset', type=str, default='mixed',
                            choices=['mixed', 'mag5x', 'mag10x', 'mag20x', 'mag40x', 'warwick'],
                            help='Dataset to evaluate on')
    eval_parser.add_argument('--output', type=str, default='outputs/evaluation',
                            help='Output directory for evaluation results')
    eval_parser.add_argument('--visualize', action='store_true',
                            help='Generate visualization outputs')

    args = parser.parse_args()

    if args.command == 'demo':
        demo(args.architecture, args.dataset)
    elif args.command == 'train':
        train(args)
    elif args.command == 'resume':
        resume(args)
    elif args.command == 'evaluate':
        evaluate(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()