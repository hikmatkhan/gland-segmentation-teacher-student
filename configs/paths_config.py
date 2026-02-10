#!/usr/bin/env python3
"""
Centralized path configuration for Combined Gland Segmentation nnU-Net
Supports unified training across different combined datasets (mixed/separate magnifications)

NEW PSEUDO-GT METRICS CONFIGURATION:
===================================

The following environment variables control the Pseudo-GT (P-Dice and P-IoU) metrics
that monitor Student vs Pseudo-GT alignment during Teacher-Student training:

PSEUDO-GT METRICS ENVIRONMENT VARIABLES:
- GLAND_PSEUDO_GT_ENABLED: Enable/disable pseudo-GT metrics computation (default: 'true')
- GLAND_PSEUDO_GT_CPU_COMPUTATION: Compute metrics on CPU to save GPU memory (default: 'true')
- GLAND_PSEUDO_GT_MEMORY_LEAK_PREVENTION: Enable memory leak prevention (default: 'true')
- GLAND_PSEUDO_GT_PROGRESS_BAR: Display pseudo-GT metrics in progress bar (default: 'true')
- GLAND_PSEUDO_GT_VISUALIZATION: Include pseudo-GT metrics in training plots (default: 'true')
- GLAND_PSEUDO_GT_GRADIENT_ISOLATION: Isolate gradients to prevent tracking (default: 'true')
- GLAND_PSEUDO_GT_EXPLICIT_CLEANUP: Enable explicit tensor cleanup (default: 'true')
- GLAND_PSEUDO_GT_FALLBACK_ON_OOM: Skip metrics on CUDA OOM errors (default: 'true')

TEACHER-STUDENT PSEUDO-GT ENVIRONMENT VARIABLES:
- GLAND_TS_PSEUDO_GT_METRICS_ENABLED: Enable pseudo-GT metrics in Teacher-Student models (default: 'true')
- GLAND_TS_PSEUDO_GT_CPU_COMPUTATION: Use CPU computation for memory optimization (default: 'true')
- GLAND_TS_PSEUDO_GT_MEMORY_OPTIMIZATION: Enable memory optimization techniques (default: 'true')
- GLAND_TS_PSEUDO_GT_PROGRESS_BAR_DISPLAY: Show in progress bar during training (default: 'true')
- GLAND_TS_PSEUDO_GT_VISUALIZATION_ENABLED: Include in enhanced training visualizations (default: 'true')

FEATURES:
- Real-time P-Dice and P-IoU display in progress bars during Teacher-Student training
- Memory leak-free implementation with explicit cleanup
- CPU-based computation to preserve GPU memory for training
- Enhanced 3x3 training visualization with dedicated Pseudo-GT sub-figures
- Zero-impact monitoring with gradient isolation
- Graceful fallback on memory errors
"""

from pathlib import Path

# =============================================================================
# DATA PATHS - Configurable via Environment Variables
# =============================================================================

import os

# Available combined datasets (dynamic based on base path)
def get_available_datasets(base_path=None):
    """Get available datasets based on base path"""
    if base_path is None:
        # Read environment variable at runtime, not import time
        base_path = os.getenv('GLAND_DATASET_BASE')
        if base_path is None:
            raise ValueError("GLAND_DATASET_BASE environment variable not set. Please run the training script which exports this variable.")

    return {
        # OSU Combined Datasets (4-class: Background, Benign, Malignant, PDC)
        # Mixed magnifications (all magnifications together)
        "mixed": f"{base_path}/Task001_Combined_Mixed_Magnifications",
        # Separate magnifications (individual datasets per magnification)
        "mag5x": f"{base_path}/Task005_Combined_Mag5x",
        "mag10x": f"{base_path}/Task010_Combined_Mag10x",
        "mag20x": f"{base_path}/Task020_Combined_Mag20x",
        "mag40x": f"{base_path}/Task040_Combined_Mag40x",
        # Warwick GlaS Teacher-Student Dataset (3-class: Background, Benign, Malignant)
        # NOTE: Warwick has nnUNet_raw/ subdirectory in the base path
        "warwick": f"{base_path}/nnUNet_raw/Task002_WarwickGlaSTeacherStudent",
    }

def get_combined_data_base():
    """Get base directory for combined datasets at runtime"""
    base_path = os.getenv('GLAND_DATASET_BASE')
    if base_path is None:
        raise ValueError("GLAND_DATASET_BASE environment variable not set. Please run the training script which exports this variable.")
    return base_path

# Default dataset to use
DEFAULT_DATASET = "mixed"  # Default to mixed magnifications

# Data paths structure (all configurable via environment variables)
DATA_PATHS = {
    # Combined dataset locations (computed at runtime)
    "combined_data_base": None,  # Will be computed at runtime via get_combined_data_base()
    "default_dataset": None,     # Will be computed at runtime via get_dataset_path()

    # nnU-Net processed data directories (configurable via environment)
    "nnunet_preprocessed": os.getenv('NNUNET_PREPROCESSED'),
    "nnunet_results": os.getenv('NNUNET_RESULTS'),

    # Output directory for experiments (configurable via environment or CLI)
    "output_base": os.getenv('GLAND_OUTPUT_DIR'),

    # Temporary files (configurable via environment)
    "temp_dir": os.getenv('GLAND_TEMP_DIR'),
}

# =============================================================================
# PROJECT PATHS
# =============================================================================

PROJECT_ROOT = Path(__file__).parent.parent
PROJECT_PATHS = {
    # Project root
    "project_root": str(PROJECT_ROOT),

    # Source code directory
    "src_dir": str(PROJECT_ROOT / "src"),

    # Configuration files
    "configs_dir": str(PROJECT_ROOT / "configs"),

    # Scripts directory
    "scripts_dir": str(PROJECT_ROOT / "scripts"),

    # Outputs directory
    "outputs_dir": str(PROJECT_ROOT / "outputs"),
}

# =============================================================================
# CONDA/PYTHON ENVIRONMENT PATHS
# =============================================================================

ENVIRONMENT_PATHS = {
    # Conda environment (configurable via environment variables)
    "conda_env": os.getenv('CONDA_PREFIX'),
    "python_bin": os.getenv('PYTHON_BIN'),

    # Python library paths (configurable via environment)
    "user_packages": os.getenv('PYTHON_USER_PACKAGES'),
}

# =============================================================================
# DEFAULT CONFIGURATION VALUES - Updated for 4-class segmentation
# =============================================================================

DEFAULT_CONFIG = {
    # Training parameters (configurable via environment variables)
    "epochs": int(os.getenv('GLAND_EPOCHS', '150')),
    "batch_size": int(os.getenv('GLAND_BATCH_SIZE', '4')),
    "learning_rate": float(os.getenv('GLAND_LEARNING_RATE', '1e-4')),
    "weight_decay": float(os.getenv('GLAND_WEIGHT_DECAY', '1e-4')),
    "num_workers": int(os.getenv('GLAND_NUM_WORKERS', '4')),

    # Architecture selection (configurable via environment)
    "architecture": os.getenv('GLAND_ARCHITECTURE', 'nnunet'),

    # Image processing (configurable via environment)
    "image_size": [int(x) for x in os.getenv('GLAND_IMAGE_SIZE', '512,512').split(',')],

    # Model parameters - Updated for 4-class segmentation
    "use_nnunet": True,
    "enable_classification": True,
    "adaptive_weighting": True,
    "num_seg_classes": 4,  # Background(0), Benign(1), Malignant(2), PDC(3)
    "num_patch_classes": 4,  # Multi-class patch classification
    "num_gland_classes": 4,  # Multi-class gland classification

    # Loss function weights (configurable via environment)
    "dice_weight": float(os.getenv('GLAND_DICE_WEIGHT', '0.5')),  # Weight for Dice loss
    "ce_weight": float(os.getenv('GLAND_CE_WEIGHT', '0.5')),      # Weight for Cross-Entropy loss

    # Multi-label classification - CRITICAL for realistic histopathology
    "use_multilabel_patch": True,  # Patches can contain multiple gland types

    # Data augmentation (configurable via environment)
    "augmentation": True,
    "rotation_limit": int(os.getenv('GLAND_ROTATION_LIMIT', '20')),
    "scale_limit": float(os.getenv('GLAND_SCALE_LIMIT', '0.1')),

    # Enhanced training (configurable via environment)
    "use_enhanced_training": os.getenv('GLAND_ENHANCED_TRAINING', 'false').lower() == 'true',

    # Stopping criteria (configurable via environment)
    "early_stop_patience": int(os.getenv('GLAND_EARLY_STOP_PATIENCE', '30')),
    "lr_scheduler_patience": int(os.getenv('GLAND_LR_SCHEDULER_PATIENCE', '15')),
    "min_lr": float(os.getenv('GLAND_MIN_LR', '1e-7')),

    # Optimization (configurable via environment)
    "optimizer": os.getenv('GLAND_OPTIMIZER', 'adamw'),
    "scheduler": os.getenv('GLAND_SCHEDULER', 'poly'),
    "step_size": int(os.getenv('GLAND_STEP_SIZE', '30')),
    "gamma": float(os.getenv('GLAND_GAMMA', '0.1')),

    # CosineAnnealingLR parameters (configurable via environment)
    "cosine_t_max": int(os.getenv('GLAND_COSINE_T_MAX', os.getenv('GLAND_EPOCHS', '150'))),
    "cosine_eta_min": float(os.getenv('GLAND_COSINE_ETA_MIN', os.getenv('GLAND_MIN_LR', '1e-7'))),

    # Output settings (configurable via environment)
    "save_best_only": True,
    "save_visualizations": True,
    "samples_per_visualization": int(os.getenv('GLAND_SAMPLES_PER_VIZ', '50')),
    "save_frequency": int(os.getenv('GLAND_SAVE_FREQUENCY', '10')),

    # Teacher-Student specific configuration (configurable via environment)
    "teacher_student_unet": {
        # Core Architecture
        "backbone_type": os.getenv('GLAND_TS_BACKBONE_TYPE', 'baseline_unet'),
        "depth": int(os.getenv('GLAND_TS_DEPTH', '4')),
        "initial_channels": int(os.getenv('GLAND_TS_INITIAL_CHANNELS', '64')),

        # EMA Configuration (from shell script TS_EMA_* variables)
        "ema_decay": float(os.getenv('GLAND_TS_EMA_DECAY', '0.999')),
        "ema_schedule": os.getenv('GLAND_TS_EMA_SCHEDULE', 'fixed'),
        "ema_decay_initial": float(os.getenv('GLAND_TS_EMA_DECAY_INITIAL', '0.999')),
        "ema_decay_final": float(os.getenv('GLAND_TS_EMA_DECAY_FINAL', '0.1')),
        "ema_annealing_start_epoch": int(os.getenv('GLAND_TS_EMA_ANNEALING_START_EPOCH', '50')),

        # Teacher Initialization
        "teacher_init_epoch": int(os.getenv('GLAND_TS_TEACHER_INIT_EPOCH', '20')),

        # Alpha Scheduling for Consistency Loss
        "min_alpha": float(os.getenv('GLAND_TS_MIN_ALPHA', '0.01')),
        "max_alpha": float(os.getenv('GLAND_TS_MAX_ALPHA', '0.9')),

        # Consistency Loss Configuration
        "consistency_loss_type": os.getenv('GLAND_TS_CONSISTENCY_LOSS_TYPE', 'mse'),
        "consistency_temperature": float(os.getenv('GLAND_TS_CONSISTENCY_TEMPERATURE', '1.0')),
        "enable_gland_consistency": os.getenv('GLAND_TS_ENABLE_GLAND_CONSISTENCY', 'false').lower() == 'true',

        # Pseudo-mask Filtering Configuration
        "pseudo_mask_filtering": os.getenv('GLAND_TS_PSEUDO_MASK_FILTERING', 'none'),
        "confidence_threshold": float(os.getenv('GLAND_TS_CONFIDENCE_THRESHOLD', '0.8')),
        "entropy_threshold": float(os.getenv('GLAND_TS_ENTROPY_THRESHOLD', '1.0')),
        "filtering_warmup_epochs": int(os.getenv('GLAND_TS_FILTERING_WARMUP_EPOCHS', '10')),

        # Confidence Annealing Configuration
        "confidence_annealing": os.getenv('GLAND_TS_CONFIDENCE_ANNEALING', 'none'),
        "confidence_max_threshold": float(os.getenv('GLAND_TS_CONFIDENCE_MAX_THRESHOLD', '0.9')),
        "confidence_min_threshold": float(os.getenv('GLAND_TS_CONFIDENCE_MIN_THRESHOLD', '0.6')),
        "confidence_annealing_start_epoch": int(os.getenv('GLAND_TS_CONFIDENCE_ANNEALING_START_EPOCH', '5')),

        # Post-training Evaluation
        "post_eval_mode": os.getenv('GLAND_TS_POST_EVAL_MODE', 'student'),

        # GT + Teacher Incorporation parameters
        "gt_teacher_incorporate_enabled": os.getenv('GLAND_TS_GT_TEACHER_INCORPORATE_ENABLED', 'false').lower() == 'true',
        "gt_incorporate_start_epoch": int(os.getenv('GLAND_TS_GT_INCORPORATE_START_EPOCH', '0')),
        "gt_incorporate_segmentation_only": os.getenv('GLAND_TS_GT_INCORPORATE_SEGMENTATION_ONLY', 'true').lower() == 'true',

        # Pseudo-GT Monitoring and Progress Bar Display parameters
        "pseudo_gt_metrics_enabled": os.getenv('GLAND_TS_PSEUDO_GT_METRICS_ENABLED', 'true').lower() == 'true',
        "pseudo_gt_cpu_computation": os.getenv('GLAND_TS_PSEUDO_GT_CPU_COMPUTATION', 'true').lower() == 'true',
        "pseudo_gt_memory_optimization": os.getenv('GLAND_TS_PSEUDO_GT_MEMORY_OPTIMIZATION', 'true').lower() == 'true',
        "pseudo_gt_progress_bar_display": os.getenv('GLAND_TS_PSEUDO_GT_PROGRESS_BAR_DISPLAY', 'true').lower() == 'true',
        "pseudo_gt_visualization_enabled": os.getenv('GLAND_TS_PSEUDO_GT_VISUALIZATION_ENABLED', 'true').lower() == 'true',
    },

    # Pseudo-GT Metrics Configuration (Student vs Pseudo-GT monitoring)
    "pseudo_gt_metrics": {
        "enabled": os.getenv('GLAND_PSEUDO_GT_ENABLED', 'true').lower() == 'true',
        "cpu_computation": os.getenv('GLAND_PSEUDO_GT_CPU_COMPUTATION', 'true').lower() == 'true',
        "memory_leak_prevention": os.getenv('GLAND_PSEUDO_GT_MEMORY_LEAK_PREVENTION', 'true').lower() == 'true',
        "progress_bar_display": os.getenv('GLAND_PSEUDO_GT_PROGRESS_BAR', 'true').lower() == 'true',
        "include_in_visualization": os.getenv('GLAND_PSEUDO_GT_VISUALIZATION', 'true').lower() == 'true',
        "gradient_isolation": os.getenv('GLAND_PSEUDO_GT_GRADIENT_ISOLATION', 'true').lower() == 'true',
        "explicit_cleanup": os.getenv('GLAND_PSEUDO_GT_EXPLICIT_CLEANUP', 'true').lower() == 'true',
        "fallback_on_oom": os.getenv('GLAND_PSEUDO_GT_FALLBACK_ON_OOM', 'true').lower() == 'true',
    },

    # Complete Shell Script Configuration for Post-Training Analysis
    "shell_script_config": {
        # Core Training Parameters
        "ARCHITECTURE": os.getenv('GLAND_ARCHITECTURE', 'nnunet'),
        "EPOCHS": int(os.getenv('GLAND_EPOCHS', '150')),
        "BATCH_SIZE": int(os.getenv('GLAND_BATCH_SIZE', '4')),
        "LEARNING_RATE": float(os.getenv('GLAND_LEARNING_RATE', '1e-4')),
        "ENHANCED_TRAINING": os.getenv('GLAND_ENHANCED_TRAINING', 'true').lower() == 'true',

        # Loss Configuration
        "DICE_WEIGHT": float(os.getenv('GLAND_DICE_WEIGHT', '0.5')),
        "CE_WEIGHT": float(os.getenv('GLAND_CE_WEIGHT', '0.5')),

        # Training Control
        "EARLY_STOP_PATIENCE": int(os.getenv('GLAND_EARLY_STOP_PATIENCE', '30')),
        "LR_SCHEDULER_PATIENCE": int(os.getenv('GLAND_LR_SCHEDULER_PATIENCE', '15')),
        "MIN_LR": float(os.getenv('GLAND_MIN_LR', '1e-7')),

        # Scheduler Configuration
        "COSINE_T_MAX": int(os.getenv('GLAND_COSINE_T_MAX', os.getenv('GLAND_EPOCHS', '150'))),
        "COSINE_ETA_MIN": float(os.getenv('GLAND_COSINE_ETA_MIN', os.getenv('GLAND_MIN_LR', '1e-7'))),
        "OPTIMIZER": os.getenv('GLAND_OPTIMIZER', 'adamw'),
        "SCHEDULER": os.getenv('GLAND_SCHEDULER', 'poly'),

        # System Configuration
        "NUM_WORKERS": int(os.getenv('GLAND_NUM_WORKERS', '4')),
        "WEIGHT_DECAY": float(os.getenv('GLAND_WEIGHT_DECAY', '1e-4')),
        "IMAGE_SIZE": os.getenv('GLAND_IMAGE_SIZE', '512,512'),

        # Teacher-Student Shell Script Parameters (TS_* variables)
        "TS_BACKBONE_TYPE": os.getenv('GLAND_TS_BACKBONE_TYPE', 'baseline_unet'),
        "TS_EMA_DECAY": float(os.getenv('GLAND_TS_EMA_DECAY', '0.999')),
        "TS_EMA_SCHEDULE": os.getenv('GLAND_TS_EMA_SCHEDULE', 'fixed'),
        "TS_EMA_DECAY_INITIAL": float(os.getenv('GLAND_TS_EMA_DECAY_INITIAL', '0.999')),
        "TS_EMA_DECAY_FINAL": float(os.getenv('GLAND_TS_EMA_DECAY_FINAL', '0.1')),
        "TS_EMA_ANNEALING_START_EPOCH": int(os.getenv('GLAND_TS_EMA_ANNEALING_START_EPOCH', '50')),
        "TS_TEACHER_INIT_EPOCH": int(os.getenv('GLAND_TS_TEACHER_INIT_EPOCH', '20')),
        "TS_MIN_ALPHA": float(os.getenv('GLAND_TS_MIN_ALPHA', '0.01')),
        "TS_MAX_ALPHA": float(os.getenv('GLAND_TS_MAX_ALPHA', '0.9')),
        "TS_CONSISTENCY_LOSS_TYPE": os.getenv('GLAND_TS_CONSISTENCY_LOSS_TYPE', 'mse'),
        "TS_CONSISTENCY_TEMPERATURE": float(os.getenv('GLAND_TS_CONSISTENCY_TEMPERATURE', '1.0')),
        "TS_ENABLE_GLAND_CONSISTENCY": os.getenv('GLAND_TS_ENABLE_GLAND_CONSISTENCY', 'false').lower() == 'true',
        "TS_DEPTH": int(os.getenv('GLAND_TS_DEPTH', '4')),
        "TS_INITIAL_CHANNELS": int(os.getenv('GLAND_TS_INITIAL_CHANNELS', '64')),
        "TS_PSEUDO_MASK_FILTERING": os.getenv('GLAND_TS_PSEUDO_MASK_FILTERING', 'none'),
        "TS_CONFIDENCE_THRESHOLD": float(os.getenv('GLAND_TS_CONFIDENCE_THRESHOLD', '0.8')),
        "TS_ENTROPY_THRESHOLD": float(os.getenv('GLAND_TS_ENTROPY_THRESHOLD', '1.0')),
        "TS_FILTERING_WARMUP_EPOCHS": int(os.getenv('GLAND_TS_FILTERING_WARMUP_EPOCHS', '10')),
        "TS_CONFIDENCE_ANNEALING": os.getenv('GLAND_TS_CONFIDENCE_ANNEALING', 'none'),
        "TS_CONFIDENCE_MAX_THRESHOLD": float(os.getenv('GLAND_TS_CONFIDENCE_MAX_THRESHOLD', '0.9')),
        "TS_CONFIDENCE_MIN_THRESHOLD": float(os.getenv('GLAND_TS_CONFIDENCE_MIN_THRESHOLD', '0.6')),
        "TS_CONFIDENCE_ANNEALING_START_EPOCH": int(os.getenv('GLAND_TS_CONFIDENCE_ANNEALING_START_EPOCH', '5')),
        "TS_GT_TEACHER_INCORPORATE_ENABLED": os.getenv('GLAND_TS_GT_TEACHER_INCORPORATE_ENABLED', 'false').lower() == 'true',
        "TS_GT_INCORPORATE_START_EPOCH": int(os.getenv('GLAND_TS_GT_INCORPORATE_START_EPOCH', '0')),
        "TS_GT_INCORPORATE_SEGMENTATION_ONLY": os.getenv('GLAND_TS_GT_INCORPORATE_SEGMENTATION_ONLY', 'true').lower() == 'true',
        "TS_POST_EVAL_MODE": os.getenv('GLAND_TS_POST_EVAL_MODE', 'student'),
        "TEACHER_STUDENT_EVALUATOR": os.getenv('GLAND_TEACHER_STUDENT_EVALUATOR', 'latest'),
    },

    # Complete Reproducibility Information
    "reproducibility": {
        # Master Seed (single source of truth for reproducibility)
        "master_seed": int(os.getenv('GLAND_MASTER_SEED', '42')),

        # Random Seeds (all derived from master seed for perfect reproducibility)
        "python_seed": int(os.getenv('GLAND_PYTHON_SEED', '42')),
        "numpy_seed": int(os.getenv('GLAND_NUMPY_SEED', '42')),
        "torch_seed": int(os.getenv('GLAND_TORCH_SEED', '42')),
        "torch_cuda_seed": int(os.getenv('GLAND_TORCH_CUDA_SEED', '42')),
        "torch_cuda_seed_all": int(os.getenv('GLAND_TORCH_CUDA_SEED_ALL', '42')),

        # System Information (captured at training time)
        "timestamp": None,  # Will be set during training
        "hostname": None,   # Will be set during training
        "username": None,   # Will be set during training
        "working_directory": None,  # Will be set during training
        "python_version": None,     # Will be set during training
        "torch_version": None,      # Will be set during training
        "cuda_version": None,       # Will be set during training
        "gpu_info": None,           # Will be set during training
        "cpu_info": None,           # Will be set during training
        "system_info": None,        # Will be set during training

        # Git Information (for code version tracking)
        "git_commit_hash": None,    # Will be set during training
        "git_branch": None,         # Will be set during training
        "git_status_clean": None,   # Will be set during training
        "git_remote_url": None,     # Will be set during training

        # Environment Variables (complete environment snapshot)
        "conda_environment": None,  # Will be set during training
        "pip_packages": None,       # Will be set during training
        "environment_variables": None,  # Will be set during training

        # Command Line and Script Information
        "command_line_args": None,  # Will be set during training
        "script_path": None,        # Will be set during training
        "config_file_path": None,   # Will be set during training
    },
}

# =============================================================================
# EVALUATION SETTINGS - Updated for 4-class segmentation
# =============================================================================

EVALUATION_CONFIG = {
    # Post-training evaluation
    "num_train_samples": 50,
    "num_test_samples": 50,
    "samples_per_figure": 5,
    "figure_dpi": 200,
    "figure_size": (20, 25),

    # Visualization colors for 4 classes
    "color_background": [0, 0, 0],      # Black
    "color_benign": [0, 255, 0],        # Green
    "color_malignant": [255, 0, 0],     # Red
    "color_pdc": [0, 0, 255],           # Blue
    "overlay_alpha": 0.6,

    # Class names
    "class_names": {
        0: "Background",
        1: "Benign Glands",
        2: "Malignant/Tumor Glands",
        3: "PDC (Poorly Differentiated Carcinoma)"
    },

    # Random seed for reproducible sampling
    "random_seed": 42,
}

# =============================================================================
# FILE NAMING CONVENTIONS
# =============================================================================

FILE_NAMES = {
    # Config files
    "training_config": "training_config.json",
    "dataset_config": "dataset_config.json",

    # Model files
    "best_model": "best_model.pth",
    "latest_model": "latest_model.pth",
    "checkpoint": "checkpoint_epoch_{:03d}.pth",

    # Loss and metrics files
    "loss_history": "loss_history.csv",
    "train_losses": "train_losses.csv",
    "val_losses": "val_losses.csv",
    "metrics_history": "metrics_history.csv",

    # Summary files
    "training_summary": "training_summary.xlsx",
    "experiment_summary": "experiment_summary.xlsx",
    "quick_summary": "quick_summary.csv",

    # Evaluation files
    "evaluation_summary": "evaluation_summary.json",
    "comprehensive_evaluation": "comprehensive_evaluation.xlsx",

    # Figure naming patterns
    "training_figure_pattern": "training_evaluation_figure_{:02d}.png",
    "test_figure_pattern": "test_evaluation_figure_{:02d}.png",
    "validation_figure_pattern": "validation_evaluation_figure_{:02d}.png",

    # Directory names
    "models_dir": "models",
    "visualizations_dir": "visualizations",
    "logs_dir": "logs",
    "checkpoints_dir": "checkpoints",
}

# =============================================================================
# ENVIRONMENT VALIDATION
# =============================================================================

def validate_environment_variables():
    """Validate that all required environment variables are set"""
    required_vars = [
        'GLAND_DATASET_BASE',
        'GLAND_OUTPUT_DIR',
        'NNUNET_PREPROCESSED',
        'NNUNET_RESULTS',
        'GLAND_TEMP_DIR'
    ]

    missing_vars = []
    for var in required_vars:
        if os.getenv(var) is None:
            missing_vars.append(var)

    if missing_vars:
        raise ValueError(
            f"Missing required environment variables: {missing_vars}. "
            f"Please run the training script (run_nnunet_training.sh) which exports these variables."
        )

    print("✅ All required environment variables are set:")
    for var in required_vars:
        print(f"  {var}={os.getenv(var)}")

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_dataset_path(dataset_key: str = None) -> str:
    """Get path to specific combined dataset"""
    if dataset_key is None:
        dataset_key = DEFAULT_DATASET

    # Get available datasets at runtime to pick up environment variables
    available_datasets = get_available_datasets()

    if dataset_key in available_datasets:
        return available_datasets[dataset_key]
    else:
        raise ValueError(f"Unknown dataset key: {dataset_key}. Available: {list(available_datasets.keys())}")

def get_output_base() -> str:
    """Get the base output directory"""
    output_base = DATA_PATHS["output_base"]
    if output_base is None:
        raise ValueError("GLAND_OUTPUT_DIR environment variable not set. Please run the training script which exports this variable.")
    return output_base

def get_experiment_path(experiment_name: str) -> Path:
    """Get full path to an experiment directory"""
    return Path(get_output_base()) / experiment_name

def create_config_dict(dataset_key: str = None, **overrides) -> dict:
    """Create a complete configuration dictionary for specified dataset"""
    config = DEFAULT_CONFIG.copy()

    # Set dataset path (computed at runtime)
    config["data_root"] = get_dataset_path(dataset_key)
    config["dataset_key"] = dataset_key or DEFAULT_DATASET
    config["output_base"] = get_output_base()

    # Add runtime dataset base for debugging
    config["runtime_dataset_base"] = get_combined_data_base()

    # Add dataset-specific information
    dataset_path = Path(config["data_root"])
    if dataset_path.exists():
        config["dataset_exists"] = True
        # Try to read dataset.json for additional info
        dataset_json_path = dataset_path / "dataset.json"
        if dataset_json_path.exists():
            import json
            with open(dataset_json_path, 'r') as f:
                dataset_info = json.load(f)
                config["dataset_info"] = dataset_info
    else:
        config["dataset_exists"] = False

    # Apply overrides
    config.update(overrides)

    # Populate runtime reproducibility information
    config = populate_runtime_reproducibility_info(config)

    return config

def validate_dataset_path(dataset_key: str) -> bool:
    """Validate that a dataset path exists and has required structure"""
    try:
        dataset_path = Path(get_dataset_path(dataset_key))

        # Check if main directory exists
        if not dataset_path.exists():
            return False

        # Check for required subdirectories
        required_dirs = ["imagesTr", "labelsTr", "imagesVal", "labelsVal", "imagesTs", "labelsTs"]
        for dir_name in required_dirs:
            if not (dataset_path / dir_name).exists():
                return False

        # Check for dataset.json
        if not (dataset_path / "dataset.json").exists():
            return False

        return True
    except:
        return False

def list_available_datasets() -> dict:
    """List all available datasets with their validation status"""
    datasets = {}
    # Get available datasets at runtime
    available_datasets = get_available_datasets()
    for key, path in available_datasets.items():
        datasets[key] = {
            "path": path,
            "exists": Path(path).exists(),
            "valid": validate_dataset_path(key) if Path(path).exists() else False
        }
    return datasets

def print_config_summary(dataset_key: str = None):
    """Print a summary of configuration for specified dataset"""
    dataset_key = dataset_key or DEFAULT_DATASET

    print("=" * 80)
    print("Combined Gland Segmentation nnU-Net - Configuration Summary")
    print("=" * 80)

    print(f"\n📊 TARGET DATASET: {dataset_key}")
    try:
        dataset_path = get_dataset_path(dataset_key)
        dataset_valid = validate_dataset_path(dataset_key)
        exists_symbol = "✅" if Path(dataset_path).exists() else "❌"
        valid_symbol = "✅" if dataset_valid else "❌"
        print(f"  {exists_symbol} Path: {dataset_path}")
        print(f"  {valid_symbol} Structure: {'Valid nnU-Net format' if dataset_valid else 'Invalid or incomplete'}")
    except Exception as e:
        print(f"  ❌ Error: {e}")

    print(f"\n📁 AVAILABLE DATASETS:")
    datasets = list_available_datasets()
    for key, info in datasets.items():
        exists_symbol = "✅" if info["exists"] else "❌"
        valid_symbol = "✅" if info["valid"] else "❌"
        current_marker = " (CURRENT)" if key == dataset_key else ""
        print(f"  {exists_symbol}{valid_symbol} {key}: {info['path']}{current_marker}")

    print(f"\n🔧 PROJECT PATHS:")
    for name, path in PROJECT_PATHS.items():
        exists = "✅" if Path(path).exists() else "❌"
        print(f"  {exists} {name}: {path}")

    print(f"\n💾 DATA PATHS:")
    for name, path in DATA_PATHS.items():
        if name not in ["default_dataset", "combined_data_base"]:  # Skip computed ones
            if path is not None:
                exists = "✅" if Path(path).exists() else "❌"
                print(f"  {exists} {name}: {path}")
            else:
                print(f"  ❌ {name}: NOT SET (environment variable missing)")

    # Show runtime computed paths
    try:
        runtime_base = get_combined_data_base()
        exists = "✅" if Path(runtime_base).exists() else "❌"
        print(f"  {exists} combined_data_base (runtime): {runtime_base}")
    except ValueError as e:
        print(f"  ❌ combined_data_base: {e}")

    print(f"\n⚙️  4-CLASS CONFIGURATION:")
    print(f"  🎯 Segmentation classes: {DEFAULT_CONFIG['num_seg_classes']}")
    print(f"  🏷️  Patch classes: {DEFAULT_CONFIG['num_patch_classes']}")
    print(f"  🔍 Gland classes: {DEFAULT_CONFIG['num_gland_classes']}")
    print(f"  🏃 Training epochs: {DEFAULT_CONFIG['epochs']}")
    print(f"  📦 Batch size: {DEFAULT_CONFIG['batch_size']}")

    print(f"\n📊 PSEUDO-GT METRICS (Student vs Pseudo-GT Monitoring):")
    pseudo_config = DEFAULT_CONFIG.get('pseudo_gt_metrics', {})
    if pseudo_config.get('enabled', True):
        print(f"  ✅ Pseudo-GT metrics: ENABLED")
        print(f"  🖥️  CPU computation: {pseudo_config.get('cpu_computation', True)}")
        print(f"  🔒 Memory leak prevention: {pseudo_config.get('memory_leak_prevention', True)}")
        print(f"  📈 Progress bar display: {pseudo_config.get('progress_bar_display', True)}")
        print(f"  📊 Visualization included: {pseudo_config.get('include_in_visualization', True)}")
        print(f"  🚫 Gradient isolation: {pseudo_config.get('gradient_isolation', True)}")
        print(f"  🧹 Explicit cleanup: {pseudo_config.get('explicit_cleanup', True)}")
        print(f"  ⚠️  OOM fallback: {pseudo_config.get('fallback_on_oom', True)}")
    else:
        print(f"  ❌ Pseudo-GT metrics: DISABLED")

    print(f"\n🎨 CLASS COLORS:")
    colors = EVALUATION_CONFIG
    for class_id, name in colors["class_names"].items():
        color_key = f"color_{name.lower().split()[0]}"
        if color_key in colors:
            color = colors[color_key]
            print(f"  🎨 Class {class_id} ({name}): RGB{color}")

    print("=" * 80)

def populate_runtime_reproducibility_info(config: dict) -> dict:
    """
    Populate runtime system information for complete reproducibility

    Args:
        config: Configuration dictionary to update

    Returns:
        Updated configuration with runtime information populated
    """
    import datetime
    import socket
    import getpass
    import os
    import platform
    import subprocess
    import sys

    # Create a copy to avoid modifying the original
    config = config.copy()

    # Get timestamp
    config["reproducibility"]["timestamp"] = datetime.datetime.now().isoformat()

    # System information
    config["reproducibility"]["hostname"] = socket.gethostname()
    config["reproducibility"]["username"] = getpass.getuser()
    config["reproducibility"]["working_directory"] = os.getcwd()

    # Python version
    config["reproducibility"]["python_version"] = sys.version

    # PyTorch version (if available)
    try:
        import torch
        config["reproducibility"]["torch_version"] = torch.__version__
        if torch.cuda.is_available():
            config["reproducibility"]["cuda_version"] = torch.version.cuda
            config["reproducibility"]["gpu_info"] = {
                "device_count": torch.cuda.device_count(),
                "devices": [torch.cuda.get_device_properties(i).__dict__ for i in range(torch.cuda.device_count())]
            }
        else:
            config["reproducibility"]["cuda_version"] = "Not available"
            config["reproducibility"]["gpu_info"] = "No GPU available"
    except ImportError:
        config["reproducibility"]["torch_version"] = "Not available"
        config["reproducibility"]["cuda_version"] = "Not available"
        config["reproducibility"]["gpu_info"] = "PyTorch not available"

    # CPU information
    config["reproducibility"]["cpu_info"] = {
        "architecture": platform.machine(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "platform": platform.platform()
    }

    # System information
    config["reproducibility"]["system_info"] = {
        "platform": platform.system(),
        "release": platform.release(),
        "version": platform.version(),
        "architecture": platform.architecture(),
        "node": platform.node()
    }

    # Git information (if in a git repository)
    try:
        # Git commit hash
        result = subprocess.run(['git', 'rev-parse', 'HEAD'],
                              capture_output=True, text=True, check=True)
        config["reproducibility"]["git_commit_hash"] = result.stdout.strip()

        # Git branch
        result = subprocess.run(['git', 'rev-parse', '--abbrev-ref', 'HEAD'],
                              capture_output=True, text=True, check=True)
        config["reproducibility"]["git_branch"] = result.stdout.strip()

        # Git status (check if clean)
        result = subprocess.run(['git', 'status', '--porcelain'],
                              capture_output=True, text=True, check=True)
        config["reproducibility"]["git_status_clean"] = len(result.stdout.strip()) == 0

        # Git remote URL
        result = subprocess.run(['git', 'remote', 'get-url', 'origin'],
                              capture_output=True, text=True, check=True)
        config["reproducibility"]["git_remote_url"] = result.stdout.strip()

    except (subprocess.CalledProcessError, FileNotFoundError):
        config["reproducibility"]["git_commit_hash"] = "Not in git repository"
        config["reproducibility"]["git_branch"] = "Not in git repository"
        config["reproducibility"]["git_status_clean"] = None
        config["reproducibility"]["git_remote_url"] = "Not in git repository"

    # Environment information
    # Conda environment
    conda_env = os.environ.get('CONDA_DEFAULT_ENV')
    if conda_env:
        config["reproducibility"]["conda_environment"] = conda_env
        # Try to get pip packages in conda env
        try:
            result = subprocess.run([sys.executable, '-m', 'pip', 'list', '--format=freeze'],
                                  capture_output=True, text=True, check=True)
            config["reproducibility"]["pip_packages"] = result.stdout.strip().split('\n')
        except subprocess.CalledProcessError:
            config["reproducibility"]["pip_packages"] = "Unable to get pip packages"
    else:
        config["reproducibility"]["conda_environment"] = "Not in conda environment"
        try:
            result = subprocess.run([sys.executable, '-m', 'pip', 'list', '--format=freeze'],
                                  capture_output=True, text=True, check=True)
            config["reproducibility"]["pip_packages"] = result.stdout.strip().split('\n')
        except subprocess.CalledProcessError:
            config["reproducibility"]["pip_packages"] = "Unable to get pip packages"

    # Environment variables (filter sensitive ones)
    sensitive_keywords = ['password', 'secret', 'key', 'token', 'auth', 'credential']
    env_vars = {}
    for key, value in os.environ.items():
        # Skip sensitive environment variables
        if not any(keyword.lower() in key.lower() for keyword in sensitive_keywords):
            env_vars[key] = value
    config["reproducibility"]["environment_variables"] = env_vars

    return config

if __name__ == "__main__":
    print_config_summary()

    print(f"\n🔍 Dataset Validation:")
    datasets = list_available_datasets()
    for key, info in datasets.items():
        status = "✅ READY" if info["valid"] else "❌ NOT READY"
        print(f"  {status} {key}")