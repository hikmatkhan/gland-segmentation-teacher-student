# Multi-Architecture Gland Segmentation Framework

A comprehensive **4-class multi-task** deep learning framework for glandular structure analysis in histopathology images. Supports **Baseline UNet**, **nnU-Net**, and **Teacher-Student UNet** architectures with unified training, evaluation, and comparison capabilities.

## ✨ Key Features

### 🏗️ Multi-Architecture Support
- **Baseline UNet**: Simple, interpretable architecture for baseline comparison (31M+ parameters)
- **nnU-Net**: State-of-the-art medical imaging architecture (20M+ parameters)
- **Teacher-Student UNet**: Independent dual networks with separate classification heads for true teacher-student learning (62M+ parameters)
- **Unified Interface**: Single CLI for all architectures with automatic prefixing

### 🎯 Multi-Task Learning
- **4-Class Segmentation**: Background (0), Benign (1), Malignant (2), PDC (3)
- **Multi-Label Patch Classification**: Realistic patches with multiple gland types
- **Individual Gland Classification**: Per-gland 4-class assessment
- **Adaptive Loss Weighting**: Automatic balancing across tasks

### 📊 Combined Datasets
- **Warwick GlaS**: 165 histopathology images with expert annotations
- **OSU Makoto**: 32 slides × 4 magnifications with 4-class labels
- **Multi-Magnification Support**: Mixed or magnification-specific training

## 🚀 Quick Start

### 🎯 SLURM Training (Recommended)
```bash
# Edit configuration in run_nnunet_training.sh
sbatch run_nnunet_training.sh
```

**⚠️ IMPORTANT**: The training script **automatically exports all required environment variables**. Always use the SLURM script or manually export the required environment variables before running any Python commands.

### Demo (Test Installation)
```bash
# First, export required environment variables (or use run_nnunet_training.sh)
export GLAND_DATASET_BASE="/path/to/your/datasets"
export GLAND_OUTPUT_DIR="/path/to/outputs"
export NNUNET_PREPROCESSED="/path/to/preprocessed"
export NNUNET_RESULTS="/path/to/results"
export GLAND_TEMP_DIR="/tmp/temp_dir"

# Test baseline UNet
python main.py demo --architecture baseline_unet

# Test nnU-Net (default)
python main.py demo --architecture nnunet

# 🆕 Test Pseudo-GT visualization (Teacher-Student specific)
python test_pseudo_gt_clean.py
```

### 🆕 Testing Pseudo-GT Visualization
Test the enhanced Teacher-Student training visualization independently:

```bash
# Generate sample pseudo-GT visualization
python test_pseudo_gt_clean.py

# Check generated plots
ls test_pseudo_gt_visualizations/
# → teacher_student_training_with_pseudo_gt.png
# → teacher_student_training_with_pseudo_gt.pdf
```

**Features Tested**:
- ✅ 3×3 grid layout with 8 specialized subplots
- ✅ Pseudo-Dice and Pseudo-IoU sub-figures with comparison curves
- ✅ Training phase visualization and alignment analysis
- ✅ High-resolution output (PNG + PDF) suitable for publications

### Manual Training (Not Recommended)
**⚠️ Note**: Manual training requires setting environment variables. Use the SLURM script instead.

```bash
# Export required environment variables first
export GLAND_DATASET_BASE="/path/to/your/datasets"
export GLAND_OUTPUT_DIR="/path/to/outputs"
export NNUNET_PREPROCESSED="/path/to/preprocessed"
export NNUNET_RESULTS="/path/to/results"
export GLAND_TEMP_DIR="/tmp/temp_dir"

# Baseline UNet training (prefixed: baseline_unet_exp_*)
python main.py train --architecture baseline_unet --dataset mixed --epochs 150 --patience 250 --output_dir /path/to/outputs

# nnU-Net training (prefixed: nnunet_exp_*)
python main.py train --architecture nnunet --dataset mixed --epochs 150 --patience 250 --output_dir /path/to/outputs
```

### Evaluation
```bash
# Export required environment variables first (or use evaluation script)
export GLAND_DATASET_BASE="/path/to/your/datasets"

# Evaluate trained models
python main.py evaluate --architecture baseline_unet --model /path/to/model.pth --dataset mixed
python main.py evaluate --architecture nnunet --model /path/to/model.pth --dataset mixed
```

## 🏗️ Architecture Comparison

| Feature | Baseline UNet | nnU-Net | Teacher-Student UNet |
|---------|---------------|---------|---------------------|
| **Parameters** | 31M+ | 20M+ | 62M+ (2×31M) |
| **Architecture** | Standard UNet | PlainConvUNet | Dual BaselineUNet |
| **Deep Supervision** | No | Yes | No |
| **Training Strategy** | Supervised | Supervised | Semi-supervised |
| **Teacher Updates** | N/A | N/A | EMA-based |
| **Consistency Loss** | No | No | Yes (Multi-type) |
| **Training Speed** | Fast | Moderate | Slower |
| **Performance** | Good baseline | State-of-the-art | Enhanced consistency |
| **Complexity** | Simple | Advanced | Moderate |

## 📁 Project Structure

```
nnUNet/
├── src/
│   ├── models/
│   │   ├── baseline_unet.py         # Baseline UNet implementation
│   │   ├── teacher_student_unet.py  # Teacher-Student UNet implementation
│   │   ├── teacher_student_loss.py  # Teacher-Student loss functions
│   │   ├── model_factory.py         # Architecture selection factory
│   │   ├── nnunet_integration.py    # nnU-Net integration
│   │   ├── multi_task_wrapper.py    # Multi-architecture wrapper
│   │   ├── projection_heads.py      # Classification heads
│   │   └── loss_functions.py        # Multi-task losses
│   ├── training/
│   │   ├── dataset.py              # Data loading & augmentation
│   │   └── trainer.py              # Training pipeline
│   └── evaluation/
│       └── evaluator.py            # Evaluation & visualization
├── configs/
│   └── paths_config.py            # Dataset configurations
├── logs/                           # SLURM training logs
├── main.py                         # Main CLI entry point
├── run_nnunet_training.sh         # SLURM training script
├── test_pseudo_gt_clean.py        # 🆕 Pseudo-GT visualization testing
└── README.md                       # This file
```

## 🎯 SLURM Training Script

The `run_nnunet_training.sh` script provides a comprehensive, configurable training interface for HPC environments:

### Key Features
- **🔧 Easy Configuration**: Modify variables at the top of the script
- **📊 Multiple Datasets**: Support for all magnification combinations
- **⚙️ Architecture Choice**: Baseline UNet or nnU-Net selection
- **🛑 Advanced Stopping**: Configurable early stopping and LR scheduling
- **📁 Automatic Organization**: Clean output structure with prefixed directories
- **🧪 Built-in Testing**: Demo run before training starts

### Configuration Options

#### Basic Training Parameters
```bash
ARCHITECTURE="baseline_unet"    # baseline_unet, nnunet, teacher_student_unet
DATASET_KEY="mag40x"           # mixed, mag5x, mag10x, mag20x, mag40x
EPOCHS=1000                    # Training epochs
BATCH_SIZE=16                  # Batch size
LEARNING_RATE="1e-2"          # Learning rate
```

#### Teacher-Student Specific Parameters
```bash
# Teacher-Student UNet Configuration (when ARCHITECTURE="teacher_student_unet")
TS_EMA_DECAY=0.999                    # EMA decay factor for teacher updates (backward compatibility, default: 0.999)
TS_TEACHER_INIT_EPOCH=20              # Epoch to initialize teacher (default: 20)
TS_MIN_ALPHA=0.01                     # Minimum alpha for consistency loss (default: 0.01)
TS_MAX_ALPHA=1.0                      # Maximum alpha for supervised loss (default: 1.0)
TS_CONSISTENCY_LOSS_TYPE="mse"        # Consistency loss type: mse, kl_div, l1, dice, iou
TS_CONSISTENCY_TEMPERATURE=2.0        # Temperature for consistency loss (1.0-4.0, default: 1.0)
TS_ENABLE_GLAND_CONSISTENCY=false     # Enable gland classification consistency (default: false)
TS_DEPTH=4                            # UNet depth (3-5, default: 4)
TS_INITIAL_CHANNELS=64                # Initial channels (32-128, default: 64)

# 🆕 EMA DECAY ANNEALING (Dynamic Teacher-Student Learning)
TS_EMA_SCHEDULE="fixed"               # EMA schedule: "fixed", "cosine", "linear", "exponential" (default: "fixed")
TS_EMA_DECAY_INITIAL=0.999            # Initial EMA decay for annealing schedules (default: 0.999)
TS_EMA_DECAY_FINAL=0.1                # Final EMA decay for annealing schedules (default: 0.1)
TS_EMA_ANNEALING_START_EPOCH=50       # Epoch to start EMA annealing (default: 50)

# 🎯 TEACHER PSEUDO-MASK FILTERING (Noise Reduction)
TS_PSEUDO_MASK_FILTERING="none"       # Filtering strategy: "none", "confidence", "entropy"
TS_CONFIDENCE_THRESHOLD=0.8           # Confidence threshold (0.7-0.95, default: 0.8)
TS_ENTROPY_THRESHOLD=1.0              # Entropy threshold (0.5-2.0, default: 1.0, lower=more selective)
TS_FILTERING_WARMUP_EPOCHS=10         # Epochs before applying filtering (0-50, default: 10)

# 🔍 TEACHER-STUDENT EVALUATION CONFIGURATION
TEACHER_STUDENT_EVALUATOR="latest"   # Which checkpoint to use ("latest" or "best")
TS_POST_EVAL_MODE="student"           # Post-training eval mode: "student", "teacher", "both"
```

#### Loss Function Weights
```bash
DICE_WEIGHT=0.5               # Weight for Dice loss (0.0-1.0, default: 0.5)
CE_WEIGHT=0.5                 # Weight for Cross-Entropy loss (0.0-1.0, default: 0.5)
                              # Note: DICE_WEIGHT + CE_WEIGHT should typically equal 1.0
```

#### Stopping Criteria
```bash
EARLY_STOP_PATIENCE=250       # Early stopping patience
LR_SCHEDULER_PATIENCE=25      # LR reduction patience
MIN_LR="1e-7"                # Minimum learning rate
```

#### Learning Rate Scheduling
```bash
COSINE_T_MAX="$EPOCHS"        # CosineAnnealingLR T_max
COSINE_ETA_MIN="$MIN_LR"      # CosineAnnealingLR eta_min
```

#### 🔄 Reproducibility Configuration (Automatic)
```bash
# Automatic seed generation (NEW!)
# A random master seed is generated for each run and used for all random operations
MASTER_SEED=auto              # Automatically generated 4-digit seed (1000-9999)

# Override for exact reproduction
export MASTER_SEED=1234       # Set specific seed to reproduce exact run
```

**Features:**
- **🎲 Automatic Generation**: Each training run gets a unique random seed
- **🔄 Unified Control**: Single master seed controls all randomness (Python, NumPy, PyTorch, CUDA)
- **📝 Easy Reproduction**: Copy the master seed to reproduce exact results
- **🚀 Optimal Performance**: Uses PyTorch defaults for maximum speed
- **💾 Full Traceability**: Master seed saved in `training_config.json`

**Example Usage:**
```bash
# Automatic seed (different each run)
sbatch run_nnunet_training.sh

# Reproduce specific run
MASTER_SEED=1234 sbatch run_nnunet_training.sh
```

#### Advanced Options (Uncomment to Use)
```bash
# NUM_WORKERS=4               # Data loading workers
# WEIGHT_DECAY=1e-4           # L2 regularization
# OPTIMIZER="adamw"           # adamw, sgd
# SCHEDULER="poly"            # poly, cosine, plateau
# IMAGE_SIZE="512,512"        # Input dimensions
```

### Enhanced Training
```bash
ENHANCED_TRAINING=true        # Stronger augmentation
```

### Usage Examples
```bash
# Quick test (2 epochs)
EPOCHS=2 sbatch run_nnunet_training.sh

# Baseline UNet on mixed data
ARCHITECTURE="baseline_unet" DATASET_KEY="mixed" sbatch run_nnunet_training.sh

# nnU-Net with enhanced training
ARCHITECTURE="nnunet" ENHANCED_TRAINING=true sbatch run_nnunet_training.sh
```

## 📊 Available Datasets

| Dataset | Description | Samples | Magnifications |
|---------|-------------|---------|----------------|
| `mixed` | All magnifications | ~25,000 | 5x, 10x, 20x, 40x |
| `mag5x` | 5x only | ~4,800 | 5x |
| `mag10x` | 10x only | ~6,200 | 10x |
| `mag20x` | 20x only | ~6,000 | 20x |
| `mag40x` | 40x only | ~6,000 | 40x |

## 🎛️ CLI Options

### Training Parameters
```bash
python main.py train [options]
  --architecture {baseline_unet,nnunet,teacher_student_unet}  # Model architecture
  --dataset {mixed,mag5x,mag10x,mag20x,mag40x}  # Dataset choice
  --epochs INT                           # Training epochs (default: 150)
  --batch_size INT                       # Batch size (default: 4)
  --learning_rate FLOAT                  # Learning rate (default: 1e-4)
  --output_dir PATH                      # Output directory (REQUIRED)
  --experiment_name STR                  # Custom experiment name
  --enhanced                             # Use stronger augmentation
  --patience INT                         # Early stopping patience (default: 30)
```

### Environment Variables

#### Required Environment Variables (Set by SLURM Script)
```bash
# Dataset and Output Paths (REQUIRED - set by run_nnunet_training.sh)
export GLAND_DATASET_BASE="/path/to/your/datasets"      # Base dataset directory
export GLAND_OUTPUT_DIR="/path/to/outputs"              # Output directory
export NNUNET_PREPROCESSED="/path/to/preprocessed"      # nnU-Net preprocessing
export NNUNET_RESULTS="/path/to/results"                # nnU-Net results
export GLAND_TEMP_DIR="/tmp/temp_dir"                   # Temporary files
```

#### Optional Training Parameters (Advanced Configuration)
```bash
# Training Parameters
export GLAND_EPOCHS=1000
export GLAND_BATCH_SIZE=16
export GLAND_LEARNING_RATE="1e-2"
export GLAND_ENHANCED_TRAINING="true"

# Loss Function Weights
export GLAND_DICE_WEIGHT=0.7
export GLAND_CE_WEIGHT=0.3

# Stopping Criteria
export GLAND_EARLY_STOP_PATIENCE=250
export GLAND_LR_SCHEDULER_PATIENCE=25
export GLAND_MIN_LR="1e-7"

# Learning Rate Scheduling
export GLAND_COSINE_T_MAX=1000
export GLAND_COSINE_ETA_MIN="1e-7"

# Optimization
export GLAND_OPTIMIZER="adamw"          # adamw, sgd
export GLAND_SCHEDULER="poly"           # poly, cosine, plateau
export GLAND_WEIGHT_DECAY="1e-4"
export GLAND_NUM_WORKERS=4

# Data Processing
export GLAND_IMAGE_SIZE="512,512"

# Teacher-Student UNet Parameters (when ARCHITECTURE="teacher_student_unet")
export GLAND_TS_EMA_DECAY=0.999
export GLAND_TS_TEACHER_INIT_EPOCH=20
export GLAND_TS_MIN_ALPHA=0.01
export GLAND_TS_MAX_ALPHA=1.0
export GLAND_TS_CONSISTENCY_LOSS_TYPE="mse"
export GLAND_TS_CONSISTENCY_TEMPERATURE=1.0
export GLAND_TS_ENABLE_GLAND_CONSISTENCY=false
export GLAND_TS_DEPTH=4
export GLAND_TS_INITIAL_CHANNELS=64
export GLAND_TEACHER_STUDENT_EVALUATOR="latest"
export GLAND_TS_POST_EVAL_MODE="student"

# 🆕 EMA Decay Annealing Parameters
export GLAND_TS_EMA_SCHEDULE="fixed"
export GLAND_TS_EMA_DECAY_INITIAL=0.999
export GLAND_TS_EMA_DECAY_FINAL=0.1
export GLAND_TS_EMA_ANNEALING_START_EPOCH=50
```

### Evaluation Parameters
```bash
python main.py evaluate [options]
  --architecture {baseline_unet,nnunet,teacher_student_unet}  # Model architecture
  --model PATH                           # Model checkpoint path (REQUIRED)
  --dataset {mixed,mag5x,mag10x,mag20x,mag40x}  # Evaluation dataset
  --output PATH                          # Output directory
  --visualize                            # Generate visualizations
```

## 🔍 Architecture Selection Guide

### Use Baseline UNet For:
- ✅ **Research baselines** and fair comparisons
- ✅ **Educational purposes** and method understanding
- ✅ **Quick experiments** with faster training
- ✅ **Resource constraints** with limited compute
- ✅ **Interpretability** studies and analysis

### Use nnU-Net For:
- ✅ **Maximum performance** and state-of-the-art results
- ✅ **Production deployment** in clinical settings
- ✅ **Complex datasets** requiring advanced features
- ✅ **Research publications** with SOTA benchmarks
- ✅ **Multi-scale analysis** with deep supervision

### Use Teacher-Student UNet For:
- ✅ **Self-training** and semi-supervised learning research
- ✅ **Consistency regularization** studies
- ✅ **Knowledge distillation** experiments
- ✅ **Model ensembling** without inference overhead
- ✅ **Robustness improvement** through teacher guidance
- ✅ **Limited labeled data** scenarios

## 📈 Expected Performance

### Baseline UNet
- Segmentation Dice: >0.80
- Multi-Label Accuracy: >0.70
- Training Time: ~6-8h (RTX 4080)
- Good baseline for comparison

### nnU-Net
- Segmentation Dice: >0.85
- Multi-Label Accuracy: >0.75
- Training Time: ~8-12h (RTX 4080)
- State-of-the-art performance

### Teacher-Student UNet
- Segmentation Dice: >0.82 (with consistency)
- Multi-Label Accuracy: >0.72
- Training Time: ~12-16h (RTX 4080)
- Enhanced robustness and consistency
- Dual model evaluation capability

## 🔧 Technical Requirements

- **Python**: 3.9+
- **PyTorch**: 2.0+
- **GPU**: ≥8GB VRAM (recommended: ≥10GB)
- **RAM**: ≥16GB (recommended: ≥32GB)
- **Storage**: ≥30GB available space

## 📝 Output Structure

Each training run creates an experiment directory with automatic architecture prefixing:

```
output_dir/
├── baseline_unet_exp_2025-09-17_14-30-45/  # Baseline UNet experiments
│   ├── models/
│   │   ├── best_model.pth                   # Best checkpoint
│   │   └── latest_model.pth                 # Latest checkpoint
│   ├── logs/
│   │   ├── training.log                     # Training logs
│   │   └── tensorboard/                     # TensorBoard logs
│   ├── visualizations/
│   │   ├── training_curves.png              # Training plots
│   │   └── training_curves.pdf
│   ├── evaluations/                         # Evaluation results
│   ├── training_config.json                 # Configuration
│   ├── loss_history.csv                     # Complete history
│   └── quick_summary.csv                    # Experiment summary
└── nnunet_exp_2025-09-17_15-45-30/         # nnU-Net experiments
    └── [same structure as above]
```

## 🧪 Example Workflows

### SLURM-Based Workflows (Recommended)

#### Baseline Comparison
```bash
# Edit run_nnunet_training.sh: ARCHITECTURE="baseline_unet"
sbatch run_nnunet_training.sh

# Edit run_nnunet_training.sh: ARCHITECTURE="nnunet"
sbatch run_nnunet_training.sh

# Compare results in output directory
```

#### Magnification Study
```bash
# Create multiple training jobs
for mag in mag5x mag10x mag20x mag40x; do
    sed "s/DATASET_KEY=\".*\"/DATASET_KEY=\"$mag\"/" run_nnunet_training.sh > run_${mag}.sh
    sbatch run_${mag}.sh
done
```

#### Extended Training with Higher Patience
```bash
# Edit run_nnunet_training.sh
EPOCHS=2000
EARLY_STOP_PATIENCE=500
sbatch run_nnunet_training.sh
```

### Manual Training (Alternative)

#### Baseline Comparison
```bash
# Train both architectures
python main.py train --architecture baseline_unet --dataset mixed --epochs 150 --patience 250 --output_dir ./outputs
python main.py train --architecture nnunet --dataset mixed --epochs 150 --patience 250 --output_dir ./outputs

# Compare results
ls ./outputs/  # Shows baseline_unet_exp_* and nnunet_exp_* directories
```

#### Enhanced Training
```bash
# Stronger augmentation for robustness
python main.py train --architecture baseline_unet --dataset mixed --enhanced --epochs 200 --patience 250 --output_dir ./outputs
```

## 🔍 Multi-Label Classification

The framework uses realistic multi-label patch classification where patches can contain multiple gland types:

```
Example Labels:
- Background + Benign + Malignant → [1, 1, 1, 0]
- Background + PDC               → [1, 0, 0, 1]
- Background + Benign            → [1, 1, 0, 0]
```

This reflects real histopathology where patches often contain multiple tissue types.

**Note**: Since this is multi-label classification, the prediction probabilities for each class are independent sigmoid outputs (0-1 range) and do not sum to 1.0. Each probability represents the confidence that the specific tissue type is present in the patch.

## 🎓 Teacher-Student UNet Architecture

### Overview
The Teacher-Student UNet implements a semi-supervised learning approach with two identical BaselineUNet networks:
- **Student Network**: Traditional training with gradients and labeled data
- **Teacher Network**: EMA-only updates, provides pseudo-labels for consistency loss

### Key Features

#### 🔄 Exponential Moving Average (EMA) Updates
```python
teacher_params = ema_decay * teacher_params + (1 - ema_decay) * student_params
```
- **Dynamic EMA Decay**: Supports fixed and annealing schedules
- **Backward Compatible**: `TS_EMA_DECAY=0.999` works as before
- Teacher weights are a moving average of student weights
- No gradient updates for teacher network

#### 🆕 EMA Decay Annealing (Dynamic Teacher-Student Learning)

**Purpose**: Implement dynamic EMA decay that starts with high teacher dependence (0.999) and gradually transitions to more student influence (0.1) during training.

**Motivation**: Early in training, maintain stable teacher weights. As training progresses, allow more frequent teacher updates to converge toward student weights.

**Annealing Schedules**:

| Schedule | Early Training | Mid Training | Late Training | Formula |
|----------|----------------|--------------|---------------|---------|
| **Fixed** | 0.999 | 0.999 | 0.999 | `decay = TS_EMA_DECAY` |
| **Cosine** | 0.999 | 0.5495 | 0.1 | `decay = final + (initial - final) * 0.5 * (1 + cos(π * progress))` |
| **Linear** | 0.999 | 0.5495 | 0.1 | `decay = initial - (initial - final) * progress` |
| **Exponential** | 0.999 | 0.3715 | 0.1 | `decay = initial * (final/initial)^progress` |

**Configuration**:
```bash
# Enable cosine annealing (recommended)
TS_EMA_SCHEDULE="cosine"              # Schedule type
TS_EMA_DECAY_INITIAL=0.999            # Starting decay (teacher-heavy)
TS_EMA_DECAY_FINAL=0.1                # Ending decay (student-heavy)
TS_EMA_ANNEALING_START_EPOCH=50       # When to begin annealing

# Backward compatibility (default)
TS_EMA_SCHEDULE="fixed"               # Use original TS_EMA_DECAY value
TS_EMA_DECAY=0.999                    # Works exactly as before
```

**Training Phases**:
1. **Pre-Annealing** (epochs 0-49): Uses `TS_EMA_DECAY_INITIAL` (0.999)
2. **Annealing Phase** (epochs 50-end): Smooth transition using selected schedule
3. **Convergence** (final epochs): Approaches `TS_EMA_DECAY_FINAL` (0.1)

**Benefits**:
- 🎯 **Curriculum Learning**: Progressive teacher-to-student knowledge transfer
- 🔥 **Enhanced Convergence**: Better final model performance
- 📈 **Optimal Training**: Teacher provides stability early, adapts quickly later
- 🛡️ **Training Stability**: Maintains teacher quality throughout training
- 🔄 **Flexible Schedules**: Choose optimal annealing strategy for your data

**Example Progress**:
```
Epoch 10:  EMA Decay: 0.9990 (cosine schedule, pre-annealing)
Epoch 50:  EMA Decay: 0.9990 (cosine schedule, 0.0% progress)
Epoch 75:  EMA Decay: 0.5495 (cosine schedule, 50.0% progress)
Epoch 100: EMA Decay: 0.1000 (cosine schedule, 100.0% progress)
```

**Recommended Settings**:
- **New Research**: `TS_EMA_SCHEDULE="cosine"` with default parameters
- **Reproduction**: `TS_EMA_SCHEDULE="fixed"` with `TS_EMA_DECAY=0.999`
- **Fast Adaptation**: `TS_EMA_SCHEDULE="exponential"` for quicker convergence
- **Smooth Transition**: `TS_EMA_SCHEDULE="linear"` for gradual decay

#### 📅 Two-Phase Training
1. **Warm-up Phase**: Student-only training until teacher initialization
   - Only student network is trained
   - Teacher initialized at epoch 20 (configurable via `TS_TEACHER_INIT_EPOCH`)
   - Alternative: validation loss threshold (configurable via `TS_TEACHER_INIT_VAL_LOSS`)

2. **Teacher-Student Phase**: Dual loss training
   - Supervised loss: Student predictions vs ground truth
   - Consistency loss: Student predictions vs teacher pseudo-labels
   - Alpha scheduling balances supervised and consistency losses

#### 🎯 Cosine Decay Alpha Scheduling
```python
alpha = max(TS_MIN_ALPHA, 0.5 * (1 + cos(π * epoch / total_epochs)))
total_loss = alpha * supervised_loss + (1 - alpha) * consistency_loss
```
- Starts at 1.0 (supervised only) → decays to `TS_MIN_ALPHA` (default: 0.01)
- Gradually shifts focus from supervised to consistency learning

#### 🔀 Multi-Type Consistency Loss
**Segmentation Consistency** (always enabled):
- **MSE**: Pixel-wise mean squared error (default)
- **KL Divergence**: Knowledge distillation with temperature scaling
- **L1**: L1 distance between predictions
- **Dice**: Regional overlap consistency
- **IoU**: Intersection-over-union consistency

**Classification Consistency** (optional):
- **Patch Classification**: Multi-label consistency (always enabled)
- **Gland Classification**: Individual gland consistency (`TS_ENABLE_GLAND_CONSISTENCY=true`)

### 🔥 Key Teacher-Student Improvements

#### ✅ True Independence
- **Separate Classification Heads**: Student and teacher have completely independent classification parameters
- **Independent Checkpoints**: No shared weights between student and teacher models
- **Independent Evaluation**: Each network evaluated separately with its own metrics

#### ✅ Flexible Evaluation
- **Checkpoint Selection**: Choose between `latest` or `best` model checkpoints
- **Evaluation Modes**: Evaluate `student`, `teacher`, or `both` networks
- **Comprehensive Comparison**: Side-by-side performance analysis when evaluating both

#### ✅ Enhanced Visualizations
- **7-Column Layout**: Original → GT Mask → GT Overlay → Teacher Pseudo-Mask → Teacher Overlay → Student Mask → Student Overlay
- **7 Rows Per Figure**: Increased from 5 to 7 image rows for comprehensive comparison
- **Equal Figure Dimensions**: Square figures (28×28 for Teacher-Student, 16×16 for standard) with optimized layout
- **Minimal White Space**: Maximized image sizes with reduced margins for better space utilization
- **Prediction Probabilities**: All columns display prediction probabilities for all 4 classes
- **Classification Display**: Both segmentation and classification predictions shown in titles
- **Visual Comparison**: Complete comparison between student and teacher predictions with overlays
- **🆕 Teacher Pseudo-Mask Filtering**: Apply confidence/entropy filtering to remove noisy teacher predictions
- **🆕 Original Image Enhancement**: Shows true unaugmented histology images with proper ImageNet denormalization
- **🆕 Adaptive Filtering Visualization**: Filtered teacher pseudo-masks shown in real-time during evaluation

### 🏗️ Teacher-Student Architecture Diagram

```
                    📊 INPUT BATCH (Images + Labels)
                                    │
                                    ▼
            ┌─────────────────────────────────────────────────────┐
            │                 🎯 MULTITASK WRAPPER                │
            └─────────────────────────────────────────────────────┘
                                    │
                        ┌───────────┴───────────┐
                        ▼                       ▼
            ┌─────────────────────────┐    ┌─────────────────────────┐
            │    🎓 STUDENT NETWORK   │    │   👨‍🏫 TEACHER NETWORK    │
            │                         │    │                         │
            │  ┌─────────────────┐    │    │  ┌─────────────────┐    │
            │  │  UNet Encoder   │    │    │  │  UNet Encoder   │    │
            │  │     (Down)      │    │    │  │     (Down)      │    │
            │  └─────────────────┘    │    │  └─────────────────┘    │
            │           │             │    │           │             │
            │  ┌─────────────────┐    │    │  ┌─────────────────┐    │
            │  │  UNet Decoder   │    │    │  │  UNet Decoder   │    │
            │  │      (Up)       │    │    │  │      (Up)       │    │
            │  └─────────────────┘    │    │  └─────────────────┘    │
            │           │             │    │           │             │
            │  ┌─────────────────┐    │    │  ┌─────────────────┐    │
            │  │ Segmentation    │    │    │  │ Segmentation    │    │
            │  │   Head (4cls)   │    │    │  │   Head (4cls)   │    │
            │  └─────────────────┘    │    │  └─────────────────┘    │
            └─────────────────────────┘    └─────────────────────────┘
                        │                              │
                        ▼                              ▼
            ┌─────────────────────────┐    ┌─────────────────────────┐
            │ 🎯 STUDENT CLASSIF HEAD │    │ 🎯 TEACHER CLASSIF HEAD │
            │                         │    │                         │
            │ ┌─────────────────────┐ │    │ ┌─────────────────────┐ │
            │ │  Patch Classifier   │ │    │ │  Patch Classifier   │ │
            │ │     (4 classes)     │ │    │ │     (4 classes)     │ │
            │ └─────────────────────┘ │    │ └─────────────────────┘ │
            │ ┌─────────────────────┐ │    │ ┌─────────────────────┐ │
            │ │  Gland Classifier   │ │    │ │  Gland Classifier   │ │
            │ │     (4 classes)     │ │    │ │     (4 classes)     │ │
            │ └─────────────────────┘ │    │ └─────────────────────┘ │
            └─────────────────────────┘    └─────────────────────────┘
                        │                              │
                        ▼                              ▼
            ┌─────────────────────────┐    ┌─────────────────────────┐
            │    📤 STUDENT OUTPUT    │    │   📤 TEACHER OUTPUT     │
            │                         │    │                         │
            │ • Segmentation Logits   │    │ • Segmentation Logits   │
            │ • Patch Classification  │    │ • Patch Classification  │
            │ • Gland Classification  │    │ • Gland Classification  │
            └─────────────────────────┘    └─────────────────────────┘
                        │                              │
                        └──────────────┬───────────────┘
                                       ▼
                        ┌─────────────────────────────────┐
                        │        🔄 EMA UPDATE            │
                        │                                 │
                        │  Teacher Weights = α × Teacher │
                        │                  + (1-α) × Student │
                        │                                 │
                        │  α = TS_EMA_DECAY (0.999)     │
                        │  Update Frequency: Every Epoch │
                        └─────────────────────────────────┘
                                       │
                                       ▼
                        ┌─────────────────────────────────┐
                        │       📊 LOSS COMPUTATION       │
                        │                                 │
                        │ 🎯 Supervised Loss (GT Labels) │
                        │    • Student vs Ground Truth   │
                        │                                 │
                        │ 🔄 Consistency Loss             │
                        │    • Student vs Teacher        │
                        │    • Segmentation + Classification │
                        │                                 │
                        │ 📐 Total Loss = β × Supervised  │
                        │               + (1-β) × Consistency │
                        │                                 │
                        │ β = Dynamic (starts 1.0 → TS_MIN_ALPHA) │
                        └─────────────────────────────────┘
                                       │
                                       ▼
                        ┌─────────────────────────────────┐
                        │      💾 INDEPENDENT SAVES       │
                        │                                 │
                        │ 🎓 student_model.pth           │
                        │    • Student UNet              │
                        │    • Student Classification    │
                        │                                 │
                        │ 👨‍🏫 teacher_model.pth           │
                        │    • Teacher UNet              │
                        │    • Teacher Classification    │
                        │                                 │
                        │ (No shared weights saved)      │
                        └─────────────────────────────────┘
```

**🔑 Key Architecture Features:**
- **🎓 Student Network**: Learns from ground truth + teacher consistency
- **👨‍🏫 Teacher Network**: Updated via EMA, provides stable pseudo-labels
- **🔄 EMA Updates**: Teacher weights = α × Teacher + (1-α) × Student
- **🎯 Independent Heads**: Separate classification parameters for each network
- **💾 Separate Checkpoints**: No shared weights between models
- **📊 Dual Loss**: Supervised (student vs GT) + Consistency (student vs teacher)

### Training Configuration

#### Basic Teacher-Student Setup
```bash
# Edit run_nnunet_training.sh
ARCHITECTURE="teacher_student_unet"
DATASET_KEY="mag20x"
EPOCHS=1000
BATCH_SIZE=8  # Lower due to dual networks

# Teacher-Student parameters
TS_EMA_DECAY=0.999
TS_TEACHER_INIT_EPOCH=20
TS_MIN_ALPHA=0.01
TS_MAX_ALPHA=1.0
TS_CONSISTENCY_LOSS_TYPE="mse"
TS_CONSISTENCY_TEMPERATURE=1.0

# Evaluation configuration
TEACHER_STUDENT_EVALUATOR="latest"   # or "best"
TS_POST_EVAL_MODE="both"              # Evaluate both networks
```

#### 🎯 Teacher Pseudo-Mask Filtering (Noise Reduction)

Reduce training noise by filtering out uncertain teacher predictions:

```bash
# Enable confidence-based filtering
TS_PSEUDO_MASK_FILTERING="confidence"
TS_CONFIDENCE_THRESHOLD=0.8           # Keep pixels with >80% confidence
TS_FILTERING_WARMUP_EPOCHS=10         # Start filtering after warmup

# Enable entropy-based filtering (alternative)
TS_PSEUDO_MASK_FILTERING="entropy"
TS_ENTROPY_THRESHOLD=1.0              # Keep low-entropy (high-confidence) pixels

# Disable filtering (default)
TS_PSEUDO_MASK_FILTERING="none"
```

**Filtering Strategies**:
- **Confidence-Based**: `confidence = max(softmax(teacher_logits))` → Keep if `confidence > threshold`
- **Entropy-Based**: `entropy = -∑(p * log(p))` → Keep if `entropy < threshold` (low entropy = high confidence)
- **Effect**: Only high-confidence teacher predictions contribute to student training

**Benefits**:
- ✅ Reduces noise in pseudo-labels
- ✅ Improves training stability
- ✅ Better final performance
- ✅ Visual feedback in 7-column visualizations

#### 🎯 Adaptive Confidence Threshold Annealing (Advanced Curriculum Learning)

Implement sophisticated curriculum learning by dynamically adjusting confidence thresholds during training:

```bash
# Enable adaptive confidence threshold annealing
TS_CONFIDENCE_ANNEALING="cosine"          # "none", "linear", "cosine"
TS_CONFIDENCE_MAX_THRESHOLD=0.9           # Starting threshold (early training)
TS_CONFIDENCE_MIN_THRESHOLD=0.6           # Ending threshold (late training)
TS_CONFIDENCE_ANNEALING_START_EPOCH=5     # When to start annealing

# Alternative schedules
TS_CONFIDENCE_ANNEALING="linear"          # Linear decay
TS_CONFIDENCE_ANNEALING="none"            # Static threshold (default)
```

**Annealing Schedules**:

| Schedule | Early Training (Epoch 0-5) | Mid Training (Epoch 25) | Late Training (Epoch 100) |
|----------|----------------------------|--------------------------|----------------------------|
| **Cosine** | 0.9 (very selective) | 0.868 (smooth transition) | 0.6 (permissive) |
| **Linear** | 0.9 (very selective) | 0.837 (gradual decline) | 0.6 (permissive) |
| **None** | 0.7 (static) | 0.7 (static) | 0.7 (static) |

**Training Strategy**:
1. **Early Training**: High threshold (0.9) → Only very confident teacher predictions used
2. **Late Training**: Low threshold (0.6) → More pseudo-labels available when teacher is stable
3. **Smooth Transition**: Cosine annealing provides optimal curriculum learning progression

**Benefits**:
- 🎯 **Curriculum Learning**: Progressive difficulty increase
- 🔥 **Better Convergence**: Optimal teacher-student knowledge transfer
- 📈 **Enhanced Performance**: Superior final model quality
- 🛡️ **Training Stability**: Reduced early-training noise

#### Advanced Consistency Loss Configuration
```bash
# Enable additional consistency losses
TS_ENABLE_GLAND_CONSISTENCY=true    # Gland classification consistency
TS_ENABLE_DICE_CONSISTENCY=true     # Dice overlap consistency
TS_ENABLE_IOU_CONSISTENCY=true      # IoU boundary consistency

# Alternative consistency loss types
TS_CONSISTENCY_LOSS_TYPE="kl_div"   # Knowledge distillation
TS_CONSISTENCY_LOSS_TYPE="dice"     # Regional consistency
TS_CONSISTENCY_LOSS_TYPE="iou"      # Boundary consistency
```

### Independent Teacher-Student Evaluation

#### Separate Model Checkpoints
Teacher-Student architecture saves **completely independent** model checkpoints:
- `latest_student_model.pth` - Student segmentation + student classification head
- `latest_teacher_model.pth` - Teacher segmentation + teacher classification head
- `best_student_model.pth` - Best student model based on validation metrics
- `best_teacher_model.pth` - Best teacher model based on validation metrics

#### Evaluation Configuration
```bash
# Choose checkpoint type
TEACHER_STUDENT_EVALUATOR="latest"   # or "best"

# Choose evaluation mode
TS_POST_EVAL_MODE="both"      # Evaluate both networks independently
TS_POST_EVAL_MODE="student"   # Evaluate student only (default)
TS_POST_EVAL_MODE="teacher"   # Evaluate teacher only
```

#### Dual Model Evaluation (`TS_POST_EVAL_MODE="both"`)
When set to "both", the system generates:
- **Independent metrics** for student and teacher on train/val/test sets
- **Separate evaluation reports** (`evaluation_student/` and `evaluation_teacher/`)
- **Performance comparison** showing which network performs better
- **Separate visualizations** for each network's predictions

#### 🎨 7-Column Enhanced Visualizations
Teacher-Student models generate comprehensive visualizations showing:
1. **Original Image** + Ground truth classes and patch classification (unaugmented histology image)
2. **Ground Truth Segmentation Mask** (colored segmentation)
3. **Ground Truth Overlay** + Ground truth classes overlaid on original image
4. **Teacher Pseudo-Mask** + Teacher prediction probabilities with filtering info (e.g., "Teacher Pseudo-Mask (Conf>0.8)" + "Bac: 0.95 | Ben: 0.78 | Mal: 0.12 | PDC: 0.03")
5. **Teacher Overlay** + Filtered teacher predictions overlaid on original image
6. **Student Predicted Mask** + Student prediction classes (colored segmentation)
7. **Student Overlay** + Student prediction probabilities and patch classification overlaid on original image

**Key Features:**
- **🆕 7-Column Layout**: Complete progression from original image through all prediction stages
- **🆕 Original Image Quality**: Unaugmented histology images instead of normalized versions
- **🆕 Filtering Visualization**: Teacher pseudo-masks show exactly what student sees during training
- **🆕 Student Mask Column**: Dedicated column for raw student predictions
- **Prediction probabilities**: All overlay columns show detailed class probabilities for all 4 classes
- **Independent classification heads**: Student and teacher have separate classification parameters
- **Complete evaluation**: Both segmentation and classification metrics for each network
- **Visual comparison**: See how student vs teacher predictions differ across all stages
- **Filtering feedback**: Visual indication of confidence/entropy filtering effects

### Training Progress Monitoring

#### Progress Bar Metrics
- **Loss**: Combined supervised + consistency loss
- **Dice**: Segmentation Dice coefficient
- **IoU**: Intersection over Union
- **PatchAcc**: Multi-label patch accuracy
- **🆕 Pseudo-Dice**: Student vs Pseudo-GT Dice coefficient (monitoring only)
- **🆕 Pseudo-IoU**: Student vs Pseudo-GT IoU metric (monitoring only)
- **Teacher_Consist_Loss**: Consistency loss value
- **Supervision_Weightage**: Current alpha value (1.0 → 0.01)
- **LR**: Learning rate

#### 🆕 Student vs Pseudo-GT Metrics
Real-time monitoring of student-teacher alignment during training:

**Purpose**: Track how well student predictions align with teacher pseudo-GT masks to assess consistency learning effectiveness.

**Key Features**:
- **Zero Impact on Optimization**: Metrics computed with `torch.no_grad()` and detached tensors
- **Same Pseudo-Mask**: Uses exact teacher pseudo-mask from consistency loss computation
- **Teacher-Student Phase Only**: Only active during teacher-student training (not warmup)
- **Progress Display**: Shows Pseudo-Dice and Pseudo-IoU in real-time during training
- **Epoch Logging**: Included in final epoch summaries alongside regular metrics

**Monitoring Display**:
```bash
Epoch  50/1000 [Teacher-Student] Loss: 0.2847 Alpha: 0.672 Pseudo-Dice: 0.823 Pseudo-IoU: 0.701
```

**Benefits**:
- ✅ **Alignment Tracking**: Monitor student-teacher consistency quality
- ✅ **Training Insights**: Understand when student starts following teacher effectively
- ✅ **Zero Overhead**: No computational impact on actual training process
- ✅ **Research Value**: Quantify effectiveness of consistency learning

#### TensorBoard Logging
- Individual loss components (segmentation, patch, gland)
- Consistency loss breakdown (segmentation, patch, gland)
- Alpha scheduling curve
- Teacher vs student metrics comparison
- **🆕 Pseudo-GT metrics tracking** (Pseudo-Dice and Pseudo-IoU trends)
- **🆕 Student-teacher alignment curves** (real-time consistency monitoring)

### Evaluation Reports

#### Comprehensive Metrics
- **Standard metrics**: Dice, IoU, Pixel Accuracy, Patch Accuracy
- **Consistency losses**: Detailed breakdown by component type
- **Mean calculations**: Averaged across Train/Val/Test splits
- **Teacher-Student comparison**: Side-by-side performance analysis

#### Consistency Loss Components Table
```
| Network | Split | Seg Consistency Loss | Patch Consistency Loss | Gland Consistency Loss |
|---------|-------|---------------------|------------------------|------------------------|
| Student | Train | 0.0245             | 0.0123                | 0.0089                |
| Student | Val   | 0.0267             | 0.0134                | 0.0098                |
| Student | Test  | 0.0251             | 0.0129                | 0.0092                |
| Student | Mean  | 0.0254             | 0.0129                | 0.0093                |
```

### 🆕 Enhanced Training Visualization with Pseudo-GT Monitoring

Teacher-Student training now generates comprehensive visualization plots with dedicated sub-figures for pseudo-GT metrics analysis.

#### Comprehensive 3×3 Training Curves Grid
The enhanced visualization includes **8 specialized subplots** in a large-format (30×18 inch) layout:

**Row 1: Core Training Metrics**
1. **Total Loss** (Training vs Validation) - Overall training progression
2. **Loss Components** (Supervised vs Consistency) - Individual loss breakdown
3. **Dice Score** (Student vs GT, Training & Validation) - Standard segmentation performance

**Row 2: Pseudo-GT Analysis (NEW!)**
4. **Alpha Schedule** (Consistency Weight) - Shows supervised-to-consistency transition
5. **🆕 Pseudo-Dice Score** (Student vs Pseudo-GT) - **Goldenrod dashed line** showing student alignment with teacher pseudo-masks, compared with regular Dice performance (forest green)
6. **🆕 Pseudo-IoU Score** (Student vs Pseudo-GT) - **Orange dashed line** tracking IoU alignment quality

**Row 3: Advanced Monitoring**
7. **Training Phase** (Warmup vs Teacher-Student) - Visual phase indication
8. **Pseudo-GT Alignment Analysis** - **|Pseudo-Dice - GT-Dice|** difference tracking (crimson line)

#### Key Visualization Features

**🎯 Pseudo-GT Sub-Figures** (Subplots 5 & 6):
- **Direct Comparison**: Student vs Pseudo-GT metrics overlaid with Student vs GT metrics for context
- **Phase Highlighting**: Shaded regions showing teacher-student phase activation
- **Alignment Tracking**: Real-time monitoring of how well student follows teacher guidance
- **Research Insights**: Visual assessment of consistency learning effectiveness

**📊 Professional Styling**:
- **High Resolution**: 300 DPI PNG + vector PDF output
- **Color Coding**: Goldenrod for Pseudo-Dice, Orange for Pseudo-IoU with distinctive dashed lines
- **Grid Background**: Easy-to-read plotting with consistent formatting
- **Comprehensive Legends**: Clear labeling for all metrics and phases

#### Automatic Generation
The enhanced visualization is **automatically generated** at the end of Teacher-Student training:

```bash
📊 Enhanced Teacher-Student training curves with Pseudo-GT metrics saved:
   📈 PNG: /path/to/outputs/teacher_student_training_curves_with_pseudo_gt.png
   📈 PDF: /path/to/outputs/teacher_student_training_curves_with_pseudo_gt.pdf
```

#### Research Applications
- **Consistency Analysis**: Visual assessment of student-teacher alignment quality
- **Training Optimization**: Identify optimal teacher initialization and alpha scheduling
- **Method Comparison**: Compare different consistency loss types and EMA schedules
- **Publication Quality**: High-resolution figures suitable for research papers
- **Training Monitoring**: Real-time feedback on pseudo-GT learning effectiveness

### Best Practices

#### Memory Management
- Use smaller batch sizes (8 vs 16) due to dual networks
- Monitor GPU memory usage during training
- Consider gradient accumulation for effective larger batch sizes

#### Hyperparameter Tuning
- **EMA Decay**: 0.999 (stable) vs 0.99 (faster adaptation)
- **Teacher Init**: Earlier (epoch 10) vs later (epoch 30)
- **Min Alpha**: 0.01 (strong consistency) vs 0.1 (balanced)
- **Consistency Type**: MSE (stable) vs Dice (semantic) vs KL (knowledge transfer)

#### Training Duration
- Longer training beneficial due to gradual alpha decay
- Patience 250-500 epochs recommended
- Monitor both supervised and consistency loss convergence

### Research Applications
- **Semi-supervised learning**: Leverage unlabeled data with teacher predictions
- **Consistency regularization**: Improve model robustness
- **Knowledge distillation**: Transfer knowledge without ensemble overhead
- **Model uncertainty**: Compare teacher-student agreement
- **Ablation studies**: Effect of different consistency loss types

## 🛠️ Troubleshooting

### SLURM Training Issues

**Job Fails to Start**
```bash
# Check job status
squeue -u $USER

# Check job details
scontrol show job <JOB_ID>

# Verify script paths and permissions
ls -la run_nnunet_training.sh
```

**Early Stopping with Patience=250**
- **Issue**: Training stops at 30 epochs despite setting `EARLY_STOP_PATIENCE=250`
- **Solution**: The patience parameter is now correctly passed via `--patience` argument
- **Check**: Look for `Early stopping patience: 250` in training logs

**Training Logs Location**
```bash
# Logs are automatically placed in:
/fs/scratch/PAS2942/Datasets/CRC/Datasets/OSU/Gland_Datasets/ALL_RUNS/logs/

# Monitor training progress
tail -f /path/to/logs/training_<JOB_ID>.out
tail -f /path/to/logs/training_<JOB_ID>.err
```

**Configuration Not Applied**
```bash
# Verify environment variables are exported
grep -n "export GLAND_" run_nnunet_training.sh

# Check parameter display in training output
grep "Training Configuration:" /path/to/logs/training_<JOB_ID>.out
```

### Common Issues

**CUDA Out of Memory**
```bash
# Edit run_nnunet_training.sh
BATCH_SIZE=8  # or smaller

# Or for manual training
python main.py train --batch_size 2 --dataset mixed --output_dir ./outputs
```

**Dataset Not Found**
```bash
# First export environment variables, then check datasets
export GLAND_DATASET_BASE="/path/to/your/datasets"
python -c "from configs.paths_config import list_available_datasets; print(list_available_datasets())"
```

**Import Errors**
```bash
python main.py demo  # Test all components
```

**CosineAnnealingLR Configuration**
```bash
# Verify scheduler settings in training output
grep "Scheduler:" /path/to/logs/training_<JOB_ID>.out
grep "T_max\|eta_min" /path/to/logs/training_<JOB_ID>.out
```

## 📊 Evaluation Metrics

### Segmentation
- Per-class Dice coefficients
- Mean Dice across all classes
- Pixel-wise accuracy
- Intersection over Union (IoU)

### Multi-Label Classification
- Exact match accuracy
- Per-class binary accuracy
- Hamming loss
- F1 scores per class

### Individual Gland Classification
- 4-class accuracy
- Per-class precision/recall
- Classification reports

## 🎯 Research Applications

- **Baseline Research**: Establish fair performance baselines
- **Architecture Comparison**: Direct performance comparison
- **Digital Pathology**: Advanced glandular analysis
- **Clinical Decision Support**: Multi-class gland assessment
- **Method Development**: Foundation for novel approaches

## 🔬 Advanced Features

- **Adaptive Loss Weighting**: Automatic multi-task balancing
- **Deep Supervision**: Multiple resolution outputs (nnU-Net)
- **Multi-Magnification Learning**: Scale-invariant models
- **Comprehensive Augmentation**: Histopathology-specific transforms
- **Rich Visualizations**: Composite images with predictions
- **Configurable Early Stopping**: Extended patience for long training runs
- **Advanced LR Scheduling**: CosineAnnealingLR with configurable parameters
- **SLURM Integration**: Complete HPC training pipeline

## 🆕 Recent Improvements

### Fixed Issues
- **✅ Dataset Path Loading**: Fixed critical bug where training used hardcoded paths instead of user-specified paths from SLURM script
- **✅ Environment Variable Timing**: Fixed import-time vs runtime environment variable reading
- **✅ Early Stopping Patience**: Fixed bug where `--patience` parameter wasn't being passed correctly
- **✅ Parameter Passing**: All script parameters now properly reach the Python trainer
- **✅ CosineAnnealingLR**: Added configurable `T_max` and `eta_min` parameters

### New Features
- **🆕 Teacher-Student UNet**: Complete semi-supervised learning architecture with EMA updates
- **🆕 EMA Decay Annealing**: Dynamic EMA decay scheduling (cosine, linear, exponential) for progressive teacher-student learning with backward compatibility
- **🆕 Multi-Type Consistency Loss**: MSE, KL-divergence, L1, Dice, and IoU consistency options
- **🆕 7-Column Visualizations**: Enhanced Teacher-Student visualizations with pseudo-mask display, ground truth overlay, teacher overlay, student mask, and student overlay with prediction probabilities
- **🆕 Dual Model Evaluation**: Separate teacher and student evaluation with comparison reports
- **🆕 Consistency Loss Tracking**: Detailed breakdown and mean calculations across splits
- **🆕 Alpha Scheduling**: Cosine decay from supervised to consistency learning
- **🆕 EMA Progress Monitoring**: Real-time logging of EMA decay values and annealing progress during training
- **🆕 Required Environment Variables**: All hardcoded paths removed, now requires proper environment setup
- **🆕 Environment Validation**: Added validation functions to ensure all required variables are set
- **🆕 SLURM Training Script**: Complete automated training pipeline for HPC
- **🆕 Extended Early Stopping**: Support for patience values up to 500+ epochs
- **🆕 Learning Rate Scheduler Config**: Configurable CosineAnnealingLR parameters
- **🆕 Enhanced Logging**: Comprehensive training configuration display
- **🆕 Teacher Pseudo-Mask Filtering**: Confidence-based and entropy-based filtering to reduce noise in teacher predictions
- **🆕 Adaptive Confidence Threshold Annealing**: Advanced curriculum learning with cosine, linear, and static schedules for dynamic confidence thresholding
- **🆕 Original Image Visualization**: Shows unaugmented histology images with proper ImageNet denormalization instead of normalized versions
- **🆕 Enhanced Figure Layout**: 7 rows per figure (increased from 5) with equal dimensions and minimal white space for better space utilization
- **🆕 Advanced Consistency Loss**: Apply selective filtering to consistency loss while preserving ground truth supervision
- **🆕 Student vs Pseudo-GT Metrics**: Real-time monitoring of student-teacher alignment with Pseudo-Dice and Pseudo-IoU metrics during training
- **🆕 Enhanced Training Visualization**: Comprehensive 3x3 grid training curves with dedicated sub-figures for pseudo-GT metrics comparison
- **🆕 Pseudo-GT Alignment Analysis**: Visual tracking of student vs pseudo-GT alignment differences throughout training
- **🆕 Automatic Testing**: Built-in demo run before training starts
- **🆕 Automatic Seed Generation**: Each training run automatically generates a unique master seed controlling all randomness (Python, NumPy, PyTorch, CUDA) with easy reproduction via MASTER_SEED environment variable
- **🆕 Simplified Reproducibility**: Removed manual deterministic settings in favor of unified master seed approach with optimal PyTorch performance defaults

### Performance Improvements
- **⚡ Path Consistency**: Training and evaluation now guaranteed to use same dataset paths
- **⚡ Better Defaults**: Optimized default parameters for long training runs
- **⚡ Memory Efficiency**: Improved batch size handling for different GPU sizes
- **⚡ Log Organization**: Centralized logging with automatic cleanup

### Breaking Changes
- **⚠️ Environment Variables Required**: Manual training now requires setting environment variables
- **⚠️ No Hardcoded Fallbacks**: All paths must be explicitly provided via environment variables

## 📚 Citation

```bibtex
@software{multi_architecture_gland_segmentation,
  title={Multi-Architecture 4-Class Gland Segmentation Framework},
  author={Claude Code - OSU CRC Research},
  year={2025},
  note={Baseline UNet vs nnU-Net comparison for histopathology analysis}
}
```

---

**Ready for comprehensive 4-class gland segmentation with architecture choice!** 🎉

For detailed documentation, see the source code and configuration files in the `src/` and `configs/` directories.