#!/bin/bash
#SBATCH --account=PAS3194
#SBATCH --job-name=_Gls3RD
#SBATCH --time=164:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-gpu=32
#SBATCH --mem=100gb
#SBATCH --output=/fs/scratch/PAS2942/Datasets/CRC/Datasets/OSU/Gland_Datasets/DeleteIT_Resuming_Testing/logs/training_%A.out
#SBATCH --error=/fs/scratch/PAS2942/Datasets/CRC/Datasets/OSU/Gland_Datasets/DeleteIT_Resuming_Testing/logs/training_%A.err
#SBATCH --cluster=ascend

# =====================================================
# Multi-Architecture Gland Segmentation Training
# Supports both Baseline UNet and nnU-Net architectures
# =====================================================
# PAS3194
# --------- Environment ----------
module load miniconda3/24.1.2-py310
module load cuda/11.8.0
conda activate trident

# --------- Environment Info ----------
echo "============================================"
echo "Multi-Architecture Gland Segmentation Training"
echo "============================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURMD_NODENAME"
echo "Start Time: $(date)"
echo "============================================"

# Check GPU availability
nvidia-smi
echo "============================================"

# Check Python environment
python --version
which python
echo "CUDA Version:"
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA Available: {torch.cuda.is_available()}, CUDA Version: {torch.version.cuda}')"
export CUDA_LAUNCH_BLOCKING=1
echo "🐛 CUDA_LAUNCH_BLOCKING=1 enabled for debugging"
echo "============================================"

# --------- Directory Setup ----------
cd /users/PAS2942/hikmat179/Code/_MLCRC/GlandSegmentation/GlandSegModels/nnUNet

# Create logs directory if it doesn't exist
mkdir -p logs

# =====================================================
# 🎯 QUICK TRAINING CONFIGURATION
# =====================================================
#
# 🔥 STEP 1: Choose Architecture
# ==============================
# ARCHITECTURE="baseline_unet"      # ← For baseline comparison (simple, fast, interpretable)
# ARCHITECTURE="nnunet"             # ← For state-of-the-art performance (advanced, slower)
ARCHITECTURE="teacher_student_unet" # ← For self-training with Teacher-Student learning

# 🔥 STEP 2: Choose Dataset (UNCOMMENT ONLY ONE LINE)
# =====================================================
# DATASET_KEY="mixed"    # ← All magnifications combined (~25k samples) - RECOMMENDED
# DATASET_KEY="mag5x"    # ← 5x magnification only (~4.8k samples)
# DATASET_KEY="mag10x"   # ← 10x magnification only (~6.2k samples)(Need to Run Below)
# DATASET_KEY="mag20x"   # ← 20x magnification only (~6k samples)
# DATASET_KEY="mag40x"   # ← 40x magnification only (~6k samples)
DATASET_KEY="warwick"  # ← Warwick GlaS Teacher-Student dataset (~165 samples, 3-class: Background, Benign, Malignant)

# 📂 DATASET PATHS - MODIFY THESE TO POINT TO YOUR DATASETS
# ==========================================================
# Base directory where all combined datasets are located
# DATASET_BASE_DIR="/fs/scratch/PAS2942/Datasets/CRC/Datasets/OSU/Gland_Datasets/nnUNetCombined"
# DATASET_BASE_DIR="/fs/scratch/PAS2942/Datasets/CRC/Datasets/OSU/Gland_Datasets/_OSU_All_Gland_Datasets_nnUNet"
# DATASET_BASE_DIR="/fs/scratch/PAS2942/Datasets/CRC/Datasets/OSU/Gland_Datasets/_OSU_All_Gland_Datasets_nnUNet_Syn_Mag"

# DATASET_BASE_DIR="/fs/scratch/PAS2942/Datasets/CRC/Datasets/OSU/Gland_Datasets/_OSU_All_4_Gland_Datasets_nnUNet_Mag_natural"
# For Warwick GlaS Teacher-Student dataset, use:
# DATASET_BASE_DIR="/fs/scratch/PAS2942/Datasets/CRC/Datasets/OSU/Gland_Datasets/_Warwick_QU_Dataset_TeacherStudent_Preprocessed"
DATASET_BASE_DIR="/fs/scratch/PAS2942/Datasets/CRC/Datasets/OSU/Gland_Datasets/_Warwick_QU_Dataset_TeacherStudent_Preprocessed_WTS"
#
# Available dataset paths (automatically constructed from base + key):
# OSU Combined Datasets (4-class: Background, Benign, Malignant, PDC):
# → mixed:   ${DATASET_BASE_DIR}/Task001_Combined_Mixed_Magnifications
# → mag5x:   ${DATASET_BASE_DIR}/Task005_Combined_Mag5x
# → mag10x:  ${DATASET_BASE_DIR}/Task010_Combined_Mag10x
# → mag20x:  ${DATASET_BASE_DIR}/Task020_Combined_Mag20x
# → mag40x:  ${DATASET_BASE_DIR}/Task040_Combined_Mag40x
# Warwick GlaS Dataset (3-class: Background, Benign, Malignant):
# → warwick: ${DATASET_BASE_DIR}/nnUNet_raw/Task002_WarwickGlaSTeacherStudent

# 🔥 STEP 3: Training Parameters
# ==============================
EPOCHS=50               # Quick test: 2-5 | Research quality: 150-200
BATCH_SIZE=16          # Small GPU: 2-4 | Medium GPU: 8-16 | Large GPU: 16-32
LEARNING_RATE="1e-4"   # Default works well for most cases

# 🎓 TEACHER-STUDENT SPECIFIC PARAMETERS
# =======================================
# (Only used when ARCHITECTURE="teacher_student_unet")

TS_BACKBONE_TYPE="nnunet"  # Backbone architecture: "baseline_unet", "nnunet" (default: baseline_unet)

# 🔄 EMA DECAY CONFIGURATION (Choose ONE approach)
# ================================================
# APPROACH 1: Fixed EMA Decay (Backward Compatible)(Teacher to Student Convergence Control)
# -------------------------------------------------
TS_EMA_DECAY=0.999           # ⚠️  ONLY used when TS_EMA_SCHEDULE="fixed"
                             # Constant EMA decay throughout entire training
                             # Original Teacher-Student implementation
                             # Range: 0.99-0.999, default: 0.999

# APPROACH 2: Dynamic EMA Annealing (Advanced) (Teacher to Student Convergence Control)
# Dynamic EMA Annealing controls how the Teacher network learns from the Student network 
# over time through adaptive convergence rates.
# 🎯 Core Concept: Teacher-Student Convergence Control

#   EMA (Exponential Moving Average) Formula:
#   teacher_weights = EMA_DECAY × teacher_weights + (1 - EMA_DECAY) × student_weights

#   EMA Decay Values Mean:
#   - High Decay (0.999): Teacher changes very slowly, maintains stability
#   - Low Decay (0.01): Teacher adapts quickly to student changes
# --------------------------------------------
TS_EMA_SCHEDULE="fixed"              # EMA schedule: "fixed", "cosine", "linear", "exponential" 
                                     # ⚠️  When "fixed": uses TS_EMA_DECAY above
                                     # ⚠️  When annealing: uses TS_EMA_DECAY_INITIAL/FINAL below

TS_EMA_DECAY_INITIAL=0.999           # ⚠️  ONLY used when TS_EMA_SCHEDULE ≠ "fixed"
                                     # Starting EMA decay for annealing schedules
                                     # Range: 0.99-0.999, default: 0.999

TS_EMA_DECAY_FINAL=0.01               # ⚠️  ONLY used when TS_EMA_SCHEDULE ≠ "fixed"
                                     # Final EMA decay for annealing schedules
                                     # Range: 0.05-0.5, default: 0.1

TS_EMA_ANNEALING_START_EPOCH=10      # ⚠️  ONLY used when TS_EMA_SCHEDULE ≠ "fixed"
                                     # Epoch to start EMA annealing
                                     # Range: 10-100, default: 50

# 📝 EMA PARAMETER USAGE SUMMARY:
# ===============================
# IF TS_EMA_SCHEDULE="fixed":
#    → Uses: TS_EMA_DECAY (constant throughout training)
#    → Ignores: TS_EMA_DECAY_INITIAL, TS_EMA_DECAY_FINAL, TS_EMA_ANNEALING_START_EPOCH
#
# IF TS_EMA_SCHEDULE="cosine"|"linear"|"exponential":
#    → Uses: TS_EMA_DECAY_INITIAL, TS_EMA_DECAY_FINAL, TS_EMA_ANNEALING_START_EPOCH
#    → Ignores: TS_EMA_DECAY
#
# EXAMPLES:
# ---------
# Backward Compatible (Original):
#   TS_EMA_SCHEDULE="fixed" + TS_EMA_DECAY=0.999
#
# Advanced Cosine Annealing:
#   TS_EMA_SCHEDULE="cosine" + TS_EMA_DECAY_INITIAL=0.999 + TS_EMA_DECAY_FINAL=0.1

TS_TEACHER_INIT_EPOCH=10     # Epoch to initialize teacher (10-50, default: 20)
TS_MIN_ALPHA=0.95            # Minimum alpha for consistency loss (0.05-0.2, default: 0.1)
TS_MAX_ALPHA=0.95            # Maximum alpha for supervised loss (0.8-1.0, default: 1.0)
TS_CONSISTENCY_LOSS_TYPE="mse"  # Consistency loss type (mse, kl_div, l1, dice, iou, default: mse) "dice" For regional overlap focus / "iou" For exact boundary alignment / "mse"For pixel-wise consistency (default)
TS_CONSISTENCY_TEMPERATURE=1.0  # Temperature for consistency loss (1.0-4.0, default: 1.0)
TS_ENABLE_GLAND_CONSISTENCY=false  # Enable gland classification consistency loss (true/false, default: false)
TS_DEPTH=4                  # UNet depth (3-5, default: 4)
TS_INITIAL_CHANNELS=64      # Initial channels (32-128, default: 64)

# 🎯 TEACHER PSEUDO-MASK FILTERING (Noise Reduction)
# ==================================================
TS_PSEUDO_MASK_FILTERING="confidence"     # Filtering strategy: "none", "confidence", "entropy" (default: "none")
TS_CONFIDENCE_THRESHOLD=0.95         # Confidence threshold for confidence-based filtering (0.7-0.95, default: 0.8)
TS_ENTROPY_THRESHOLD=1.0            # Entropy threshold for entropy-based filtering (0.5-2.0, default: 1.0, lower=more selective)
TS_FILTERING_WARMUP_EPOCHS=10        # Number of epochs before applying filtering (0-50, default: 10)

# 🎯 ADAPTIVE CONFIDENCE THRESHOLD (Advanced Curriculum Learning)
# ==============================================================
TS_CONFIDENCE_ANNEALING="cosine"          # Annealing schedule: "none", "linear", "cosine" (default: "none")
TS_CONFIDENCE_MAX_THRESHOLD=0.95           # Starting threshold (early training) (0.8-0.95, default: 0.9)
TS_CONFIDENCE_MIN_THRESHOLD=0.25           # Ending threshold (late training) (0.5-0.7, default: 0.6)
TS_CONFIDENCE_ANNEALING_START_EPOCH=10     # When to start annealing (after warmup) (1-20, default: 5)

# 🧬 GT + TEACHER INCORPORATION (Enhanced Pseudo-Mask Fusion)
# ===========================================================
# GT + Teacher Incorporation Fusion Control:
# - When TRUE:  Teacher uses ENHANCED pseudo-masks = GT + Teacher predictions combined
#               GT foreground pixels (high priority) + Teacher discoveries in GT background
#               Better learning from both human annotations AND teacher's discoveries
#               ⚠️  REQUIRES: Teacher must be initialized first (after warmup phase)
# - When FALSE: Teacher uses PURE teacher predictions only (standard Teacher-Student)
#               No GT influence on pseudo-masks, purely learned teacher knowledge
#
# IMPORTANT: GT+Teacher fusion is automatically delayed until AFTER teacher initialization!
#            - During warmup phase (teacher not initialized): fusion is DISABLED
#            - After teacher init: fusion starts at max(TS_GT_INCORPORATE_START_EPOCH, teacher_init_epoch)
TS_GT_TEACHER_INCORPORATE_ENABLED=false   # Enable enhanced GT+Teacher fusion (default: false)
TS_GT_INCORPORATE_START_EPOCH=0           # Epoch to start fusion AFTER teacher init (0=immediately after teacher init, >0=additional delay, default: 0)
TS_GT_INCORPORATE_SEGMENTATION_ONLY=true  # Apply only to segmentation masks (true/false, default: true)

# 🔍 TEACHER-STUDENT EVALUATION CONFIGURATION
# ============================================
TEACHER_STUDENT_EVALUATOR="latest"  # Which checkpoint to use for evaluation ("latest" or "best")
TS_POST_EVAL_MODE="both"         # Post-training evaluation mode: "student", "teacher", "both" (default: "student")

# ⚖️ LOSS FUNCTION WEIGHTS
# ========================
DICE_WEIGHT=0.95           # Weight for Dice loss (0.0-1.0, default: 0.5)
CE_WEIGHT=0.05             # Weight for Cross-Entropy loss (0.0-1.0, default: 0.5)
                          # Note: DICE_WEIGHT + CE_WEIGHT should typically equal 1.0

# 🛑 STOPPING CRITERIA PARAMETERS
# ================================
EARLY_STOP_PATIENCE=8500  # Stop if no validation improvement for N epochs (default: 30)
LR_SCHEDULER_PATIENCE=100 # Reduce LR if no improvement for N epochs (default: 15)
MIN_LR="1e-9"           # Minimum learning rate for scheduler (default: 1e-7)

# 📉 LEARNING RATE SCHEDULER PARAMETERS
# =====================================
COSINE_T_MAX="$EPOCHS"  # T_max for CosineAnnealingLR (should equal total epochs for full cycle)
COSINE_ETA_MIN="$MIN_LR" # eta_min for CosineAnnealingLR (minimum LR at end of cycle)

# 🔥 STEP 4: Output Directory - MODIFY TO YOUR PREFERENCE
# ========================================================
# OUTPUT_DIR="/fs/scratch/PAS2942/Datasets/CRC/Datasets/OSU/Gland_Datasets/ALL_RUNS"
# OUTPUT_DIR="/fs/scratch/PAS2942/Datasets/CRC/Datasets/OSU/Gland_Datasets/ALL_RUNS/_Sept18"
# OUTPUT_DIR="/fs/scratch/PAS2942/Datasets/CRC/Datasets/OSU/Gland_Datasets/_RESULTS_2"
# OUTPUT_DIR="/fs/scratch/PAS2942/Datasets/CRC/Datasets/OSU/Gland_Datasets/_RESULTS_2"
# OUTPUT_DIR="/fs/scratch/PAS2942/Datasets/CRC/Datasets/OSU/Gland_Datasets/_RESULTS_OCT13"
# OUTPUT_DIR="/fs/scratch/PAS2942/Datasets/CRC/Datasets/OSU/Gland_Datasets/_RESULTS_Oct30_Paper"
# OUTPUT_DIR="/fs/scratch/PAS2942/Datasets/CRC/Datasets/OSU/Gland_Datasets/_RESULTS_PPR"


# OUTPUT_DIR="/fs/scratch/PAS2942/Datasets/CRC/Datasets/OSU/Gland_Datasets/_RESULTS_PPR/2ndRuns"
# OUTPUT_DIR="/fs/scratch/PAS2942/Datasets/CRC/Datasets/OSU/Gland_Datasets/_RESULTS_PPR/3rdRuns"

# OUTPUT_DIR="/fs/scratch/PAS2942/Datasets/CRC/Datasets/OSU/Gland_Datasets/_RESULTS_PPR/GlasDS"
# OUTPUT_DIR="/fs/scratch/PAS2942/Datasets/CRC/Datasets/OSU/Gland_Datasets/_RESULTS_PPR/GlasDS/2ndRuns"
# OUTPUT_DIR="/fs/scratch/PAS2942/Datasets/CRC/Datasets/OSU/Gland_Datasets/_RESULTS_PPR/GlasDS/3rdRuns"

OUTPUT_DIR="/fs/scratch/PAS2942/Datasets/CRC/Datasets/OSU/Gland_Datasets/DeleteIT_Resuming_Testing"

# 📁 PROCESSING DIRECTORIES - MODIFY IF NEEDED
# =============================================
# Where nnU-Net stores intermediate preprocessing files
NNUNET_PREPROCESSED_DIR="/fs/scratch/PAS2942/Datasets/CRC/Datasets/OSU/Gland_Datasets/nnUNet_preprocessed"
NNUNET_RESULTS_DIR="/fs/scratch/PAS2942/Datasets/CRC/Datasets/OSU/Gland_Datasets/nnUNet_results"
TEMP_DIR="/tmp/nnunet_gland_seg_$$"  # Unique temp dir per job

# 🔬 ADVANCED OPTIONS
# ===================
ENHANCED_TRAINING=true  # Set to true for stronger augmentation (better but slower)
CUSTOM_NAME=""          # Leave empty for auto-generated names with architecture prefix

# 🔄 REPRODUCIBILITY CONFIGURATION
# =================================
# Critical for ensuring exact reproducibility of results
# Generate a random seed for this training run (or use provided seed)
MASTER_SEED=${MASTER_SEED:-$(python3 -c "import random; print(random.randint(1000, 9999))")}
echo "🎲 Using Master Seed: $MASTER_SEED for this training run"
echo "📝 To reproduce this exact run later, set: export MASTER_SEED=$MASTER_SEED"

# Use the same seed for all random number generators for perfect reproducibility
PYTHON_SEED=$MASTER_SEED              # Python random seed
NUMPY_SEED=$MASTER_SEED               # NumPy random seed
TORCH_SEED=$MASTER_SEED               # PyTorch random seed
TORCH_CUDA_SEED=$MASTER_SEED          # PyTorch CUDA seed
TORCH_CUDA_SEED_ALL=$MASTER_SEED      # PyTorch CUDA seed for all devices

# 🛠️ OPTIONAL ADVANCED PARAMETERS
# =================================
# Uncomment and modify any of these if you want to customize further:
#
# NUM_WORKERS=4           # Number of data loading workers (2-8)
# WEIGHT_DECAY=1e-4       # Weight decay for regularization (1e-5 to 1e-3)
IMAGE_SIZE="512,512"    # Input image size (256,256 | 512,512 | 1024,1024)
# OPTIMIZER="adamw"       # Optimizer type (adamw | sgd)
SCHEDULER="CosineAnnealingLR"        # Learning rate scheduler (poly | CosineAnnealingLR | ReduceLROnPlateau)

# =====================================================
# 🚀 QUICK START EXAMPLES
# =====================================================
#
# 📖 Example 1: Baseline UNet on Mixed Dataset (Fast Baseline)
# ARCHITECTURE="baseline_unet"
# DATASET_KEY="mixed"
# EPOCHS=150
# BATCH_SIZE=8
#
# 📖 Example 2: nnU-Net on High-Resolution Dataset (Best Performance)
# ARCHITECTURE="nnunet"
# DATASET_KEY="mag40x"
# EPOCHS=200
# BATCH_SIZE=4
#
# 📖 Example 3: Quick Test Run (2 epochs for debugging)
# ARCHITECTURE="baseline_unet"
# DATASET_KEY="mixed"
# EPOCHS=2
# BATCH_SIZE=16
#
# 📖 Example 4: Enhanced Training with Stronger Augmentation
# ARCHITECTURE="nnunet"
# DATASET_KEY="mixed"
# EPOCHS=150
# ENHANCED_TRAINING=true
# BATCH_SIZE=4

# --------- Export Environment Variables ----------
# Export all configuration so Python can access it (no hardcoded values!)
export GLAND_DATASET_BASE="$DATASET_BASE_DIR"
export GLAND_OUTPUT_DIR="$OUTPUT_DIR"
export NNUNET_PREPROCESSED="$NNUNET_PREPROCESSED_DIR"
export NNUNET_RESULTS="$NNUNET_RESULTS_DIR"
export GLAND_TEMP_DIR="$TEMP_DIR"

# Export training parameters
export GLAND_ARCHITECTURE="$ARCHITECTURE"
export GLAND_EPOCHS="$EPOCHS"
export GLAND_BATCH_SIZE="$BATCH_SIZE"
export GLAND_LEARNING_RATE="$LEARNING_RATE"
export GLAND_ENHANCED_TRAINING="$ENHANCED_TRAINING"

# Export loss function weights
export GLAND_DICE_WEIGHT="$DICE_WEIGHT"
export GLAND_CE_WEIGHT="$CE_WEIGHT"

# Export stopping criteria parameters
export GLAND_EARLY_STOP_PATIENCE="$EARLY_STOP_PATIENCE"
export GLAND_LR_SCHEDULER_PATIENCE="$LR_SCHEDULER_PATIENCE"
export GLAND_MIN_LR="$MIN_LR"

# Export learning rate scheduler parameters
export GLAND_COSINE_T_MAX="${COSINE_T_MAX:-$EPOCHS}"
export GLAND_COSINE_ETA_MIN="${COSINE_ETA_MIN:-$MIN_LR}"

# Export additional configurable parameters with defaults
export GLAND_NUM_WORKERS="${NUM_WORKERS:-4}"
export GLAND_WEIGHT_DECAY="${WEIGHT_DECAY:-1e-4}"
export GLAND_IMAGE_SIZE="${IMAGE_SIZE:-512,512}"
export GLAND_OPTIMIZER="${OPTIMIZER:-adamw}"
export GLAND_SCHEDULER="${SCHEDULER:-poly}"

# Export Teacher-Student specific parameters
export GLAND_TS_BACKBONE_TYPE="$TS_BACKBONE_TYPE"
export GLAND_TS_EMA_DECAY="$TS_EMA_DECAY"
export GLAND_TS_EMA_SCHEDULE="$TS_EMA_SCHEDULE"
export GLAND_TS_EMA_DECAY_INITIAL="$TS_EMA_DECAY_INITIAL"
export GLAND_TS_EMA_DECAY_FINAL="$TS_EMA_DECAY_FINAL"
export GLAND_TS_EMA_ANNEALING_START_EPOCH="$TS_EMA_ANNEALING_START_EPOCH"
export GLAND_TS_TEACHER_INIT_EPOCH="$TS_TEACHER_INIT_EPOCH"
export GLAND_TS_MIN_ALPHA="$TS_MIN_ALPHA"
export GLAND_TS_MAX_ALPHA="$TS_MAX_ALPHA"
export GLAND_TS_CONSISTENCY_LOSS_TYPE="$TS_CONSISTENCY_LOSS_TYPE"
export GLAND_TS_CONSISTENCY_TEMPERATURE="$TS_CONSISTENCY_TEMPERATURE"
export GLAND_TS_ENABLE_GLAND_CONSISTENCY="$TS_ENABLE_GLAND_CONSISTENCY"
export GLAND_TS_DEPTH="$TS_DEPTH"
export GLAND_TS_INITIAL_CHANNELS="$TS_INITIAL_CHANNELS"
export GLAND_TS_PSEUDO_MASK_FILTERING="$TS_PSEUDO_MASK_FILTERING"
export GLAND_TS_CONFIDENCE_THRESHOLD="$TS_CONFIDENCE_THRESHOLD"
export GLAND_TS_ENTROPY_THRESHOLD="$TS_ENTROPY_THRESHOLD"
export GLAND_TS_FILTERING_WARMUP_EPOCHS="$TS_FILTERING_WARMUP_EPOCHS"
export GLAND_TS_CONFIDENCE_ANNEALING="$TS_CONFIDENCE_ANNEALING"
export GLAND_TS_CONFIDENCE_MAX_THRESHOLD="$TS_CONFIDENCE_MAX_THRESHOLD"

# Export reproducibility parameters
export GLAND_MASTER_SEED="$MASTER_SEED"
export GLAND_PYTHON_SEED="$PYTHON_SEED"
export GLAND_NUMPY_SEED="$NUMPY_SEED"
export GLAND_TORCH_SEED="$TORCH_SEED"
export GLAND_TORCH_CUDA_SEED="$TORCH_CUDA_SEED"
export GLAND_TORCH_CUDA_SEED_ALL="$TORCH_CUDA_SEED_ALL"
export GLAND_TS_CONFIDENCE_MIN_THRESHOLD="$TS_CONFIDENCE_MIN_THRESHOLD"
export GLAND_TS_CONFIDENCE_ANNEALING_START_EPOCH="$TS_CONFIDENCE_ANNEALING_START_EPOCH"
export GLAND_TEACHER_STUDENT_EVALUATOR="$TEACHER_STUDENT_EVALUATOR"
export GLAND_TS_POST_EVAL_MODE="$TS_POST_EVAL_MODE"
export GLAND_TS_GT_TEACHER_INCORPORATE_ENABLED="$TS_GT_TEACHER_INCORPORATE_ENABLED"
export GLAND_TS_GT_INCORPORATE_START_EPOCH="$TS_GT_INCORPORATE_START_EPOCH"
export GLAND_TS_GT_INCORPORATE_SEGMENTATION_ONLY="$TS_GT_INCORPORATE_SEGMENTATION_ONLY"

# --------- Derived Configuration ----------
if [ "$ENHANCED_TRAINING" = true ]; then
    ENHANCED_FLAG="--enhanced"
    ENHANCED_SUFFIX="_enhanced"
else
    ENHANCED_FLAG=""
    ENHANCED_SUFFIX=""
fi

if [ -n "$CUSTOM_NAME" ]; then
    EXPERIMENT_NAME="$CUSTOM_NAME"
else
    # For Teacher-Student UNet, include backbone type in the experiment name
    if [ "$ARCHITECTURE" = "teacher_student_unet" ]; then
        # Convert backbone type to short form for cleaner names
        if [ "$TS_BACKBONE_TYPE" = "baseline_unet" ]; then
            BACKBONE_SUFFIX="_baseline"
        elif [ "$TS_BACKBONE_TYPE" = "nnunet" ]; then
            BACKBONE_SUFFIX="_nnunet"
        else
            BACKBONE_SUFFIX="_${TS_BACKBONE_TYPE}"
        fi
        EXPERIMENT_NAME="teacher_student${BACKBONE_SUFFIX}_${DATASET_KEY}${ENHANCED_SUFFIX}_$(date +%Y%m%d_%H%M%S)"
    else
        EXPERIMENT_NAME="${ARCHITECTURE}_${DATASET_KEY}${ENHANCED_SUFFIX}_$(date +%Y%m%d_%H%M%S)"
    fi
fi

echo "Training Configuration:"
echo "  🏗️  Architecture: $ARCHITECTURE"
echo "  📊 Dataset: $DATASET_KEY"
echo "  📁 Dataset Base: $DATASET_BASE_DIR"
echo "  🔄 Epochs: $EPOCHS"
echo "  📦 Batch Size: $BATCH_SIZE"
echo "  📈 Learning Rate: $LEARNING_RATE"
echo "  🚀 Enhanced Training: $ENHANCED_TRAINING"
echo "  ⏱️  Early Stop Patience: $EARLY_STOP_PATIENCE epochs"
echo "  📉 LR Scheduler Patience: $LR_SCHEDULER_PATIENCE epochs"
echo "  🔻 Minimum LR: $MIN_LR"
echo "  🌊 Cosine T_max: ${COSINE_T_MAX:-$EPOCHS} epochs"
echo "  🌊 Cosine eta_min: ${COSINE_ETA_MIN:-$MIN_LR}"
echo "  📁 Output Directory: $OUTPUT_DIR"
echo "  🔧 Preprocessed: $NNUNET_PREPROCESSED_DIR"
echo "  📊 Results: $NNUNET_RESULTS_DIR"
echo "  💾 Temp Dir: $TEMP_DIR"
echo "  🏷️  Experiment Name: $EXPERIMENT_NAME"
echo "  🤖 Auto Prefix: ${ARCHITECTURE}_exp_*"
echo "============================================"

# --------- Architecture Info ----------
case "$ARCHITECTURE" in
    "baseline_unet")
        echo "🔧 Baseline UNet Configuration:"
        echo "   • Simple, interpretable architecture"
        echo "   • ~31M+ parameters"
        echo "   • Faster training (~6-8h for 150 epochs)"
        echo "   • Good baseline performance"
        echo "   • Experiment prefix: baseline_unet_exp_*"
        ;;
    "nnunet")
        echo "🚀 nnU-Net Configuration:"
        echo "   • State-of-the-art architecture"
        echo "   • ~20M+ parameters"
        echo "   • Advanced training (~8-12h for 150 epochs)"
        echo "   • Maximum performance"
        echo "   • Experiment prefix: nnunet_exp_*"
        ;;
    "teacher_student_unet")
        echo "🎓 Teacher-Student UNet Configuration:"
        echo "   • Self-training with EMA teacher updates"
        echo "   • ~62M+ parameters (Student + Teacher)"
        echo "   • Two-phase training (~10-16h for 200 epochs)"
        echo "   • Advanced consistency learning"
        echo "   • Backbone Type: $TS_BACKBONE_TYPE"
        echo "   • EMA Schedule: $TS_EMA_SCHEDULE"
        if [ "$TS_EMA_SCHEDULE" = "fixed" ]; then
            echo "   • EMA Decay: $TS_EMA_DECAY (fixed)"
        else
            echo "   • EMA Decay: $TS_EMA_DECAY_INITIAL → $TS_EMA_DECAY_FINAL ($TS_EMA_SCHEDULE annealing)"
            echo "   • EMA Annealing Start: Epoch $TS_EMA_ANNEALING_START_EPOCH"
        fi
        echo "   • Teacher Init Epoch: $TS_TEACHER_INIT_EPOCH"
        echo "   • Consistency Loss: $TS_CONSISTENCY_LOSS_TYPE"
        echo "   • Alpha Range: $TS_MAX_ALPHA → $TS_MIN_ALPHA"
        echo "   • Experiment prefix: teacher_student_${TS_BACKBONE_TYPE}_exp_*"
        ;;
    *)
        echo "❌ ERROR: Unknown architecture: $ARCHITECTURE"
        echo "Please use 'baseline_unet', 'nnunet', or 'teacher_student_unet'"
        exit 1
        ;;
esac
echo "============================================"

# --------- Dataset Info ----------
case "$DATASET_KEY" in
    "mixed")
        echo "📊 Dataset: Mixed Magnifications"
        echo "   • All magnifications combined"
        echo "   • ~25,000 samples"
        echo "   • Best for general-purpose models"
        ;;
    "mag5x"|"mag10x"|"mag20x"|"mag40x")
        echo "📊 Dataset: $DATASET_KEY"
        echo "   • Single magnification"
        echo "   • ~4,800-6,200 samples"
        echo "   • Magnification-specific optimization"
        ;;
    "warwick")
        echo "📊 Dataset: Warwick GlaS Teacher-Student"
        echo "   • 3-class semantic segmentation (Background, Benign, Malignant)"
        echo "   • 165 samples total (train+val: 85, test: 80)"
        echo "   • Includes validation split for teacher-student learning"
        ;;
    *)
        echo "❌ ERROR: Unknown dataset: $DATASET_KEY"
        echo "Please use: mixed, mag5x, mag10x, mag20x, mag40x, or warwick"
        exit 1
        ;;
esac
echo "============================================"

# --------- Demo Test (Optional) ----------
echo "🧪 Running quick demo test with $DATASET_KEY dataset..."
python main.py demo --architecture $ARCHITECTURE --dataset $DATASET_KEY

if [ $? -ne 0 ]; then
    echo "❌ Demo test failed! Please check your setup."
    exit 1
fi
echo "✅ Demo test passed for $DATASET_KEY dataset!"
echo "============================================"

# --------- Training Command ----------
echo "🚀 Starting Multi-Architecture Training on $DATASET_KEY dataset..."
echo "Command: python main.py train --architecture $ARCHITECTURE --dataset $DATASET_KEY --epochs $EPOCHS --batch_size $BATCH_SIZE --learning_rate $LEARNING_RATE --patience $EARLY_STOP_PATIENCE --output_dir $OUTPUT_DIR --experiment_name $EXPERIMENT_NAME $ENHANCED_FLAG"
echo "============================================"

# Build and run training command
python main.py train \
    --architecture $ARCHITECTURE \
    --dataset $DATASET_KEY \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --learning_rate $LEARNING_RATE \
    --patience $EARLY_STOP_PATIENCE \
    --output_dir $OUTPUT_DIR \
    --experiment_name $EXPERIMENT_NAME \
    $ENHANCED_FLAG

# Check exit status
if [ $? -eq 0 ]; then
    echo "============================================"
    echo "✅ Training completed successfully on $DATASET_KEY dataset!"
    echo "End Time: $(date)"
    echo "Results saved to: $OUTPUT_DIR/${ARCHITECTURE}_exp_*"
    echo "============================================"
else
    echo "============================================"
    echo "❌ Training failed on $DATASET_KEY dataset with exit code: $?"
    echo "End Time: $(date)"
    echo "Check error logs for details."
    echo "============================================"
    exit 1
fi

# --------- Results Summary ----------
echo "============================================"
echo "📊 TRAINING RESULTS SUMMARY"
echo "============================================"
echo "🏗️  Architecture: $ARCHITECTURE"
echo "📊 Dataset: $DATASET_KEY"
echo "📁 Output Location: $OUTPUT_DIR/${ARCHITECTURE}_exp_*"
echo ""
echo "📋 Available Results:"
echo "   ✅ Best Model: models/best_model.pth"
echo "   ✅ Training Logs: logs/training.log"
echo "   ✅ TensorBoard: logs/tensorboard/"
echo "   ✅ Training Curves: visualizations/training_curves.png"
echo "   ✅ Metrics History: loss_history.csv"
echo "   ✅ Configuration: training_config.json"
echo "   ✅ Quick Summary: quick_summary.csv"
echo ""
echo "🔬 Automatic Evaluation:"
echo "   ✅ Complete dataset evaluation included"
echo "   ✅ Comprehensive metrics computed"
echo "   ✅ Visualization samples generated"
echo "   ✅ Publication-ready results"

# --------- Architecture Comparison Helper ----------
echo ""
echo "============================================"
echo "🔄 ARCHITECTURE COMPARISON"
echo "============================================"
if [ "$ARCHITECTURE" = "baseline_unet" ]; then
    echo "You trained: Baseline UNet on $DATASET_KEY dataset"
    echo "💡 To compare with nnU-Net, run:"
    echo "   1. Change ARCHITECTURE='nnunet' in this script"
    echo "   2. Keep DATASET_KEY='$DATASET_KEY' for fair comparison"
    echo "   3. Rerun: sbatch run_nnunet_training.sh"
    echo "   4. Compare results in $OUTPUT_DIR/"
else
    echo "You trained: nnU-Net on $DATASET_KEY dataset"
    echo "💡 To compare with Baseline UNet, run:"
    echo "   1. Change ARCHITECTURE='baseline_unet' in this script"
    echo "   2. Keep DATASET_KEY='$DATASET_KEY' for fair comparison"
    echo "   3. Rerun: sbatch run_nnunet_training.sh"
    echo "   4. Compare results in $OUTPUT_DIR/"
fi

# --------- Manual Evaluation (Optional) ----------
echo ""
echo "============================================"
echo "🔍 OPTIONAL: Manual Evaluation"
echo "============================================"
echo "Training includes automatic evaluation, but for custom analysis:"

# Find the actual experiment directory (with auto-prefix)
ACTUAL_EXP_DIR=$(find "$OUTPUT_DIR" -maxdepth 1 -name "${ARCHITECTURE}_exp_*" -type d | sort | tail -n 1)

if [ -n "$ACTUAL_EXP_DIR" ] && [ -f "$ACTUAL_EXP_DIR/models/best_model.pth" ]; then
    echo "Best model: $ACTUAL_EXP_DIR/models/best_model.pth"

    # For Teacher-Student UNet, show additional models
    if [ "$ARCHITECTURE" = "teacher_student_unet" ]; then
        if [ -f "$ACTUAL_EXP_DIR/models/best_student_model.pth" ]; then
            echo "Best Student: $ACTUAL_EXP_DIR/models/best_student_model.pth"
        fi
        if [ -f "$ACTUAL_EXP_DIR/models/best_teacher_model.pth" ]; then
            echo "Best Teacher: $ACTUAL_EXP_DIR/models/best_teacher_model.pth"
        fi
    fi

    echo ""
    echo "Manual evaluation command:"
    echo "python main.py evaluate \\"
    echo "    --architecture $ARCHITECTURE \\"
    echo "    --model \"$ACTUAL_EXP_DIR/models/best_model.pth\" \\"
    echo "    --dataset $DATASET_KEY \\"
    echo "    --output \"$OUTPUT_DIR/custom_evaluation\" \\"
    echo "    --visualize"
else
    echo "⚠️  Model files not found where expected. Check training logs for details."

    # For Teacher-Student UNet, provide specific guidance
    if [ "$ARCHITECTURE" = "teacher_student_unet" ]; then
        echo "   For Teacher-Student UNet evaluation, the system uses INDEPENDENT models:"
        echo "   - Checkpoint Type: TEACHER_STUDENT_EVALUATOR='$TEACHER_STUDENT_EVALUATOR'"
        echo "   - Evaluation Mode: TS_POST_EVAL_MODE='$TS_POST_EVAL_MODE'"

        if [ "$TEACHER_STUDENT_EVALUATOR" = "best" ]; then
            checkpoint_prefix="best"
        else
            checkpoint_prefix="latest"
        fi

        if [ "$TS_POST_EVAL_MODE" = "both" ]; then
            echo "   - ${checkpoint_prefix}_student_model.pth (student evaluation + metrics)"
            echo "   - ${checkpoint_prefix}_teacher_model.pth (teacher evaluation + metrics)"
            echo "   - Will generate separate reports for both student and teacher"
        elif [ "$TS_POST_EVAL_MODE" = "teacher" ]; then
            echo "   - ${checkpoint_prefix}_teacher_model.pth (teacher evaluation + metrics)"
            echo "   - Will evaluate teacher network only"
        else
            echo "   - ${checkpoint_prefix}_student_model.pth (student evaluation + metrics)"
            echo "   - ${checkpoint_prefix}_teacher_model.pth (for visualization pseudo-masks only)"
            echo "   - Will evaluate student network only"
        fi
        echo "   Note: Each model is completely independent with separate classification heads"
        if [ -n "$ACTUAL_EXP_DIR" ]; then
            echo "   Available files in $ACTUAL_EXP_DIR/models/:"
            ls -la "$ACTUAL_EXP_DIR/models/" 2>/dev/null || echo "   Directory not accessible"
        fi
    fi
fi

echo ""
echo "============================================"
echo "🎉 Job completed at: $(date)"
echo "============================================"

# --------- Quick Commands Reference ----------
echo ""
echo "============================================"
echo "📚 QUICK REFERENCE"
echo "============================================"
echo "Architecture Options:"
echo "  ARCHITECTURE='baseline_unet'      # Simple, fast, good baseline"
echo "  ARCHITECTURE='nnunet'             # Advanced, slower, best performance"
echo "  ARCHITECTURE='teacher_student_unet' # Self-training, dual networks (includes backbone in name)"
echo ""
echo "Dataset Configuration:"
echo "  DATASET_BASE_DIR='/path/to/nnUNetCombined'     # Base dataset directory"
echo "  DATASET_KEY='mixed'                            # Dataset selection"
echo "  Available: mixed, mag5x, mag10x, mag20x, mag40x"
echo ""
echo "Path Configuration:"
echo "  OUTPUT_DIR='/path/to/outputs'                  # Where to save results"
echo "  NNUNET_PREPROCESSED_DIR='/path/to/preprocessed' # nnU-Net processing"
echo "  NNUNET_RESULTS_DIR='/path/to/results'          # nnU-Net results"
echo "  TEMP_DIR='/tmp/unique_temp'                     # Temporary files"
echo ""
echo "Training Options:"
echo "  EPOCHS=2              # Fast test (2-5 epochs)"
echo "  EPOCHS=150            # Research quality (150+ epochs)"
echo "  BATCH_SIZE=4          # Adjust based on GPU memory (2-16)"
echo "  LEARNING_RATE='1e-4'  # Learning rate"
echo "  DICE_WEIGHT=0.7       # Weight for Dice loss (0.0-1.0)"
echo "  CE_WEIGHT=0.3         # Weight for Cross-Entropy loss (0.0-1.0)"
echo "  ENHANCED_TRAINING=true   # Stronger augmentation"
echo "  EARLY_STOP_PATIENCE=30  # Stop if no improvement for N epochs"
echo "  LR_SCHEDULER_PATIENCE=15 # Reduce LR if no improvement for N epochs"
echo "  MIN_LR='1e-7'           # Minimum learning rate for scheduler"
echo "  COSINE_T_MAX=150        # CosineAnnealingLR T_max (default: same as epochs)"
echo "  COSINE_ETA_MIN='1e-7'   # CosineAnnealingLR eta_min (default: same as MIN_LR)"
echo ""
echo "Advanced Options (uncomment in script to use):"
echo "  NUM_WORKERS=4         # Data loading workers"
echo "  WEIGHT_DECAY=1e-4     # Regularization strength"
echo "  IMAGE_SIZE='512,512'  # Input image dimensions"
echo "  OPTIMIZER='adamw'     # adamw, sgd"
echo "  SCHEDULER='poly'      # poly, cosine, plateau"
echo ""
echo "Teacher-Student Options (when ARCHITECTURE='teacher_student_unet'):"
echo "  TS_BACKBONE_TYPE='baseline_unet' # Backbone ('baseline_unet', 'nnunet')"
echo "  TS_EMA_DECAY=0.999           # Teacher EMA decay (0.99-0.999)"
echo "  TS_TEACHER_INIT_EPOCH=20     # Teacher initialization epoch (10-50)"
echo "  TS_MIN_ALPHA=0.1            # Min consistency weight (0.05-0.2)"
echo "  TS_MAX_ALPHA=1.0            # Max supervised weight (0.8-1.0)"
echo "  TS_CONSISTENCY_LOSS_TYPE='mse' # Loss type (mse, kl_div, l1)"
echo "  TS_CONSISTENCY_TEMPERATURE=1.0 # Temperature (1.0-4.0)"
echo "  TS_DEPTH=4                  # UNet depth (3-5) - for baseline_unet backbone"
echo "  TS_INITIAL_CHANNELS=64      # Initial channels (32-128) - for baseline_unet backbone"
echo "  TS_PSEUDO_MASK_FILTERING='none'  # Pseudo-mask filtering ('none', 'confidence', 'entropy')"
echo "  TS_CONFIDENCE_THRESHOLD=0.8     # Confidence threshold (0.7-0.95)"
echo "  TS_ENTROPY_THRESHOLD=1.0        # Entropy threshold (0.5-2.0, lower=more selective)"
echo "  TS_FILTERING_WARMUP_EPOCHS=10   # Warmup epochs before filtering (0-50)"
echo "  TEACHER_STUDENT_EVALUATOR='latest' # Evaluation checkpoint ('latest' or 'best')"
echo "  TS_POST_EVAL_MODE='student'     # Evaluation mode ('student', 'teacher', 'both')"
echo "============================================"