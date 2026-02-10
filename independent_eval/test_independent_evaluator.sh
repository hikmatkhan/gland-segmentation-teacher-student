#!/bin/bash
#SBATCH --account=PAS2942
#SBATCH --job-name=IndepEval_Gland
#SBATCH --time=24:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-gpu=16
#SBATCH --mem=64gb
#SBATCH --output=/fs/scratch/PAS2942/Datasets/CRC/Datasets/OSU/Gland_Datasets/_INDEPENDENT_EVAL/logs/independent_eval_%A.out
#SBATCH --error=/fs/scratch/PAS2942/Datasets/CRC/Datasets/OSU/Gland_Datasets/_INDEPENDENT_EVAL/logs/independent_eval_%A.err
#SBATCH --cluster=ascend

# =====================================================
# Independent Evaluator Test Script for GlandSegModels nnU-Net
# Supports both baseline_unet and nnunet architectures
# =====================================================

# --------- Environment ----------
module load miniconda3/24.1.2-py310
module load cuda/11.8.0
conda activate llm

# --------- Environment Info ----------
echo "============================================"
echo "Independent Model Evaluator for GlandSegModels nnU-Net"
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
echo "============================================"

echo "🧪 Testing Independent Model Evaluator for GlandSegModels nnU-Net..."
echo ""

# =====================================================
# 🎯 EVALUATION CONFIGURATION
# =====================================================

# 🔥 STEP 1: Choose Architecture (must match training)
# ====================================================
ARCHITECTURE="baseline_unet"  # ← baseline_unet OR nnunet (MUST match training architecture)
# ARCHITECTURE="nnunet"       # ← Uncomment for nnunet evaluation

# 🔥 STEP 2: Choose Dataset (must match training)
# ===============================================
DATASET_KEY="mixed"         # ← mixed, mag5x, mag10x, mag20x, mag40x (MUST match training dataset)

# 🔥 STEP 3: Experiment Path (root experiment directory from training)
# ====================================================================
# Replace with actual experiment path after training is complete
EXPERIMENT_PATH="/fs/scratch/PAS2942/Datasets/CRC/Datasets/OSU/Gland_Datasets/_RESULTS_2/baseline_unet_mixed_20250919_181253"
# EXPERIMENT_PATH="/fs/scratch/PAS2942/Datasets/CRC/Datasets/OSU/Gland_Datasets/_Output_nnUNet_Sanity/nnunet_exp_20250918_150045"

# 📁 Dataset Configuration (from training script)
# ===============================================
# DATASET_BASE_DIR="/fs/scratch/PAS2942/Datasets/CRC/Datasets/OSU/Gland_Datasets/nnUNetCombined"
DATASET_BASE_DIR="/fs/scratch/PAS2942/Datasets/CRC/Datasets/OSU/Gland_Datasets/_OSU_All_Gland_Datasets_nnUNet"

# 📂 Output Configuration
# ======================
# OUTPUT_PATH="/fs/scratch/PAS2942/Datasets/CRC/Datasets/OSU/Gland_Datasets/_Independent_Eval"
OUTPUT_PATH="/fs/scratch/PAS2942/Datasets/CRC/Datasets/OSU/Gland_Datasets/_INDEPENDENT_EVAL"

# 🔬 Evaluation Configuration
# ===========================
SPLIT="all"              # train, val, test, or all (evaluate all three splits)
NUM_SAMPLES=100          # 100 samples per split for visualization (as requested)
BATCH_SIZE=32             # Evaluation batch size (reduced for memory efficiency)

# =====================================================
# 🚀 QUICK CONFIGURATION EXAMPLES
# =====================================================
#
# 📖 Example 1: Evaluate Baseline UNet on mag40x dataset
# ARCHITECTURE="baseline_unet"
# DATASET_KEY="mag40x"
# EXPERIMENT_PATH="/path/to/baseline_unet_exp_20250918_143022"
#
# 📖 Example 2: Evaluate nnU-Net on mixed dataset
# ARCHITECTURE="nnunet"
# DATASET_KEY="mixed"
# EXPERIMENT_PATH="/path/to/nnunet_exp_20250918_150045"
#
# 📖 Example 3: Evaluate only test split
# SPLIT="test"
# NUM_SAMPLES=50
#
# 📖 Example 4: Quick evaluation with fewer samples
# SPLIT="val"
# NUM_SAMPLES=25

# Create output directory
mkdir -p "$OUTPUT_PATH"

# =====================================================
# 🌍 EXPORT REQUIRED ENVIRONMENT VARIABLES
# =====================================================
# The independent evaluator requires these environment variables to be set
# for the nnU-Net integration to work properly

echo "🌍 Exporting required environment variables..."

export GLAND_DATASET_BASE="$DATASET_BASE_DIR"
export GLAND_OUTPUT_DIR="$OUTPUT_PATH"
export NNUNET_PREPROCESSED="/fs/scratch/PAS2942/Datasets/CRC/Datasets/OSU/Gland_Datasets/nnUNet_preprocessed"
export NNUNET_RESULTS="/fs/scratch/PAS2942/Datasets/CRC/Datasets/OSU/Gland_Datasets/nnUNet_results"
export GLAND_TEMP_DIR="/tmp/nnunet_gland_seg_$$"

echo "✅ Environment variables exported:"
echo "   GLAND_DATASET_BASE: $GLAND_DATASET_BASE"
echo "   GLAND_OUTPUT_DIR: $GLAND_OUTPUT_DIR"
echo "   NNUNET_PREPROCESSED: $NNUNET_PREPROCESSED"
echo "   NNUNET_RESULTS: $NNUNET_RESULTS"
echo "   GLAND_TEMP_DIR: $GLAND_TEMP_DIR"

echo "============================================"
echo "📍 Evaluation Configuration:"
echo "   🏗️  Architecture: $ARCHITECTURE"
echo "   📊 Dataset Key: $DATASET_KEY"
echo "   📁 Experiment Path: $EXPERIMENT_PATH"
echo "   📊 Dataset Base: $DATASET_BASE_DIR"
echo "   📂 Output Path: $OUTPUT_PATH"
echo "   🔄 Split(s): $SPLIT"
echo "   🖼️  Samples per Split: $NUM_SAMPLES"
echo "   📦 Batch Size: $BATCH_SIZE"
echo "============================================"

# =====================================================
# 🔍 VALIDATION CHECKS
# =====================================================

echo "🔍 Validating paths and configuration..."

# Check if experiment path exists
if [ ! -d "$EXPERIMENT_PATH" ]; then
    echo "❌ Experiment path does not exist: $EXPERIMENT_PATH"
    echo ""
    echo "💡 Common experiment paths to check:"
    echo "   /fs/scratch/PAS2942/Datasets/CRC/Datasets/OSU/Gland_Datasets/_Output_nnUNet_Sanity/baseline_unet_exp_*"
    echo "   /fs/scratch/PAS2942/Datasets/CRC/Datasets/OSU/Gland_Datasets/_Output_nnUNet_Sanity/nnunet_exp_*"
    echo ""
    echo "To find your experiment directory, run:"
    echo "   ls -la /fs/scratch/PAS2942/Datasets/CRC/Datasets/OSU/Gland_Datasets/_Output_nnUNet_Sanity/"
    exit 1
fi

# Check for best model
if [ ! -f "$EXPERIMENT_PATH/models/best_model.pth" ]; then
    echo "❌ Best model not found: $EXPERIMENT_PATH/models/best_model.pth"
    echo ""
    echo "📂 Checking for available models in $EXPERIMENT_PATH/models/:"
    ls -la "$EXPERIMENT_PATH/models/" || echo "   Models directory not found"
    exit 1
fi

# Check if dataset base directory exists
if [ ! -d "$DATASET_BASE_DIR" ]; then
    echo "❌ Dataset base directory does not exist: $DATASET_BASE_DIR"
    echo ""
    echo "💡 Expected dataset structure:"
    echo "   $DATASET_BASE_DIR/Task001_Combined_Mixed_Magnifications/"
    echo "   $DATASET_BASE_DIR/Task005_Combined_Mag5x/"
    echo "   $DATASET_BASE_DIR/Task010_Combined_Mag10x/"
    echo "   $DATASET_BASE_DIR/Task020_Combined_Mag20x/"
    echo "   $DATASET_BASE_DIR/Task040_Combined_Mag40x/"
    exit 1
fi

# Check if specific dataset exists
case "$DATASET_KEY" in
    "mixed")
        DATASET_DIR="$DATASET_BASE_DIR/Task001_Combined_Mixed_Magnifications"
        ;;
    "mag5x")
        DATASET_DIR="$DATASET_BASE_DIR/Task005_Combined_Mag5x"
        ;;
    "mag10x")
        DATASET_DIR="$DATASET_BASE_DIR/Task010_Combined_Mag10x"
        ;;
    "mag20x")
        DATASET_DIR="$DATASET_BASE_DIR/Task020_Combined_Mag20x"
        ;;
    "mag40x")
        DATASET_DIR="$DATASET_BASE_DIR/Task040_Combined_Mag40x"
        ;;
    *)
        echo "❌ Unknown dataset key: $DATASET_KEY"
        echo "   Valid options: mixed, mag5x, mag10x, mag20x, mag40x"
        exit 1
        ;;
esac

if [ ! -d "$DATASET_DIR" ]; then
    echo "❌ Dataset directory does not exist: $DATASET_DIR"
    echo ""
    echo "📂 Available datasets:"
    ls -la "$DATASET_BASE_DIR" || echo "   Base directory not accessible"
    exit 1
fi

# Check if dataset.json exists
if [ ! -f "$DATASET_DIR/dataset.json" ]; then
    echo "❌ Dataset configuration not found: $DATASET_DIR/dataset.json"
    exit 1
fi

echo "✅ All paths exist and validation passed!"
echo ""

# =====================================================
# 🚀 RUN EVALUATION
# =====================================================

echo "🚀 Starting independent model evaluation..."
echo ""
echo "Command:"
echo "python independent_evaluator.py \\"
echo "    --experiment_path \"$EXPERIMENT_PATH\" \\"
echo "    --architecture \"$ARCHITECTURE\" \\"
echo "    --dataset_key \"$DATASET_KEY\" \\"
echo "    --dataset_base_dir \"$DATASET_BASE_DIR\" \\"
echo "    --output \"$OUTPUT_PATH\" \\"
echo "    --split \"$SPLIT\" \\"
echo "    --num_samples \"$NUM_SAMPLES\" \\"
echo "    --batch_size \"$BATCH_SIZE\""
echo ""
echo "============================================"

# --------- Directory Setup ----------
# Change to the independent_eval directory (absolute path)
INDEPENDENT_EVAL_DIR="/users/PAS2942/hikmat179/Code/_MLCRC/GlandSegmentation/GlandSegModels/nnUNet/independent_eval"
cd "$INDEPENDENT_EVAL_DIR"

echo "📁 Working directory: $(pwd)"
echo "🔍 Checking for independent_evaluator.py..."
if [ -f "independent_evaluator.py" ]; then
    echo "✅ Found independent_evaluator.py"
else
    echo "❌ independent_evaluator.py not found in $(pwd)"
    echo "📂 Current directory contents:"
    ls -la
    exit 1
fi
echo "============================================"

# Run the independent evaluator
python independent_evaluator.py \
    --experiment_path "$EXPERIMENT_PATH" \
    --architecture "$ARCHITECTURE" \
    --dataset_key "$DATASET_KEY" \
    --dataset_base_dir "$DATASET_BASE_DIR" \
    --output "$OUTPUT_PATH" \
    --split "$SPLIT" \
    --num_samples "$NUM_SAMPLES" \
    --batch_size "$BATCH_SIZE"

# =====================================================
# 🎉 RESULTS SUMMARY
# =====================================================

# Check results
if [ $? -eq 0 ]; then
    echo ""
    echo "============================================"
    echo "✅ Evaluation completed successfully!"
    echo "============================================"
    echo ""
    echo "📊 Results Summary:"
    echo "   🏗️  Architecture: $ARCHITECTURE"
    echo "   📊 Dataset: $DATASET_KEY"
    echo "   🔄 Evaluated Splits: $SPLIT"
    echo "   🖼️  Visualizations: $NUM_SAMPLES samples per split"
    echo ""
    echo "📁 Output files in: $OUTPUT_PATH/evaluation_${ARCHITECTURE}_${DATASET_KEY}_*"
    echo ""
    echo "📋 Available Results:"
    echo "   ✅ Excel Report: ${ARCHITECTURE}_${DATASET_KEY}_comprehensive_evaluation.xlsx"
    echo "   ✅ Summary CSV: ${ARCHITECTURE}_${DATASET_KEY}_evaluation_summary.csv"
    echo "   ✅ Visualizations: visualizations/ directory"
    echo ""
    echo "🔍 Quick file check:"

    # Find the most recent evaluation directory
    LATEST_EVAL_DIR=$(find "$OUTPUT_PATH" -maxdepth 1 -name "evaluation_${ARCHITECTURE}_${DATASET_KEY}_*" -type d | sort | tail -n 1)

    if [ -n "$LATEST_EVAL_DIR" ] && [ -d "$LATEST_EVAL_DIR" ]; then
        echo "   📁 Latest evaluation: $(basename "$LATEST_EVAL_DIR")"
        echo "   📊 Excel reports:"
        find "$LATEST_EVAL_DIR" -name "*.xlsx" | head -5 | sed 's/^/      /'
        echo "   📊 CSV files:"
        find "$LATEST_EVAL_DIR" -name "*.csv" | head -5 | sed 's/^/      /'
        echo "   🖼️  Visualization figures:"
        find "$LATEST_EVAL_DIR/visualizations" -name "*.png" | wc -l | sed 's/^/      /' | sed 's/$/ PNG files/'

        echo ""
        echo "💡 To view results:"
        echo "   cd \"$LATEST_EVAL_DIR\""
        echo "   ls -la"
        echo "   # Open Excel file to see comprehensive metrics"
        echo "   # View PNG files in visualizations/ for sample predictions"
    else
        echo "   ⚠️  Could not find evaluation output directory"
    fi

else
    echo ""
    echo "============================================"
    echo "❌ Evaluation failed!"
    echo "============================================"
    echo ""
    echo "💡 Troubleshooting tips:"
    echo "   1. Check that the experiment path contains a trained model"
    echo "   2. Verify that architecture matches the training configuration"
    echo "   3. Ensure dataset_key matches the training dataset"
    echo "   4. Check GPU memory if using large batch sizes"
    echo "   5. Review error messages above for specific issues"
    echo ""
    echo "🔧 Common fixes:"
    echo "   - Reduce BATCH_SIZE if out of memory"
    echo "   - Reduce NUM_SAMPLES if storage is limited"
    echo "   - Check paths for typos or permission issues"
    exit 1
fi

# =====================================================
# 📚 USAGE EXAMPLES AND TIPS
# =====================================================

echo ""
echo "============================================"
echo "📚 USAGE EXAMPLES FOR NEXT TIME"
echo "============================================"
echo ""
echo "🔧 Architecture Options:"
echo "   ARCHITECTURE='baseline_unet'  # Simple, fast, good baseline"
echo "   ARCHITECTURE='nnunet'         # Advanced, slower, best performance"
echo ""
echo "📊 Dataset Options:"
echo "   DATASET_KEY='mixed'   # All magnifications combined (~25k samples)"
echo "   DATASET_KEY='mag5x'   # 5x magnification only (~4.8k samples)"
echo "   DATASET_KEY='mag10x'  # 10x magnification only (~6.2k samples)"
echo "   DATASET_KEY='mag20x'  # 20x magnification only (~6k samples)"
echo "   DATASET_KEY='mag40x'  # 40x magnification only (~6k samples)"
echo ""
echo "🔄 Split Options:"
echo "   SPLIT='all'    # Evaluate train, val, and test (recommended)"
echo "   SPLIT='test'   # Evaluate test set only"
echo "   SPLIT='val'    # Evaluate validation set only"
echo "   SPLIT='train'  # Evaluate training set only"
echo ""
echo "🖼️  Visualization Options:"
echo "   NUM_SAMPLES=100  # 100 samples per split (as requested)"
echo "   NUM_SAMPLES=50   # Fewer samples for quicker evaluation"
echo "   NUM_SAMPLES=25   # Minimal samples for testing"
echo ""
echo "⚙️  Performance Options:"
echo "   BATCH_SIZE=4   # Standard batch size"
echo "   BATCH_SIZE=2   # Smaller batch for limited GPU memory"
echo "   BATCH_SIZE=8   # Larger batch for faster evaluation (if GPU allows)"
echo ""
echo "============================================"
echo "🎉 Independent Evaluation Job completed!"
echo "============================================"
echo "End Time: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "============================================"