#!/bin/bash
#SBATCH --account=PAS2942
#SBATCH --job-name=2ndmag10x
#SBATCH --time=164:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-gpu=32
#SBATCH --mem=100gb
#SBATCH --output=/fs/scratch/PAS2942/Datasets/CRC/Datasets/OSU/Gland_Datasets/_RESULTS_PPR/2ndRuns/logs_resumed_interupted/resume_%j.out
#SBATCH --error=/fs/scratch/PAS2942/Datasets/CRC/Datasets/OSU/Gland_Datasets/_RESULTS_PPR/2ndRuns/logs_resumed_interupted/resume_%j.err
#SBATCH --cluster=ascend

# ==============================================================================
# Resume Training Script for Gland Segmentation
# ==============================================================================
# This script resumes training from an interrupted experiment.
#
# Usage:
#   sbatch resume_nnunet_training.sh
#
# Configuration:
#   Set EXPERIMENT_DIR to the path of the interrupted experiment
# ==============================================================================
#PAS2942
#PAS3194

# ============================
# USER CONFIGURATION
# ============================
# /users/PAS2942/hikmat179/Code/_MLCRC/GlandSegmentation/GlandSegModels/nnUNet
# REQUIRED: Path to experiment directory to resume from
EXPERIMENT_DIR="/fs/scratch/PAS2942/Datasets/CRC/Datasets/OSU/Gland_Datasets/_RESULTS_PPR/2ndRuns/teacher_student_nnunet_mag10x_enhanced_20251026_181808"

# OPTIONAL: Resume from best checkpoint instead of latest (default: false)
USE_BEST=false

# ============================
# ENVIRONMENT SETUP
# ============================
# --------- Environment ----------
module load miniconda3/24.1.2-py310
module load cuda/11.8.0
conda activate trident

# ============================
# VALIDATION
# ============================
echo "========================================"
echo "Resume Training Configuration"
echo "========================================"
echo "Experiment Directory: ${EXPERIMENT_DIR}"
echo "Use Best Checkpoint: ${USE_BEST}"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURM_NODELIST}"
echo "Start Time: $(date)"
echo "========================================"

# Check if experiment directory exists
if [ ! -d "${EXPERIMENT_DIR}" ]; then
    echo "ERROR: Experiment directory not found: ${EXPERIMENT_DIR}"
    exit 1
fi

# Check if training_config.json exists
if [ ! -f "${EXPERIMENT_DIR}/training_config.json" ]; then
    echo "ERROR: training_config.json not found in experiment directory"
    exit 1
fi

# Check if models directory exists
if [ ! -d "${EXPERIMENT_DIR}/models" ]; then
    echo "ERROR: models directory not found in experiment directory"
    exit 1
fi

echo "✓ Validation passed"
echo ""

# ============================
# EXTRACT AND EXPORT ENVIRONMENT VARIABLES FROM CONFIG
# ============================

echo "📋 Extracting environment variables from training_config.json..."

# Read the comprehensive_paths from training_config.json and export environment variables
CONFIG_FILE="${EXPERIMENT_DIR}/training_config.json"

# Use Python to extract and export environment variables
eval $(python - <<PYTHON_SCRIPT
import json
import sys

config_file = "${CONFIG_FILE}"

try:
    with open(config_file, 'r') as f:
        config = json.load(f)

    # Get the comprehensive_paths section
    comprehensive_paths = config.get('comprehensive_paths', {})
    env_vars = comprehensive_paths.get('environment_variables', {})

    # Export the essential environment variables
    if 'GLAND_DATASET_BASE' in env_vars:
        print(f"export GLAND_DATASET_BASE='{env_vars['GLAND_DATASET_BASE']}'")
    if 'GLAND_OUTPUT_DIR' in env_vars:
        print(f"export GLAND_OUTPUT_DIR='{env_vars['GLAND_OUTPUT_DIR']}'")
    if 'NNUNET_PREPROCESSED' in env_vars:
        print(f"export NNUNET_PREPROCESSED='{env_vars['NNUNET_PREPROCESSED']}'")
    if 'NNUNET_RESULTS' in env_vars:
        print(f"export NNUNET_RESULTS='{env_vars['NNUNET_RESULTS']}'")
    if 'GLAND_TEMP_DIR' in env_vars:
        print(f"export GLAND_TEMP_DIR='{env_vars['GLAND_TEMP_DIR']}'")
    else:
        # Create a new temp dir if not in config (for older experiments)
        import os
        print(f"export GLAND_TEMP_DIR='/tmp/nnunet_resume_{os.getpid()}'")

except Exception as e:
    print(f"# Error extracting environment variables: {e}", file=sys.stderr)
    sys.exit(1)
PYTHON_SCRIPT
)

echo "✓ Environment variables exported:"
echo "  GLAND_DATASET_BASE: ${GLAND_DATASET_BASE}"
echo "  GLAND_OUTPUT_DIR: ${GLAND_OUTPUT_DIR}"
echo "  NNUNET_PREPROCESSED: ${NNUNET_PREPROCESSED}"
echo "  NNUNET_RESULTS: ${NNUNET_RESULTS}"
echo "  GLAND_TEMP_DIR: ${GLAND_TEMP_DIR}"
echo ""

# ============================
# RESUME TRAINING
# ============================

# Build resume command
RESUME_CMD="python main.py resume --experiment_dir ${EXPERIMENT_DIR}"

if [ "${USE_BEST}" = true ]; then
    RESUME_CMD="${RESUME_CMD} --use_best"
fi

echo "Executing: ${RESUME_CMD}"
echo ""

# Execute resume
${RESUME_CMD}

# ============================
# COMPLETION
# ============================

echo ""
echo "========================================"
echo "Resume Training Completed"
echo "End Time: $(date)"
echo "========================================"
