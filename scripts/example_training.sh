#!/bin/bash
"""
Example Training Scripts for 4-Class nnU-Net Multi-Task Learning
==============================================================

This script provides example commands for training the 4-class nnU-Net
multi-task model on combined Warwick GlaS + OSU Makoto datasets.

Usage: bash scripts/example_training.sh
"""

# Navigate to the nnUNet directory
cd /users/PAS2942/hikmat179/Code/DLPath/CRC/GlandSegmentation/GlandSegModels/nnUNet

echo "🚀 4-Class nnU-Net Multi-Task Training Examples"
echo "=============================================="

# Test all components first
echo ""
echo "1. 🧪 Testing all components..."
python main.py demo

echo ""
echo "2. 📋 Available training commands:"
echo ""

# Basic training on mixed magnifications (recommended)
echo "   🎯 Standard Training (Mixed Magnifications):"
echo "   python main.py train --dataset mixed --epochs 150 --batch_size 4 --output_dir /path/to/outputs"
echo ""

# Enhanced training with stronger augmentation
echo "   🚀 Enhanced Training (Stronger Augmentation):"
echo "   python main.py train --dataset mixed --enhanced --epochs 150 --batch_size 4 --output_dir /path/to/outputs"
echo ""

# Magnification-specific training
echo "   🔍 Magnification-Specific Training:"
echo "   python main.py train --dataset mag20x --epochs 150 --batch_size 4 --output_dir /path/to/outputs"
echo "   python main.py train --dataset mag40x --epochs 150 --batch_size 4 --output_dir /path/to/outputs"
echo ""

# Research training with custom parameters
echo "   🔬 Research Training (Custom Configuration):"
echo "   python main.py train \\"
echo "       --dataset mixed \\"
echo "       --epochs 200 \\"
echo "       --batch_size 6 \\"
echo "       --learning_rate 5e-5 \\"
echo "       --enhanced \\"
echo "       --patience 40 \\"
echo "       --output_dir /path/to/outputs \\"
echo "       --experiment_name 'research_4class_v1'"
echo ""

# Quick training for testing
echo "   ⚡ Quick Test Training (Short Run):"
echo "   python main.py train --dataset mixed --epochs 5 --batch_size 2 --output_dir /path/to/outputs --experiment_name 'test_run'"
echo ""

# Evaluation examples
echo "   📊 Evaluation Examples:"
echo "   python main.py evaluate \\"
echo "       --model outputs/research_4class_v1/checkpoints/best_model.pth \\"
echo "       --dataset mixed \\"
echo "       --output outputs/evaluation_results \\"
echo "       --visualize"
echo ""

echo "💡 Tips:"
echo "   - Start with 'python main.py demo' to test all components"
echo "   - Use 'mixed' dataset for general-purpose models"
echo "   - Use '--enhanced' for stronger augmentation (longer training)"
echo "   - Monitor training with TensorBoard: tensorboard --logdir outputs/*/logs/tensorboard"
echo "   - Check outputs in: outputs/[experiment_name]/"
echo ""

echo "🔧 Hardware Recommendations:"
echo "   - GPU: ≥8GB VRAM (RTX 3080/4080 or better)"
echo "   - RAM: ≥32GB system memory"
echo "   - Storage: ≥50GB available space"
echo ""

echo "📚 For more details, see:"
echo "   - TRAINING_README.md - Comprehensive documentation"
echo "   - configs/paths_config.py - Configuration options"
echo "   - src/training/trainer.py - Training implementation"
echo ""

echo "✅ Ready to start training!"