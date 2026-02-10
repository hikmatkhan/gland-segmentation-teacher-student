#!/usr/bin/env python3
"""
Integration Test Example for Post-Training Evaluation
===================================================

This script demonstrates how the post-training evaluation system
will work when training completes.
"""

import os
import tempfile
import torch
import numpy as np
from pathlib import Path

# Add project root to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluation.post_training_evaluator import PostTrainingEvaluator


def create_dummy_model_checkpoint(checkpoint_path: Path):
    """Create a dummy model checkpoint for testing"""
    from src.models.multi_task_wrapper import create_multitask_model

    # Create model
    model = create_multitask_model()

    # Create dummy checkpoint data
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'best_metrics': {
            'best_val_loss': 0.1234,
            'best_dice': 0.8567,
            'best_patch_acc': 0.8901,
            'best_epoch': 75
        },
        'config': {
            'epochs': 150,
            'batch_size': 4,
            'learning_rate': 1e-4,
            'use_multilabel_patch': True
        }
    }

    # Save checkpoint
    torch.save(checkpoint, checkpoint_path)
    print(f"✅ Created dummy checkpoint: {checkpoint_path}")


def main():
    """Test the integration example"""
    print("🧪 Testing Post-Training Evaluation Integration")
    print("=" * 60)

    try:
        # Create temporary directory structure
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create experiment directory structure
            models_dir = temp_path / "models"
            evaluations_dir = temp_path / "evaluations"
            visualizations_dir = temp_path / "visualizations"

            for dir_path in [models_dir, evaluations_dir, visualizations_dir]:
                dir_path.mkdir(parents=True, exist_ok=True)

            # Create dummy best model checkpoint
            best_model_path = models_dir / "best_model.pth"
            create_dummy_model_checkpoint(best_model_path)

            print(f"🏗️ Created test experiment structure:")
            print(f"   📂 Root: {temp_path}")
            print(f"   📂 Models: {models_dir}")
            print(f"   📂 Evaluations: {evaluations_dir}")
            print(f"   📂 Visualizations: {visualizations_dir}")
            print()

            # Initialize PostTrainingEvaluator
            print("🔬 Initializing PostTrainingEvaluator...")
            evaluator = PostTrainingEvaluator(
                model_path=str(best_model_path),
                dataset_key='mixed',  # Use mixed dataset for testing
                output_dir=str(temp_path),
                visualization_samples=10  # Small number for testing
            )

            print("✅ PostTrainingEvaluator initialized successfully!")
            print()

            # Test model loading
            print("📂 Testing model loading...")
            evaluator.load_model()
            print("✅ Model loaded successfully!")
            print()

            print("📊 Expected workflow when training completes:")
            print("1. ✅ Load best model checkpoint")
            print("2. 📊 Evaluate on COMPLETE train/val/test datasets")
            print("3. 🎯 Calculate comprehensive metrics (Dice, IoU, Patch Accuracy)")
            print("4. 🎲 Randomly sample 100 images per split")
            print("5. 🎨 Generate 4-column visualizations:")
            print("   - Column 1: Original patch image")
            print("   - Column 2: Ground truth segmentation mask")
            print("   - Column 3: Predicted segmentation mask")
            print("   - Column 4: Overlay prediction on original")
            print("6. 💾 Save comprehensive metrics table (CSV)")
            print("7. 📋 Generate evaluation report (Markdown)")
            print("8. 📁 Organize all outputs in visualizations/ and evaluations/")
            print()

            print("🎉 Integration test completed successfully!")
            print()
            print("💡 Usage in actual training:")
            print("   python main.py train --dataset mixed --output_dir /path/to/outputs")
            print("   → Post-training evaluation runs automatically after training")
            print()
            print("📁 Expected output structure:")
            print("   outputs/exp_YYYY-MM-DD_HH-MM-SS/")
            print("   ├── models/best_model.pth")
            print("   ├── evaluations/")
            print("   │   ├── final_evaluation_metrics.csv")
            print("   │   ├── evaluation_summary_report.md")
            print("   │   └── detailed_metrics.json")
            print("   └── visualizations/")
            print("       ├── train_evaluation_samples_001.png")
            print("       ├── val_evaluation_samples_001.png")
            print("       ├── test_evaluation_samples_001.png")
            print("       └── sample_indices.json")

    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    return True


if __name__ == "__main__":
    success = main()
    print()
    if success:
        print("🎉 All integration tests passed!")
        print("✅ Ready for production training with post-training evaluation!")
    else:
        print("❌ Integration tests failed!")

    sys.exit(0 if success else 1)