#!/usr/bin/env python3
"""
Quick Test Script for 4-Class nnU-Net Multi-Task Implementation
==============================================================

This script demonstrates the key components of the 4-class nnU-Net
multi-task implementation following the GlaS_MultiTask approach.

Usage: python scripts/quick_test.py
"""

import sys
import torch
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

def test_model_creation():
    """Test model creation and architecture"""
    print("🏗️ Testing model creation...")

    from src.models.multi_task_wrapper import create_multitask_model

    # Create 4-class multi-task model
    model = create_multitask_model(
        input_channels=3,
        num_seg_classes=4,
        enable_classification=True
    )

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"✅ Model created with {total_params:,} parameters")

    return model

def test_loss_functions():
    """Test multi-task loss functions"""
    print("🔧 Testing loss functions...")

    from src.models.loss_functions import MultiTaskLoss

    # Create multi-label loss function
    loss_fn = MultiTaskLoss(
        use_multilabel_patch=True,
        use_adaptive_weighting=True,
        dice_weight=0.5,
        ce_weight=0.5
    )

    print("✅ Multi-task loss function created with multi-label support")
    return loss_fn

def test_forward_pass(model):
    """Test forward pass with realistic dimensions"""
    print("🧪 Testing forward pass...")

    # Create dummy input (batch_size=2, channels=3, height=256, width=256)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    dummy_input = torch.randn(2, 3, 256, 256).to(device)

    with torch.no_grad():
        outputs = model(dummy_input)

    print(f"✅ Forward pass successful:")
    print(f"   📊 Segmentation output: {outputs['segmentation'].shape}")
    print(f"   🏷️ Patch classification: {outputs['patch_classification'].shape}")
    print(f"   🔍 Gland classification: {outputs['gland_classification'].shape}")
    print(f"   🎯 Deep supervision: {'deep_supervision' in outputs}")

    return outputs

def test_multi_label_generation():
    """Test multi-label patch generation"""
    print("🎨 Testing multi-label generation...")

    from src.models.projection_heads import create_multilabel_patch_labels_from_segmentation

    # Create sample 4-class segmentation mask
    segmentation_mask = torch.randint(0, 4, (2, 256, 256))

    # Generate multi-label patch labels
    multilabel_patches = create_multilabel_patch_labels_from_segmentation(
        segmentation_mask,
        min_pixels_threshold=50
    )

    print(f"✅ Multi-label generation successful:")
    print(f"   📥 Input segmentation: {segmentation_mask.shape}")
    print(f"   📤 Output multi-label: {multilabel_patches.shape}")
    print(f"   🏷️ Example labels: {multilabel_patches[0].tolist()}")

    return multilabel_patches

def test_configuration():
    """Test configuration system"""
    print("⚙️ Testing configuration...")

    from configs.paths_config import (
        DEFAULT_CONFIG, list_available_datasets,
        get_dataset_path, validate_dataset_path
    )

    # Test configuration
    print(f"   🔧 Default epochs: {DEFAULT_CONFIG['epochs']}")
    print(f"   📦 Default batch size: {DEFAULT_CONFIG['batch_size']}")
    print(f"   🎯 Segmentation classes: {DEFAULT_CONFIG['num_seg_classes']}")
    print(f"   🏷️ Patch classes: {DEFAULT_CONFIG['num_patch_classes']}")
    print(f"   🔍 Gland classes: {DEFAULT_CONFIG['num_gland_classes']}")
    print(f"   🎨 Multi-label patches: {DEFAULT_CONFIG['use_multilabel_patch']} (CRITICAL!)")

    # Test dataset availability
    available = list_available_datasets()
    print(f"   📊 Available datasets: {list(available.keys())}")

    # Test dataset path validation
    if available:
        test_dataset = 'mixed'
        if test_dataset in available:
            path = get_dataset_path(test_dataset)
            is_valid = validate_dataset_path(path)
            print(f"   ✅ Dataset '{test_dataset}' path validation: {is_valid}")

    print("✅ Configuration system working")

def main():
    """Main test function"""
    print("🚀 4-Class nnU-Net Multi-Task Implementation Test")
    print("=" * 60)

    try:
        # Test configuration
        test_configuration()
        print()

        # Test model creation
        model = test_model_creation()
        print()

        # Test loss functions
        loss_fn = test_loss_functions()
        print()

        # Test forward pass
        outputs = test_forward_pass(model)
        print()

        # Test multi-label generation
        multilabel_patches = test_multi_label_generation()
        print()

        print("🎉 All tests passed!")
        print("✅ Implementation ready for training")
        print("📝 Use: python main.py train --dataset mixed --epochs 150")

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)