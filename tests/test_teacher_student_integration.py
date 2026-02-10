#!/usr/bin/env python3
"""
Teacher-Student Integration Test
===============================

Comprehensive test of Teacher-Student UNet integration with the training pipeline.
Tests the full end-to-end integration including:
- Model creation and initialization
- Loss function integration
- Training loop integration
- Multi-task wrapper compatibility

Author: Claude Code - Generated for OSU CRC Research
Date: 2025-09-22
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import tempfile
from pathlib import Path
import os
import warnings

# Set environment variables for testing
os.environ['GLAND_DATASET_BASE'] = '/tmp/test_dataset'
os.environ['NNUNET_RESULTS'] = '/tmp/test_results'
os.environ['GLAND_TS_EMA_DECAY'] = '0.999'
os.environ['GLAND_TS_TEACHER_INIT_EPOCH'] = '5'
os.environ['GLAND_TS_MIN_ALPHA'] = '0.1'
os.environ['GLAND_TS_MAX_ALPHA'] = '1.0'
os.environ['GLAND_TS_CONSISTENCY_LOSS_TYPE'] = 'mse'
os.environ['GLAND_TS_CONSISTENCY_TEMPERATURE'] = '1.0'
os.environ['GLAND_TS_ENABLE_GLAND_CONSISTENCY'] = 'false'
os.environ['GLAND_TS_DEPTH'] = '3'
os.environ['GLAND_TS_INITIAL_CHANNELS'] = '32'

# Local imports - Add project root to path
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.teacher_student_unet import TeacherStudentUNet
from src.models.teacher_student_loss import TeacherStudentLoss
from src.models.multi_task_wrapper import MultiTaskWrapper
from src.models.model_factory import ModelFactory


def create_dummy_batch(batch_size=2, height=128, width=128, num_classes=4):
    """Create dummy training batch for testing"""
    # Input images
    images = torch.randn(batch_size, 3, height, width)

    # Segmentation targets
    seg_targets = torch.randint(0, num_classes, (batch_size, height, width))

    # Patch classification targets (multi-label)
    patch_targets = torch.randint(0, 2, (batch_size, num_classes)).float()

    # Gland classification targets
    gland_targets = torch.randint(0, num_classes, (batch_size,))

    targets = {
        'segmentation': seg_targets,
        'patch_labels': patch_targets,
        'gland_labels': gland_targets
    }

    return images, targets


def test_teacher_student_model_creation():
    """Test Teacher-Student UNet model creation"""
    print("🧪 Testing Teacher-Student Model Creation...")

    try:
        # Test direct creation
        model = TeacherStudentUNet(
            input_channels=3,
            num_classes=4,
            ema_decay=0.999,
            teacher_init_epoch=5,
            depth=3,
            initial_channels=32
        )

        print(f"✅ Direct creation: {model.get_total_parameters()['total']:,} parameters")

        # Test factory creation
        factory_model = ModelFactory.create_segmentation_model(
            architecture='teacher_student_unet',
            input_channels=3,
            num_classes=4,
            ema_decay=0.999,
            teacher_init_epoch=5,
            depth=3,
            initial_channels=32
        )

        print(f"✅ Factory creation: {factory_model.get_total_parameters()['total']:,} parameters")

        return True

    except Exception as e:
        print(f"❌ Model creation failed: {e}")
        return False


def test_teacher_student_forward_pass():
    """Test Teacher-Student forward pass modes"""
    print("\n🧪 Testing Teacher-Student Forward Pass...")

    try:
        model = TeacherStudentUNet(
            input_channels=3,
            num_classes=4,
            ema_decay=0.999,
            teacher_init_epoch=5,
            depth=3,
            initial_channels=32
        )

        batch_size = 2
        x = torch.randn(batch_size, 3, 128, 128)

        # Test student-only mode (before teacher initialization)
        model.train()
        outputs_student = model(x, mode="student_only")
        print(f"✅ Student-only output shape: {outputs_student['student'].shape}")

        # Initialize teacher
        model.initialize_teacher()

        # Test teacher-student mode
        outputs_ts = model(x, mode="teacher_student")
        print(f"✅ Teacher-student student shape: {outputs_ts['student'].shape}")
        print(f"✅ Teacher-student teacher shape: {outputs_ts['teacher'].shape}")

        # Test teacher-only mode
        model.eval()
        outputs_teacher = model(x, mode="teacher_only")
        print(f"✅ Teacher-only output shape: {outputs_teacher['teacher'].shape}")

        return True

    except Exception as e:
        print(f"❌ Forward pass failed: {e}")
        return False


def test_teacher_student_loss_integration():
    """Test Teacher-Student loss function integration"""
    print("\n🧪 Testing Teacher-Student Loss Integration...")

    try:
        # Create loss function
        loss_fn = TeacherStudentLoss(
            total_epochs=10,
            warmup_epochs=3,
            min_alpha=0.1,
            max_alpha=1.0,
            consistency_loss_config={
                'loss_type': 'mse',
                'enable_gland_consistency': False
            }
        )

        # Create dummy outputs and targets
        student_outputs = {
            'segmentation': torch.randn(2, 4, 128, 128),
            'patch_classification': torch.randn(2, 4),
            'gland_classification': torch.randn(2, 4)
        }

        teacher_outputs = {
            'segmentation': torch.randn(2, 4, 128, 128),
            'patch_classification': torch.randn(2, 4),
            'gland_classification': torch.randn(2, 4)
        }

        targets = {
            'segmentation': torch.randint(0, 4, (2, 128, 128)),
            'patch_labels': torch.randint(0, 2, (2, 4)).float(),
            'gland_labels': torch.randint(0, 4, (2,))
        }

        # Test warmup phase (epoch 1)
        warmup_loss = loss_fn(student_outputs, None, targets, current_epoch=1)
        print(f"✅ Warmup loss: {warmup_loss['total_loss'].item():.4f}")
        print(f"   Phase: {warmup_loss['phase']}")
        print(f"   Alpha: {warmup_loss['alpha'].item():.4f}")

        # Test teacher-student phase (epoch 5)
        ts_loss = loss_fn(student_outputs, teacher_outputs, targets, current_epoch=5)
        print(f"✅ Teacher-Student loss: {ts_loss['total_loss'].item():.4f}")
        print(f"   Phase: {ts_loss['phase']}")
        print(f"   Alpha: {ts_loss['alpha'].item():.4f}")
        print(f"   Supervised: {ts_loss['supervised_loss'].item():.4f}")
        print(f"   Consistency: {ts_loss['consistency_loss'].item():.4f}")

        return True

    except Exception as e:
        print(f"❌ Loss integration failed: {e}")
        return False


def test_multi_task_wrapper_integration():
    """Test Teacher-Student UNet with MultiTaskWrapper"""
    print("\n🧪 Testing Multi-Task Wrapper Integration...")

    try:
        # Create multi-task wrapper with Teacher-Student UNet
        wrapper = MultiTaskWrapper(
            architecture='teacher_student_unet',
            input_channels=3,
            num_seg_classes=4,
            enable_classification=True,
            ema_decay=0.999,
            teacher_init_epoch=5,
            depth=3,
            initial_channels=32
        )

        batch_size = 2
        x = torch.randn(batch_size, 3, 128, 128)

        # Test before teacher initialization
        wrapper.train()
        outputs_before = wrapper(x, mode="student_only")
        print(f"✅ Before teacher init - outputs: {list(outputs_before.keys())}")
        print(f"   Segmentation: {outputs_before['segmentation'].shape}")
        print(f"   Patch classification: {outputs_before['patch_classification'].shape}")
        print(f"   Gland classification: {outputs_before['gland_classification'].shape}")

        # Initialize teacher
        wrapper.segmentation_model.initialize_teacher()

        # Test after teacher initialization
        outputs_after = wrapper(x, mode="teacher_student")
        print(f"✅ After teacher init - outputs: {list(outputs_after.keys())}")
        print(f"   Student segmentation: {outputs_after['student'].shape}")
        print(f"   Teacher segmentation: {outputs_after['teacher'].shape}")
        print(f"   Patch classification: {outputs_after['patch_classification'].shape}")

        return True

    except Exception as e:
        print(f"❌ Multi-task wrapper integration failed: {e}")
        return False


def test_ema_update_mechanism():
    """Test EMA update mechanism"""
    print("\n🧪 Testing EMA Update Mechanism...")

    try:
        model = TeacherStudentUNet(
            input_channels=3,
            num_classes=4,
            ema_decay=0.9,  # Lower decay for visible changes
            teacher_init_epoch=1,
            depth=3,
            initial_channels=32
        )

        # Initialize teacher
        model.initialize_teacher()

        # Get initial teacher parameters
        initial_teacher_params = [p.clone() for p in model.teacher.parameters()]

        # Simulate training step - modify student parameters
        with torch.no_grad():
            for p in model.student.parameters():
                p.add_(torch.randn_like(p) * 0.1)  # Add noise to student

        # Update teacher with EMA
        model.update_teacher_ema()

        # Check if teacher parameters changed
        current_teacher_params = [p.clone() for p in model.teacher.parameters()]

        param_changes = []
        for initial, current in zip(initial_teacher_params, current_teacher_params):
            change = (current - initial).abs().max().item()
            param_changes.append(change)

        max_change = max(param_changes)
        print(f"✅ EMA update completed")
        print(f"   Maximum parameter change: {max_change:.6f}")
        print(f"   EMA decay: {model.ema_decay}")

        # Verify change is reasonable (should be small due to EMA)
        assert max_change > 0, "No parameter changes detected"
        assert max_change < 0.1, "Parameter changes too large"

        return True

    except Exception as e:
        print(f"❌ EMA update failed: {e}")
        return False


def test_training_simulation():
    """Simulate a mini training loop"""
    print("\n🧪 Testing Training Simulation...")

    try:
        # Create model and loss
        model = TeacherStudentUNet(
            input_channels=3,
            num_classes=4,
            ema_decay=0.999,
            teacher_init_epoch=2,
            depth=3,
            initial_channels=32
        )

        loss_fn = TeacherStudentLoss(
            total_epochs=5,
            warmup_epochs=2,
            min_alpha=0.1,
            max_alpha=1.0
        )

        optimizer = optim.Adam(model.student.parameters(), lr=0.001)

        print("📊 Mini Training Loop:")

        for epoch in range(5):
            model.train()

            # Create batch
            x, targets = create_dummy_batch(batch_size=2, height=64, width=64)

            # Forward pass
            if not model.teacher_initialized and model.should_initialize_teacher(epoch):
                model.initialize_teacher()
                print(f"   🚀 Teacher initialized at epoch {epoch}")

            # Get model outputs
            if model.teacher_initialized:
                outputs = model(x, mode="teacher_student")
                student_outputs = {'segmentation': outputs['student']}
                teacher_outputs = {'segmentation': outputs['teacher']}
            else:
                outputs = model(x, mode="student_only")
                student_outputs = {'segmentation': outputs['student']}
                teacher_outputs = None

            # Compute loss
            loss_dict = loss_fn(student_outputs, teacher_outputs, targets, epoch)
            total_loss = loss_dict['total_loss']

            # Backward pass
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            # Update teacher EMA
            if model.teacher_initialized:
                model.update_teacher_ema()

            print(f"   Epoch {epoch}: Loss={total_loss.item():.4f}, "
                  f"Alpha={loss_dict['alpha'].item():.3f}, "
                  f"Phase={loss_dict['phase']}")

        print("✅ Training simulation completed successfully")
        return True

    except Exception as e:
        print(f"❌ Training simulation failed: {e}")
        return False


def main():
    """Run all Teacher-Student integration tests"""
    print("🎓 Teacher-Student UNet Integration Tests")
    print("=" * 50)

    tests = [
        test_teacher_student_model_creation,
        test_teacher_student_forward_pass,
        test_teacher_student_loss_integration,
        test_multi_task_wrapper_integration,
        test_ema_update_mechanism,
        test_training_simulation
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"❌ Test {test.__name__} crashed: {e}")
            failed += 1

    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed} passed, {failed} failed")

    if failed == 0:
        print("🎉 All Teacher-Student UNet integration tests passed!")
        return True
    else:
        print("❌ Some tests failed. Please review the issues above.")
        return False


if __name__ == "__main__":
    main()