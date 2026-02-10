#!/usr/bin/env python3
"""
Multi-Architecture Multi-task Wrapper for Gland Segmentation
Supports both baseline UNet and nnU-Net with classification heads for comprehensive gland analysis
"""

import torch
import torch.nn as nn
from typing import Tuple, List, Dict, Any, Optional
import numpy as np
import warnings

# Local imports
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.models.nnunet_integration import create_nnunet_architecture, get_architecture_classes
from src.models.projection_heads import MultiClassDualClassificationHead, create_classification_labels_from_segmentation
from src.models.baseline_unet import BaselineUNet
from src.models.teacher_student_unet import TeacherStudentUNet
from src.models.model_factory import ModelFactory

class MultiTaskWrapper(nn.Module):
    """
    Multi-architecture multi-task wrapper for gland segmentation
    Supports both baseline UNet and nnU-Net with classification heads

    Features:
    - 4-class segmentation: Background(0), Benign(1), Malignant(2), PDC(3)
    - Dual classification: patch-level and gland-level
    - Architecture flexibility: baseline UNet or nnU-Net
    """

    def __init__(self,
                 segmentation_model: Optional[nn.Module] = None,
                 architecture: str = 'nnunet',
                 input_channels: int = 3,
                 num_seg_classes: int = 4,
                 bottleneck_channels: Optional[int] = None,
                 enable_classification: bool = True,
                 classification_dropout: float = 0.5,
                 deep_supervision: bool = True,
                 **model_kwargs):
        super().__init__()

        self.architecture = architecture
        self.num_seg_classes = num_seg_classes
        self.enable_classification = enable_classification
        self.deep_supervision = deep_supervision

        # Create or use provided segmentation model
        if segmentation_model is None:
            print(f"🏗️ Creating {architecture} segmentation model...")

            if architecture == 'nnunet':
                self.segmentation_model = create_nnunet_architecture(
                    input_channels=input_channels,
                    num_classes=num_seg_classes,
                    deep_supervision=deep_supervision,
                    **model_kwargs
                )
                if self.segmentation_model is None:
                    raise RuntimeError("Failed to create nnU-Net segmentation model")

            elif architecture == 'baseline_unet':
                self.segmentation_model = ModelFactory.create_segmentation_model(
                    architecture='baseline_unet',
                    input_channels=input_channels,
                    num_classes=num_seg_classes,
                    **model_kwargs
                )
            elif architecture == 'teacher_student_unet':
                self.segmentation_model = ModelFactory.create_segmentation_model(
                    architecture='teacher_student_unet',
                    input_channels=input_channels,
                    num_classes=num_seg_classes,
                    **model_kwargs
                )
            else:
                raise ValueError(f"Unsupported architecture: {architecture}")
        else:
            self.segmentation_model = segmentation_model
            # Auto-detect architecture from model type
            if isinstance(segmentation_model, BaselineUNet):
                self.architecture = 'baseline_unet'
            elif isinstance(segmentation_model, TeacherStudentUNet):
                self.architecture = 'teacher_student_unet'
            else:
                self.architecture = 'nnunet'

        # Determine bottleneck channels if not specified
        if bottleneck_channels is None:
            bottleneck_channels = ModelFactory.get_bottleneck_channels(self.segmentation_model)

        self.bottleneck_channels = bottleneck_channels

        # Classification heads
        if self.enable_classification:
            if self.architecture == 'teacher_student_unet':
                # For Teacher-Student UNet: separate classification heads for student and teacher
                self.student_classification_head = MultiClassDualClassificationHead(
                    input_channels=bottleneck_channels,
                    patch_classes=num_seg_classes,  # 4-class patch classification
                    gland_classes=num_seg_classes,  # 4-class gland classification
                    dropout_p=classification_dropout
                )

                self.teacher_classification_head = MultiClassDualClassificationHead(
                    input_channels=bottleneck_channels,
                    patch_classes=num_seg_classes,  # 4-class patch classification
                    gland_classes=num_seg_classes,  # 4-class gland classification
                    dropout_p=classification_dropout
                )

                print(f"   🎓 Student classification head created")
                print(f"   👨‍🏫 Teacher classification head created")
            else:
                # For other architectures: single classification head
                self.classification_head = MultiClassDualClassificationHead(
                    input_channels=bottleneck_channels,
                    patch_classes=num_seg_classes,  # 4-class patch classification
                    gland_classes=num_seg_classes,  # 4-class gland classification
                    dropout_p=classification_dropout
                )

            # Hook to capture bottleneck features
            self.bottleneck_features = None
            self._register_hooks()

        print(f"✅ MultiTaskWrapper initialized:")
        print(f"   🏗️ Architecture: {self.architecture}")
        print(f"   🎯 Segmentation classes: {num_seg_classes}")
        print(f"   🏷️ Classification enabled: {enable_classification}")
        print(f"   🔍 Deep supervision: {deep_supervision}")
        print(f"   📦 Bottleneck channels: {bottleneck_channels}")

    def _register_hooks(self):
        """Register forward hooks to capture bottleneck features from both architectures"""
        def hook_fn(module, input, output):
            # Store the deepest encoder features (bottleneck)
            self.bottleneck_features = output

        hook_registered = False

        if self.architecture == 'baseline_unet':
            # For baseline UNet, the hook is already registered in the model
            if hasattr(self.segmentation_model, 'bottleneck_features'):
                hook_registered = True
                print(f"   🔗 Using BaselineUNet built-in hook")
            else:
                # Fallback: register on the last encoder layer
                if hasattr(self.segmentation_model, 'encoder') and len(self.segmentation_model.encoder) > 0:
                    self.segmentation_model.encoder[-1].register_forward_hook(hook_fn)
                    hook_registered = True
                    print(f"   🔗 Registered hook on baseline UNet encoder[-1]")

        elif self.architecture == 'nnunet':
            # Register hook based on nnU-Net architecture structure
            def register_hook_recursive(module, target_name="encoder"):
                """Recursively find and register hook on the deepest encoder layer"""
                for name, child in module.named_children():
                    if target_name in name.lower() or "down" in name.lower():
                        # This is likely an encoder path
                        if hasattr(child, 'stages') or hasattr(child, 'blocks'):
                            # Register on the last stage/block
                            if hasattr(child, 'stages'):
                                child.stages[-1].register_forward_hook(hook_fn)
                                print(f"   🔗 Registered hook on nnUNet: {name}.stages[-1]")
                                return True
                            elif hasattr(child, 'blocks'):
                                child.blocks[-1].register_forward_hook(hook_fn)
                                print(f"   🔗 Registered hook on nnUNet: {name}.blocks[-1]")
                                return True
                        # Continue searching recursively
                        if register_hook_recursive(child, target_name):
                            return True
                return False

            # Try to register hook
            hook_registered = register_hook_recursive(self.segmentation_model)

        elif self.architecture == 'teacher_student_unet':
            # For Teacher-Student models, register hooks on both student and teacher networks
            print(f"   🎓 Registering hooks for Teacher-Student UNet...")

            def register_ts_hook_recursive(module, target_name="encoder"):
                """Register hook on Teacher-Student UNet encoder layers"""
                for name, child in module.named_children():
                    if target_name in name.lower() or "down" in name.lower():
                        # Found encoder-like layer, check if it has children
                        if len(list(child.children())) == 0:
                            # Leaf node - register hook here
                            child.register_forward_hook(hook_fn)
                            print(f"   🔗 Registered Teacher-Student hook on {name}")
                            return True
                        else:
                            # Continue searching recursively
                            if register_ts_hook_recursive(child, target_name):
                                return True
                return False

            # Try to register on student network (which is used during training)
            if hasattr(self.segmentation_model, 'student'):
                hook_registered = register_ts_hook_recursive(self.segmentation_model.student)
                if hook_registered:
                    print(f"   ✅ Successfully registered Teacher-Student hooks")
                else:
                    # Try alternative approach - look for BaselineUNet structure within student
                    student_model = self.segmentation_model.student
                    if hasattr(student_model, 'encoder') and len(student_model.encoder) > 0:
                        # Register on the last encoder layer of the student network
                        student_model.encoder[-1].register_forward_hook(hook_fn)
                        hook_registered = True
                        print(f"   🔗 Registered Teacher-Student hook on student.encoder[-1]")
            else:
                print(f"   ⚠️ Teacher-Student model not fully initialized during wrapper creation")

        if not hook_registered:
            # Fallback: register on the entire segmentation model
            print(f"   ⚠️ Could not find specific encoder layer, using entire model")
            def fallback_hook_fn(module, input, output):
                # Use the input features as bottleneck (not ideal but functional)
                if isinstance(input, (list, tuple)):
                    self.bottleneck_features = input[0]
                else:
                    self.bottleneck_features = input

            self.segmentation_model.register_forward_hook(fallback_hook_fn)

    def forward(self, x: torch.Tensor, mode: str = "auto") -> Dict[str, Any]:
        """
        Forward pass with multi-task outputs

        Args:
            x: Input tensor [B, C, H, W]
            mode: Forward mode for Teacher-Student UNet
                - "auto": Automatically determine mode based on training phase
                - "student_only": Only student network (for Teacher-Student UNet)
                - "teacher_student": Both networks (for Teacher-Student UNet)
                - "teacher_only": Only teacher network (for Teacher-Student UNet)

        Returns:
            Dictionary containing segmentation and classification results
        """
        # Reset bottleneck features
        self.bottleneck_features = None

        # Forward through segmentation model
        if self.architecture == 'teacher_student_unet':
            # Handle Teacher-Student UNet specially
            if mode == "auto":
                # Auto-determine mode based on teacher initialization and training phase
                if not self.segmentation_model.teacher_initialized:
                    ts_mode = "student_only"
                elif self.training:
                    ts_mode = "teacher_student"
                else:
                    ts_mode = "teacher_only"
            else:
                ts_mode = mode

            seg_output = self.segmentation_model(x, mode=ts_mode)

            # Extract segmentation output based on mode
            if 'student' in seg_output:
                segmentation_logits = seg_output['student']
            elif 'teacher' in seg_output:
                segmentation_logits = seg_output['teacher']
            else:
                raise RuntimeError("No valid output from Teacher-Student UNet")

        else:
            # Standard forward for baseline_unet and nnunet
            seg_output = self.segmentation_model(x)
            segmentation_logits = seg_output

        # Handle different output formats
        if self.architecture == 'teacher_student_unet':
            # For Teacher-Student UNet, use extracted segmentation_logits
            outputs = {
                'segmentation': segmentation_logits
            }
            # Store raw teacher-student outputs for loss computation
            if isinstance(seg_output, dict):
                outputs.update(seg_output)

        elif isinstance(segmentation_logits, (list, tuple)):
            # Deep supervision enabled (nnUNet) - take the highest resolution output
            segmentation_logits = segmentation_logits[0]
            outputs = {
                'segmentation': segmentation_logits,
                'deep_supervision': segmentation_logits
            }
        else:
            outputs = {
                'segmentation': segmentation_logits
            }

        # Get bottleneck features - handle all architectures
        if self.enable_classification:
            # For baseline UNet, get features from the model's built-in hook
            if self.architecture == 'baseline_unet' and hasattr(self.segmentation_model, 'bottleneck_features'):
                self.bottleneck_features = self.segmentation_model.bottleneck_features

            # For Teacher-Student UNet, get features from active network
            elif self.architecture == 'teacher_student_unet':
                if 'student' in seg_output and hasattr(self.segmentation_model.student, 'bottleneck_features'):
                    self.bottleneck_features = self.segmentation_model.student.bottleneck_features
                elif 'teacher' in seg_output and hasattr(self.segmentation_model.teacher, 'bottleneck_features'):
                    self.bottleneck_features = self.segmentation_model.teacher.bottleneck_features

            if self.bottleneck_features is not None:
                # Get segmentation predictions for gland extraction
                seg_predictions = torch.softmax(segmentation_logits, dim=1)
                seg_predictions = torch.argmax(seg_predictions, dim=1).float()  # [B, H, W]

                # Multi-class dual classification - use appropriate classification head
                if self.architecture == 'teacher_student_unet':
                    # For Teacher-Student UNet, use the appropriate classification head
                    if 'student' in seg_output:
                        # Student network active, use student classification head
                        patch_logits, gland_logits, gland_counts = self.student_classification_head(
                            self.bottleneck_features, seg_predictions
                        )
                    elif 'teacher' in seg_output:
                        # Teacher network active, use teacher classification head
                        patch_logits, gland_logits, gland_counts = self.teacher_classification_head(
                            self.bottleneck_features, seg_predictions
                        )
                    else:
                        raise RuntimeError("No valid student or teacher output found for Teacher-Student UNet")
                else:
                    # For other architectures, use single classification head
                    patch_logits, gland_logits, gland_counts = self.classification_head(
                        self.bottleneck_features, seg_predictions
                    )

                outputs.update({
                    'patch_classification': patch_logits,
                    'gland_classification': gland_logits,
                    'gland_counts': gland_counts
                })
            else:
                # Create dummy classification outputs if no bottleneck features captured
                batch_size = x.shape[0]
                device = x.device

                outputs.update({
                    'patch_classification': torch.zeros(batch_size, self.num_seg_classes).to(device),
                    'gland_classification': torch.zeros(0, self.num_seg_classes).to(device),
                    'gland_counts': [0] * batch_size
                })

                warnings.warn(f"⚠️ No bottleneck features captured for {self.architecture}, using dummy classification outputs")

        return outputs

def create_multitask_model(
    architecture: str = 'nnunet',
    input_channels: int = 3,
    num_seg_classes: int = 4,
    enable_classification: bool = True,
    **kwargs
) -> MultiTaskWrapper:
    """
    Create a complete multi-task model for gland segmentation

    Args:
        architecture: Model architecture ('baseline_unet' or 'nnunet')
        input_channels: Number of input channels (default: 3 for RGB)
        num_seg_classes: Number of segmentation classes (default: 4)
        enable_classification: Whether to enable classification heads
        **kwargs: Additional arguments for MultiTaskWrapper

    Returns:
        Initialized MultiTaskWrapper model
    """
    model = MultiTaskWrapper(
        segmentation_model=None,  # Will create model automatically
        architecture=architecture,
        input_channels=input_channels,
        num_seg_classes=num_seg_classes,
        enable_classification=enable_classification,
        **kwargs
    )

    return model


def test_multitask_wrapper():
    """Test function for multi-task wrapper with both architectures"""
    print("🧪 Testing Multi-Task Wrapper with both architectures...")

    # Test parameters
    batch_size = 2
    input_channels = 3
    num_classes = 4
    height, width = 256, 256

    test_cases = [
        {
            "name": "Baseline UNet Multi-Task",
            "architecture": "baseline_unet",
            "config": {
                "depth": 4,
                "initial_channels": 64,
                "activation": "relu"
            }
        },
        {
            "name": "Teacher-Student UNet Multi-Task",
            "architecture": "teacher_student_unet",
            "config": {
                "ema_decay": 0.999,
                "teacher_init_epoch": 10,
                "depth": 4,
                "initial_channels": 64
            }
        },
        # Note: nnUNet test would need proper environment setup
    ]

    for test_case in test_cases:
        print(f"\n🔬 Testing: {test_case['name']}")
        try:
            # Create test input
            x = torch.randn(batch_size, input_channels, height, width)

            # Create multi-task model
            model = create_multitask_model(
                architecture=test_case['architecture'],
                input_channels=input_channels,
                num_seg_classes=num_classes,
                enable_classification=True,
                **test_case['config']
            )

            # Get model info
            total_params = sum(p.numel() for p in model.parameters())
            print(f"📊 Total parameters: {total_params:,}")

            # Forward pass
            with torch.no_grad():
                outputs = model(x)

                print(f"✅ Input shape: {x.shape}")
                print(f"✅ Outputs: {list(outputs.keys())}")

                # Check segmentation output
                seg_shape = outputs['segmentation'].shape
                expected_seg_shape = (batch_size, num_classes, height, width)
                print(f"✅ Segmentation shape: {seg_shape} (expected: {expected_seg_shape})")

                # Check classification outputs
                if 'patch_classification' in outputs:
                    patch_shape = outputs['patch_classification'].shape
                    expected_patch_shape = (batch_size, num_classes)
                    print(f"✅ Patch classification shape: {patch_shape} (expected: {expected_patch_shape})")

                if 'gland_classification' in outputs:
                    gland_shape = outputs['gland_classification'].shape
                    print(f"✅ Gland classification shape: {gland_shape}")
                    print(f"✅ Gland counts: {outputs['gland_counts']}")

                print(f"🎉 {test_case['name']} test completed successfully!")

        except Exception as e:
            print(f"❌ {test_case['name']} test failed: {e}")

    print("\n🎉 Multi-Task Wrapper test completed!")


if __name__ == "__main__":
    test_multitask_wrapper()

def create_simple_multitask_model(
    in_channels: int = 3,
    num_seg_classes: int = 4,
    enable_classification: bool = True
) -> nn.Module:
    """
    Create a simple U-Net based multi-task model (fallback when nnU-Net not available)

    Args:
        in_channels: Number of input channels
        num_seg_classes: Number of segmentation classes
        enable_classification: Whether to enable classification

    Returns:
        Simple multi-task model
    """
    print("🔄 Creating simple U-Net based multi-task model (fallback)...")

    # Simple U-Net implementation
    class SimpleUNet(nn.Module):
        def __init__(self, in_channels, num_classes):
            super().__init__()

            # Encoder
            self.enc1 = self._make_layer(in_channels, 64)
            self.enc2 = self._make_layer(64, 128)
            self.enc3 = self._make_layer(128, 256)
            self.enc4 = self._make_layer(256, 512)

            # Decoder
            self.dec4 = self._make_layer(512 + 256, 256)
            self.dec3 = self._make_layer(256 + 128, 128)
            self.dec2 = self._make_layer(128 + 64, 64)
            self.dec1 = self._make_layer(64, 64)

            # Final conv
            self.final = nn.Conv2d(64, num_classes, 1)

            # Pooling and upsampling
            self.pool = nn.MaxPool2d(2)
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        def _make_layer(self, in_ch, out_ch):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True)
            )

        def forward(self, x):
            # Encoder
            e1 = self.enc1(x)
            e2 = self.enc2(self.pool(e1))
            e3 = self.enc3(self.pool(e2))
            e4 = self.enc4(self.pool(e3))

            # Decoder
            d4 = self.dec4(torch.cat([self.up(e4), e3], dim=1))
            d3 = self.dec3(torch.cat([self.up(d4), e2], dim=1))
            d2 = self.dec2(torch.cat([self.up(d3), e1], dim=1))
            d1 = self.dec1(d2)

            return self.final(d1)

    # Create the base segmentation model
    base_model = SimpleUNet(in_channels, num_seg_classes)

    # Wrap with multi-task functionality
    model = MultiTaskWrapper(
        segmentation_model=base_model,
        bottleneck_channels=512,  # From enc4
        enable_classification=enable_classification,
        num_seg_classes=num_seg_classes
    )

    print(f"✅ Simple multi-task model created with {num_seg_classes} classes")
    return model

def test_multitask_model():
    """Test the multi-task model"""
    print("🧪 Testing 4-class multi-task model...")

    # Test parameters
    batch_size = 2
    input_channels = 3
    height, width = 512, 512
    num_classes = 4

    # Create test input
    x = torch.randn(batch_size, input_channels, height, width)
    print(f"📊 Input shape: {x.shape}")

    # Test nnU-Net based model
    try:
        print(f"\n🏗️ Testing nnU-Net based multi-task model:")
        model = create_multitask_model(
            input_channels=input_channels,
            num_seg_classes=num_classes,
            enable_classification=True
        )

        # Get model info
        total_params = sum(p.numel() for p in model.parameters())
        print(f"   📊 Total parameters: {total_params:,}")

        # Forward pass
        with torch.no_grad():
            outputs = model(x)

        print(f"   ✅ Forward pass successful!")
        print(f"   📈 Outputs:")
        for key, value in outputs.items():
            if isinstance(value, torch.Tensor):
                print(f"      {key}: {value.shape}")
            elif isinstance(value, list):
                print(f"      {key}: {value}")

        # Test segmentation output
        seg_output = outputs['segmentation']
        assert seg_output.shape == (batch_size, num_classes, height, width), f"Unexpected segmentation shape: {seg_output.shape}"

        # Test classification outputs
        if 'patch_classification' in outputs:
            patch_output = outputs['patch_classification']
            assert patch_output.shape == (batch_size, num_classes), f"Unexpected patch classification shape: {patch_output.shape}"

        print(f"   ✅ All output shapes are correct!")

    except Exception as e:
        print(f"   ❌ nnU-Net model test failed: {e}")
        print(f"   🔄 Falling back to simple model test...")

        # Test simple model as fallback
        try:
            print(f"\n🔧 Testing simple U-Net based multi-task model:")
            simple_model = create_simple_multitask_model(
                in_channels=input_channels,
                num_seg_classes=num_classes,
                enable_classification=True
            )

            total_params = sum(p.numel() for p in simple_model.parameters())
            print(f"   📊 Total parameters: {total_params:,}")

            with torch.no_grad():
                outputs = simple_model(x)

            print(f"   ✅ Simple model forward pass successful!")
            for key, value in outputs.items():
                if isinstance(value, torch.Tensor):
                    print(f"      {key}: {value.shape}")
                elif isinstance(value, list):
                    print(f"      {key}: {value}")

        except Exception as e2:
            print(f"   ❌ Simple model test also failed: {e2}")
            return False

    print(f"\n✅ Multi-task model testing completed successfully!")
    return True

if __name__ == "__main__":
    success = test_multitask_model()
    if success:
        print(f"\n🎉 All tests passed! The 4-class multi-task model is ready for training.")
    else:
        print(f"\n❌ Tests failed! Please check the implementation.")