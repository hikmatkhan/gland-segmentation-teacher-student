#!/usr/bin/env python3
"""
Model Factory for Architecture Selection
Centralized creation of different segmentation model architectures
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Union
import warnings

from .baseline_unet import BaselineUNet, create_baseline_unet
from .nnunet_integration import create_nnunet_architecture
from .teacher_student_unet import TeacherStudentUNet, create_teacher_student_unet


class ModelFactory:
    """Factory class for creating different segmentation model architectures"""

    SUPPORTED_ARCHITECTURES = ['baseline_unet', 'nnunet', 'teacher_student_unet']

    @staticmethod
    def create_segmentation_model(architecture: str,
                                input_channels: int = 3,
                                num_classes: int = 4,
                                **kwargs) -> nn.Module:
        """
        Create a segmentation model based on architecture type

        Args:
            architecture: Model architecture type ('baseline_unet', 'nnunet', 'teacher_student_unet')
            input_channels: Number of input channels (default: 3 for RGB)
            num_classes: Number of output classes (default: 4)
            **kwargs: Architecture-specific parameters

        Returns:
            Segmentation model instance

        Raises:
            ValueError: If architecture is not supported
        """
        if architecture not in ModelFactory.SUPPORTED_ARCHITECTURES:
            raise ValueError(
                f"Unsupported architecture: {architecture}. "
                f"Supported architectures: {ModelFactory.SUPPORTED_ARCHITECTURES}"
            )

        print(f"🏗️ Creating {architecture} model...")

        if architecture == 'baseline_unet':
            return ModelFactory._create_baseline_unet(
                input_channels=input_channels,
                num_classes=num_classes,
                **kwargs
            )
        elif architecture == 'nnunet':
            return ModelFactory._create_nnunet(
                input_channels=input_channels,
                num_classes=num_classes,
                **kwargs
            )
        elif architecture == 'teacher_student_unet':
            return ModelFactory._create_teacher_student_unet(
                input_channels=input_channels,
                num_classes=num_classes,
                **kwargs
            )

    @staticmethod
    def _create_baseline_unet(input_channels: int = 3,
                            num_classes: int = 4,
                            depth: int = 4,
                            initial_channels: int = 64,
                            channel_multiplier: int = 2,
                            activation: str = 'relu',
                            normalization: str = 'batch',
                            dropout: float = 0.1,
                            bilinear: bool = True,
                            enable_hooks: bool = True,
                            **kwargs) -> BaselineUNet:
        """
        Create baseline UNet model

        Args:
            input_channels: Number of input channels
            num_classes: Number of output classes
            depth: Network depth (number of downsampling levels)
            initial_channels: Starting channel count
            channel_multiplier: Factor to multiply channels at each level
            activation: Activation function ('relu', 'leaky_relu', 'gelu')
            normalization: Normalization type ('batch', 'group', 'instance')
            dropout: Dropout probability
            bilinear: Use bilinear upsampling instead of transpose conv
            enable_hooks: Enable feature hooks for classification
            **kwargs: Additional arguments (ignored with warning)

        Returns:
            BaselineUNet instance
        """
        # Warn about unused arguments
        if kwargs:
            warnings.warn(f"Unused arguments for baseline_unet: {list(kwargs.keys())}")

        model = BaselineUNet(
            input_channels=input_channels,
            num_classes=num_classes,
            depth=depth,
            initial_channels=initial_channels,
            channel_multiplier=channel_multiplier,
            activation=activation,
            normalization=normalization,
            dropout=dropout,
            bilinear=bilinear,
            enable_hooks=enable_hooks
        )

        param_count = sum(p.numel() for p in model.parameters())
        print(f"✅ Baseline UNet created with {param_count:,} parameters")
        print(f"📊 Architecture: depth={depth}, channels={initial_channels}, activation={activation}")

        return model

    @staticmethod
    def _create_nnunet(input_channels: int = 3,
                      num_classes: int = 4,
                      deep_supervision: bool = True,
                      **kwargs) -> nn.Module:
        """
        Create nnUNet model

        Args:
            input_channels: Number of input channels
            num_classes: Number of output classes
            deep_supervision: Enable deep supervision
            **kwargs: Additional nnUNet-specific arguments

        Returns:
            nnUNet model instance
        """
        try:
            model = create_nnunet_architecture(
                input_channels=input_channels,
                num_classes=num_classes,
                deep_supervision=deep_supervision,
                **kwargs
            )

            if model is not None:
                param_count = sum(p.numel() for p in model.parameters())
                print(f"✅ nnUNet created with {param_count:,} parameters")
                print(f"📊 Deep supervision: {deep_supervision}")
                return model
            else:
                raise RuntimeError("Failed to create nnUNet model")

        except Exception as e:
            print(f"❌ Error creating nnUNet: {e}")
            raise

    @staticmethod
    def _create_teacher_student_unet(input_channels: int = 3,
                                   num_classes: int = 4,
                                   ema_decay: float = 0.999,
                                   teacher_init_epoch: Optional[int] = None,
                                   teacher_init_val_loss: Optional[float] = None,
                                   backbone_type: str = 'baseline_unet',
                                   **kwargs) -> TeacherStudentUNet:
        """
        Create Teacher-Student UNet model

        Args:
            input_channels: Number of input channels
            num_classes: Number of output classes
            ema_decay: EMA decay factor for teacher updates
            teacher_init_epoch: Epoch for teacher initialization
            teacher_init_val_loss: Validation loss threshold for teacher init
            backbone_type: Backbone architecture type ('baseline_unet' or 'nnunet')
            **kwargs: Additional arguments for backbone components

        Returns:
            TeacherStudentUNet instance
        """
        # Separate loss configuration from model configuration
        loss_config_keys = {'min_alpha', 'max_alpha', 'consistency_loss_type', 'consistency_temperature'}
        loss_config = {k: v for k, v in kwargs.items() if k in loss_config_keys}

        # Define architecture-specific arguments
        baseline_unet_args = {
            'depth', 'initial_channels', 'channel_multiplier', 'activation',
            'normalization', 'dropout', 'bilinear', 'enable_hooks'
        }
        nnunet_args = {
            'deep_supervision', 'architecture_type'
        }

        # Determine supported arguments based on backbone type
        if backbone_type == 'baseline_unet':
            supported_args = baseline_unet_args
        elif backbone_type == 'nnunet':
            supported_args = nnunet_args
        else:
            supported_args = set()

        # Warn about unused arguments that don't apply to the selected backbone
        unsupported_args = set(kwargs.keys()) - supported_args - loss_config_keys

        if unsupported_args:
            warnings.warn(f"Unused arguments for teacher_student_unet with {backbone_type} backbone: {list(unsupported_args)}")

        # Store loss configuration for later use by trainer
        if loss_config:
            print(f"📋 Loss configuration available: {loss_config}")

        # Filter out loss configuration and unsupported arguments for model creation
        model_kwargs = {k: v for k, v in kwargs.items() if k not in unsupported_args and k not in loss_config_keys}

        model = TeacherStudentUNet(
            input_channels=input_channels,
            num_classes=num_classes,
            ema_decay=ema_decay,
            teacher_init_epoch=teacher_init_epoch,
            teacher_init_val_loss=teacher_init_val_loss,
            backbone_type=backbone_type,
            **model_kwargs
        )

        param_count = sum(p.numel() for p in model.parameters())
        print(f"✅ Teacher-Student UNet created with {param_count:,} parameters")
        print(f"📊 Architecture: Backbone={backbone_type}, EMA decay={ema_decay}, Teacher init epoch={teacher_init_epoch}")

        return model

    @staticmethod
    def get_model_info(model: nn.Module) -> Dict[str, Any]:
        """
        Get information about a model

        Args:
            model: PyTorch model

        Returns:
            Dictionary with model information
        """
        info = {
            'type': type(model).__name__,
            'parameters': sum(p.numel() for p in model.parameters()),
            'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
        }

        # Architecture-specific information
        if isinstance(model, BaselineUNet):
            info.update({
                'architecture': 'baseline_unet',
                'depth': model.depth,
                'input_channels': model.input_channels,
                'num_classes': model.num_classes,
                'bottleneck_channels': model.get_bottleneck_channels(),
                'config': model.config
            })
        elif isinstance(model, TeacherStudentUNet):
            info.update({
                'architecture': 'teacher_student_unet',
                'input_channels': model.input_channels,
                'num_classes': model.num_classes,
                'ema_decay': model.ema_decay,
                'teacher_initialized': model.teacher_initialized,
                'bottleneck_channels': model.get_bottleneck_channels(),
                'config': model.get_config()
            })
        else:
            info['architecture'] = 'nnunet'

        return info

    @staticmethod
    def get_bottleneck_channels(model: nn.Module) -> int:
        """
        Get the number of channels in the model's bottleneck layer

        Args:
            model: PyTorch model

        Returns:
            Number of bottleneck channels
        """
        if isinstance(model, BaselineUNet):
            return model.get_bottleneck_channels()
        elif isinstance(model, TeacherStudentUNet):
            return model.get_bottleneck_channels()
        else:
            # For nnUNet, try to infer from the model structure
            # This is a heuristic and may need adjustment
            try:
                # Look for the deepest encoder features
                return 512  # Common nnUNet bottleneck size
            except:
                warnings.warn("Could not determine bottleneck channels for nnUNet, using default 512")
                return 512

    @staticmethod
    def validate_model_compatibility(model: nn.Module,
                                   input_channels: int,
                                   num_classes: int) -> bool:
        """
        Validate that a model is compatible with expected input/output dimensions

        Args:
            model: PyTorch model
            input_channels: Expected input channels
            num_classes: Expected number of classes

        Returns:
            True if compatible, False otherwise
        """
        try:
            # Test with dummy input
            dummy_input = torch.randn(1, input_channels, 256, 256)

            with torch.no_grad():
                output = model(dummy_input)

                # Handle Teacher-Student UNet output (dictionary)
                if isinstance(output, dict):
                    if 'student' in output:
                        # Teacher-Student mode, use student output for validation
                        output = output['student']['segmentation']
                    elif 'segmentation' in output:
                        # Direct output mode
                        output = output['segmentation']
                    else:
                        print(f"❌ Model compatibility failed: unexpected dict output format")
                        return False

                # Handle nnUNet deep supervision output
                elif isinstance(output, (list, tuple)):
                    output = output[0]  # Take the highest resolution output

                # Check output shape
                expected_shape = (1, num_classes, 256, 256)
                actual_shape = output.shape

                if actual_shape == expected_shape:
                    print(f"✅ Model compatibility validated: {actual_shape}")
                    return True
                else:
                    print(f"❌ Model compatibility failed: expected {expected_shape}, got {actual_shape}")
                    return False

        except Exception as e:
            print(f"❌ Model compatibility test failed: {e}")
            return False


def create_model_from_config(config: Dict[str, Any]) -> nn.Module:
    """
    Create a model from configuration dictionary

    Args:
        config: Configuration dictionary containing model parameters

    Returns:
        Created model instance

    Example config:
        {
            "architecture": "baseline_unet",
            "input_channels": 3,
            "num_classes": 4,
            "baseline_unet": {
                "depth": 4,
                "initial_channels": 64,
                "activation": "relu"
            }
        }
    """
    architecture = config.get('architecture', 'baseline_unet')
    input_channels = config.get('input_channels', 3)
    num_classes = config.get('num_classes', 4)

    # Get architecture-specific parameters
    arch_params = config.get(architecture, {})

    # Create model
    model = ModelFactory.create_segmentation_model(
        architecture=architecture,
        input_channels=input_channels,
        num_classes=num_classes,
        **arch_params
    )

    # Skip validation for Teacher-Student UNet for now (special case)
    if architecture == 'teacher_student_unet':
        print(f"✅ Teacher-Student UNet created (skipping compatibility validation)")
        return model

    # Validate compatibility for other models
    if not ModelFactory.validate_model_compatibility(model, input_channels, num_classes):
        raise RuntimeError(f"Created model is not compatible with expected dimensions")

    return model


def test_model_factory():
    """Test function for model factory"""
    print("🧪 Testing Model Factory...")

    test_configs = [
        {
            "name": "Baseline UNet - Standard",
            "config": {
                "architecture": "baseline_unet",
                "input_channels": 3,
                "num_classes": 4,
                "baseline_unet": {
                    "depth": 4,
                    "initial_channels": 64,
                    "activation": "relu"
                }
            }
        },
        {
            "name": "Baseline UNet - Deep",
            "config": {
                "architecture": "baseline_unet",
                "input_channels": 3,
                "num_classes": 4,
                "baseline_unet": {
                    "depth": 5,
                    "initial_channels": 32,
                    "activation": "gelu",
                    "normalization": "group"
                }
            }
        },
        {
            "name": "Teacher-Student UNet - Standard",
            "config": {
                "architecture": "teacher_student_unet",
                "input_channels": 3,
                "num_classes": 4,
                "teacher_student_unet": {
                    "ema_decay": 0.999,
                    "teacher_init_epoch": 20,
                    "depth": 4,
                    "initial_channels": 64
                }
            }
        }
    ]

    for test_case in test_configs:
        print(f"\n🔬 Testing: {test_case['name']}")
        try:
            model = create_model_from_config(test_case['config'])
            info = ModelFactory.get_model_info(model)
            print(f"✅ Model info: {info}")

            # Test bottleneck channels
            bottleneck_channels = ModelFactory.get_bottleneck_channels(model)
            print(f"✅ Bottleneck channels: {bottleneck_channels}")

        except Exception as e:
            print(f"❌ Test failed: {e}")

    print("\n🎉 Model Factory test completed!")


if __name__ == "__main__":
    test_model_factory()