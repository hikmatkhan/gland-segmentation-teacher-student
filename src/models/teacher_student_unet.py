#!/usr/bin/env python3
"""
Teacher-Student UNet Implementation for Self-Training Gland Segmentation
=========================================================================

Teacher-Student architecture with Exponential Moving Average (EMA) updates.
- Student: Traditional training with gradients (based on BaselineUNet)
- Teacher: EMA-only updates, provides pseudo-labels for consistency loss
- Multi-task: Both support segmentation + classification heads

Features:
- Constant EMA decay for teacher weight updates
- Two-phase training: warm-up (student only) → teacher-student (dual loss)
- Multi-task consistency loss across all heads
- Cosine decay loss weighting (supervised → consistency)

Author: Claude Code - Generated for OSU CRC Research
Date: 2025-09-22
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, Any, Optional, Tuple
import warnings

# Local imports
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.models.baseline_unet import BaselineUNet
from src.models.nnunet_integration import create_nnunet_architecture


class TeacherStudentUNet(nn.Module):
    """
    Teacher-Student UNet architecture for self-training gland segmentation

    Features:
    - Student network: Traditional training with gradients
    - Teacher network: EMA-only updates, no gradients
    - Two-phase training: warm-up → teacher-student
    - Multi-task consistency loss
    """

    def __init__(self,
                 input_channels: int = 3,
                 num_classes: int = 4,
                 ema_decay: float = 0.999,
                 ema_schedule: str = 'fixed',
                 ema_decay_initial: float = 0.999,
                 ema_decay_final: float = 0.1,
                 ema_annealing_start_epoch: int = 50,
                 teacher_init_epoch: Optional[int] = None,
                 teacher_init_val_loss: Optional[float] = None,
                 backbone_type: str = 'baseline_unet',
                 enable_hooks: bool = True,
                 depth: int = 4,
                 initial_channels: int = 64,
                 **unet_kwargs):
        """
        Initialize Teacher-Student UNet architecture

        Args:
            input_channels: Number of input channels (default: 3 for RGB)
            num_classes: Number of output classes (default: 4)
            ema_decay: EMA decay factor for fixed schedule (default: 0.999, used only when ema_schedule='fixed')
            ema_schedule: EMA scheduling type ('fixed', 'cosine', 'linear', 'exponential')
            ema_decay_initial: Initial EMA decay for annealing schedules (default: 0.999)
            ema_decay_final: Final EMA decay for annealing schedules (default: 0.1)
            ema_annealing_start_epoch: Epoch to start EMA annealing (default: 50)
            teacher_init_epoch: Epoch after which to initialize teacher
            teacher_init_val_loss: Validation loss threshold for teacher init
            backbone_type: Backbone architecture type ('baseline_unet' or 'nnunet')
            enable_hooks: Enable feature hooks for classification
            depth: Network depth (number of downsampling levels, default: 4)
            initial_channels: Starting channel count (default: 64)
            **unet_kwargs: Additional arguments for backbone architecture
        """
        super().__init__()

        self.input_channels = input_channels
        self.num_classes = num_classes
        self.ema_decay = ema_decay  # For backward compatibility (fixed schedule)
        self.ema_schedule = ema_schedule
        self.ema_decay_initial = ema_decay_initial
        self.ema_decay_final = ema_decay_final
        self.ema_annealing_start_epoch = ema_annealing_start_epoch
        self.current_ema_decay = ema_decay_initial if ema_schedule != 'fixed' else ema_decay
        self.teacher_init_epoch = teacher_init_epoch
        self.teacher_init_val_loss = teacher_init_val_loss
        self.backbone_type = backbone_type
        self.teacher_initialized = False

        # Create student network (trainable)
        print(f"🎓 Creating Student {backbone_type} network...")
        self.student = self._create_backbone(
            backbone_type=backbone_type,
            input_channels=input_channels,
            num_classes=num_classes,
            depth=depth,
            initial_channels=initial_channels,
            enable_hooks=enable_hooks,
            **unet_kwargs
        )

        # Create teacher network (EMA-only, no gradients)
        print(f"👨‍🏫 Creating Teacher {backbone_type} network...")
        self.teacher = self._create_backbone(
            backbone_type=backbone_type,
            input_channels=input_channels,
            num_classes=num_classes,
            depth=depth,
            initial_channels=initial_channels,
            enable_hooks=enable_hooks,
            **unet_kwargs
        )

        # Disable gradients for teacher network
        for param in self.teacher.parameters():
            param.requires_grad = False

        # Set teacher to eval mode permanently
        self.teacher.eval()

        # Architecture info
        student_params = sum(p.numel() for p in self.student.parameters())
        teacher_params = sum(p.numel() for p in self.teacher.parameters())

        print(f"✅ Teacher-Student UNet initialized:")
        print(f"   🎓 Student parameters: {student_params:,}")
        print(f"   👨‍🏫 Teacher parameters: {teacher_params:,}")
        print(f"   📊 Total parameters: {student_params + teacher_params:,}")
        print(f"   🔄 EMA decay: {ema_decay}")
        print(f"   📅 Teacher init epoch: {teacher_init_epoch}")
        print(f"   📉 Teacher init val loss: {teacher_init_val_loss}")

    def _create_backbone(self, backbone_type: str, **kwargs):
        """
        Create backbone network based on type

        Args:
            backbone_type: Type of backbone ('baseline_unet' or 'nnunet')
            **kwargs: Arguments for backbone creation

        Returns:
            Backbone network instance
        """
        if backbone_type == 'baseline_unet':
            return BaselineUNet(**kwargs)
        elif backbone_type == 'nnunet':
            # Filter out BaselineUNet-specific arguments for nnUNet
            nnunet_kwargs = {k: v for k, v in kwargs.items()
                           if k not in ['depth', 'initial_channels', 'channel_multiplier',
                                      'activation', 'normalization', 'dropout', 'bilinear', 'enable_hooks']}
            return create_nnunet_architecture(**nnunet_kwargs)
        else:
            raise ValueError(f"Unsupported backbone type: {backbone_type}. "
                           f"Supported types: 'baseline_unet', 'nnunet'")

    def should_initialize_teacher(self, current_epoch: int, val_loss: float) -> bool:
        """
        Check if teacher should be initialized based on criteria

        Args:
            current_epoch: Current training epoch
            val_loss: Current validation loss

        Returns:
            True if teacher should be initialized
        """
        if self.teacher_initialized:
            return False

        # Epoch-based initialization
        if self.teacher_init_epoch is not None and current_epoch >= self.teacher_init_epoch:
            return True

        # Validation loss-based initialization
        if self.teacher_init_val_loss is not None and val_loss <= self.teacher_init_val_loss:
            return True

        return False

    def initialize_teacher(self):
        """
        Initialize teacher network with student weights
        """
        if self.teacher_initialized:
            print("⚠️ Teacher already initialized, skipping...")
            return

        print("🚀 Initializing Teacher network with Student weights...")

        # Copy student weights to teacher
        for teacher_param, student_param in zip(self.teacher.parameters(), self.student.parameters()):
            teacher_param.data.copy_(student_param.data)

        self.teacher_initialized = True
        print("✅ Teacher network initialized successfully!")

    def get_dynamic_ema_decay(self, current_epoch: int, total_epochs: int) -> float:
        """
        Calculate dynamic EMA decay based on the selected schedule

        Args:
            current_epoch: Current training epoch
            total_epochs: Total number of training epochs

        Returns:
            Current EMA decay value
        """
        if self.ema_schedule == "fixed":
            return self.ema_decay

        if current_epoch < self.ema_annealing_start_epoch:
            return self.ema_decay_initial

        # Calculate progress for annealing schedules
        progress = (current_epoch - self.ema_annealing_start_epoch) / (total_epochs - self.ema_annealing_start_epoch)
        progress = min(progress, 1.0)

        if self.ema_schedule == "cosine":
            # Smooth cosine transition from initial to final
            cosine_factor = 0.5 * (1 + math.cos(math.pi * progress))
            return self.ema_decay_final + (self.ema_decay_initial - self.ema_decay_final) * cosine_factor

        elif self.ema_schedule == "linear":
            # Linear interpolation from initial to final
            return self.ema_decay_initial + (self.ema_decay_final - self.ema_decay_initial) * progress

        elif self.ema_schedule == "exponential":
            # Exponential decay: faster transition early, slower later
            exp_factor = math.exp(-5 * progress)  # Adjustable rate
            return self.ema_decay_final + (self.ema_decay_initial - self.ema_decay_final) * exp_factor

        else:
            raise ValueError(f"Unknown EMA schedule: {self.ema_schedule}. Supported: 'fixed', 'cosine', 'linear', 'exponential'")

    def update_teacher_ema(self, current_epoch: Optional[int] = None, total_epochs: Optional[int] = None):
        """
        Update teacher weights using Exponential Moving Average (EMA)

        Args:
            current_epoch: Current training epoch (for dynamic scheduling)
            total_epochs: Total number of training epochs (for dynamic scheduling)

        Formula: teacher = current_ema_decay * teacher + (1 - current_ema_decay) * student
        """
        if not self.teacher_initialized:
            return

        # Calculate dynamic EMA decay if epoch information is provided
        if current_epoch is not None and total_epochs is not None:
            self.current_ema_decay = self.get_dynamic_ema_decay(current_epoch, total_epochs)
        else:
            # Fallback to fixed EMA decay for backward compatibility
            self.current_ema_decay = self.ema_decay

        with torch.no_grad():
            for teacher_param, student_param in zip(self.teacher.parameters(), self.student.parameters()):
                teacher_param.data.mul_(self.current_ema_decay).add_(
                    student_param.data, alpha=1.0 - self.current_ema_decay
                )

    def forward(self, x: torch.Tensor, mode: str = "student_only") -> Dict[str, Any]:
        """
        Forward pass with different modes

        Args:
            x: Input tensor [B, C, H, W]
            mode: Forward mode
                - "student_only": Only student forward (warm-up phase)
                - "teacher_student": Both networks forward (teacher-student phase)
                - "teacher_only": Only teacher forward (evaluation)

        Returns:
            Dictionary with outputs based on mode
        """
        outputs = {}

        if mode == "student_only":
            # Warm-up phase: only student network
            outputs['student'] = self.student(x)

        elif mode == "teacher_student":
            # Teacher-student phase: both networks
            if not self.teacher_initialized:
                raise RuntimeError("Teacher not initialized for teacher_student mode")

            outputs['student'] = self.student(x)

            # Teacher forward (no gradients)
            with torch.no_grad():
                outputs['teacher'] = self.teacher(x)

        elif mode == "teacher_only":
            # Evaluation phase: only teacher network
            if not self.teacher_initialized:
                raise RuntimeError("Teacher not initialized for teacher_only mode")

            with torch.no_grad():
                outputs['teacher'] = self.teacher(x)

        else:
            raise ValueError(f"Unknown forward mode: {mode}")

        return outputs

    def get_bottleneck_channels(self) -> int:
        """
        Get bottleneck channels for classification heads

        Returns:
            Number of channels in bottleneck layer
        """
        if hasattr(self.student, 'get_bottleneck_channels'):
            return self.student.get_bottleneck_channels()
        elif self.backbone_type == 'nnunet':
            # nnUNet typically uses 512 channels in the bottleneck
            return 512
        else:
            # Default fallback
            return 512

    def get_config(self) -> Dict[str, Any]:
        """
        Get model configuration

        Returns:
            Configuration dictionary
        """
        return {
            'architecture': 'teacher_student_unet',
            'backbone_type': self.backbone_type,
            'input_channels': self.input_channels,
            'num_classes': self.num_classes,
            'ema_decay': self.ema_decay,
            'ema_schedule': self.ema_schedule,
            'ema_decay_initial': self.ema_decay_initial,
            'ema_decay_final': self.ema_decay_final,
            'ema_annealing_start_epoch': self.ema_annealing_start_epoch,
            'current_ema_decay': self.current_ema_decay,
            'teacher_init_epoch': self.teacher_init_epoch,
            'teacher_init_val_loss': self.teacher_init_val_loss,
            'teacher_initialized': self.teacher_initialized,
            'student_config': self.student.config if hasattr(self.student, 'config') else {},
            'parameters': {
                'student': sum(p.numel() for p in self.student.parameters()),
                'teacher': sum(p.numel() for p in self.teacher.parameters()),
                'total': sum(p.numel() for p in self.parameters())
            }
        }

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        """
        Custom state dict to save both student and teacher weights
        """
        state_dict = super().state_dict(destination, prefix, keep_vars)

        # Add initialization status
        state_dict[prefix + 'teacher_initialized'] = self.teacher_initialized

        return state_dict

    def load_state_dict(self, state_dict, strict=True):
        """
        Custom load state dict to handle teacher initialization status and key mapping
        """
        # Handle key mapping for models saved with segmentation_model prefix
        if any(key.startswith('segmentation_model.') for key in state_dict.keys()):
            # Create new state dict with mapped keys
            new_state_dict = {}
            for key, value in state_dict.items():
                if key.startswith('segmentation_model.'):
                    # Remove segmentation_model prefix
                    new_key = key.replace('segmentation_model.', '')
                    new_state_dict[new_key] = value
                elif key == 'teacher_initialized':
                    new_state_dict[key] = value
                # Skip classification_head keys as they're not part of TeacherStudentUNet
            state_dict = new_state_dict

        # Extract teacher initialization status
        teacher_init_key = 'teacher_initialized'
        if teacher_init_key in state_dict:
            self.teacher_initialized = state_dict.pop(teacher_init_key)

        return super().load_state_dict(state_dict, strict)


def create_teacher_student_unet(input_channels: int = 3,
                               num_classes: int = 4,
                               ema_decay: float = 0.999,
                               teacher_init_epoch: Optional[int] = 20,
                               teacher_init_val_loss: Optional[float] = None,
                               backbone_type: str = 'baseline_unet',
                               depth: int = 4,
                               initial_channels: int = 64,
                               **kwargs) -> TeacherStudentUNet:
    """
    Factory function to create TeacherStudentUNet

    Args:
        input_channels: Number of input channels
        num_classes: Number of output classes
        ema_decay: EMA decay factor
        teacher_init_epoch: Epoch for teacher initialization
        teacher_init_val_loss: Validation loss threshold for teacher init
        backbone_type: Backbone architecture type ('baseline_unet' or 'nnunet')
        depth: Network depth (number of downsampling levels)
        initial_channels: Starting channel count
        **kwargs: Additional arguments for backbone architecture

    Returns:
        TeacherStudentUNet instance
    """
    model = TeacherStudentUNet(
        input_channels=input_channels,
        num_classes=num_classes,
        ema_decay=ema_decay,
        teacher_init_epoch=teacher_init_epoch,
        teacher_init_val_loss=teacher_init_val_loss,
        backbone_type=backbone_type,
        depth=depth,
        initial_channels=initial_channels,
        **kwargs
    )

    return model


def test_teacher_student_unet():
    """Test function for Teacher-Student UNet"""
    print("🧪 Testing Teacher-Student UNet...")

    # Create model
    model = create_teacher_student_unet(
        input_channels=3,
        num_classes=4,
        ema_decay=0.999,
        teacher_init_epoch=10
    )

    # Test input
    x = torch.randn(2, 3, 256, 256)

    # Test warm-up phase
    print("\n🔥 Testing warm-up phase (student only)...")
    outputs = model(x, mode="student_only")
    print(f"✅ Student output shape: {outputs['student'].shape}")

    # Initialize teacher
    print("\n🚀 Initializing teacher...")
    model.initialize_teacher()

    # Test teacher-student phase
    print("\n🤝 Testing teacher-student phase...")
    outputs = model(x, mode="teacher_student")
    print(f"✅ Student output shape: {outputs['student'].shape}")
    print(f"✅ Teacher output shape: {outputs['teacher'].shape}")

    # Test EMA update
    print("\n🔄 Testing EMA update...")
    model.update_teacher_ema()
    print("✅ EMA update successful")

    # Test teacher-only mode
    print("\n👨‍🏫 Testing teacher-only mode...")
    outputs = model(x, mode="teacher_only")
    print(f"✅ Teacher output shape: {outputs['teacher'].shape}")

    # Test configuration
    print("\n📋 Testing configuration...")
    config = model.get_config()
    print(f"✅ Configuration: {config}")

    print("\n🎉 Teacher-Student UNet test completed successfully!")


if __name__ == "__main__":
    test_teacher_student_unet()