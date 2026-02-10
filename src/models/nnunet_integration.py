#!/usr/bin/env python3
"""
nnU-Net Integration for Combined Gland Segmentation
Handles imports and environment setup for 4-class segmentation with multi-task learning
"""

import sys
import os
from pathlib import Path
import warnings
from typing import Dict, Any, Optional
import torch

# Import from our config
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))
from configs.paths_config import DATA_PATHS

def setup_nnunet_environment():
    """
    Setup nnU-Net environment variables for combined gland segmentation
    """
    # Import here to avoid circular imports
    from configs.paths_config import get_combined_data_base

    try:
        # Set nnU-Net environment variables using runtime functions
        raw_path = get_combined_data_base()
        preprocessed_path = DATA_PATHS["nnunet_preprocessed"]
        results_path = DATA_PATHS["nnunet_results"]

        # Validate that environment variables are set
        if raw_path is None:
            raise ValueError("GLAND_DATASET_BASE environment variable not set")
        if preprocessed_path is None:
            raise ValueError("NNUNET_PREPROCESSED environment variable not set")
        if results_path is None:
            raise ValueError("NNUNET_RESULTS environment variable not set")

        os.environ['nnUNet_raw'] = raw_path
        os.environ['nnUNet_preprocessed'] = preprocessed_path
        os.environ['nnUNet_results'] = results_path

        print(f"🔧 nnU-Net Environment Setup:")
        print(f"   nnUNet_raw: {os.environ.get('nnUNet_raw')}")
        print(f"   nnUNet_preprocessed: {os.environ.get('nnUNet_preprocessed')}")
        print(f"   nnUNet_results: {os.environ.get('nnUNet_results')}")

        # Create directories if they don't exist
        for env_var in ['nnUNet_preprocessed', 'nnUNet_results']:
            path = Path(os.environ[env_var])
            path.mkdir(parents=True, exist_ok=True)
            print(f"   ✅ {env_var} directory ready")

    except ValueError as e:
        print(f"❌ nnU-Net environment setup failed: {e}")
        print("💡 Please run the training script (run_nnunet_training.sh) which exports required environment variables")
        raise

def import_nnunet_components() -> Dict[str, Any]:
    """
    Import nnU-Net components with comprehensive error handling
    Returns dictionary of available components
    """
    components = {
        'architectures': None,
        'trainer': None,
        'preprocessor': None,
        'planner': None,
        'transforms': None
    }

    print("🔄 Importing nnU-Net components...")

    # Try to import dynamic network architectures (most important for us)
    try:
        from dynamic_network_architectures.architectures.unet import PlainConvUNet, ResidualEncoderUNet
        from dynamic_network_architectures.building_blocks.helper import (
            get_matching_instancenorm,
            convert_dim_to_conv_op,
            convert_conv_op_to_dim
        )

        components['architectures'] = {
            'PlainConvUNet': PlainConvUNet,
            'ResidualEncoderUNet': ResidualEncoderUNet,
            'get_matching_instancenorm': get_matching_instancenorm,
            'convert_dim_to_conv_op': convert_dim_to_conv_op,
            'convert_conv_op_to_dim': convert_conv_op_to_dim
        }
        print("   ✅ Dynamic network architectures imported successfully")
    except ImportError as e:
        print(f"   ❌ Could not import dynamic_network_architectures: {e}")
        print("   📝 This will limit nnU-Net functionality")

    # Try to import nnU-Net v2 trainer (optional for our use case)
    try:
        from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
        components['trainer'] = {'nnUNetTrainer': nnUNetTrainer}
        print("   ✅ nnUNet trainer imported successfully")
    except ImportError as e:
        print(f"   ⚠️ Could not import nnUNetTrainer: {e}")
        print("   📝 Will use custom trainer instead")

    # Try to import data transforms
    try:
        import batchgenerators.transforms.spatial_transforms as spatial_transforms
        import batchgenerators.transforms.color_transforms as color_transforms
        import batchgenerators.transforms.noise_transforms as noise_transforms

        components['transforms'] = {
            'spatial_transforms': spatial_transforms,
            'color_transforms': color_transforms,
            'noise_transforms': noise_transforms
        }
        print("   ✅ Batchgenerators transforms imported successfully")
    except ImportError as e:
        print(f"   ⚠️ Could not import batchgenerators: {e}")
        print("   📝 Will use basic transforms")

    # Try to import preprocessor (optional)
    try:
        from nnunetv2.preprocessing.preprocessors.default_preprocessor import DefaultPreprocessor
        components['preprocessor'] = {'DefaultPreprocessor': DefaultPreprocessor}
        print("   ✅ nnU-Net preprocessor imported successfully")
    except ImportError as e:
        print(f"   ⚠️ Could not import DefaultPreprocessor: {e}")

    # Try to import experiment planner (optional)
    try:
        from nnunetv2.experiment_planning.experiment_planners.default_experiment_planner import ExperimentPlanner
        components['planner'] = {'ExperimentPlanner': ExperimentPlanner}
        print("   ✅ nnU-Net planner imported successfully")
    except ImportError as e:
        print(f"   ⚠️ Could not import ExperimentPlanner: {e}")

    return components

def create_nnunet_architecture(
    input_channels: int = 3,
    num_classes: int = 4,  # Updated for 4-class segmentation
    deep_supervision: bool = True,
    architecture_type: str = 'PlainConvUNet'
) -> Optional[Any]:
    """
    Create nnU-Net architecture adapted for 4-class gland segmentation

    Args:
        input_channels: Number of input channels (3 for RGB)
        num_classes: Number of output classes (4: Background, Benign, Malignant, PDC)
        deep_supervision: Whether to use deep supervision
        architecture_type: Type of architecture ('PlainConvUNet' or 'ResidualEncoderUNet')

    Returns:
        nnU-Net model instance or None if not available
    """
    components = import_nnunet_components()

    if not components['architectures']:
        print("❌ nnU-Net architectures not available")
        return None

    try:
        # Standard nnU-Net configuration for 2D segmentation
        # Adapted for gland segmentation task
        conv_op = components['architectures']['convert_dim_to_conv_op'](2)  # 2D convolution
        norm_op = components['architectures']['get_matching_instancenorm'](conv_op)

        # nnU-Net standard configuration for 2D
        # These parameters are optimized for medical image segmentation
        num_stages = 6
        features_per_stage = [32, 64, 128, 256, 512, 512]
        conv_kernel_sizes = [[3, 3], [3, 3], [3, 3], [3, 3], [3, 3], [3, 3]]
        pool_op_kernel_sizes = [[1, 1], [2, 2], [2, 2], [2, 2], [2, 2], [2, 2]]

        # Get the architecture class
        if architecture_type == 'ResidualEncoderUNet' and 'ResidualEncoderUNet' in components['architectures']:
            ArchitectureClass = components['architectures']['ResidualEncoderUNet']
        else:
            ArchitectureClass = components['architectures']['PlainConvUNet']

        # Create the model
        model = ArchitectureClass(
            input_channels=input_channels,
            n_stages=num_stages,
            features_per_stage=features_per_stage,
            conv_op=conv_op,
            kernel_sizes=conv_kernel_sizes,
            strides=pool_op_kernel_sizes,
            n_conv_per_stage=[2, 2, 2, 2, 2, 2],
            num_classes=num_classes,
            n_conv_per_stage_decoder=[2, 2, 2, 2, 2],
            conv_bias=True,
            norm_op=norm_op,
            norm_op_kwargs={'eps': 1e-5, 'affine': True},
            dropout_op=None,  # No dropout in encoder/decoder
            dropout_op_kwargs=None,
            nonlin=torch.nn.LeakyReLU,  # LeakyReLU
            nonlin_kwargs={'negative_slope': 1e-2, 'inplace': True},
            deep_supervision=deep_supervision,
            nonlin_first=False
        )

        print(f"✅ Created {architecture_type} model:")
        print(f"   📊 Input channels: {input_channels}")
        print(f"   🎯 Output classes: {num_classes}")
        print(f"   🏗️ Stages: {num_stages}")
        print(f"   📈 Features per stage: {features_per_stage}")
        print(f"   🔍 Deep supervision: {deep_supervision}")

        return model

    except Exception as e:
        print(f"❌ Failed to create nnU-Net architecture: {e}")
        return None

def get_architecture_classes() -> Optional[Dict[str, Any]]:
    """Get architecture classes with error handling"""
    components = import_nnunet_components()
    return components.get('architectures')

def get_nnunet_trainer_class() -> Optional[Any]:
    """Get nnU-Net trainer class with fallback"""
    components = import_nnunet_components()

    if components['trainer']:
        return components['trainer']['nnUNetTrainer']
    else:
        return None

def check_nnunet_availability() -> Dict[str, bool]:
    """
    Check availability of nnU-Net components
    Returns dictionary with component availability status
    """
    components = import_nnunet_components()

    availability = {}
    for name, component in components.items():
        availability[name] = component is not None

    return availability

def print_nnunet_status():
    """Print detailed status of nnU-Net integration"""
    print("=" * 80)
    print("nnU-Net Integration Status for Combined Gland Segmentation")
    print("=" * 80)

    # Environment
    print("\n🌍 Environment Variables:")
    for var in ['nnUNet_raw', 'nnUNet_preprocessed', 'nnUNet_results']:
        value = os.environ.get(var, 'Not set')
        exists = "✅" if Path(value).exists() else "❌"
        print(f"   {exists} {var}: {value}")

    # Component availability
    print("\n🔧 Component Availability:")
    availability = check_nnunet_availability()
    for component, available in availability.items():
        status = "✅ Available" if available else "❌ Not Available"
        print(f"   {status} {component}")

    # Test architecture creation
    print("\n🏗️ Architecture Test:")
    try:
        model = create_nnunet_architecture()
        if model is not None:
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"   ✅ Architecture creation successful")
            print(f"   📊 Total parameters: {total_params:,}")
            print(f"   🏃 Trainable parameters: {trainable_params:,}")
        else:
            print(f"   ❌ Architecture creation failed")
    except Exception as e:
        print(f"   ❌ Architecture test failed: {e}")

    print("=" * 80)

# Initialize environment on import (only if environment variables are available)
# This allows independent evaluator to import the module without requiring training environment
try:
    setup_nnunet_environment()
except ValueError as e:
    # Environment variables not set - this is expected for independent evaluator
    print(f"ℹ️ nnU-Net environment setup skipped: {e}")
    print("ℹ️ This is normal for independent evaluation - environment will be set when needed")

if __name__ == "__main__":
    # Test the integration
    print_nnunet_status()