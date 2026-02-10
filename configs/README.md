# Configuration Module - Dataset and Path Management

This module provides centralized configuration management for the 4-class multi-task nnU-Net implementation.

## 📁 **Files Overview**

### **`paths_config.py`**
**Purpose**: Unified configuration system for dataset paths, training parameters, and environment setup.

---

## 🎯 **Key Features**

### **1. Dataset-Agnostic Configuration**
- **Unified Interface**: Single configuration for all dataset variations
- **Automatic Validation**: Built-in dataset structure verification
- **Flexible Switching**: Easy switching between magnifications and combinations

### **2. Environment Management**
- **nnU-Net Integration**: Automatic environment variable setup
- **Path Validation**: Ensures all required paths exist and are accessible
- **Error Prevention**: Catches configuration issues before training

### **3. Research-Ready Defaults**
- **4-Class Optimized**: Pre-configured for 4-class segmentation
- **Multi-Label Support**: Default settings for multi-label patch classification
- **Full Training**: 150-epoch configuration for comprehensive research

---

## 🏗️ **Configuration Structure**

### **Available Datasets**:
```python
AVAILABLE_DATASETS = {
    # Mixed magnifications (all magnifications together)
    "mixed": "Task001_Combined_Mixed_Magnifications",

    # Separate magnifications (individual datasets per magnification)
    "mag5x": "Task005_Combined_Mag5x",          # 4,872 samples
    "mag10x": "Task010_Combined_Mag10x",        # 6,188 samples
    "mag20x": "Task020_Combined_Mag20x",        # ~6,000 samples
    "mag40x": "Task040_Combined_Mag40x",        # ~6,000 samples
}
```

### **Default Training Configuration**:
```python
DEFAULT_CONFIG = {
    # Training parameters (Research-grade)
    "epochs": 150,                    # Full research training
    "batch_size": 4,                  # GPU memory optimized
    "learning_rate": 1e-4,            # Adam/AdamW learning rate
    "weight_decay": 1e-4,             # L2 regularization
    "num_workers": 4,                 # Data loading workers

    # Image processing
    "image_size": [512, 512],         # Standard histopathology size

    # Model parameters - 4-class setup
    "use_nnunet": True,               # Enable nnU-Net backbone
    "enable_classification": True,     # Multi-task learning
    "adaptive_weighting": True,       # Learn task balance
    "num_seg_classes": 4,             # Background, Benign, Malignant, PDC
    "num_patch_classes": 4,           # Multi-class patch classification
    "num_gland_classes": 4,           # Multi-class gland classification

    # Loss function weights
    "dice_weight": 0.5,               # Weight for Dice loss (configurable)
    "ce_weight": 0.5,                 # Weight for Cross-Entropy loss (configurable)

    # Data augmentation
    "augmentation": True,             # Enable training augmentation
    "rotation_limit": 20,             # Rotation range
    "scale_limit": 0.1,               # Scaling range

    # Optimization
    "optimizer": "adamw",             # AdamW optimizer
    "scheduler": "poly",              # Polynomial learning rate decay
    "step_size": 30,                  # Scheduler step size
    "gamma": 0.1,                     # Learning rate decay factor

    # Output settings
    "save_best_only": True,           # Save only best model
    "save_visualizations": True,      # Generate training visualizations
    "samples_per_visualization": 50,  # Samples per visualization batch
    "save_frequency": 10,             # Checkpoint save frequency (epochs)
}
```

### **4-Class Evaluation Configuration**:
```python
EVALUATION_CONFIG = {
    # Post-training evaluation
    "num_train_samples": 50,          # Samples for training evaluation
    "num_test_samples": 50,           # Samples for test evaluation
    "samples_per_figure": 5,          # Samples per visualization figure
    "figure_dpi": 200,                # High-quality figures
    "figure_size": (20, 25),          # Large figure size

    # 4-class visualization colors
    "color_background": [0, 0, 0],    # Black
    "color_benign": [0, 255, 0],      # Green
    "color_malignant": [255, 0, 0],   # Red
    "color_pdc": [0, 0, 255],         # Blue
    "overlay_alpha": 0.6,             # Transparency for overlays

    # Class names for analysis
    "class_names": {
        0: "Background",
        1: "Benign Glands",
        2: "Malignant/Tumor Glands",
        3: "PDC (Poorly Differentiated Carcinoma)"
    },

    # Random seed for reproducible sampling
    "random_seed": 42,
}
```

---

## 🔧 **Key Functions**

### **Dataset Management**:

#### **`get_dataset_path(dataset_key)`**
```python
# Get path to specific combined dataset
dataset_path = get_dataset_path('mag5x')
# Returns: "/fs/scratch/PAS2942/Datasets/CRC/Datasets/OSU/Gland_Datasets/nnUNetCombined/Task005_Combined_Mag5x"
```

#### **`validate_dataset_path(dataset_key)`**
```python
# Validate dataset structure
is_valid = validate_dataset_path('mag10x')
# Returns: True if dataset has proper nnU-Net structure
```

#### **`list_available_datasets()`**
```python
# Get status of all datasets
datasets = list_available_datasets()
# Returns: {'mag5x': {'path': '...', 'exists': True, 'valid': True}, ...}
```

### **Configuration Creation**:

#### **`create_config_dict(dataset_key, **overrides)`**
```python
# Create complete configuration for training
config = create_config_dict(
    dataset_key='mag5x',
    epochs=200,                    # Override default epochs
    batch_size=8,                  # Override default batch size
    use_multilabel_patch=True      # Enable multi-label patches
)
```

### **Environment Setup**:

#### **`print_config_summary(dataset_key)`**
```python
# Display comprehensive configuration summary
print_config_summary('mag10x')
```

**Output Example**:
```
================================================================================
Combined Gland Segmentation nnU-Net - Configuration Summary
================================================================================

📊 TARGET DATASET: mag10x
  ✅ Path: /fs/scratch/.../Task010_Combined_Mag10x
  ✅ Structure: Valid nnU-Net format

📁 AVAILABLE DATASETS:
  ✅✅ mag5x: /fs/scratch/.../Task005_Combined_Mag5x
  ✅✅ mag10x: /fs/scratch/.../Task010_Combined_Mag10x (CURRENT)
  ✅✅ mag20x: /fs/scratch/.../Task020_Combined_Mag20x
  ✅✅ mag40x: /fs/scratch/.../Task040_Combined_Mag40x
  ✅❌ mixed: /fs/scratch/.../Task001_Combined_Mixed_Magnifications

⚙️ 4-CLASS CONFIGURATION:
  🎯 Segmentation classes: 4
  🏷️ Patch classes: 4
  🔍 Gland classes: 4
  🏃 Training epochs: 150
  📦 Batch size: 4

🎨 CLASS COLORS:
  🎨 Class 0 (Background): RGB[0, 0, 0]
  🎨 Class 1 (Benign Glands): RGB[0, 255, 0]
  🎨 Class 2 (Malignant/Tumor Glands): RGB[255, 0, 0]
  🎨 Class 3 (PDC): RGB[0, 0, 255]
================================================================================
```

---

## 🎯 **Usage Examples**

### **1. Quick Configuration Check**:
```python
from configs.paths_config import print_config_summary

# Check current configuration
print_config_summary()

# Check specific dataset
print_config_summary('mag20x')
```

### **2. Training Configuration**:
```python
from configs.paths_config import create_config_dict

# Create configuration for training
config = create_config_dict(
    dataset_key='mag5x',
    epochs=100,                    # Shorter training
    batch_size=8,                  # Larger batch size
    learning_rate=5e-5,            # Lower learning rate
    adaptive_weighting=True,       # Enable adaptive loss weighting
    save_frequency=5               # Save checkpoints more frequently
)

# Use in training
model = create_multitask_model(**config)
optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'])
```

### **3. Dataset Validation**:
```python
from configs.paths_config import list_available_datasets, validate_dataset_path

# Check all datasets
datasets = list_available_datasets()
for name, info in datasets.items():
    status = "✅ READY" if info['valid'] else "❌ NOT READY"
    print(f"{status} {name}: {info['path']}")

# Validate specific dataset
if validate_dataset_path('mag10x'):
    print("✅ Dataset ready for training!")
else:
    print("❌ Dataset validation failed!")
```

### **4. Environment Setup**:
```python
# Configuration is automatically loaded on import
from configs.paths_config import DEFAULT_CONFIG, EVALUATION_CONFIG

# Access pre-configured settings
print(f"Training epochs: {DEFAULT_CONFIG['epochs']}")
print(f"Class names: {EVALUATION_CONFIG['class_names']}")
```

---

## 🔧 **Customization**

### **Custom Dataset Paths**:
```python
# Add custom dataset location
AVAILABLE_DATASETS['custom'] = "/path/to/custom/dataset"

# Or override at runtime
config = create_config_dict(
    dataset_key='mag5x',
    data_root='/custom/path/to/dataset'  # Override default path
)
```

### **Training Parameters**:
```python
# Research configuration
research_config = create_config_dict(
    dataset_key='mixed',           # Use all magnifications
    epochs=300,                    # Extended training
    batch_size=2,                  # Large model, small batch
    learning_rate=1e-5,            # Conservative learning rate
    save_frequency=25              # Less frequent saving
)

# Quick experimentation
experiment_config = create_config_dict(
    dataset_key='mag5x',
    epochs=50,                     # Quick training
    batch_size=16,                 # Small model, large batch
    learning_rate=1e-3,            # Aggressive learning rate
    save_frequency=5               # Frequent saving
)
```

### **Evaluation Settings**:
```python
# Custom evaluation colors
custom_colors = {
    "color_background": [50, 50, 50],     # Dark gray
    "color_benign": [0, 255, 0],          # Green
    "color_malignant": [255, 165, 0],     # Orange
    "color_pdc": [128, 0, 128]            # Purple
}

config = create_config_dict(
    dataset_key='mag10x',
    **custom_colors
)
```

---

## 📊 **Dataset Information**

### **Magnification-Specific Datasets**:

| Key | Magnification | Samples | Training Time | Use Case |
|-----|---------------|---------|---------------|----------|
| `mag5x` | 5x only | 4,872 | ~12 hours | Low-resolution analysis |
| `mag10x` | 10x only | 6,188 | ~15 hours | Standard clinical resolution |
| `mag20x` | 20x only | ~6,000 | ~15 hours | High-detail analysis |
| `mag40x` | 40x only | ~6,000 | ~15 hours | Ultra-high resolution |
| `mixed` | All combined | ~25,000 | ~72 hours | Multi-scale learning |

### **Dataset Structure Requirements**:
```
TaskXXX_Combined_MagXx/
├── imagesTr/           # Training images
├── labelsTr/           # Training segmentation masks
├── imagesVal/          # Validation images
├── labelsVal/          # Validation segmentation masks
├── imagesTs/           # Test images
├── labelsTs/           # Test segmentation masks
├── dataset.json        # nnU-Net dataset configuration
└── processing_summary.json  # Dataset creation metadata
```

---

## 🎯 **Best Practices**

### **1. Dataset Selection**:
- **Start Small**: Begin with `mag5x` or `mag10x` for initial experiments
- **Scale Up**: Move to `mixed` for final research models
- **Compare**: Train on individual magnifications for magnification-specific analysis

### **2. Configuration Management**:
- **Use Defaults**: Start with default configuration for proven performance
- **Incremental Changes**: Modify one parameter at a time
- **Document Overrides**: Keep track of configuration changes for reproducibility

### **3. Environment Validation**:
- **Always Check**: Run `print_config_summary()` before training
- **Validate Paths**: Ensure datasets exist and are properly formatted
- **Monitor Space**: Check disk space for large mixed dataset training

### **4. Performance Optimization**:
- **Batch Size**: Adjust based on GPU memory (4-16 typical range)
- **Workers**: Set `num_workers` to match CPU cores (4-8 typical)
- **Frequency**: Adjust `save_frequency` based on training duration

---

## 🔍 **Troubleshooting**

### **Common Issues**:

1. **"Dataset not found"**:
   ```python
   # Check dataset status
   python configs/paths_config.py
   ```

2. **"Invalid dataset structure"**:
   ```python
   # Validate specific dataset
   from configs.paths_config import validate_dataset_path
   print(validate_dataset_path('mag5x'))
   ```

3. **"Configuration conflicts"**:
   ```python
   # Check current configuration
   from configs.paths_config import print_config_summary
   print_config_summary('mag10x')
   ```

### **Performance Issues**:
- **GPU Memory**: Reduce `batch_size` if out-of-memory errors
- **Training Speed**: Increase `num_workers` for faster data loading
- **Storage**: Ensure sufficient disk space for checkpoints and logs

---

**Centralized configuration management for seamless 4-class multi-task training!** 🚀