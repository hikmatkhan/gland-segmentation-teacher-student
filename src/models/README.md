# Models Module - 4-Class Multi-Task nnU-Net

This module contains the core model components for 4-class gland segmentation with multi-label patch classification.

## 📁 **Files Overview**

### **1. `nnunet_integration.py`**
**Purpose**: nnU-Net environment setup and architecture creation

**Key Functions**:
- `setup_nnunet_environment()`: Configure nnU-Net paths and directories
- `create_nnunet_architecture()`: Create 4-class nnU-Net model
- `import_nnunet_components()`: Import nnU-Net dependencies with fallbacks

**Usage**:
```python
from src.models.nnunet_integration import create_nnunet_architecture

# Create 4-class nnU-Net model
model = create_nnunet_architecture(
    input_channels=3,        # RGB histopathology images
    num_classes=4,          # Background, Benign, Malignant, PDC
    deep_supervision=True   # Enable multi-resolution training
)
```

**Architecture Details**:
- **Base**: PlainConvUNet from dynamic_network_architectures
- **Stages**: 6-stage encoder-decoder
- **Features**: [32, 64, 128, 256, 512, 512] per stage
- **Parameters**: ~20M parameters
- **Output**: 4-class segmentation logits

---

### **2. `projection_heads.py`**
**Purpose**: Multi-label classification heads for patch and gland-level classification

**Key Classes**:
- `PatchClassificationHead`: Multi-label patch classification (can have multiple gland types)
- `GlandClassificationHead`: Single-label individual gland classification
- `MultiClassDualClassificationHead`: Combined patch + gland classification

**Key Functions**:
- `create_multilabel_patch_labels_from_segmentation()`: Generate multi-label targets from masks
- `analyze_patch_class_distribution()`: Analyze multi-class presence in patches

**Multi-Label Patch Classification**:
```python
from src.models.projection_heads import PatchClassificationHead

# Multi-label patch classifier
patch_head = PatchClassificationHead(
    input_channels=512,      # Bottleneck features
    num_classes=4,          # 4-class output
    dropout_p=0.5
)

# Input: [B, 512, H, W] features
# Output: [B, 4] multi-label logits (no sigmoid - applied in loss)
```

**Multi-Label Generation**:
```python
# Automatic multi-label generation from segmentation
segmentation_mask = torch.randint(0, 4, (batch_size, 512, 512))
multilabel_targets = create_multilabel_patch_labels_from_segmentation(
    segmentation_mask,
    min_pixels_threshold=50  # Minimum pixels for class presence
)
# Output: [B, 4] binary tensor indicating class presence
```

**Real Examples**:
- Patch with Benign + Malignant glands → `[1, 1, 1, 0]` (Background + Benign + Malignant + No PDC)
- Patch with only PDC → `[1, 0, 0, 1]` (Background + No Benign + No Malignant + PDC)

---

### **3. `multi_task_wrapper.py`**
**Purpose**: Complete multi-task model combining nnU-Net with classification heads

**Key Classes**:
- `MultiTaskWrapper`: Main multi-task model wrapper
- `create_multitask_model()`: Factory function for complete model creation

**Architecture Flow**:
```
Input Image [B, 3, 512, 512]
    ↓
nnU-Net Encoder-Decoder
    ↓
├── Segmentation Output [B, 4, 512, 512]
└── Bottleneck Features [B, 512, H, W]
        ↓
    Classification Heads
        ↓
    ├── Patch Classification [B, 4] (Multi-label)
    └── Gland Classification [N, 4] (Single-label per gland)
```

**Usage**:
```python
from src.models.multi_task_wrapper import create_multitask_model

# Create complete multi-task model
model = create_multitask_model(
    input_channels=3,
    num_seg_classes=4,
    enable_classification=True
)

# Forward pass
outputs = model(images)
# Returns:
# {
#     'segmentation': [B, 4, H, W],
#     'patch_classification': [B, 4],
#     'gland_classification': [N, 4],
#     'gland_counts': [B],
#     'deep_supervision': [list of multi-resolution outputs]
# }
```

**Hook Registration**:
- Automatically registers hooks on nnU-Net encoder to capture bottleneck features
- Handles different nnU-Net architectures robustly
- Fallback mechanisms for hook registration

---

### **4. `loss_functions.py`**
**Purpose**: Advanced loss functions for 4-class multi-task learning

**Key Classes**:
- `FourClassDiceLoss`: Dice loss for 4-class segmentation
- `WeightedCrossEntropyLoss`: Handles class imbalance
- `MultiTaskLoss`: Combined multi-task loss with adaptive weighting
- `FocalLoss`: For hard example mining in classification

**Multi-Label Loss**:
```python
from src.models.loss_functions import MultiTaskLoss

# Multi-label multi-task loss
loss_fn = MultiTaskLoss(
    use_multilabel_patch=True,      # Enable multi-label patches
    use_adaptive_weighting=True,    # Learn optimal task weights
    dice_weight=0.5,               # Dice vs CE balance (configurable via shell script)
    ce_weight=0.5,                 # Configurable via DICE_WEIGHT/CE_WEIGHT parameters
    use_focal_loss=False           # Standard CE for glands
)

# Loss computation
losses = loss_fn(outputs, targets)
# Returns:
# {
#     'total': combined_loss,
#     'segmentation': seg_loss,
#     'patch_classification': patch_loss,    # BCEWithLogitsLoss for multi-label
#     'gland_classification': gland_loss,    # CrossEntropyLoss for single-label
#     'weights': [task_weights]              # If adaptive weighting
# }
```

**Adaptive Weighting**:
- Learns optimal balance between segmentation, patch, and gland classification
- Based on uncertainty estimation (homoscedastic uncertainty)
- Prevents one task from dominating training

**Class Imbalance Handling**:
- Weighted Cross-Entropy with automatic weight calculation
- Focal Loss option for hard examples
- Background class downweighting

---

## 🔧 **Integration Examples**

### **Complete Model Creation**:
```python
from src.models.multi_task_wrapper import create_multitask_model
from src.models.loss_functions import MultiTaskLoss

# Create model
model = create_multitask_model(
    input_channels=3,
    num_seg_classes=4,
    enable_classification=True
)

# Create loss function
loss_fn = MultiTaskLoss(use_multilabel_patch=True)

# Forward pass
outputs = model(images)
losses = loss_fn(outputs, targets)

# Training step
loss = losses['total']
loss.backward()
optimizer.step()
```

### **Multi-Label Target Generation**:
```python
from src.models.projection_heads import create_multilabel_patch_labels_from_segmentation

# Generate multi-label targets from segmentation masks
batch_segmentation = batch['segmentation_targets']  # [B, H, W]
multilabel_patches = create_multilabel_patch_labels_from_segmentation(
    batch_segmentation,
    min_pixels_threshold=50
)

# Use in loss computation
targets = {
    'segmentation': batch_segmentation,
    'patch_labels': multilabel_patches,    # [B, 4] multi-label
    'gland_labels': batch['gland_labels']  # [N] single-label
}
```

### **Model Analysis**:
```python
from src.models.projection_heads import analyze_patch_class_distribution

# Analyze multi-class presence in a batch
analysis = analyze_patch_class_distribution(batch_segmentation)

print(f"Multi-class patches: {analysis['patches_with_multiple_classes']}")
print(f"Class combinations: {analysis['class_combinations']}")
print(f"Pure class patches: {analysis['pure_class_patches']}")
```

## 🎯 **Key Design Decisions**

### **1. Multi-Label vs Single-Label**
- **Patches**: Multi-label (realistic for histopathology)
- **Glands**: Single-label (each gland has primary type)
- **Segmentation**: Single-label per pixel (mutually exclusive classes)

### **2. Loss Function Strategy**
- **Segmentation**: Dice + Weighted CE for class imbalance
- **Patch Classification**: BCEWithLogitsLoss for multi-label
- **Gland Classification**: CrossEntropyLoss for single-label
- **Adaptive Weighting**: Learns task importance automatically

### **3. Feature Extraction**
- **Bottleneck Features**: Deepest encoder features (highest semantic content)
- **Global Pooling**: For patch-level classification
- **Region-Based**: For individual gland classification

### **4. Architecture Integration**
- **Minimal Modification**: Standard nnU-Net + classification heads
- **Hook-Based**: Non-intrusive feature extraction
- **Fallback Support**: Handles different nnU-Net versions

---

## 📊 **Performance Characteristics**

### **Model Size**:
- **nnU-Net Backbone**: ~20M parameters
- **Classification Heads**: ~260K parameters
- **Total**: ~20.3M parameters

### **Memory Usage**:
- **Training**: ~8GB GPU (batch_size=4, 512x512)
- **Inference**: ~2GB GPU
- **Feature Maps**: Efficient hook-based extraction

### **Computational Complexity**:
- **Segmentation**: O(HW) per pixel
- **Patch Classification**: O(C) global features
- **Gland Classification**: O(N*C) per detected gland

---

**Ready for state-of-the-art 4-class multi-task gland segmentation!** 🚀