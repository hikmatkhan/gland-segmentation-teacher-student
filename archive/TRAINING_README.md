# 4-Class nnU-Net Multi-Task Training for Combined Gland Segmentation

This project implements a comprehensive 4-class multi-task deep learning approach for glandular structure analysis in histopathology images. The system performs **4-class gland segmentation** and **multi-level classification** (patch-level and individual gland-level) using combined Warwick GlaS + OSU Makoto datasets.

## Key Features

### Multi-Task Architecture
- **4-Class Segmentation Branch**: nnU-Net based architecture for precise gland boundary detection
  - Background (0), Benign Glands (1), Malignant Glands (2), PDC - Poorly Differentiated Carcinoma (3)
- **Multi-Label Patch Classification Head**: Handles patches containing multiple gland types simultaneously
- **Gland Classification Head**: Individual gland-level 4-class classification
- **Shared nnU-Net Encoder**: Efficient feature learning across all tasks with 20M+ parameters

### Advanced Training Pipeline
- **Multi-Label Learning**: Realistic histopathology patches with multiple gland types
- **Adaptive Loss Weighting**: Automatic balancing of segmentation and classification losses
- **Comprehensive Data Augmentation**: Optimized for histopathology domain
- **Research-Ready Training**: 100+ epochs with early stopping and advanced scheduling
- **Rich Visualizations**: Composite images showing predictions at multiple levels

### Combined Datasets
- **Warwick GlaS Dataset**: 165 histopathology images with 3-class gland annotations
- **OSU Makoto Dataset**: 32 slides × 4 magnifications (5x, 10x, 20x, 40x) with 4-class annotations
- **Unified 4-Class System**: Background, Benign, Malignant, PDC
- **Multi-Magnification Support**: Mixed or separate magnification training strategies

## Architecture Overview

```
Input Image → nnU-Net Encoder (PlainConvUNet) → Bottleneck Features (512 channels)
                                                        ↓
                    ┌──────────────────────────────────────────────────────┐
                    │                                                      │
                    ▼                                                      ▼
            nnU-Net Decoder                                    Classification Heads
                    │                                                      │
                    ▼                                                      ▼
          4-Class Segmentation Output              ┌────────────────────────────────┐
          (Background, Benign, Malignant, PDC)     │                                │
                                                  ▼                                ▼
                                        Multi-Label Patch Head            Gland Head
                                        (BCEWithLogitsLoss)              (CrossEntropyLoss)
                                                  │                                │
                                                  ▼                                ▼
                                        Multi-Label Predictions           Gland Predictions
                                        [B, 4] binary labels              4-class per gland
```

## Project Structure

```
nnUNet/
├── src/
│   ├── models/                    # Model architectures
│   │   ├── nnunet_integration.py  # nnU-Net backbone integration
│   │   ├── multi_task_wrapper.py  # Multi-task model wrapper
│   │   ├── projection_heads.py    # Classification heads
│   │   └── loss_functions.py      # Multi-task loss functions
│   ├── training/                  # Training pipeline
│   │   ├── dataset.py            # Data loading and augmentation
│   │   └── trainer.py            # Training loop
│   └── evaluation/               # Evaluation and visualization
│       └── evaluator.py          # Comprehensive evaluation
├── configs/                      # Configuration files
│   └── paths_config.py          # Dataset paths and configurations
├── outputs/                      # Training outputs
│   └── exp_YYYY-MM-DD_HH-MM-SS/  # Auto-generated experiment directory
│       ├── models/               # Model checkpoints (best_model.pth, latest_model.pth)
│       ├── logs/                 # Training logs and TensorBoard
│       ├── evaluations/          # 🔥 NEW: Post-training evaluation results
│       │   ├── final_evaluation_metrics.csv      # Complete dataset metrics
│       │   ├── evaluation_summary_report.md      # Detailed analysis report
│       │   └── detailed_metrics.json             # Full metrics breakdown
│       └── visualizations/       # 🔥 NEW: 4-column visualization figures
│           ├── train_evaluation_samples_001.png  # 100 train samples
│           ├── val_evaluation_samples_001.png    # 100 val samples
│           ├── test_evaluation_samples_001.png   # 100 test samples
│           ├── sample_indices.json               # Reproducible sample selection
│           └── training_curves.png               # Training progress plots
├── scripts/                      # Training scripts
├── main.py                       # Main entry point
└── TRAINING_README.md            # This documentation
```

## Quick Start

### 1. Environment Setup

First, ensure you have the required dependencies:

```bash
# Activate your conda environment
conda activate llm  # or your preferred environment

# Verify PyTorch and CUDA
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"

# Verify nnU-Net components (optional but recommended)
python -c "from dynamic_network_architectures.architectures.unet import PlainConvUNet; print('nnU-Net components available')"
```

### 2. Quick Demo

Test all components without full training:

```bash
cd /users/PAS2942/hikmat179/Code/DLPath/CRC/GlandSegmentation/GlandSegModels/nnUNet
python main.py demo
```

This tests:
- Model creation and architecture
- Loss function initialization
- Data loading capabilities
- Forward pass functionality
- Configuration validation

### 3. Training Options

#### Standard Training (Mixed Magnifications - Recommended)
```bash
python main.py train --dataset mixed --epochs 150 --batch_size 4 --output_dir /path/to/outputs
```

#### Enhanced Training with Stronger Augmentation
```bash
python main.py train --dataset mixed --enhanced --epochs 150 --batch_size 4 --output_dir /path/to/outputs
```

#### Magnification-Specific Training
```bash
# Train on specific magnifications
python main.py train --dataset mag20x --epochs 150 --batch_size 4 --output_dir /path/to/outputs
python main.py train --dataset mag40x --epochs 150 --batch_size 4 --output_dir /path/to/outputs
```

#### Research Training with Custom Configuration
```bash
python main.py train \
    --dataset mixed \
    --epochs 200 \
    --batch_size 6 \
    --learning_rate 5e-5 \
    --enhanced \
    --patience 40 \
    --output_dir /path/to/outputs \
    --experiment_name "research_4class_v1"
```

### 4. Evaluation

**🔥 NEW: Automatic Post-Training Evaluation**

The system now automatically runs comprehensive evaluation after training completes! This includes:
- **Complete Dataset Evaluation**: Evaluates on ENTIRE train/val/test datasets for robust statistics
- **4-Column Visualizations**: 100 randomly sampled images per split with publication-ready figures
- **Comprehensive Metrics**: Detailed performance tables and reports

**Automatic Integration (Recommended):**
```bash
# Training automatically includes post-training evaluation
python main.py train --dataset mixed --epochs 150 --output_dir /path/to/outputs
# → Post-training evaluation runs automatically after training completes!
```

**Manual Evaluation (Optional):**
```bash
python main.py evaluate \
    --model outputs/research_4class_v1/models/best_model.pth \
    --dataset mixed \
    --output outputs/evaluation_results \
    --visualize
```

## Implementation Details

### Multi-Task Loss Function

The system uses adaptive loss weighting to automatically balance three loss components:

```
Total Loss = α × Segmentation Loss + β × Patch Classification Loss + γ × Gland Classification Loss
```

Where α, β, γ are learned parameters that adapt during training based on task uncertainty.

### Segmentation Loss (4-Class)

Combined Dice + Cross-Entropy loss optimized for 4-class histopathology data:

```python
L_seg = 0.5 × Dice Loss + 0.5 × Weighted Cross-Entropy Loss
```

### Multi-Label Patch Classification Loss

Binary Cross-Entropy with Logits for realistic multi-label scenarios:

```python
L_patch = BCEWithLogitsLoss(patch_predictions, multi_label_targets)
```

### Single-Label Gland Classification Loss

Cross-Entropy for individual gland classification:

```python
L_gland = CrossEntropyLoss(gland_predictions, gland_labels)
```

### Multi-Label Patch Generation

Automatically generates multi-label targets from segmentation masks:

```python
def create_multilabel_patch_labels_from_segmentation(mask, min_pixels=50):
    """
    Creates [B, 4] binary tensor indicating presence of each class
    - Background: pixel count > threshold
    - Benign: benign gland pixels > threshold
    - Malignant: malignant gland pixels > threshold
    - PDC: PDC gland pixels > threshold
    """
```

### Data Augmentation

Histopathology-specific augmentations:

**Geometric Transformations:**
- Random rotation (0-360°)
- Random horizontal/vertical flipping
- Random scaling (0.8-1.2x)
- Elastic deformation

**Photometric Augmentations:**
- Brightness adjustment (±20%)
- Contrast enhancement (±20%)
- Hue/Saturation shifts (±10%)
- Gaussian noise injection

**Advanced Augmentations:**
- Gaussian blur (σ=0.5-1.5)
- Motion blur simulation
- Grid distortion
- Coarse dropout

### Training Configuration

**Default Research Configuration:**
```python
DEFAULT_CONFIG = {
    'epochs': 150,
    'batch_size': 4,
    'learning_rate': 1e-4,
    'optimizer': 'AdamW',
    'scheduler': 'CosineAnnealingLR',
    'weight_decay': 1e-5,
    'num_seg_classes': 4,
    'num_patch_classes': 4,
    'num_gland_classes': 4,
    'use_multilabel_patch': True,
    'adaptive_weighting': True,
    'early_stop_patience': 30,
    'use_deep_supervision': True
}
```

### Evaluation Metrics

**4-Class Segmentation:**
- Per-class Dice Coefficient (Background, Benign, Malignant, PDC)
- Mean Dice across all classes
- Per-class Intersection over Union (IoU)
- Pixel-wise accuracy
- Confusion matrices

**Multi-Label Patch Classification:**
- Exact Match Accuracy (all labels correct)
- Per-class binary accuracy
- Per-class F1 scores
- Per-class AUC-ROC scores
- Hamming Loss
- Multi-label confusion matrices

**Single-Label Gland Classification:**
- 4-class accuracy
- Per-class precision, recall, F1
- Multi-class AUC-ROC
- Classification reports

### Visualization Outputs

The system generates comprehensive visualizations:

**🔥 NEW: 4-Column Post-Training Visualizations**
1. **4-Column Evaluation Figures** (100 samples per split):
   - **Column 1**: Original patch image
   - **Column 2**: Ground truth segmentation mask (colored)
   - **Column 3**: Predicted segmentation mask (colored)
   - **Column 4**: Overlay prediction on original image
   - **Format**: 10 figures × 10 samples each per split (train/val/test)
   - **Quality**: 300 DPI publication-ready PNG files

2. **Comprehensive Evaluation Reports**:
   - **CSV Metrics Table**: Final performance across all splits
   - **Markdown Report**: Detailed analysis with findings and recommendations
   - **JSON Details**: Complete metrics breakdown for programmatic access

3. **Training Monitoring Visualizations**:
   - Loss curves (training/validation)
   - Dice score progression
   - Multi-label accuracy trends
   - Learning rate schedule

4. **Statistical Analysis Visualizations**:
   - 4×4 segmentation confusion matrix
   - Per-class binary classification matrices
   - Multi-class gland classification matrix
   - Bar plots of key performance indicators
   - Per-class performance breakdowns

## Dataset-Specific Training

### Mixed Magnifications (Recommended)
```bash
python main.py train --dataset mixed --epochs 150 --output_dir /path/to/outputs
```
- **Best for**: General-purpose models, magnification-invariant learning
- **Data**: All magnifications combined from OSU Makoto + Warwick GlaS
- **Use case**: Clinical deployment across different imaging systems

### Magnification-Specific Training
```bash
# High-resolution detailed analysis
python main.py train --dataset mag40x --epochs 150 --output_dir /path/to/outputs

# Medium resolution balance
python main.py train --dataset mag20x --epochs 150 --output_dir /path/to/outputs

# Lower resolution overview
python main.py train --dataset mag10x --epochs 150 --output_dir /path/to/outputs

# Very low resolution screening
python main.py train --dataset mag5x --epochs 150 --output_dir /path/to/outputs
```

### Training Performance Expected

**Hardware Requirements:**
- **GPU**: NVIDIA GPU with ≥8GB VRAM (RTX 3080/4080 or better)
- **RAM**: ≥32GB system memory recommended
- **Storage**: ≥50GB available space for outputs

**Training Times (Approximate):**
- **Mixed Dataset**: ~8-12 hours for 150 epochs (RTX 4080)
- **Single Magnification**: ~4-6 hours for 150 epochs
- **Enhanced Training**: +20-30% longer due to stronger augmentation

**Expected Performance:**
- **Segmentation Dice**: >0.85 for well-trained models
- **Multi-Label Accuracy**: >0.75 exact match
- **Per-Class F1**: >0.80 for common classes (Benign, Malignant)
- **PDC Detection**: Challenging class, expect >0.70 F1

## Advanced Usage

### Custom Training Configuration

Create custom configuration files:

```python
# custom_config.py
CUSTOM_CONFIG = {
    'epochs': 200,
    'batch_size': 8,
    'learning_rate': 2e-4,
    'optimizer': 'AdamW',
    'scheduler': 'ReduceLROnPlateau',
    'weight_decay': 1e-4,
    'use_focal_loss': True,
    'augmentation_strength': 'strong',
    'early_stop_patience': 50
}
```

### Resuming Training

```bash
# Resume from checkpoint (to be implemented)
python main.py train --dataset mixed --resume outputs/experiment/checkpoints/latest_model.pth --output_dir /path/to/outputs
```

### Multi-GPU Training

```bash
# Use multiple GPUs (to be implemented)
CUDA_VISIBLE_DEVICES=0,1 python main.py train --dataset mixed --multi_gpu --output_dir /path/to/outputs
```

### Hyperparameter Optimization

```bash
# Different learning rates
python main.py train --dataset mixed --learning_rate 1e-5 --experiment_name "lr_1e5" --output_dir /path/to/outputs
python main.py train --dataset mixed --learning_rate 5e-4 --experiment_name "lr_5e4" --output_dir /path/to/outputs

# Different batch sizes
python main.py train --dataset mixed --batch_size 2 --experiment_name "batch_2" --output_dir /path/to/outputs
python main.py train --dataset mixed --batch_size 8 --experiment_name "batch_8" --output_dir /path/to/outputs
```

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   ```bash
   # Reduce batch size
   python main.py train --dataset mixed --batch_size 2 --output_dir /path/to/outputs

   # Or use gradient accumulation (to be implemented)
   python main.py train --dataset mixed --batch_size 2 --accumulate_grad_batches 2 --output_dir /path/to/outputs
   ```

2. **Dataset Not Found**
   ```bash
   # Check available datasets
   python -c "from configs.paths_config import list_available_datasets; print(list_available_datasets())"

   # Verify dataset path
   python -c "from configs.paths_config import get_dataset_path, validate_dataset_path; print(validate_dataset_path(get_dataset_path('mixed')))"
   ```

3. **Import Errors**
   ```bash
   # Test imports
   python main.py demo

   # Check Python path
   cd /users/PAS2942/hikmat179/Code/DLPath/CRC/GlandSegmentation/GlandSegModels/nnUNet
   python -c "import sys; print(sys.path)"
   ```

4. **Training Divergence**
   - Reduce learning rate: `--learning_rate 5e-5`
   - Increase patience: `--patience 50`
   - Check data quality and class distribution

### Performance Optimization

1. **Speed up training**:
   - Increase `num_workers` in dataset configuration
   - Use mixed precision training (to be implemented)
   - Optimize data loading pipeline

2. **Improve model performance**:
   - Increase training epochs: `--epochs 200`
   - Use enhanced augmentation: `--enhanced`
   - Experiment with different optimizers and schedulers

3. **Memory optimization**:
   - Reduce batch size and use gradient accumulation
   - Enable gradient checkpointing (to be implemented)
   - Use smaller input image sizes

## Key Results Expected

1. **Precise 4-Class Segmentation**: Accurate boundary detection for all glandular structures including PDC
2. **Robust Multi-Label Classification**: Realistic patch-level analysis handling multiple gland types
3. **Individual Gland Analysis**: Per-gland 4-class assessment for detailed diagnosis
4. **Rich Feature Learning**: Shared nnU-Net representations benefiting all tasks through multi-task learning
5. **Magnification Generalization**: Models that work across different imaging magnifications

## Technical Requirements

- **Python**: 3.9+
- **PyTorch**: 2.0+
- **nnU-Net**: v2 (optional, uses local implementation)
- **CUDA**: 11.8+ for GPU acceleration
- **Additional**: OpenCV, scikit-image, albumentations, matplotlib, seaborn
- **Monitoring**: TensorBoard for training visualization

## Citation and Acknowledgments

This implementation combines:
- **Warwick GlaS Dataset**: For established gland segmentation benchmarks
- **OSU Makoto Dataset**: For multi-magnification 4-class annotations
- **nnU-Net Architecture**: For medical image segmentation backbone
- **Multi-Task Learning**: For comprehensive histopathology analysis

## Research Applications

This framework is designed for:
- **Digital Pathology Research**: Advanced glandular structure analysis
- **Clinical Decision Support**: Multi-class gland classification
- **Histopathology Education**: Detailed visualization and analysis
- **Method Development**: Baseline for novel segmentation and classification approaches

---

**Note**: This is a research implementation for educational and scientific purposes. For clinical applications, additional validation and regulatory approval would be required.

## Next Steps

After successful training:

1. **🔥 NEW: Automatic Evaluation Complete**: Post-training evaluation runs automatically!
   - Check `outputs/exp_*/evaluations/final_evaluation_metrics.csv` for comprehensive metrics
   - Review `outputs/exp_*/evaluations/evaluation_summary_report.md` for detailed analysis
   - Examine `outputs/exp_*/visualizations/` for 4-column visualization figures

2. **Analyze Results**:
   - **Performance Metrics**: Review CSV table with train/val/test performance
   - **Visual Analysis**: Examine 4-column figures showing predictions vs ground truth
   - **Class-Specific Performance**: Check per-class Dice scores and classification accuracies

3. **Quality Assessment**:
   - **Segmentation Quality**: Look for clean boundaries in 4-class predictions
   - **Multi-Label Performance**: Verify patch-level multi-class detection accuracy
   - **Generalization**: Compare train vs validation vs test performance

4. **Research Applications**:
   - Compare magnifications: Train models on different magnification strategies
   - Ablation studies: Test different architectures or loss functions
   - Dataset expansion: Incorporate additional datasets or annotations

5. **Deployment Preparation**:
   - Export best models for clinical or research deployment
   - Document model performance and limitations
   - Prepare inference pipelines for production use

**🔥 NEW Output Structure**: All results are automatically organized in:
```
outputs/exp_YYYY-MM-DD_HH-MM-SS/
├── models/best_model.pth                        # Best trained model
├── evaluations/                                 # Complete evaluation results
│   ├── final_evaluation_metrics.csv           # Performance table
│   ├── evaluation_summary_report.md           # Detailed analysis
│   └── detailed_metrics.json                  # Full metrics
└── visualizations/                             # Publication-ready figures
    ├── train_evaluation_samples_001.png       # 100 train samples
    ├── val_evaluation_samples_001.png         # 100 val samples
    ├── test_evaluation_samples_001.png        # 100 test samples
    └── sample_indices.json                    # Reproducible sampling
```

For questions or issues, check the comprehensive evaluation outputs and logs in the `outputs/` directory.