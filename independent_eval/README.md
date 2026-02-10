# Independent Model Evaluator for GlandSegModels nnU-Net

This directory contains tools for independent evaluation of trained GlandSegModels nnU-Net models. The evaluator loads a trained model from an experiment directory and performs comprehensive evaluation on training, validation, and test datasets.

## 📁 Directory Structure

```
independent_eval/
├── independent_evaluator.py     # Main evaluation script
├── test_independent_evaluator.sh # Test shell script with configuration
├── README.md                     # This documentation
└── [evaluation results will be saved here when run]
```

## 🚀 Quick Start

### Step 1: Configure the Evaluation

Edit `test_independent_evaluator.sh` and set these key parameters:

```bash
# MUST match your training configuration
ARCHITECTURE="baseline_unet"     # or "nnunet"
DATASET_KEY="mag40x"            # mixed, mag5x, mag10x, mag20x, mag40x

# Path to your trained experiment
EXPERIMENT_PATH="/path/to/your/experiment_directory"

# Base dataset directory (YOU specify the exact dataset location)
DATASET_BASE_DIR="/fs/scratch/PAS2942/Datasets/CRC/Datasets/OSU/Gland_Datasets/_OSU_All_Gland_Datasets_nnUNet"

# Output path for results
OUTPUT_PATH="/fs/scratch/PAS2942/Datasets/CRC/Datasets/OSU/Gland_Datasets/independent_evaluation"
```

### Step 2: Run the Evaluation

```bash
cd independent_eval
chmod +x test_independent_evaluator.sh
./test_independent_evaluator.sh
```

### Step 3: View Results

Results will be saved in a timestamped directory under `OUTPUT_PATH`:
- **Excel Report**: Comprehensive metrics across all splits
- **CSV Summary**: Quick summary of key metrics
- **Visualizations**: 100 sample predictions per split with overlays

## 🆕 Key Improvements (Latest Version)

### Enhanced Dataset Path Control
- **User-Specified Paths**: Now uses your exact `dataset_base_dir` instead of training config paths
- **Full Path Control**: No dependency on training configuration dataset paths
- **Flexible Evaluation**: Evaluate any compatible dataset regardless of training configuration

### Improved Metrics Accuracy
- **Exact Per-Class Metrics**: No longer approximated - computed exactly like training phase
- **Complete Gland Classification**: Full implementation of gland-level classification evaluation
- **Training-Identical Computation**: Metrics now match training phase results exactly

### Enhanced Visualizations
- **Larger Figures**: Increased from 20×16 to 24×20 for better detail visibility
- **Reduced White Space**: Optimized subplot spacing with equal horizontal/vertical margins (wspace=0.20, hspace=0.20)
- **Better Layout**: Improved legend positioning and overall figure arrangement

### Usage Benefits
```bash
# Example: Evaluate with YOUR dataset path
python independent_evaluator.py \
    --experiment_path "/path/to/experiment" \
    --dataset_base_dir "/your/exact/dataset/path" \
    --dataset_key "mag20x" \
    # ... other args
```

**Result**: The evaluator uses YOUR specified dataset location, computes exact training-like metrics, and generates enhanced visualizations.

---

## 📊 Features

### ✅ Architecture Support
- **Baseline UNet**: Simple, fast, good baseline performance
- **nnU-Net**: State-of-the-art, advanced segmentation architecture

### ✅ Dataset Support
- **mixed**: All magnifications combined (~25k samples)
- **mag5x**: 5x magnification only (~4.8k samples)
- **mag10x**: 10x magnification only (~6.2k samples)
- **mag20x**: 20x magnification only (~6k samples)
- **mag40x**: 40x magnification only (~6k samples)

### ✅ Comprehensive Metrics
- **Segmentation Metrics**: Dice Score, IoU, Pixel Accuracy (per-class and overall)
- **Multi-task Metrics**: Patch classification accuracy, gland classification accuracy
- **4-Class Analysis**: Background, Benign, Malignant, PDC individual performance
- **Statistical Metrics**: Precision, recall, F1-score for classification tasks

### ✅ Rich Visualizations
- **100 Random Samples** per dataset split for visualization
- **4-Column Layout**: Original image, ground truth, prediction, overlay with metrics
- **4-Class Color Scheme**: Background (black), Benign (green), Malignant (red), PDC (blue)
- **Comprehensive Overlays**: Prediction confidence, individual Dice scores, classification results

### ✅ Multi-Split Evaluation
- **Training Set**: Performance on training data (reproducibility check)
- **Validation Set**: Performance on validation data
- **Test Set**: Performance on held-out test data
- **Combined Analysis**: Cross-split comparison and analysis

## 🔧 Command Line Usage

For advanced users, run the evaluator directly:

```bash
python independent_evaluator.py \
    --experiment_path /path/to/experiment \
    --architecture baseline_unet \
    --dataset_key mag20x \
    --dataset_base_dir "/fs/scratch/PAS2942/Datasets/CRC/Datasets/OSU/Gland_Datasets/_OSU_All_Gland_Datasets_nnUNet" \
    --output /path/to/results \
    --split all \
    --num_samples 100 \
    --batch_size 4
```

### Arguments

| Argument | Options | Description |
|----------|---------|-------------|
| `--experiment_path` | Path | **Required**. Path to experiment directory with trained model |
| `--architecture` | `baseline_unet`, `nnunet` | **Required**. Architecture used in training |
| `--dataset_key` | `mixed`, `mag5x`, `mag10x`, `mag20x`, `mag40x` | **Required**. Dataset configuration |
| `--dataset_base_dir` | Path | **Required**. Base directory containing datasets |
| `--output` | Path | **Required**. Output directory for results |
| `--split` | `train`, `val`, `test`, `all` | Dataset split(s) to evaluate (default: `all`) |
| `--num_samples` | Integer | Samples per split for visualization (default: 100) |
| `--batch_size` | Integer | Evaluation batch size (default: 4) |

## 📁 Expected Directory Structure

### Experiment Directory
Your experiment directory should contain:
```
experiment_directory/
├── models/
│   └── best_model.pth           # Best trained model
├── training_config.json         # Training configuration (optional)
└── [other training artifacts]
```

### Dataset Directory
The dataset base directory should contain:
```
nnUNetCombined/
├── Task001_Combined_Mixed_Magnifications/   # mixed dataset
├── Task005_Combined_Mag5x/                  # mag5x dataset
├── Task010_Combined_Mag10x/                 # mag10x dataset
├── Task020_Combined_Mag20x/                 # mag20x dataset
└── Task040_Combined_Mag40x/                 # mag40x dataset
```

## 📊 Output Structure

Results are saved in timestamped directories:
```
evaluation_baseline_unet_mag40x_2025-09-18_14-30-22/
├── baseline_unet_mag40x_evaluation_config.json           # 🆕 Evaluation configuration & paths
├── baseline_unet_mag40x_comprehensive_evaluation.xlsx    # Detailed Excel report
├── baseline_unet_mag40x_evaluation_summary.csv           # Quick CSV summary
└── visualizations/
    ├── train_baseline_unet_mag40x_figure_1.png          # Training visualizations
    ├── val_baseline_unet_mag40x_figure_1.png            # Validation visualizations
    └── test_baseline_unet_mag40x_figure_1.png           # Test visualizations
```

### Output Files Explained

#### 🆕 **Evaluation Configuration JSON**
- **Purpose**: Sanity check to verify you evaluated the correct dataset
- **Contains**: All dataset paths, model paths, evaluation parameters
- **Key sections**:
  - `dataset_configuration`: Actual paths used for train/val/test data
  - `model_configuration`: Model and experiment paths
  - `paths_sanity_check`: Quick verification of user-specified vs actual paths
  - `dataset_verification`: Confirms all required directories exist

#### **Excel Report Sheets**
- **Split_Comparison**: Side-by-side comparison of train/val/test performance
- **Train_Detailed**: Comprehensive training set metrics
- **Val_Detailed**: Comprehensive validation set metrics
- **Test_Detailed**: Comprehensive test set metrics
- **Configuration**: Evaluation parameters and model information

## 🎯 Example Configurations

### Baseline UNet on Mixed Dataset
```bash
ARCHITECTURE="baseline_unet"
DATASET_KEY="mixed"
SPLIT="all"
NUM_SAMPLES=100
```

### nnU-Net on High-Resolution Dataset
```bash
ARCHITECTURE="nnunet"
DATASET_KEY="mag40x"
SPLIT="test"
NUM_SAMPLES=50
```

### Quick Test Evaluation
```bash
ARCHITECTURE="baseline_unet"
DATASET_KEY="mag5x"
SPLIT="val"
NUM_SAMPLES=25
BATCH_SIZE=2
```

## 🔍 Troubleshooting

### Common Issues

1. **Model Not Found**
   - Ensure `experiment_path` contains `models/best_model.pth`
   - Check file permissions and paths

2. **Architecture Mismatch**
   - Verify `architecture` matches the training configuration
   - Check model state dict keys for compatibility

3. **Dataset Not Found**
   - Ensure `dataset_base_dir` contains the expected Task directories
   - Verify `dataset_key` matches available datasets

4. **Out of Memory**
   - Reduce `batch_size` (try 2 or 1)
   - Reduce `num_samples` if needed

5. **Permission Errors**
   - Check read access to experiment and dataset directories
   - Check write access to output directory

### Performance Tips

- **Faster Evaluation**: Use smaller `num_samples` and higher `batch_size`
- **Memory Optimization**: Use smaller `batch_size` for large models
- **Storage Optimization**: Use fewer `num_samples` to reduce visualization storage

## 🔬 Technical Details

### Model Loading
- Automatically detects and loads both baseline_unet and nnunet architectures
- Handles multi-task models with segmentation + classification heads
- Validates model compatibility and provides detailed loading information

### Metrics Computation
- Uses the existing comprehensive metrics system from training
- Computes per-class and overall Dice, IoU, and pixel accuracy
- Includes multi-task classification metrics (patch and gland level)

### Visualization System
- Random sampling ensures representative visualization across dataset
- 4-class color coding matches training visualization standards
- Comprehensive overlays show both segmentation and classification results

### Data Pipeline
- Reuses training dataset loaders for consistency
- Applies same preprocessing pipeline as training
- Supports all dataset configurations used in training

---

**Note**: This evaluator is designed to work with models trained using the GlandSegModels nnU-Net training pipeline. Ensure all paths and configurations match your training setup for optimal results.