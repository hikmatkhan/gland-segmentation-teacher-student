# Dataset Preparation Guide

This guide explains how to prepare and configure datasets for the Teacher-Student Gland Segmentation Framework.

## Supported Datasets

### 1. Warwick GlaS Dataset

**Description**: Gland segmentation challenge dataset with expert annotations

- **Images**: 165 histopathology images
- **Resolution**: High-resolution H&E stained images
- **Classes**: 3 classes (Background, Benign, Malignant)
- **Split**: Train/Test splits provided
- **Source**: [GlaS Challenge Website](https://warwick.ac.uk/fac/cross_fac/tia/data/glascontest)

### 2. OSU Makoto Dataset

**Description**: Multi-magnification gland dataset

- **Slides**: 32 histopathology slides
- **Magnifications**: 4 levels (5x, 10x, 20x, 40x)
- **Classes**: 4 classes (Background, Benign, Malignant, PDC)
- **Images**: ~25,000 images across all magnifications
- **Source**: [Contact authors for access]

## Directory Structure

### Expected Directory Layout

```
$GLAND_DATASET_BASE/
├── warwick_glas/
│   ├── train/
│   │   ├── images/
│   │   │   ├── train_1.png
│   │   │   ├── train_2.png
│   │   │   └── ...
│   │   └── masks/
│   │       ├── train_1_anno.png
│   │       ├── train_2_anno.png
│   │       └── ...
│   ├── val/
│   │   ├── images/
│   │   └── masks/
│   └── test/
│       ├── images/
│       └── masks/
│
└── osu_makoto/
    ├── mag5x/
    │   ├── images/
    │   └── masks/
    ├── mag10x/
    │   ├── images/
    │   └── masks/
    ├── mag20x/
    │   ├── images/
    │   └── masks/
    ├── mag40x/
    │   ├── images/
    │   └── masks/
    └── mixed/                    # Combined magnifications
        ├── images/
        └── masks/
```

## Dataset Configuration

### Environment Variables

Set the base dataset directory:

```bash
# Linux/macOS
export GLAND_DATASET_BASE="/path/to/datasets"
export GLAND_OUTPUT_DIR="/path/to/outputs"

# Windows (PowerShell)
$env:GLAND_DATASET_BASE = "C:\path\to\datasets"
$env:GLAND_OUTPUT_DIR = "C:\path\to\outputs"
```

### Configuration File

Edit `configs/paths_config.py` if needed:

```python
DATA_PATHS = {
    'dataset_base': os.getenv('GLAND_DATASET_BASE', '/default/path/to/datasets'),
    'output_base': os.getenv('GLAND_OUTPUT_DIR', './outputs'),

    # Dataset-specific paths
    'warwick_glas': 'warwick_glas',
    'osu_makoto_5x': 'osu_makoto/mag5x',
    'osu_makoto_10x': 'osu_makoto/mag10x',
    'osu_makoto_20x': 'osu_makoto/mag20x',
    'osu_makoto_40x': 'osu_makoto/mag40x',
    'osu_makoto_mixed': 'osu_makoto/mixed',
}
```

## Data Format Specifications

### Image Format

- **File Format**: PNG, JPEG, or TIFF
- **Color Space**: RGB (3 channels)
- **Resolution**: 512×512 pixels (configurable)
- **Bit Depth**: 8-bit per channel
- **Staining**: H&E (Hematoxylin and Eosin)

### Mask Format

- **File Format**: PNG (grayscale)
- **Channels**: Single channel (grayscale)
- **Values**:
  - `0`: Background
  - `1`: Benign gland
  - `2`: Malignant gland
  - `3`: Poorly Differentiated Carcinoma (PDC)

### Naming Convention

**Images**:
```
{slide_id}_{region_id}.png
Example: patient001_roi_01.png
```

**Masks** (must match image names):
```
{slide_id}_{region_id}_anno.png
Example: patient001_roi_01_anno.png
```

## Dataset Preparation Steps

### Step 1: Download Datasets

#### Warwick GlaS

1. Visit [GlaS Challenge](https://warwick.ac.uk/fac/cross_fac/tia/data/glascontest)
2. Register and download dataset
3. Extract to `$GLAND_DATASET_BASE/warwick_glas/`

#### OSU Makoto

1. Contact dataset authors for access
2. Download all magnification levels
3. Extract to `$GLAND_DATASET_BASE/osu_makoto/`

### Step 2: Organize Directory Structure

```bash
cd $GLAND_DATASET_BASE

# Create directory structure
mkdir -p warwick_glas/{train,val,test}/{images,masks}
mkdir -p osu_makoto/{mag5x,mag10x,mag20x,mag40x,mixed}/{images,masks}
```

### Step 3: Process Raw Data

If your data is in a different format, use preprocessing scripts:

```bash
# Example: Convert TIFF to PNG
python scripts/preprocess_images.py \
    --input_dir /path/to/raw/data \
    --output_dir $GLAND_DATASET_BASE/warwick_glas/train \
    --format png \
    --resize 512
```

### Step 4: Split Dataset

If you need to create train/val/test splits:

```bash
python scripts/split_dataset.py \
    --dataset_dir $GLAND_DATASET_BASE/warwick_glas \
    --train_ratio 0.7 \
    --val_ratio 0.15 \
    --test_ratio 0.15 \
    --seed 42
```

### Step 5: Verify Dataset

```bash
# Check dataset structure and statistics
python -c "from configs.paths_config import validate_dataset; validate_dataset('mixed')"
```

## Dataset Statistics

### Warwick GlaS

| Split | Images | Benign | Malignant | Total Glands |
|-------|--------|--------|-----------|--------------|
| Train | 85     | 1,271  | 892       | 2,163        |
| Test  | 80     | 1,156  | 741       | 1,897        |

### OSU Makoto

| Magnification | Images  | Background | Benign | Malignant | PDC    |
|---------------|---------|------------|--------|-----------|--------|
| 5x            | 3,200   | 65%        | 18%    | 12%       | 5%     |
| 10x           | 6,400   | 60%        | 20%    | 15%       | 5%     |
| 20x           | 8,500   | 55%        | 22%    | 18%       | 5%     |
| 40x           | 6,900   | 50%        | 25%    | 20%       | 5%     |
| **Mixed**     | **25,000** | **57%** | **21%** | **16%** | **5%** |

## Dataset Selection in Training

### Use Specific Dataset

```bash
# Train on Warwick GlaS only
python main.py train --dataset warwick_glas

# Train on OSU Makoto 20x magnification
python main.py train --dataset mag20x

# Train on mixed magnifications (recommended)
python main.py train --dataset mixed
```

### Dataset Codes

| Code | Description |
|------|-------------|
| `warwick_glas` | Warwick GlaS dataset |
| `mag5x` | OSU Makoto 5x magnification |
| `mag10x` | OSU Makoto 10x magnification |
| `mag20x` | OSU Makoto 20x magnification |
| `mag40x` | OSU Makoto 40x magnification |
| `mixed` | All OSU Makoto magnifications combined |

## Data Augmentation

The framework applies automatic augmentation during training:

### Augmentation Pipeline

1. **Geometric Transformations**
   - Random rotation (±45°)
   - Random flip (horizontal/vertical)
   - Random scaling (0.8-1.2×)
   - Elastic deformation

2. **Color Augmentations**
   - H&E stain augmentation
   - Brightness/contrast adjustment
   - Hue/saturation jitter

3. **Advanced Augmentations**
   - Cutout/CoarseDropout
   - Grid distortion
   - Gaussian noise

### Configure Augmentation

Edit augmentation settings in training:

```bash
python main.py train \
    --dataset mixed \
    --augmentation_level medium  # Options: none, light, medium, strong
```

## Creating Custom Datasets

To add a new dataset:

### 1. Prepare Data

Follow the directory structure and format specifications above.

### 2. Update Configuration

Edit `configs/paths_config.py`:

```python
DATA_PATHS = {
    # ... existing paths ...
    'my_custom_dataset': 'path/to/custom/dataset',
}
```

### 3. Register Dataset

In `src/training/dataset.py`, add dataset loading logic:

```python
def load_custom_dataset(dataset_path):
    # Your custom loading logic
    images, masks = ...
    return images, masks
```

### 4. Use Custom Dataset

```bash
python main.py train --dataset my_custom_dataset
```

## Dataset Validation

### Check Dataset Integrity

```bash
# Verify all images have corresponding masks
python scripts/validate_dataset.py \
    --dataset_dir $GLAND_DATASET_BASE/warwick_glas \
    --check_missing \
    --check_format

# Check class distribution
python scripts/dataset_statistics.py \
    --dataset mixed \
    --plot_distribution
```

### Common Issues

#### Issue: Missing masks

**Error**: `FileNotFoundError: Mask not found for image XXX`

**Solution**: Ensure every image has a corresponding mask with `_anno` suffix.

#### Issue: Incorrect mask values

**Error**: `ValueError: Mask contains invalid class labels`

**Solution**: Ensure masks only contain values 0-3 (4 classes).

#### Issue: Mismatched dimensions

**Error**: `ValueError: Image and mask dimensions don't match`

**Solution**: Resize images and masks to same dimensions (512×512).

## Data Privacy and Ethics

⚠️ **Important Considerations**:

- Ensure you have proper authorization to use medical imaging data
- Follow HIPAA/GDPR guidelines for patient data
- Remove or anonymize patient identifiable information (PII)
- Obtain necessary IRB approvals for research use
- Respect dataset licenses and citation requirements

## Troubleshooting

### Dataset not found

```bash
# Check environment variable
echo $GLAND_DATASET_BASE

# Verify directory exists
ls -la $GLAND_DATASET_BASE

# Check permissions
chmod -R 755 $GLAND_DATASET_BASE
```

### Slow data loading

- Use SSD storage instead of HDD
- Increase `num_workers` in data loader
- Enable data caching (if RAM permits)

### Out of memory during loading

- Reduce batch size
- Use image resize preprocessing
- Enable gradient checkpointing

## Next Steps

After preparing datasets:

1. **Verify installation**: Run `python tests/test_basic_demo.py`
2. **Start training**: Follow [TRAINING.md](TRAINING.md)
3. **Monitor progress**: Use TensorBoard visualization

## Support

For dataset-related questions:
- Check existing GitHub issues
- Contact dataset authors for access
- Email: [support email - TO BE FILLED]
