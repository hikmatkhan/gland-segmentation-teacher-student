# Installation Guide

Comprehensive installation instructions for the Teacher-Student Gland Segmentation Framework.

## Prerequisites

### Hardware Requirements

- **GPU**: NVIDIA GPU with ≥8GB VRAM (recommended: ≥10GB for batch size 8)
- **RAM**: ≥16GB system RAM (recommended: ≥32GB)
- **Storage**: ≥30GB free space (datasets + outputs)
- **CPU**: Modern multi-core processor

### Software Requirements

- **Operating System**: Linux (Ubuntu 18.04+), macOS (10.14+), or Windows 10/11
- **Python**: 3.9 or higher
- **CUDA**: 11.8 or higher (for GPU support)
- **Git**: For cloning the repository

## Step 1: Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/gland-segmentation-teacher-student.git
cd "gland-segmentation-teacher-student"
```

## Step 2: Environment Setup

### Option A: Using Conda (Recommended)

Conda provides better environment isolation and dependency management.

```bash
# Create new environment
conda create -n gland-seg python=3.9 -y
conda activate gland-seg

# Verify Python version
python --version  # Should show Python 3.9.x
```

### Option B: Using venv

```bash
# Create virtual environment
python3.9 -m venv venv

# Activate environment
source venv/bin/activate      # Linux/macOS
# OR
venv\Scripts\activate         # Windows
```

## Step 3: Install PyTorch

Install PyTorch with CUDA support (adjust CUDA version as needed):

```bash
# For CUDA 11.8
pip install torch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1
pip install torch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 --index-url https://download.pytorch.org/whl/cu121

# For CPU only (not recommended for training)
pip install torch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0
```

### Verify PyTorch Installation

```bash
python -c "import torch; print(f'PyTorch {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}')"
```

Expected output:
```
PyTorch 2.0.0
CUDA available: True
CUDA version: 11.8
```

## Step 4: Install Dependencies

```bash
# Install all required packages
pip install -r requirements.txt
```

This will install:
- Core dependencies (numpy, scikit-learn, scikit-image, opencv-python, pillow)
- Data augmentation (albumentations)
- Visualization (matplotlib, seaborn, tensorboard)
- Utilities (tqdm, pyyaml, pandas)

### Verify Installations

```bash
# Test critical imports
python -c "import numpy as np; import torch; import albumentations; import matplotlib; print('✓ All imports successful')"
```

## Step 5: Environment Configuration

Set required environment variables for dataset paths:

### Linux/macOS

Add to your `~/.bashrc` or `~/.zshrc`:

```bash
export GLAND_DATASET_BASE="/path/to/your/datasets"
export GLAND_OUTPUT_DIR="/path/to/your/outputs"

# Optional: If using nnU-Net backbone
export NNUNET_PREPROCESSED="/path/to/nnunet/preprocessed"
export NNUNET_RESULTS="/path/to/nnunet/results"
```

Then reload:
```bash
source ~/.bashrc  # or ~/.zshrc
```

### Windows

Set environment variables via Command Prompt (as Administrator):

```cmd
setx GLAND_DATASET_BASE "C:\path\to\your\datasets"
setx GLAND_OUTPUT_DIR "C:\path\to\your\outputs"
```

Or via PowerShell:

```powershell
[Environment]::SetEnvironmentVariable("GLAND_DATASET_BASE", "C:\path\to\your\datasets", "User")
[Environment]::SetEnvironmentVariable("GLAND_OUTPUT_DIR", "C:\path\to\your\outputs", "User")
```

### Verify Environment Variables

```bash
python -c "import os; print('GLAND_DATASET_BASE:', os.getenv('GLAND_DATASET_BASE')); print('GLAND_OUTPUT_DIR:', os.getenv('GLAND_OUTPUT_DIR'))"
```

## Step 6: Verify Installation

### Test Imports

```bash
# Test core framework imports
python -c "from src.models.teacher_student_unet import TeacherStudentUNet; print('✓ TeacherStudentUNet imported')"
python -c "from src.models.teacher_student_loss import TeacherStudentLoss; print('✓ TeacherStudentLoss imported')"
python -c "from src.training.teacher_student_trainer import TeacherStudentTrainer; print('✓ TeacherStudentTrainer imported')"
```

### Run Demo

```bash
# Run quick demo (requires dataset)
python main.py demo --architecture teacher_student_unet
```

### Run Tests

```bash
# Run basic integration test
python tests/test_basic_demo.py

# Run teacher-student integration test
python tests/test_teacher_student_integration.py

# Run pseudo-GT refinement demo
python tests/demo_pseudo_gt_refinement.py
```

## Optional: nnU-Net Integration

If you want to use the nnU-Net backbone:

```bash
# Install nnU-Net dependencies
pip install dynamic-network-architectures>=0.2

# Set nnU-Net environment variables (see Step 5)
```

## Troubleshooting

### Issue: CUDA out of memory

**Solution**: Reduce batch size in training:
```bash
python main.py train --batch_size 2  # Reduce from default 8
```

### Issue: Import errors for `src.models.*`

**Solution**: Ensure you're in the project root directory:
```bash
cd /path/to/gland-segmentation-teacher-student
python main.py demo
```

### Issue: Environment variables not found

**Solution**:
1. Verify variables are set: `echo $GLAND_DATASET_BASE`
2. Reload shell configuration: `source ~/.bashrc`
3. Restart terminal session

### Issue: `ModuleNotFoundError: No module named 'albumentations'`

**Solution**: Reinstall requirements:
```bash
pip install --upgrade -r requirements.txt
```

### Issue: PyTorch can't detect GPU

**Solution**:
1. Verify NVIDIA drivers: `nvidia-smi`
2. Check CUDA installation: `nvcc --version`
3. Reinstall PyTorch with correct CUDA version

### Issue: Permission denied errors

**Solution**:
- **Linux/macOS**: Check directory permissions, use `chmod` if needed
- **Windows**: Run terminal as Administrator

## System-Specific Notes

### Linux

- Recommended OS for development
- Full CUDA support
- Easiest environment management

### macOS

- Limited GPU support (CPU training only on M1/M2)
- Use conda for best compatibility
- Training will be slower without GPU

### Windows

- Requires WSL2 for best performance (optional)
- Use Anaconda Prompt for conda environments
- May need Visual C++ Build Tools for some packages

## Updating the Installation

To update to the latest version:

```bash
# Pull latest changes
git pull origin main

# Update dependencies
pip install --upgrade -r requirements.txt

# Verify updates
python main.py demo
```

## Uninstallation

To remove the environment:

```bash
# Conda
conda deactivate
conda env remove -n gland-seg

# venv
deactivate
rm -rf venv/
```

## Next Steps

After successful installation:

1. **Prepare datasets**: Follow [DATASETS.md](DATASETS.md)
2. **Review training guide**: See [TRAINING.md](TRAINING.md)
3. **Understand architecture**: Read [ARCHITECTURE.md](ARCHITECTURE.md)
4. **Run first experiment**: Try the examples in [TRAINING.md](TRAINING.md)

## Support

If you encounter issues not covered here:
- Open an issue on GitHub
- Check existing issues for solutions
- Contact: [email - TO BE FILLED]
