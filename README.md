# Weakly Supervised Teacher-Student Framework with Progressive Pseudo-Mask Refinement for Gland Segmentation

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **Official PyTorch implementation** of "Weakly Supervised Teacher-Student Framework with Progressive Pseudo-Mask Refinement for Gland Segmentation"

## Overview

This repository contains the official implementation of our **progressive pseudo-mask refinement framework** for weakly supervised gland segmentation in histopathology images. Our method achieves state-of-the-art performance on gland tissue classification by dynamically refining teacher pseudo-masks through confidence-based filtering, adaptive annealing, and GT-Teacher incorporation.

The framework addresses the challenge of learning accurate gland segmentation from weakly-annotated histopathology data by leveraging a teacher-student architecture with intelligent pseudo-label quality control.

### Key Contributions

🔬 **Progressive Pseudo-Mask Refinement**
- Confidence and entropy-based filtering with adaptive annealing
- Curriculum learning approach: starts strict (high confidence threshold), gradually relaxes
- Removes noisy teacher predictions while preserving high-quality supervision

🎯 **GT + Teacher Incorporation**
- Novel fusion strategy combining ground truth and teacher discoveries
- GT foreground always preserved (perfect annotations)
- GT background uses teacher predictions (discovers missed structures)
- Achieves best of both worlds: expert knowledge + learned patterns

🔄 **Adaptive Two-Phase Training**
- **Phase 1 (Warmup)**: Supervised student-only training from ground truth
- **Phase 2 (Teacher-Student)**: Dual loss with cosine alpha scheduling
- Smooth transition from supervised to semi-supervised learning

📊 **Dynamic EMA Teacher Updates**
- Multiple schedules: Fixed, Cosine, Linear, Exponential
- Progressive knowledge transfer from stable teacher to adaptive student
- Configurable decay rates for different learning dynamics

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     Teacher-Student Framework                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  Input Image                                                      │
│      │                                                            │
│      ├──────────────┬──────────────┐                            │
│      │              │               │                            │
│  ┌───▼────┐    ┌───▼────┐     ┌───▼────┐                       │
│  │Student │    │Teacher │     │  GT    │                        │
│  │ Model  │    │ Model  │     │ Mask   │                        │
│  └───┬────┘    └───┬────┘     └───┬────┘                       │
│      │              │               │                            │
│      │         ┌────▼───────────────▼────┐                      │
│      │         │ Pseudo-Mask Refinement  │                      │
│      │         │ • Confidence Filtering  │                      │
│      │         │ • Entropy Filtering     │                      │
│      │         │ • GT + Teacher Fusion   │                      │
│      │         └────┬────────────────────┘                      │
│      │              │                                            │
│      │         Enhanced Pseudo-Mask                             │
│      │              │                                            │
│      ├──────────────┴────────────┐                              │
│      │                            │                              │
│  ┌───▼────────────┐     ┌────────▼───────┐                     │
│  │  Supervised    │     │  Consistency   │                      │
│  │     Loss       │     │      Loss      │                      │
│  └───┬────────────┘     └────────┬───────┘                     │
│      │                            │                              │
│      └──────────┬─────────────────┘                             │
│                 │                                                │
│         Combined Loss (α·L_sup + (1-α)·L_cons)                  │
│                 │                                                │
│            Backprop to Student                                   │
│                 │                                                │
│            EMA Update Teacher                                    │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
```

## Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/YOUR_USERNAME/gland-segmentation-teacher-student.git
cd "gland-segmentation-teacher-student"

# Create conda environment
conda create -n gland-seg python=3.9
conda activate gland-seg

# Install dependencies
pip install -r requirements.txt
```

### Environment Configuration

Set required environment variables:

```bash
export GLAND_DATASET_BASE="/path/to/datasets"
export GLAND_OUTPUT_DIR="./outputs"
export NNUNET_PREPROCESSED="/path/to/nnunet/preprocessed"  # Optional, if using nnU-Net backbone
export NNUNET_RESULTS="/path/to/nnunet/results"            # Optional
```

### Demo

Test the framework with a quick demonstration:

```bash
# Test teacher-student architecture
python main.py demo --architecture teacher_student_unet

# Visualize pseudo-GT refinement mechanism
python tests/demo_pseudo_gt_refinement.py

# Run integration test
python tests/test_teacher_student_integration.py
```

## Training

### Local Training (Single GPU)

**Basic Training:**
```bash
python main.py train \
    --architecture teacher_student_unet \
    --dataset mixed \
    --epochs 150 \
    --batch_size 8 \
    --learning_rate 1e-4 \
    --output_dir ./outputs/experiment_1
```

**Advanced Training with Progressive Refinement:**
```bash
python main.py train \
    --architecture teacher_student_unet \
    --dataset mag20x \
    --epochs 200 \
    --batch_size 4 \
    --ts_pseudo_mask_filtering confidence \
    --ts_confidence_threshold 0.8 \
    --ts_confidence_annealing cosine \
    --ts_confidence_max_threshold 0.9 \
    --ts_confidence_min_threshold 0.6 \
    --ts_gt_teacher_incorporate_enabled true \
    --ts_ema_schedule cosine \
    --ts_ema_decay_initial 0.999 \
    --ts_ema_decay_final 0.1 \
    --output_dir ./outputs/experiment_2
```

### SLURM Training (HPC Cluster)

For users with access to HPC clusters:

```bash
# Edit configuration in the script
vim run_nnunet_training.sh

# Submit job
sbatch run_nnunet_training.sh

# Resume training from checkpoint
sbatch resume_nnunet_training.sh
```

### Training Configuration Options

#### Progressive Pseudo-Mask Filtering

Control teacher prediction quality:

```bash
# Confidence-based filtering (recommended)
--ts_pseudo_mask_filtering confidence
--ts_confidence_threshold 0.8              # Keep predictions with >80% confidence

# Entropy-based filtering (alternative)
--ts_pseudo_mask_filtering entropy
--ts_entropy_threshold 1.0                 # Keep predictions with entropy <1.0

# Adaptive annealing (curriculum learning)
--ts_confidence_annealing cosine           # Cosine decay schedule
--ts_confidence_max_threshold 0.9          # Early training: very selective (90%)
--ts_confidence_min_threshold 0.6          # Late training: more permissive (60%)
```

#### GT + Teacher Incorporation

Enable enhanced pseudo-mask fusion:

```bash
--ts_gt_teacher_incorporate_enabled true
--ts_gt_incorporate_start_epoch 20         # Start fusion at epoch 20
--ts_gt_teacher_priority gt_foreground     # GT foreground + Teacher background
```

#### EMA Decay Scheduling

Dynamic teacher-student knowledge transfer:

```bash
# Cosine annealing (recommended for best results)
--ts_ema_schedule cosine
--ts_ema_decay_initial 0.999               # Stable early teacher
--ts_ema_decay_final 0.1                   # Adaptive late teacher

# Fixed decay (backward compatible)
--ts_ema_schedule fixed
--ts_ema_decay 0.999

# Other schedules
--ts_ema_schedule linear                   # Linear interpolation
--ts_ema_schedule exponential              # Exponential decay
```

#### Consistency Loss Types

```bash
# MSE consistency (default)
--ts_consistency_loss_type mse

# Knowledge distillation
--ts_consistency_loss_type kl_div
--ts_consistency_temperature 2.0

# Regional consistency
--ts_consistency_loss_type dice
--ts_consistency_loss_type iou
```

## Evaluation

```bash
# Evaluate trained model
python main.py evaluate \
    --architecture teacher_student_unet \
    --model outputs/experiment_1/models/best_student_model.pth \
    --dataset mixed \
    --visualize \
    --output_dir ./evaluation_results
```

## Project Structure

```
.
├── README.md                    # This file
├── LICENSE                      # MIT License
├── requirements.txt             # Python dependencies
├── main.py                      # Main CLI entry point
├── run_nnunet_training.sh      # SLURM training script
├── resume_nnunet_training.sh   # Resume training script
│
├── docs/                        # Documentation
│   ├── INSTALLATION.md         # Detailed installation guide
│   ├── TRAINING.md             # Comprehensive training guide
│   ├── ARCHITECTURE.md         # Technical architecture details
│   └── DATASETS.md             # Dataset preparation instructions
│
├── src/                         # Source code
│   ├── models/                  # Model architectures
│   │   ├── teacher_student_unet.py          # Teacher-Student architecture (MAIN)
│   │   ├── teacher_student_loss.py          # Refinement mechanism (MAIN)
│   │   ├── teacher_student_trainer.py       # Training protocol (MAIN)
│   │   ├── baseline_unet.py                 # Baseline U-Net
│   │   ├── nnunet_integration.py            # nnU-Net backbone
│   │   ├── multi_task_wrapper.py            # Multi-task wrapper
│   │   ├── projection_heads.py              # Classification heads
│   │   ├── loss_functions.py                # Multi-task losses
│   │   ├── metrics.py                       # Evaluation metrics
│   │   └── model_factory.py                 # Model creation
│   ├── training/                # Training pipeline
│   │   ├── teacher_student_trainer.py       # Teacher-Student trainer
│   │   ├── trainer.py                       # Base trainer
│   │   └── dataset.py                       # Data loading
│   └── evaluation/              # Evaluation utilities
│       ├── evaluator.py                     # Standard evaluator
│       └── post_training_evaluator.py       # Post-training analysis
│
├── configs/                     # Configuration
│   ├── paths_config.py         # Path configuration
│   └── README.md               # Config documentation
│
├── scripts/                     # Utility scripts
│   ├── example_training.sh     # Local training example
│   └── quick_test.py           # Quick functionality test
│
├── tests/                       # Demonstration & testing
│   ├── demo_pseudo_gt_refinement.py         # Pseudo-GT demo (shows novelty)
│   ├── test_teacher_student_integration.py  # Integration test
│   └── test_basic_demo.py                   # Basic functionality test
│
└── independent_eval/            # Independent evaluation
    ├── independent_evaluator.py             # Standalone evaluator
    └── README.md                            # Evaluation documentation
```

## Datasets

This framework supports multi-class gland segmentation on:

- **Warwick GlaS**: 165 histopathology images with expert annotations
- **OSU Makoto**: 32 slides × 4 magnifications (5x, 10x, 20x, 40x)

**4-Class Segmentation:**
- Class 0: Background
- Class 1: Benign gland
- Class 2: Malignant gland
- Class 3: Poorly Differentiated Carcinoma (PDC)

See [docs/DATASETS.md](docs/DATASETS.md) for dataset preparation instructions.

## Documentation

- **[Installation Guide](docs/INSTALLATION.md)** - Prerequisites, environment setup, troubleshooting
- **[Training Guide](docs/TRAINING.md)** - Training modes, hyperparameters, monitoring
- **[Architecture Details](docs/ARCHITECTURE.md)** - Technical deep-dive into teacher-student architecture
- **[Dataset Preparation](docs/DATASETS.md)** - Dataset setup and configuration

## Citation

If you use this code for your research, please cite our paper:

```bibtex
@article{author2026weakly,
  title={Weakly Supervised Teacher-Student Framework with Progressive Pseudo-Mask Refinement for Gland Segmentation},
  author={[Author Names - TO BE FILLED]},
  journal={[Journal/Conference Name - TO BE FILLED]},
  year={2026},
  volume={XX},
  pages={XXX-XXX},
  doi={[DOI - TO BE FILLED]}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Warwick GlaS Dataset creators for gland segmentation benchmark
- OSU Makoto Dataset contributors for multi-magnification histopathology data
- nnU-Net architecture developers for medical imaging framework
- PyTorch team for the deep learning framework

## Contact

For questions, issues, or collaborations:
- **GitHub Issues**: [Open an issue](https://github.com/YOUR_USERNAME/gland-segmentation-teacher-student/issues)
- **Email**: [corresponding_author@institution.edu - TO BE FILLED]

---

**Note**: This is a research implementation accompanying our paper. For clinical applications, additional validation and regulatory approval are required.

**⚠️ Important**: Before publishing, please update the placeholders marked with `[TO BE FILLED]` including author names, journal/conference details, DOI, email, and GitHub username.
