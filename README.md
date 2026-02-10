# Weakly Supervised Teacher–Student Framework with Progressive Pseudo-Mask Refinement for Gland Segmentation

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **Official PyTorch implementation** of "Weakly Supervised Teacher-Student Framework with Progressive Pseudo-Mask Refinement for Gland Segmentation"
>
> **Authors:** Hikmat Khan*, Wei Chen, Muhammad Khalid Khan Niazi
> **Affiliation:** Department of Pathology, College of Medicine, The Ohio State University Wexner Medical Center
> **Funding:** Supported by R01 CA276301 from the National Cancer Institute

---

## 📋 Overview

Colorectal cancer histopathological grading relies on accurate gland segmentation, but current deep learning methods require extensive pixel-level annotations. Our framework provides a **weakly supervised** solution that achieves **competitive performance with fully supervised methods** while requiring only **sparse annotations** — reducing annotation time by approximately **60-fold**.

### 🎯 Key Innovation

We introduce a teacher-student framework with **progressive pseudo-mask refinement** that:
- Leverages sparse pathologist annotations via adaptive GT-teacher fusion
- Uses an EMA-stabilized teacher network for reliable pseudo-labels
- Employs confidence-based filtering with curriculum learning
- Gradually discovers and segments unannotated glandular structures

### 📊 Performance Highlights

**GlaS Dataset (MICCAI 2015 Benchmark):**
- **mIoU**: 80.10% (±1.52) — Competitive with state-of-the-art
- **mDice**: 89.10% (±2.10) — On par with fully supervised methods
- **Superior stability**: Lower variance than MAA (±2.26 mIoU, ±3.31 mDice)
- **Comparable to EWASwin UNet**: Fully supervised SOTA (81.5% mIoU)

**Cross-Domain Generalization:**
- Strong performance on TCGA-COAD and TCGA-READ without fine-tuning
- Robust to inter-institutional staining variations
- 60× reduction in annotation effort vs. fully supervised approaches

---

## 🔬 Method

### Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│              Teacher-Student Self-Training Framework             │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  Input H&E Image                                                 │
│         │                                                         │
│         ├──────────────┬──────────────┬──────────────┐          │
│         │              │              │              │          │
│    ┌────▼────┐    ┌────▼────┐   ┌────▼────┐   ┌────▼────┐    │
│    │ Student │    │ Teacher │   │   GT    │   │Confidence│    │
│    │ Network │    │ Network │   │  Mask   │   │Filtering │    │
│    │   (θₛ)  │    │   (θₜ)  │   │(sparse) │   │  Module  │    │
│    └────┬────┘    └────┬────┘   └────┬────┘   └────┬────┘    │
│         │              │              │              │          │
│         │         ┌────▼──────────────▼──────────────▼────┐   │
│         │         │  Progressive Pseudo-Mask Refinement   │   │
│         │         │  • Confidence-based filtering         │   │
│         │         │  • GT + Teacher adaptive fusion       │   │
│         │         │  • Curriculum-guided threshold decay  │   │
│         │         └────────────────┬──────────────────────┘   │
│         │                          │                            │
│         │              Enhanced Pseudo-Mask                     │
│         │                          │                            │
│         ├──────────────────────────┴────────────┐              │
│         │                                        │              │
│    ┌────▼──────────┐                   ┌────────▼────────┐    │
│    │  Supervised   │                   │  Consistency    │    │
│    │     Loss      │                   │      Loss       │    │
│    │ ℒₛᵤₚ (Dice+CE)│                   │   ℒcons (MSE)   │    │
│    └────┬──────────┘                   └────────┬────────┘    │
│         │                                        │              │
│         └──────────┬─────────────────────────────┘            │
│                    │                                            │
│        ℒtotal = α·ℒₛᵤₚ + (1-α)·ℒcons                          │
│                    │    (α: 0.9 → 0.01, cosine)                │
│                    │                                            │
│              Backprop to Student                                │
│                    │                                            │
│         EMA Update: θₜ ← 0.999·θₜ + 0.001·θₛ                  │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
```

### Two-Phase Training Protocol

#### Phase 1: Supervised Warm-Up (Epochs 0-50)
- Student trained on sparse ground truth only
- Builds robust baseline representations
- Loss: ℒ = λ_Dice·ℒ_Dice + λ_CE·ℒ_CE (λ_Dice=0.6, λ_CE=0.4)
- Prepares stable initialization for teacher

#### Phase 2: Teacher-Student Co-Training (Epochs 51-200)
- Teacher initialized from student weights (θₜ ← θₛ)
- Student trained with hybrid supervision:
  - **Supervised loss** (ℒₛᵤₚ) on sparse GT labels
  - **Consistency loss** (ℒcons) on teacher pseudo-labels
- Teacher updated via EMA: θₜ ← 0.999·θₜ + 0.001·θₛ
- **Cosine alpha scheduling**: Gradually shifts emphasis from GT to pseudo-labels

### Progressive Pseudo-Mask Refinement

**1. Confidence-Based Filtering (Curriculum Learning):**
```python
confidence = max(softmax(teacher_logits), dim=1)
confidence_mask = (confidence > threshold_t)
# threshold_t decays: 0.95 → 0.25 (cosine schedule)
```

**2. GT + Teacher Adaptive Fusion:**
```python
# Preserve sparse GT, fill background with filtered teacher predictions
enhanced_mask = (
    gt_mask * gt_foreground_mask +          # Keep GT foreground
    teacher_pred * confidence_mask * gt_background_mask  # Teacher discovery
)
```

**3. Consistency Regularization:**
```python
ℒcons = MSE(student_logits, enhanced_mask.detach())
```

**Key Principles:**
- **Early training** (high threshold): Only high-confidence regions, conservative
- **Late training** (low threshold): More permissive, explores ambiguous boundaries
- **GT priority**: Expert annotations always preserved
- **Stable teacher**: EMA updates prevent noisy pseudo-labels

---

## 📚 Datasets

### 1. OSUWMC In-House Dataset
- **Source**: The Ohio State University Wexner Medical Center
- **Images**: 60 H&E-stained WSIs
- **Patches**: 74,179 (512×512 @ 5× magnification)
- **Annotations**: **Sparse labels** from 2 pathology residents
- **Classes**: 4 (Background, Benign, Malignant, PDC)
- **Split**: 63,191 train / 5,460 val / 5,528 test
- **Key Challenge**: Most patches contain both annotated and unannotated glands

### 2. GlaS Benchmark (MICCAI 2015)
- **Images**: 165 H&E-stained images @ 20× magnification
- **Annotations**: Dense pixel-level ground truth
- **Train**: 85 images (37 benign, 48 malignant)
- **Test**: 80 images (used for final evaluation)
- **Resolution**: Resized to 512×512 pixels
- **Challenge**: Inter-subject staining variability

### 3. External Generalization Cohorts (Qualitative)
- **TCGA-COAD**: Colon adenocarcinoma
- **TCGA-READ**: Rectum adenocarcinoma
- **SPIDER**: Multi-organ pathology dataset
- **Purpose**: Assess cross-domain generalization without fine-tuning

---

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/hikmatkhan/gland-segmentation-teacher-student.git
cd gland-segmentation-teacher-student

# Create conda environment
conda create -n gland-seg python=3.9
conda activate gland-seg

# Install PyTorch (CUDA 11.8)
pip install torch==2.0.0 torchvision==0.15.0 --index-url https://download.pytorch.org/whl/cu118

# Install dependencies
pip install -r requirements.txt
```

### Environment Setup

```bash
export GLAND_DATASET_BASE="/path/to/datasets"
export GLAND_OUTPUT_DIR="./outputs"
```

### Demo

```bash
# Quick functionality test
python tests/test_basic_demo.py

# Visualize pseudo-GT refinement mechanism
python tests/demo_pseudo_gt_refinement.py

# Teacher-Student integration test
python tests/test_teacher_student_integration.py
```

---

## 🎓 Training

### Basic Training (GlaS Dataset)

```bash
python main.py train \
    --architecture teacher_student_unet \
    --dataset glas \
    --epochs 200 \
    --batch_size 8 \
    --learning_rate 1e-4 \
    --output_dir ./outputs/glas_experiment
```

### Reproduce Paper Results (GlaS)

```bash
python main.py train \
    --architecture teacher_student_unet \
    --dataset glas \
    --epochs 200 \
    --batch_size 8 \
    --learning_rate 1e-4 \
    --dice_weight 0.6 \
    --ce_weight 0.4 \
    --ts_teacher_init_epoch 50 \
    --ts_ema_decay 0.999 \
    --ts_pseudo_mask_filtering confidence \
    --ts_confidence_threshold 0.8 \
    --ts_confidence_annealing cosine \
    --ts_confidence_max_threshold 0.95 \
    --ts_confidence_min_threshold 0.25 \
    --ts_gt_teacher_incorporate_enabled true \
    --ts_min_alpha 0.01 \
    --ts_max_alpha 0.9 \
    --output_dir ./outputs/glas_paper
```

### SLURM Training (HPC)

```bash
# Edit run_nnunet_training.sh with your configuration
sbatch run_nnunet_training.sh
```

**Implementation Details:**
- **Optimizer**: AdamW (lr=1e-4, weight_decay=1e-5)
- **Scheduler**: Cosine annealing (lr_min=1e-6)
- **Batch size**: 8
- **Input size**: 512×512 pixels
- **Augmentation**: Rotation (±90°), H-flip (p=0.5), HSV jittering, Gaussian noise/blur
- **Early stopping**: Patience=20 epochs
- **Gradient clipping**: Max norm=1.0
- **Random seed**: 42 (reproducibility)

---

## 📊 Results

### Comparison with Weakly Supervised Methods (GlaS)

| Method | Year | mIoU (%) | mDice (%) | Variance (mIoU / mDice) |
|--------|------|----------|-----------|-------------------------|
| SEAM | 2020 | 71.36 | 79.59 | ±0.49 / ±4.88 |
| AMR | 2022 | 72.83 | - | ±0.37 / - |
| MLPS | 2022 | 73.60 | - | ±0.16 / - |
| OEEM | 2022 | 76.48 | 83.40 | ±0.10 / ±5.36 |
| HAMIL | 2023 | 77.37 | - | ±0.73 / - |
| CBFNet | 2024 | 76.30 | - | ±0.26 / - |
| MPFP | 2025 | 80.44 | - | ±0.05 / - |
| **MAA** | **2025** | **81.99** | **90.10** | **±2.26 / ±3.31** |
| **Ours** | **2025** | **80.10** | **89.10** | **±1.52 / ±2.10** ✓ |

**Key Findings:**
- ✅ Competitive with state-of-the-art MAA (0.81.99% vs 80.10% mIoU)
- ✅ **Superior training stability**: Lower variance (±1.52 vs ±2.26 mIoU)
- ✅ More reliable for clinical deployment (consistent performance)

### Comparison with Fully Supervised Methods (GlaS)

| Method | Year | mIoU | mDice | Architecture |
|--------|------|------|-------|--------------|
| UNet | 2015 | 0.648 | 0.776 | CNN |
| TransUNet | 2021 | 0.701 | 0.815 | Transformer |
| UNet++ | 2018 | 0.702 | 0.819 | Nested U-Net |
| ResUNet++ | 2019 | 0.738 | 0.841 | Residual U-Net |
| TransAttUNet | 2023 | 0.777 | 0.867 | Transformer + Attention |
| **EWASwin UNet** (Fully Sup.) | **2025** | **0.815** | **0.894** | **Swin Transformer** |
| **Ours (Weakly Sup.)** | **2025** | **0.801** | **0.891** | **Teacher-Student** |

**Key Findings:**
- ✅ Comparable to fully supervised SOTA (EWASwin UNet: 0.815 vs. Ours: 0.801 mIoU)
- ✅ Outperforms classical architectures (UNet++, ResUNet++)
- ✅ Requires **60× less annotation effort** than fully supervised methods
- ✅ Performance improves with denser annotations (if available)

### Generalization Performance

| Dataset | Domain | Performance | Annotation | Notes |
|---------|--------|-------------|------------|-------|
| **OSUWMC** | In-domain (trained) | Strong | Sparse | Effective gland discovery |
| **GlaS** | In-domain | **80.10% mIoU** | Dense | Competitive with SOTA |
| **TCGA-COAD** | Out-of-domain | Robust | None | No fine-tuning |
| **TCGA-READ** | Out-of-domain | Robust | None | Maintains quality |
| **SPIDER** | Out-of-domain | Degraded | None | Extreme domain shift* |

*Performance degradation on SPIDER attributed to lower image quality and highly diverse staining profiles.

---

## 📁 Project Structure

```
.
├── README.md                    # This file
├── LICENSE                      # MIT License
├── requirements.txt             # Python dependencies
├── .gitignore                   # Git ignore patterns
│
├── main.py                      # CLI entry point
├── run_nnunet_training.sh      # SLURM training script
├── resume_nnunet_training.sh   # Resume training
│
├── docs/                        # Documentation
│   ├── INSTALLATION.md         # Setup guide
│   ├── TRAINING.md             # Training guide
│   ├── ARCHITECTURE.md         # Technical details
│   └── DATASETS.md             # Dataset preparation
│
├── src/                         # Source code
│   ├── models/                  # Model architectures
│   │   ├── teacher_student_unet.py          # Main architecture ⭐
│   │   ├── teacher_student_loss.py          # Refinement mechanism ⭐
│   │   ├── baseline_unet.py                 # Baseline U-Net
│   │   ├── nnunet_integration.py            # nnU-Net backbone
│   │   └── ...
│   ├── training/                # Training pipeline
│   │   ├── teacher_student_trainer.py       # Training protocol ⭐
│   │   ├── trainer.py                       # Base trainer
│   │   └── dataset.py                       # Data loading
│   └── evaluation/              # Evaluation utilities
│       ├── evaluator.py
│       └── post_training_evaluator.py
│
├── configs/                     # Configuration
│   └── paths_config.py
│
├── tests/                       # Demonstration & testing
│   ├── demo_pseudo_gt_refinement.py         # Shows core novelty
│   ├── test_teacher_student_integration.py  # Integration test
│   └── test_basic_demo.py                   # Basic functionality
│
└── independent_eval/            # Standalone evaluation
    └── independent_evaluator.py
```

---

## 🎯 Key Contributions

### 1. Sparse-Annotation–Aware Pseudo-Label Fusion
- Explicitly preserves pathologist annotations via pixel-wise integration
- GT foreground always preserved (anatomically faithful)
- Teacher fills GT background (discovers unannotated glands)
- Maintains clinical relevance under extreme annotation sparsity

### 2. Curriculum-Driven Pseudo-Mask Refinement
- **Confidence threshold**: Cosine-decayed from 0.95 → 0.25
- **Dynamic loss weighting**: α decays from 0.9 → 0.01
- **Gradual expansion**: High-confidence regions → ambiguous boundaries
- Addresses morphological complexity of glandular structures

### 3. Comprehensive Clinical Evaluation
- Multi-dataset validation (1 institutional + 4 public benchmarks)
- Cross-domain generalization assessment (TCGA-COAD, TCGA-READ, SPIDER)
- Rigorous quantification of robustness and failure modes
- Actionable insights for clinical translation

---

## 📖 Documentation

- **[Installation Guide](docs/INSTALLATION.md)** - Prerequisites, environment setup, troubleshooting
- **[Training Guide](docs/TRAINING.md)** - Training modes, hyperparameters, monitoring
- **[Architecture Details](docs/ARCHITECTURE.md)** - Technical deep-dive into teacher-student framework
- **[Dataset Preparation](docs/DATASETS.md)** - Dataset setup and configuration

---

## 📝 Citation

If you use this code for your research, please cite our paper:

```bibtex
@article{khan2025weakly,
  title={Weakly Supervised Teacher-Student Framework with Progressive Pseudo-Mask Refinement for Gland Segmentation},
  author={Khan, Hikmat and Chen, Wei and Niazi, Muhammad Khalid Khan},
  journal={Under Review},
  year={2025},
  note={Supported by R01 CA276301 from the National Cancer Institute}
}
```

---

## 🏥 Clinical Impact

### Annotation Efficiency
- **60× reduction** in annotation time vs fully supervised methods
- Pathologists provide sparse scribbles instead of dense pixel-level labels
- Practical for large-scale clinical deployment

### Performance vs. Effort Trade-off
- Achieves 80.10% mIoU on GlaS with sparse annotations
- Only 1.89% below fully supervised SOTA (81.99%)
- Significantly reduces pathologist workload

### Generalization & Robustness
- Generalizes to TCGA-COAD and TCGA-READ without fine-tuning
- Robust to inter-institutional staining variations
- Identifies failure modes (SPIDER) for responsible deployment

### Clinical Translation
- Lower variance (±1.52 mIoU) ensures consistent performance
- Critical prerequisite for clinical adoption
- Maintains high segmentation fidelity with minimal annotations

---

## ⚖️ License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

### Funding
This work was supported by:
- **R01 CA276301** from the National Cancer Institute (PIs: Niazi, Chen)
- The Ohio State University Comprehensive Cancer Center
- Pelotonia Research Funds
- Department of Pathology, OSU Wexner Medical Center

The content is solely the responsibility of the authors and does not necessarily represent the official views of the National Institutes of Health or National Cancer Institute.

### Datasets
- **GlaS Dataset**: MICCAI 2015 Gland Segmentation Challenge
- **TCGA**: The Cancer Genome Atlas Program
- **SPIDER**: Multi-organ pathology dataset

### Frameworks & Tools
- **nnU-Net**: Self-configuring method for medical image segmentation
- **PyTorch**: Deep learning framework
- **Albumentations**: Image augmentation library

---

## 📧 Contact

**Corresponding Author:**

**Hikmat Khan**
Department of Pathology
The Ohio State University Wexner Medical Center
Pelotonia Research Center, 2nd Floor
2281 Kenny Road, Suite 450
Columbus, OH 43210, USA

📧 Email: [Hikmat.Khan@osumc.edu](mailto:Hikmat.Khan@osumc.edu)
📱 Tel: +1-XXX-XXX-XXXX

**Co-Authors:**
- **Wei Chen** (Co-PI, Funding)
- **Muhammad Khalid Khan Niazi** (Co-PI, Funding)

---

## 🔬 Data & Code Availability

### In-House Dataset
The OSUWMC dataset is available upon reasonable request by contacting the corresponding author at [Hikmat.Khan@osumc.edu](mailto:Hikmat.Khan@osumc.edu).

**Ethical Approval**: IRB No. 2018C0098, The Ohio State University Wexner Medical Center

### Public Datasets
- **GlaS**: https://warwick.ac.uk/fac/cross_fac/tia/data/glascontest
- **TCGA-COAD/READ**: https://portal.gdc.cancer.gov/
- **SPIDER**: https://github.com/nechaev-d/SPIDER

### Code
All code is available in this repository under MIT License.

---

## ⚠️ Important Notes

### Research Implementation
- This is a **research implementation** for reproducibility and further development
- For **clinical applications**, additional validation and regulatory approval are required

### Domain Specificity
- Framework designed for **H&E-stained histopathology images**
- May require adaptation for other imaging modalities or tissue types

### Performance Considerations
- **In-domain** (GlaS, OSUWMC): Strong performance
- **Out-of-domain** (TCGA-COAD/READ): Robust generalization
- **Extreme domain shift** (SPIDER): Performance degradation may require domain adaptation

### Reproducibility
- Random seed fixed (42) for deterministic results
- All hyperparameters documented
- Comprehensive test suite provided

---

## 🚀 Future Directions

### Clinical Extensions
- Extend to other adenocarcinoma types (prostate, breast, lung, endometrial)
- Integrate with whole-slide image analysis pipelines
- Develop interactive annotation tools for pathologists

### Technical Improvements
- Incorporate advanced domain adaptation for cross-institutional deployment
- Explore multi-task learning with prognosis prediction
- Investigate few-shot learning scenarios

### Validation Studies
- Multi-center validation studies
- Comparison with pathologist inter-observer variability
- Prospective clinical trials

---

**Built with ❤️ at The Ohio State University Wexner Medical Center**

*Advancing computational pathology through weakly supervised learning*
