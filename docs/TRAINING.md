# Training Guide

Comprehensive guide for training the Teacher-Student Gland Segmentation Framework.

## Table of Contents

1. [Training Modes](#training-modes)
2. [Basic Training](#basic-training)
3. [Advanced Configuration](#advanced-configuration)
4. [Teacher-Student Specific](#teacher-student-specific)
5. [Hyperparameter Tuning](#hyperparameter-tuning)
6. [Monitoring Training](#monitoring-training)
7. [Resuming Training](#resuming-training)
8. [Troubleshooting](#troubleshooting)

---

## Training Modes

### 1. Local Training (Single GPU)

Suitable for:
- Development and experimentation
- Small datasets
- Quick prototyping

```bash
python main.py train --architecture teacher_student_unet --dataset mixed --epochs 150
```

### 2. SLURM Training (HPC Cluster)

Suitable for:
- Production training runs
- Large-scale experiments
- Hyperparameter grid searches

```bash
sbatch run_nnunet_training.sh
```

### 3. Multi-GPU Training (Future)

Currently not supported, but planned for future releases.

---

## Basic Training

### Minimal Training Command

```bash
python main.py train \
    --architecture teacher_student_unet \
    --dataset mixed \
    --epochs 150 \
    --batch_size 8 \
    --output_dir ./outputs/experiment_1
```

### Recommended Training Command

```bash
python main.py train \
    --architecture teacher_student_unet \
    --dataset mag20x \
    --epochs 200 \
    --batch_size 4 \
    --learning_rate 1e-4 \
    --dice_weight 0.6 \
    --ce_weight 0.4 \
    --early_stop_patience 30 \
    --output_dir ./outputs/teacher_student_exp1
```

### Core Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--architecture` | - | Model architecture: `baseline_unet`, `nnunet`, `teacher_student_unet` |
| `--dataset` | `mixed` | Dataset: `mixed`, `mag5x`, `mag10x`, `mag20x`, `mag40x`, `warwick_glas` |
| `--epochs` | 150 | Total training epochs |
| `--batch_size` | 8 | Batch size (reduce if OOM) |
| `--learning_rate` | 1e-4 | Initial learning rate |
| `--output_dir` | `./outputs` | Output directory for checkpoints/logs |

---

## Advanced Configuration

### Loss Configuration

```bash
python main.py train \
    --architecture teacher_student_unet \
    --dice_weight 0.6 \        # Dice loss weight (segmentation)
    --ce_weight 0.4 \          # Cross-entropy weight (segmentation)
    --patch_weight 0.3 \       # Patch classification weight
    --gland_weight 0.2         # Gland classification weight
```

**Recommended Loss Weights:**
- **Segmentation-focused**: `--dice_weight 0.7 --ce_weight 0.3`
- **Balanced**: `--dice_weight 0.6 --ce_weight 0.4` (default)
- **Classification-focused**: `--patch_weight 0.4 --gland_weight 0.3`

### Learning Rate Scheduling

```bash
python main.py train \
    --learning_rate 1e-4 \
    --lr_scheduler plateau \          # Options: plateau, cosine, step, none
    --lr_scheduler_patience 10 \      # Patience for ReduceLROnPlateau
    --lr_scheduler_factor 0.5         # LR reduction factor
```

**Scheduler Options:**
- `plateau`: Reduce LR when validation loss plateaus (recommended)
- `cosine`: Cosine annealing schedule
- `step`: Step decay at fixed intervals
- `none`: No scheduling (constant LR)

### Early Stopping

```bash
python main.py train \
    --early_stop_patience 30 \        # Stop after 30 epochs without improvement
    --early_stop_min_delta 0.001      # Minimum improvement threshold
```

### Data Augmentation

```bash
python main.py train \
    --augmentation_level medium \     # Options: none, light, medium, strong
    --augmentation_prob 0.8           # Probability of applying augmentation
```

**Augmentation Levels:**
- `none`: No augmentation (for debugging)
- `light`: Basic flips and rotations
- `medium`: Geometric + color augmentations (recommended)
- `strong`: Aggressive augmentations (may hurt performance)

---

## Teacher-Student Specific

### Two-Phase Training

The Teacher-Student architecture uses a two-phase training protocol:

**Phase 1: Warmup (Supervised Only)**
- Epochs: 0 to `teacher_init_epoch` (default: 50)
- Only student is trained on ground truth labels
- No teacher updates or consistency loss
- Builds strong foundation

**Phase 2: Teacher-Student**
- Epochs: `teacher_init_epoch` to end
- Teacher initialized from student weights
- Both supervised and consistency losses
- Teacher updated via EMA

```bash
python main.py train \
    --architecture teacher_student_unet \
    --ts_teacher_init_epoch 50 \      # Start teacher-student at epoch 50
    --epochs 200                       # Total epochs
```

### Progressive Pseudo-Mask Refinement

#### Confidence-Based Filtering

```bash
python main.py train \
    --architecture teacher_student_unet \
    --ts_pseudo_mask_filtering confidence \
    --ts_confidence_threshold 0.8 \             # Keep predictions >80% confidence
    --ts_confidence_annealing cosine \          # Adaptive threshold decay
    --ts_confidence_max_threshold 0.9 \         # Start threshold (strict)
    --ts_confidence_min_threshold 0.6           # End threshold (relaxed)
```

**How it works:**
1. Teacher generates pseudo-masks for unlabeled/weakly-labeled data
2. Only confident predictions (>threshold) are kept
3. Threshold gradually decreases: 0.9 → 0.6 (curriculum learning)
4. Student learns from high-quality filtered pseudo-labels

#### Entropy-Based Filtering

```bash
python main.py train \
    --ts_pseudo_mask_filtering entropy \
    --ts_entropy_threshold 1.0 \                # Keep predictions with entropy <1.0
    --ts_entropy_annealing cosine \             # Adaptive threshold increase
    --ts_entropy_min_threshold 0.5 \            # Start threshold (strict)
    --ts_entropy_max_threshold 1.5              # End threshold (relaxed)
```

**Entropy vs Confidence:**
- **Confidence**: Maximum softmax probability (0-1, higher = better)
- **Entropy**: Prediction uncertainty (0-∞, lower = better)
- **Recommendation**: Use confidence filtering (more intuitive)

### GT + Teacher Incorporation

Novel fusion strategy combining ground truth and teacher discoveries:

```bash
python main.py train \
    --architecture teacher_student_unet \
    --ts_gt_teacher_incorporate_enabled true \  # Enable GT+Teacher fusion
    --ts_gt_incorporate_start_epoch 20 \        # Start fusion at epoch 20
    --ts_gt_teacher_priority gt_foreground      # GT foreground + Teacher background
```

**Fusion Strategy:**
- **GT foreground preserved**: Expert annotations always used for labeled glands
- **GT background + Teacher**: Teacher can discover missed glands in background
- **Result**: Best of both worlds (expert + learned)

### EMA Teacher Updates

Control teacher learning from student:

```bash
python main.py train \
    --architecture teacher_student_unet \
    --ts_ema_schedule cosine \                  # Schedule: fixed, cosine, linear, exponential
    --ts_ema_decay_initial 0.999 \              # Initial EMA decay (stable teacher)
    --ts_ema_decay_final 0.1                    # Final EMA decay (adaptive teacher)
```

**EMA Update Formula:**
```
θ_teacher = α × θ_teacher + (1 - α) × θ_student
```

**Schedules:**
- **Fixed** (α=0.999): Constant, traditional approach
- **Cosine** (0.999→0.1): Smooth transition, recommended
- **Linear** (0.999→0.1): Linear interpolation
- **Exponential** (0.999→0.1): Faster early, slower late

### Consistency Loss

```bash
python main.py train \
    --architecture teacher_student_unet \
    --ts_consistency_loss_type mse \            # Options: mse, kl_div, l1, dice, iou
    --ts_consistency_temperature 2.0            # For kl_div (knowledge distillation)
```

**Loss Types:**
- `mse`: Mean squared error (pixel-wise, default)
- `kl_div`: KL divergence (knowledge distillation)
- `l1`: L1 distance (robust to outliers)
- `dice`: Dice similarity (regional consistency)
- `iou`: IoU similarity (overlap-based)

### Alpha Scheduling

Controls supervised vs consistency loss balance:

```bash
python main.py train \
    --architecture teacher_student_unet \
    --ts_min_alpha 0.2 \                        # Minimum supervised weight
    --ts_max_alpha 1.0                          # Maximum supervised weight (warmup)
```

**Schedule:**
```
Loss = α × L_supervised + (1 - α) × L_consistency
```

- **Warmup phase**: α = 1.0 (supervised only)
- **Early TS phase**: α ≈ 0.8 (mostly supervised)
- **Late TS phase**: α ≈ 0.2 (mostly consistency)

---

## Hyperparameter Tuning

### Recommended Starting Points

**For small datasets (<5K images):**
```bash
python main.py train \
    --batch_size 8 \
    --learning_rate 1e-4 \
    --epochs 200 \
    --early_stop_patience 40
```

**For large datasets (>20K images):**
```bash
python main.py train \
    --batch_size 16 \
    --learning_rate 5e-5 \
    --epochs 150 \
    --early_stop_patience 20
```

### Grid Search (SLURM)

For systematic hyperparameter exploration:

```bash
# Edit grid search parameters in run_nnunet_training.sh
# Then submit multiple jobs:
for lr in 1e-4 5e-5 1e-5; do
    sbatch run_nnunet_training.sh --learning_rate $lr
done
```

### Hyperparameter Sensitivity

**High impact:**
- Learning rate (1e-4 works well)
- Batch size (8-16 for GPUs with 10-16GB VRAM)
- Teacher initialization epoch (50 recommended)
- Confidence threshold (0.7-0.9 range)

**Medium impact:**
- EMA decay schedule (cosine recommended)
- Loss weights (default is balanced)
- Augmentation level (medium recommended)

**Low impact:**
- Early stop patience (20-40 range)
- LR scheduler patience (5-15 range)

---

## Monitoring Training

### TensorBoard

```bash
# Start TensorBoard
tensorboard --logdir outputs/experiment_1/logs/tensorboard

# Open browser to http://localhost:6006
```

**Key Metrics to Monitor:**
- **Training Loss**: Should decrease steadily
- **Validation Dice Score**: Should increase then plateau
- **Pseudo-Dice Score** (TS only): Student-teacher alignment
- **Alpha Schedule** (TS only): Transition from supervised to consistency

### Training Logs

```bash
# View training progress
tail -f outputs/experiment_1/logs/training.log

# Monitor GPU usage
watch nvidia-smi
```

### Visualization Outputs

Training generates visualizations automatically:

```
outputs/experiment_1/visualizations/
├── training_curves.png          # Loss, Dice, IoU over epochs
├── training_curves.pdf          # High-res version
├── sample_predictions/          # Sample predictions per epoch
└── pseudo_gt_analysis/          # Teacher-student alignment (TS only)
```

---

## Resuming Training

### From Latest Checkpoint

```bash
python main.py train \
    --architecture teacher_student_unet \
    --resume outputs/experiment_1/models/latest_model.pth \
    --output_dir outputs/experiment_1_resumed
```

### From Best Checkpoint

```bash
python main.py train \
    --architecture teacher_student_unet \
    --resume outputs/experiment_1/models/best_student_model.pth \
    --output_dir outputs/experiment_1_resumed
```

### SLURM Resume

```bash
# Edit resume_nnunet_training.sh with checkpoint path
sbatch resume_nnunet_training.sh
```

**What gets restored:**
- Model weights (student + teacher for TS)
- Optimizer state
- Learning rate scheduler state
- Training epoch counter
- Best validation metrics

---

## Troubleshooting

### CUDA Out of Memory

**Symptoms:** `RuntimeError: CUDA out of memory`

**Solutions:**
1. Reduce batch size: `--batch_size 4` or `--batch_size 2`
2. Reduce image size (edit `dataset.py`)
3. Enable gradient checkpointing (edit `teacher_student_unet.py`)
4. Use mixed precision training (edit `trainer.py`)

### Training Not Converging

**Symptoms:** Loss not decreasing, Dice score stuck

**Solutions:**
1. Check data loading: `python tests/test_basic_demo.py`
2. Reduce learning rate: `--learning_rate 5e-5`
3. Increase warmup epochs: `--ts_teacher_init_epoch 80`
4. Verify dataset labels are correct

### Slow Training

**Symptoms:** <5 iterations/sec

**Solutions:**
1. Enable more data workers: Edit `dataset.py` → `num_workers=4`
2. Use faster storage (SSD instead of HDD)
3. Reduce image resolution
4. Profile bottlenecks: Use PyTorch profiler

### Teacher-Student Divergence

**Symptoms:** Pseudo-Dice score decreasing, student worse than teacher

**Solutions:**
1. Lower EMA decay final: `--ts_ema_decay_final 0.5`
2. Increase confidence threshold: `--ts_confidence_threshold 0.9`
3. Delay teacher initialization: `--ts_teacher_init_epoch 80`
4. Increase supervised loss weight: `--ts_min_alpha 0.4`

### Overfitting

**Symptoms:** Training loss low, validation loss high

**Solutions:**
1. Stronger augmentation: `--augmentation_level strong`
2. Early stopping: `--early_stop_patience 20`
3. Reduce model capacity (use baseline_unet)
4. Add dropout (edit model architecture)

---

## Training Best Practices

### ✅ Do

- Start with default hyperparameters
- Monitor TensorBoard regularly
- Save checkpoints frequently
- Use early stopping
- Validate on separate test set
- Document experiment configurations
- Use version control (git) for code

### ❌ Don't

- Train without validation set
- Use test set for hyperparameter tuning
- Ignore CUDA OOM warnings (reduce batch size)
- Change multiple hyperparameters at once
- Train without monitoring metrics
- Overwrite previous experiments (use timestamped directories)

---

## Next Steps

After training:

1. **Evaluate model**: See `main.py evaluate` command
2. **Analyze results**: Check `outputs/experiment_1/evaluations/`
3. **Visualize predictions**: Review `outputs/experiment_1/visualizations/`
4. **Compare models**: Use `independent_eval/` for fair comparison
5. **Fine-tune**: Adjust hyperparameters based on results

## Support

For training issues:
- Check [Troubleshooting](#troubleshooting) section
- Review [GitHub Issues](https://github.com/YOUR_USERNAME/gland-segmentation-teacher-student/issues)
- Contact: [email - TO BE FILLED]
