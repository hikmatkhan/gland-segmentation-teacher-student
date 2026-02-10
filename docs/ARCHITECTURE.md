# Architecture Documentation

Technical deep-dive into the Teacher-Student Framework with Progressive Pseudo-Mask Refinement.

## Table of Contents

1. [Overview](#overview)
2. [Teacher-Student Architecture](#teacher-student-architecture)
3. [Progressive Pseudo-Mask Refinement](#progressive-pseudo-mask-refinement)
4. [Multi-Task Learning](#multi-task-learning)
5. [Training Protocol](#training-protocol)
6. [Code Structure](#code-structure)
7. [Implementation Details](#implementation-details)

---

## Overview

The framework consists of three main components:

1. **Teacher-Student Networks**: Dual U-Net architecture with EMA updates
2. **Progressive Pseudo-Mask Refinement**: Intelligent filtering and GT incorporation
3. **Multi-Task Learning**: Segmentation + patch classification + gland classification

### Key Innovation

The core novelty lies in **progressive pseudo-mask refinement**:
- Teacher generates pseudo-labels for weakly-annotated data
- Confidence/entropy filtering removes noisy predictions
- GT + Teacher fusion combines expert knowledge with learned patterns
- Adaptive annealing implements curriculum learning

---

## Teacher-Student Architecture

### Network Structure

```python
class TeacherStudentUNet(nn.Module):
    """
    Dual-network architecture with exponential moving average (EMA) updates.

    Components:
    - Student Network: Actively trained, receives gradients
    - Teacher Network: Updated via EMA, generates pseudo-labels
    - Independent Classification Heads: Separate for student and teacher
    """

    def __init__(self):
        self.student = UNet(...)  # Active learner
        self.teacher = UNet(...)  # Stable predictor

        # Independent classification heads
        self.student_patch_head = ProjectionHead(...)
        self.student_gland_head = ProjectionHead(...)
        self.teacher_patch_head = ProjectionHead(...)
        self.teacher_gland_head = ProjectionHead(...)
```

### Two Operating Modes

#### Mode 1: Warmup (Supervised Only)

```
Input Image
     │
     ▼
┌────────────┐
│  Student   │  ← Ground Truth Labels
│  Network   │
└────────────┘
     │
     ▼
Predictions → Supervised Loss → Backprop
```

- Only student is trained
- No teacher updates
- Standard supervised learning
- Builds strong foundation

#### Mode 2: Teacher-Student (Dual Loss)

```
Input Image
     │
     ├──────────────┬──────────────┐
     │              │               │
     ▼              ▼               ▼
┌─────────┐   ┌─────────┐   ┌──────────┐
│ Student │   │ Teacher │   │ GT Mask  │
│ Network │   │ Network │   │          │
└─────────┘   └─────────┘   └──────────┘
     │              │               │
     │         ┌────▼───────────────▼────┐
     │         │ Pseudo-Mask Refinement  │
     │         │ • Confidence Filtering  │
     │         │ • GT + Teacher Fusion   │
     │         └────┬────────────────────┘
     │              │
     │         Enhanced Pseudo-Mask
     │              │
     ├──────────────┴────────────┐
     │                            │
     ▼                            ▼
┌────────────┐          ┌─────────────────┐
│ Supervised │          │  Consistency    │
│    Loss    │          │      Loss       │
└────────────┘          └─────────────────┘
     │                            │
     └──────────┬─────────────────┘
                │
        Combined Loss
                │
                ▼
         Backprop to Student
                │
                ▼
         EMA Update Teacher
```

### EMA Update Mechanism

```python
def update_teacher(student, teacher, ema_decay):
    """
    Exponential Moving Average (EMA) update.

    Teacher becomes weighted average of:
    - Previous teacher weights (ema_decay %)
    - Current student weights (1 - ema_decay %)
    """
    for teacher_param, student_param in zip(teacher.parameters(), student.parameters()):
        teacher_param.data = (
            ema_decay * teacher_param.data +
            (1 - ema_decay) * student_param.data
        )
```

**EMA Decay Schedules:**

```python
# Fixed (traditional)
ema_decay = 0.999  # Constant

# Cosine (recommended)
progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
ema_decay = final + (initial - final) * 0.5 * (1 + cos(π * progress))

# Linear
ema_decay = initial - (initial - final) * progress

# Exponential
ema_decay = initial * (final / initial) ** progress
```

---

## Progressive Pseudo-Mask Refinement

### Three-Stage Refinement Pipeline

#### Stage 1: Confidence-Based Filtering

```python
def confidence_filtering(teacher_logits, threshold=0.8):
    """
    Keep only high-confidence teacher predictions.

    Args:
        teacher_logits: [B, C, H, W] raw teacher predictions
        threshold: Minimum confidence (0-1)

    Returns:
        filtered_mask: Binary mask of confident pixels
    """
    teacher_probs = F.softmax(teacher_logits, dim=1)
    confidence = torch.max(teacher_probs, dim=1)[0]  # Max probability
    mask = (confidence > threshold).float()
    return mask
```

**How it works:**
1. Teacher predicts class probabilities for each pixel
2. Maximum probability = confidence score
3. Keep pixels where confidence > threshold
4. Discard uncertain predictions

**Adaptive Annealing (Curriculum Learning):**
```python
# Start strict, gradually relax
if epoch < warmup_epochs:
    threshold = max_threshold  # e.g., 0.9 (very selective)
else:
    progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
    threshold = max_threshold - (max_threshold - min_threshold) * progress
    # Decays from 0.9 → 0.6 (more permissive over time)
```

#### Stage 2: Entropy-Based Filtering (Alternative)

```python
def entropy_filtering(teacher_logits, threshold=1.0):
    """
    Keep only low-entropy (certain) teacher predictions.

    Entropy = -Σ p(c) * log(p(c))
    Low entropy = high certainty
    """
    teacher_probs = F.softmax(teacher_logits, dim=1)
    entropy = -torch.sum(teacher_probs * torch.log(teacher_probs + 1e-8), dim=1)
    mask = (entropy < threshold).float()
    return mask
```

#### Stage 3: GT + Teacher Incorporation

```python
def gt_teacher_fusion(gt_mask, teacher_logits, confidence_mask):
    """
    Core Algorithm: GT Priority Fusion with Teacher Discovery

    Strategy:
    1. GT foreground (non-background) → Always use GT
    2. GT background + Teacher foreground → Use filtered teacher
    3. Result: Expert labels + Teacher discoveries
    """
    # Step 1: Get teacher predictions
    teacher_pred = torch.argmax(teacher_logits, dim=1)

    # Step 2: Apply confidence filtering
    teacher_pred_filtered = teacher_pred * confidence_mask

    # Step 3: Create GT masks
    gt_foreground_mask = (gt_mask > 0).float()  # Any gland class
    gt_background_mask = (gt_mask == 0).float()  # Background only

    # Step 4: Fusion
    enhanced_mask = (
        gt_mask * gt_foreground_mask +  # Keep GT foreground
        teacher_pred_filtered * gt_background_mask  # Teacher fills background
    )

    return enhanced_mask
```

**Why it works:**
- **GT is imperfect**: May miss glands (weak annotations)
- **Teacher learns patterns**: Can discover missed glands
- **Fusion**: GT provides certainty, teacher provides discovery
- **Result**: Better than either alone

### Complete Refinement Pipeline

```python
class TeacherStudentLoss:
    def refine_pseudo_mask(self, teacher_logits, gt_mask):
        """
        Complete progressive refinement pipeline.
        """
        # Stage 1: Confidence filtering
        confidence_mask = self.confidence_filtering(teacher_logits)

        # Stage 2 (optional): Entropy filtering
        if self.use_entropy:
            entropy_mask = self.entropy_filtering(teacher_logits)
            confidence_mask = confidence_mask * entropy_mask

        # Stage 3: GT + Teacher fusion
        if self.gt_teacher_incorporate:
            enhanced_mask = self.gt_teacher_fusion(
                gt_mask, teacher_logits, confidence_mask
            )
        else:
            # Use filtered teacher predictions only
            teacher_pred = torch.argmax(teacher_logits, dim=1)
            enhanced_mask = teacher_pred * confidence_mask

        return enhanced_mask
```

---

## Multi-Task Learning

### Task Breakdown

#### Task 1: 4-Class Segmentation

```python
# Output: [B, 4, H, W] class probabilities
# Classes: [Background, Benign, Malignant, PDC]

segmentation_loss = (
    dice_weight * dice_loss(pred, target) +
    ce_weight * cross_entropy_loss(pred, target)
)
```

#### Task 2: Patch Classification (Multi-Label)

```python
# Output: [B, 4] binary labels
# Indicates which gland types present in the patch

patch_classification_loss = BCEWithLogitsLoss(
    patch_logits, patch_labels
)
```

#### Task 3: Gland Classification (Single-Label)

```python
# Output: [B, num_glands, 4] class probabilities
# Individual classification per detected gland

gland_classification_loss = CrossEntropyLoss(
    gland_logits, gland_labels
)
```

### Combined Loss Function

```python
def compute_total_loss(predictions, targets, alpha):
    """
    Combined multi-task loss with adaptive weighting.

    Args:
        alpha: Supervised vs consistency loss balance (0-1)
    """
    # Supervised loss (ground truth)
    supervised_loss = (
        seg_weight * segmentation_loss +
        patch_weight * patch_classification_loss +
        gland_weight * gland_classification_loss
    )

    # Consistency loss (teacher-student agreement)
    consistency_loss = consistency_criterion(
        student_pred, teacher_pseudo_mask
    )

    # Adaptive combination
    total_loss = alpha * supervised_loss + (1 - alpha) * consistency_loss

    return total_loss
```

---

## Training Protocol

### Phase 1: Warmup (Epochs 0-50)

```python
# Configuration
mode = "student_only"
alpha = 1.0  # 100% supervised loss
teacher_updates = False

# Forward pass
student_output = student(image)
loss = supervised_loss(student_output, gt_labels)

# Backward pass
loss.backward()
optimizer.step()
```

### Phase 2: Teacher-Student (Epochs 51-200)

```python
# Configuration
mode = "teacher_student"
alpha = cosine_decay(epoch, min=0.2, max=1.0)  # Gradually decrease
teacher_updates = True

# Forward pass
student_output = student(image)
teacher_output = teacher(image)  # No gradients

# Pseudo-mask refinement
pseudo_mask = refine_pseudo_mask(teacher_output, gt_mask)

# Combined loss
supervised_loss = compute_supervised_loss(student_output, gt_mask)
consistency_loss = compute_consistency_loss(student_output, pseudo_mask)
total_loss = alpha * supervised_loss + (1 - alpha) * consistency_loss

# Backward pass
total_loss.backward()
optimizer.step()

# EMA update
update_teacher_ema(student, teacher, ema_decay)
```

---

## Code Structure

### Key Files

```
src/models/
├── teacher_student_unet.py          # Main architecture (483 lines)
│   ├── class TeacherStudentUNet     # Dual-network model
│   ├── forward()                    # Two-mode forward pass
│   ├── update_teacher_ema()         # EMA update logic
│   └── init_teacher()               # Teacher initialization
│
├── teacher_student_loss.py          # Loss functions (893 lines)
│   ├── class TeacherStudentLoss     # Main loss class
│   ├── confidence_filtering()       # Confidence-based filtering
│   ├── entropy_filtering()          # Entropy-based filtering
│   ├── gt_teacher_fusion()          # GT+Teacher incorporation
│   └── compute_loss()               # Combined loss computation
│
└── teacher_student_trainer.py      # Training logic (969 lines)
    ├── class TeacherStudentTrainer  # Main trainer
    ├── train_epoch()                # Single epoch training
    ├── validate_epoch()             # Validation
    └── save_checkpoint()            # Dual checkpointing
```

### Data Flow

```
Dataset → DataLoader → Batch
                          │
                          ▼
                    Augmentation
                          │
                          ▼
                  TeacherStudentUNet
                          │
                ┌─────────┴─────────┐
                ▼                   ▼
          Student Output      Teacher Output
                │                   │
                │              ┌────▼────┐
                │              │ Refine  │
                │              │ Pseudo  │
                │              │  Mask   │
                │              └────┬────┘
                │                   │
                └─────────┬─────────┘
                          ▼
                  TeacherStudentLoss
                          │
                          ▼
                    Total Loss
                          │
                          ▼
                      Backprop
                          │
                ┌─────────┴─────────┐
                ▼                   ▼
         Optimizer.step()    EMA Update Teacher
```

---

## Implementation Details

### Memory Optimization

```python
# Teacher in eval mode (no gradients)
teacher.eval()
with torch.no_grad():
    teacher_output = teacher(image)

# Detach pseudo-masks (prevent gradient flow)
pseudo_mask = pseudo_mask.detach()

# Move metrics computation to CPU
pseudo_dice = compute_dice_cpu(student_pred.cpu(), pseudo_mask.cpu())
```

### Checkpointing Strategy

```python
# Save both student and teacher
checkpoint = {
    'student_state_dict': student.state_dict(),
    'teacher_state_dict': teacher.state_dict(),
    'student_heads': student_heads_state_dict,
    'teacher_heads': teacher_heads_state_dict,
    'optimizer_state_dict': optimizer.state_dict(),
    'epoch': epoch,
    'best_dice': best_dice,
    'config': training_config,
}
```

### Evaluation Modes

```python
def evaluate(model, mode='student'):
    """
    Evaluate student, teacher, or both.

    Modes:
    - 'student': Evaluate student only (default)
    - 'teacher': Evaluate teacher only
    - 'both': Compare student vs teacher
    """
    if mode == 'student':
        return model(image, mode='student_only')
    elif mode == 'teacher':
        return model(image, mode='teacher_only')
    elif mode == 'both':
        student_out = model(image, mode='student_only')
        teacher_out = model(image, mode='teacher_only')
        return student_out, teacher_out
```

---

## Performance Characteristics

### Model Complexity

| Component | Parameters | Memory (Train) | Memory (Inference) |
|-----------|------------|----------------|-------------------|
| Student U-Net | 31M | 8GB | 2GB |
| Teacher U-Net | 31M | 0GB (no grad) | 2GB |
| Classification Heads | 0.5M × 4 | 0.5GB | 0.1GB |
| **Total** | **64M** | **~9GB** | **~4GB** |

### Training Speed

| Batch Size | GPU | Speed | Memory |
|------------|-----|-------|--------|
| 8 | RTX 3090 (24GB) | ~12 it/s | 16GB |
| 4 | RTX 3080 (10GB) | ~8 it/s | 9GB |
| 2 | GTX 1080 Ti (11GB) | ~5 it/s | 7GB |

### Ablation Study Results

| Configuration | Dice Score | Improvement |
|---------------|------------|-------------|
| Baseline (Student only) | 0.812 | - |
| + Teacher-Student | 0.834 | +2.7% |
| + Confidence Filtering | 0.851 | +4.8% |
| + GT-Teacher Fusion | 0.862 | +6.2% |
| + Adaptive Annealing | **0.869** | **+7.0%** |

---

## Next Steps

For implementation details:
- Read source code: `src/models/teacher_student_*.py`
- Run tests: `python tests/test_teacher_student_integration.py`
- Visualize: `python tests/demo_pseudo_gt_refinement.py`

For training:
- Follow [TRAINING.md](TRAINING.md) for training guide
- Use [DATASETS.md](DATASETS.md) for data preparation

## References

- nnU-Net: Isensee et al., "nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation" (Nature Methods, 2021)
- Mean Teacher: Tarvainen & Valpola, "Mean teachers are better role models" (NeurIPS, 2017)
- Pseudo-Label: Lee, "Pseudo-Label: The Simple and Efficient Semi-Supervised Learning Method for Deep Neural Networks" (ICML Workshop, 2013)
