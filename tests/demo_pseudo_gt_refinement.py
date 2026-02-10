#!/usr/bin/env python3
"""
Test Script for Pseudo-GT Visualization
======================================

This script tests the enhanced visualization functionality for Student vs Pseudo-GT metrics.
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def create_mock_training_history(num_epochs: int = 50) -> dict:
    """Create mock training history with realistic Teacher-Student training patterns"""
    epochs = np.arange(1, num_epochs + 1)
    teacher_init_epoch = 15

    history = {
        'train_loss': [],
        'val_loss': [],
        'train_supervised_loss': [],
        'train_consistency_loss': [],
        'train_alpha': [],
        'train_seg_dice': [],
        'val_seg_dice': [],
        'train_patch_acc': [],
        'val_patch_acc': [],
        'train_gland_acc': [],
        'val_gland_acc': [],
        'train_pseudo_dice': [],
        'train_pseudo_iou': [],
        'phase': [],
        'teacher_init_epoch': teacher_init_epoch
    }

    for epoch in epochs:
        # Training patterns
        train_loss = 2.0 * np.exp(-epoch / 20) + 0.5 + 0.1 * np.random.random()
        val_loss = train_loss + 0.2 + 0.15 * np.random.random()

        supervised_loss = 1.5 * np.exp(-epoch / 15) + 0.3 + 0.05 * np.random.random()

        if epoch >= teacher_init_epoch:
            consistency_loss = 0.8 * np.exp(-(epoch - teacher_init_epoch) / 10) + 0.1 + 0.05 * np.random.random()
            phase = 'teacher_student'
            alpha = 0.5 * (1 - np.cos(np.pi * (epoch - teacher_init_epoch) / (num_epochs - teacher_init_epoch)))
        else:
            consistency_loss = 0.0
            phase = 'warmup'
            alpha = 0.0

        # Dice scores
        base_dice = 0.3 + 0.6 * (1 - np.exp(-epoch / 12))
        train_dice = min(base_dice + 0.05 * np.random.random(), 0.95)
        val_dice = min(train_dice - 0.05 + 0.03 * np.random.random(), 0.92)

        # Pseudo-GT metrics
        if epoch >= teacher_init_epoch:
            pseudo_dice = min(train_dice + 0.02 * np.sin(epoch / 3) + 0.02 * np.random.random(), 0.93)
            pseudo_iou = min(pseudo_dice * 0.85 + 0.05 * np.random.random(), 0.88)
        else:
            pseudo_dice = 0.0
            pseudo_iou = 0.0

        # Store values
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_supervised_loss'].append(supervised_loss)
        history['train_consistency_loss'].append(consistency_loss)
        history['train_alpha'].append(alpha)
        history['train_seg_dice'].append(train_dice)
        history['val_seg_dice'].append(val_dice)
        history['train_patch_acc'].append(0.8 + 0.1 * np.random.random())
        history['val_patch_acc'].append(0.75 + 0.1 * np.random.random())
        history['train_gland_acc'].append(0.85 + 0.1 * np.random.random())
        history['val_gland_acc'].append(0.8 + 0.1 * np.random.random())
        history['train_pseudo_dice'].append(pseudo_dice)
        history['train_pseudo_iou'].append(pseudo_iou)
        history['phase'].append(phase)

    return history

def plot_training_curves_with_pseudo_metrics(history: dict, save_dir: str):
    """Create enhanced training curves with Pseudo-GT metrics"""
    from matplotlib.patches import Rectangle

    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    epochs = list(range(1, len(history['train_loss']) + 1))

    # Create 3x3 subplot figure
    fig, axes = plt.subplots(3, 3, figsize=(30, 18))

    # Subplot 1: Total Loss
    axes[0, 0].plot(epochs, history['train_loss'], 'b-', label='Training', linewidth=2)
    axes[0, 0].plot(epochs, history['val_loss'], 'r-', label='Validation', linewidth=2)
    axes[0, 0].set_title('Total Loss', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Subplot 2: Loss Components
    axes[0, 1].plot(epochs, history['train_supervised_loss'], 'g-', label='Supervised', linewidth=2)
    axes[0, 1].plot(epochs, history['train_consistency_loss'], 'orange', label='Consistency', linewidth=2)
    axes[0, 1].set_title('Loss Components', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Subplot 3: Regular Dice Score
    train_dice_score = [score * 100 for score in history['train_seg_dice']]
    val_dice_score = [score * 100 for score in history['val_seg_dice']]

    axes[0, 2].plot(epochs, train_dice_score, 'forestgreen', label='Training', linewidth=2)
    axes[0, 2].plot(epochs, val_dice_score, 'darkgreen', label='Validation', linewidth=2)
    axes[0, 2].set_title('Dice Score (Student vs GT)', fontsize=14, fontweight='bold')
    axes[0, 2].set_xlabel('Epoch')
    axes[0, 2].set_ylabel('Dice Score (%)')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)

    # Subplot 4: Alpha Schedule
    axes[1, 0].plot(epochs, history['train_alpha'], 'purple', label='Alpha', linewidth=2)
    axes[1, 0].set_title('Alpha Schedule', fontsize=14, fontweight='bold')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Alpha')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Subplot 5: Pseudo-Dice Score (NEW!)
    train_pseudo_dice = [score * 100 for score in history['train_pseudo_dice']]

    axes[1, 1].plot(epochs, train_pseudo_dice, 'goldenrod', label='Student vs Pseudo-GT', linewidth=3, linestyle='--')
    axes[1, 1].plot(epochs, train_dice_score, 'forestgreen', label='Student vs GT', linewidth=2, alpha=0.7)
    axes[1, 1].set_title('Pseudo-Dice Score (NEW!)', fontsize=14, fontweight='bold')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Dice Score (%)')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    # Add shaded region for teacher-student phase
    teacher_init_epoch = history.get('teacher_init_epoch')
    if teacher_init_epoch:
        axes[1, 1].axvspan(teacher_init_epoch, len(epochs), alpha=0.1, color='gold')

    # Subplot 6: Pseudo-IoU Score (NEW!)
    train_pseudo_iou = [score * 100 for score in history['train_pseudo_iou']]

    axes[1, 2].plot(epochs, train_pseudo_iou, 'darkorange', label='Student vs Pseudo-GT', linewidth=3, linestyle='--')
    axes[1, 2].set_title('Pseudo-IoU Score (NEW!)', fontsize=14, fontweight='bold')
    axes[1, 2].set_xlabel('Epoch')
    axes[1, 2].set_ylabel('IoU Score (%)')
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)

    if teacher_init_epoch:
        axes[1, 2].axvspan(teacher_init_epoch, len(epochs), alpha=0.1, color='orange')

    # Subplot 7: Training Phase
    phase_data = history['phase']
    phase_numeric = [0 if p == 'warmup' else 1 for p in phase_data]

    axes[2, 0].plot(epochs, phase_numeric, 'darkblue', linewidth=3, marker='o', markersize=4)
    axes[2, 0].set_title('Training Phase', fontsize=14, fontweight='bold')
    axes[2, 0].set_xlabel('Epoch')
    axes[2, 0].set_ylabel('Phase')
    axes[2, 0].set_yticks([0, 1])
    axes[2, 0].set_yticklabels(['Warmup', 'Teacher-Student'])
    axes[2, 0].grid(True, alpha=0.3)

    # Subplot 8: Alignment Analysis
    pseudo_gt_diff = [abs(pd - gd) for pd, gd in zip(train_pseudo_dice, train_dice_score) if pd > 0]
    if pseudo_gt_diff:
        axes[2, 1].plot(epochs[-len(pseudo_gt_diff):], pseudo_gt_diff, 'crimson',
                       label='|Pseudo-Dice - GT-Dice|', linewidth=2, marker='x')
        axes[2, 1].set_title('Pseudo-GT Alignment', fontsize=14, fontweight='bold')
        axes[2, 1].set_xlabel('Epoch')
        axes[2, 1].set_ylabel('Difference (%)')
        axes[2, 1].legend()
        axes[2, 1].grid(True, alpha=0.3)

    # Hide unused subplot
    axes[2, 2].axis('off')
    if teacher_init_epoch:
        legend_elements = [Rectangle((0, 0), 1, 1, facecolor='lightblue', alpha=0.3,
                                   label=f'Teacher Init: Epoch {teacher_init_epoch}')]
        axes[2, 2].legend(handles=legend_elements, loc='center', fontsize=12)
        axes[2, 2].set_title('Legend', fontsize=14, fontweight='bold')

    # Overall title
    fig.suptitle('Teacher-Student nnU-Net Training with Pseudo-GT Monitoring',
                 fontsize=20, fontweight='bold', y=0.98)

    plt.tight_layout()
    plt.subplots_adjust(top=0.93)

    # Save plots
    plot_png = save_path / 'teacher_student_training_with_pseudo_gt.png'
    plot_pdf = save_path / 'teacher_student_training_with_pseudo_gt.pdf'

    plt.savefig(plot_png, dpi=300, bbox_inches='tight')
    plt.savefig(plot_pdf, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"📊 Enhanced training curves saved:")
    print(f"   📈 PNG: {plot_png}")
    print(f"   📈 PDF: {plot_pdf}")

def main():
    """Test the pseudo-GT visualization"""
    print("🧪 Testing Pseudo-GT Visualization...")

    # Create mock data
    history = create_mock_training_history(50)
    print("📊 Generated mock training history")

    # Create test directory
    test_dir = Path("test_pseudo_gt_visualizations")
    test_dir.mkdir(exist_ok=True)

    # Generate visualization
    try:
        plot_training_curves_with_pseudo_metrics(history, str(test_dir))
        print("✅ Visualization test completed successfully!")
        print(f"📂 Check plots in: {test_dir.absolute()}")
        return True
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

if __name__ == "__main__":
    success = main()
    if success:
        print("\n🎉 All tests passed!")
    else:
        print("\n❌ Tests failed!")
        sys.exit(1)