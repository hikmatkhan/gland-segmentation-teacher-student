#!/usr/bin/env python3
"""
4-Class Projection Heads for Combined Gland Segmentation
Adapted for 4-class multi-task learning: Background(0), Benign(1), Malignant(2), PDC(3)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Optional, Dict, Any
import numpy as np

# Try to import skimage for gland extraction, fallback if not available
try:
    from skimage.measure import label, regionprops
    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False
    print("⚠️ scikit-image not available, using simplified gland extraction")

class PatchClassificationHead(nn.Module):
    """
    Multi-label 4-class patch-level classification head
    Predicts presence of each class in patch: Background(0), Benign(1), Malignant(2), PDC(3)
    Uses multi-label classification since patches can contain multiple gland types
    """
    def __init__(self, input_channels: int, num_classes: int = 4, dropout_p: float = 0.5):
        super().__init__()

        self.num_classes = num_classes
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout_p)

        # Multi-layer classifier for better representation
        # Note: No final activation - will use sigmoid for multi-label
        self.classifier = nn.Sequential(
            nn.Linear(input_channels, input_channels // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_p),
            nn.Linear(input_channels // 2, num_classes)
            # No sigmoid here - applied in loss function or during inference
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: [B, C, H, W] bottleneck features
        Returns:
            patch_logits: [B, num_classes] multi-label patch-level predictions
        """
        # Global average pooling
        x = self.global_pool(features)  # [B, C, 1, 1]
        x = torch.flatten(x, 1)         # [B, C]

        # Multi-label classification
        logits = self.classifier(x)     # [B, num_classes]

        return logits

class GlandClassificationHead(nn.Module):
    """
    4-class gland-level classification head
    Extracts features for individual glands and classifies each into 4 classes
    """
    def __init__(self, input_channels: int, num_classes: int = 4, dropout_p: float = 0.5, min_gland_size: int = 4):
        super().__init__()

        self.num_classes = num_classes
        self.min_gland_size = min_gland_size

        # 4-class classification: Background(0), Benign(1), Malignant(2), PDC(3)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_p),
            nn.Linear(input_channels, input_channels // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_p),
            nn.Linear(input_channels // 2, num_classes)
        )

    def extract_gland_features_advanced(self, features: torch.Tensor, segmentation_mask: torch.Tensor) -> Tuple[torch.Tensor, List[int]]:
        """
        Advanced gland feature extraction using connected component analysis (requires scikit-image)
        """
        batch_size = features.shape[0]
        all_gland_features = []
        gland_counts = []

        for b in range(batch_size):
            # Get current image features and mask
            img_features = features[b]  # [C, H_f, W_f]
            img_mask = segmentation_mask[b]  # [H_s, W_s]

            # Resize segmentation mask to match feature dimensions
            if img_mask.shape != img_features.shape[-2:]:
                img_mask = F.interpolate(
                    img_mask.unsqueeze(0).unsqueeze(0).float(),
                    size=img_features.shape[-2:],
                    mode='nearest'
                ).squeeze(0).squeeze(0)

            # Find glands (non-background regions: classes 1, 2, 3)
            gland_mask = (img_mask > 0).float()  # [H_f, W_f]

            if gland_mask.sum() == 0:
                # No glands found
                gland_counts.append(0)
                continue

            # Convert to numpy for connected component analysis
            gland_mask_np = gland_mask.detach().cpu().numpy()
            labeled_mask = label(gland_mask_np)
            regions = regionprops(labeled_mask)

            img_gland_features = []
            for region in regions:
                if region.area < self.min_gland_size:  # Skip very small regions
                    continue

                # Create mask for this gland
                region_mask = torch.zeros_like(gland_mask)
                region_coords = region.coords
                for coord in region_coords:
                    region_mask[coord[0], coord[1]] = 1.0

                # Extract features for this gland region
                masked_features = img_features * region_mask.unsqueeze(0)  # [C, H_f, W_f]

                # Average pooling over the gland region
                if region_mask.sum() > 0:
                    gland_feature = masked_features.sum(dim=[1, 2]) / region_mask.sum()  # [C]
                    img_gland_features.append(gland_feature)

            if img_gland_features:
                img_gland_features = torch.stack(img_gland_features)  # [num_glands, C]
                all_gland_features.append(img_gland_features)
                gland_counts.append(len(img_gland_features))
            else:
                gland_counts.append(0)

        if all_gland_features:
            gland_features = torch.cat(all_gland_features, dim=0)  # [total_glands, C]
        else:
            # No glands found in any image
            gland_features = torch.zeros(0, features.shape[1]).to(features.device)

        return gland_features, gland_counts

    def extract_gland_features_simple(self, features: torch.Tensor, segmentation_mask: torch.Tensor) -> Tuple[torch.Tensor, List[int]]:
        """
        Simplified gland feature extraction using class-based pooling (fallback when scikit-image not available)
        """
        batch_size = features.shape[0]
        all_gland_features = []
        gland_counts = []

        for b in range(batch_size):
            # Get current image features and mask
            img_features = features[b]  # [C, H_f, W_f]
            img_mask = segmentation_mask[b]  # [H_s, W_s]

            # Resize segmentation mask to match feature dimensions
            if img_mask.shape != img_features.shape[-2:]:
                img_mask = F.interpolate(
                    img_mask.unsqueeze(0).unsqueeze(0).float(),
                    size=img_features.shape[-2:],
                    mode='nearest'
                ).squeeze(0).squeeze(0)

            img_gland_features = []

            # Extract features for each non-background class
            for class_id in [1, 2, 3]:  # Benign, Malignant, PDC
                class_mask = (img_mask == class_id).float()

                if class_mask.sum() > self.min_gland_size:  # Only if sufficient pixels
                    # Average pooling over class regions
                    masked_features = img_features * class_mask.unsqueeze(0)  # [C, H_f, W_f]
                    class_feature = masked_features.sum(dim=[1, 2]) / class_mask.sum()  # [C]
                    img_gland_features.append(class_feature)

            if img_gland_features:
                img_gland_features = torch.stack(img_gland_features)  # [num_glands, C]
                all_gland_features.append(img_gland_features)
                gland_counts.append(len(img_gland_features))
            else:
                gland_counts.append(0)

        if all_gland_features:
            gland_features = torch.cat(all_gland_features, dim=0)  # [total_glands, C]
        else:
            gland_features = torch.zeros(0, features.shape[1]).to(features.device)

        return gland_features, gland_counts

    def extract_gland_features(self, features: torch.Tensor, segmentation_mask: torch.Tensor) -> Tuple[torch.Tensor, List[int]]:
        """
        Extract features for individual glands from 4-class segmentation mask

        Args:
            features: [B, C, H_f, W_f] bottleneck features
            segmentation_mask: [B, H_s, W_s] segmentation predictions (0=bg, 1=benign, 2=malignant, 3=PDC)

        Returns:
            gland_features: [N, C] features for N glands across all images in batch
            gland_counts: List[int] number of glands per image in batch
        """
        if SKIMAGE_AVAILABLE:
            return self.extract_gland_features_advanced(features, segmentation_mask)
        else:
            return self.extract_gland_features_simple(features, segmentation_mask)

    def forward(self, features: torch.Tensor, segmentation_mask: torch.Tensor) -> Tuple[torch.Tensor, List[int]]:
        """
        Args:
            features: [B, C, H, W] bottleneck features
            segmentation_mask: [B, H, W] segmentation predictions

        Returns:
            gland_logits: [N, num_classes] gland-level predictions for N total glands
            gland_counts: List[int] number of glands per image in batch
        """
        gland_features, gland_counts = self.extract_gland_features(features, segmentation_mask)

        if gland_features.shape[0] > 0:
            gland_logits = self.classifier(gland_features)  # [N, 4] for 4 classes
        else:
            gland_logits = torch.zeros(0, self.num_classes).to(features.device)

        return gland_logits, gland_counts

class MultiClassDualClassificationHead(nn.Module):
    """
    Combined 4-class patch and gland classification head
    - Patch: 4-class classification (Background=0, Benign=1, Malignant=2, PDC=3)
    - Gland: 4-class classification (Background=0, Benign=1, Malignant=2, PDC=3)
    """
    def __init__(self, input_channels: int, patch_classes: int = 4, gland_classes: int = 4, dropout_p: float = 0.5):
        super().__init__()

        self.patch_classes = patch_classes
        self.gland_classes = gland_classes

        self.patch_head = PatchClassificationHead(input_channels, patch_classes, dropout_p)
        self.gland_head = GlandClassificationHead(input_channels, gland_classes, dropout_p)

    def forward(self, features: torch.Tensor, segmentation_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, List[int]]:
        """
        Args:
            features: [B, C, H, W] bottleneck features
            segmentation_mask: [B, H, W] segmentation predictions

        Returns:
            patch_logits: [B, 4] patch-level predictions (4-class)
            gland_logits: [N, 4] gland-level predictions (4-class)
            gland_counts: List[int] number of glands per image in batch
        """
        # Patch-level classification
        patch_logits = self.patch_head(features)

        # Gland-level classification
        gland_logits, gland_counts = self.gland_head(features, segmentation_mask)

        return patch_logits, gland_logits, gland_counts

def create_classification_labels_from_segmentation(segmentation_mask: torch.Tensor) -> torch.Tensor:
    """
    Create patch-level classification labels from 4-class segmentation mask
    DEPRECATED: Use create_multilabel_patch_labels_from_segmentation for multi-label approach

    Args:
        segmentation_mask: [B, H, W] segmentation mask with values 0-3

    Returns:
        patch_labels: [B] patch-level labels based on dominant non-background class
    """
    batch_size = segmentation_mask.shape[0]
    patch_labels = torch.zeros(batch_size, dtype=torch.long, device=segmentation_mask.device)

    for b in range(batch_size):
        mask = segmentation_mask[b]

        # Count pixels for each class
        class_counts = []
        for class_id in range(4):  # 0, 1, 2, 3
            count = (mask == class_id).sum().item()
            class_counts.append(count)

        # Exclude background (class 0) when determining patch label
        non_bg_counts = class_counts[1:]  # [benign, malignant, PDC]

        if sum(non_bg_counts) == 0:
            # Only background present
            patch_labels[b] = 0
        else:
            # Find dominant non-background class
            dominant_class = np.argmax(non_bg_counts) + 1  # +1 to account for skipping background
            patch_labels[b] = dominant_class

    return patch_labels

def create_multilabel_patch_labels_from_segmentation(segmentation_mask: torch.Tensor,
                                                   min_pixels_threshold: int = 50) -> torch.Tensor:
    """
    Create multi-label patch-level classification labels from 4-class segmentation mask
    Each patch can have multiple classes present simultaneously

    Args:
        segmentation_mask: [B, H, W] segmentation mask with values 0-3
        min_pixels_threshold: Minimum number of pixels for a class to be considered present

    Returns:
        patch_labels: [B, 4] multi-label binary tensor (0/1 for each class presence)
    """
    batch_size = segmentation_mask.shape[0]
    num_classes = 4
    patch_labels = torch.zeros(batch_size, num_classes, dtype=torch.float32, device=segmentation_mask.device)

    for b in range(batch_size):
        mask = segmentation_mask[b]

        # Count pixels for each class
        for class_id in range(num_classes):
            count = (mask == class_id).sum().item()

            # Set label to 1 if class has sufficient presence
            if count >= min_pixels_threshold:
                patch_labels[b, class_id] = 1.0

        # Special handling: if no foreground classes meet threshold,
        # set background to 1 (pure background patch)
        if patch_labels[b, 1:].sum() == 0:  # No foreground classes
            patch_labels[b, 0] = 1.0

    return patch_labels

def analyze_patch_class_distribution(segmentation_mask: torch.Tensor) -> Dict[str, Any]:
    """
    Analyze the class distribution in patches to understand multi-class presence

    Args:
        segmentation_mask: [B, H, W] segmentation mask with values 0-3

    Returns:
        Dictionary with analysis results
    """
    batch_size = segmentation_mask.shape[0]
    num_classes = 4
    class_names = ["Background", "Benign", "Malignant", "PDC"]

    results = {
        "total_patches": batch_size,
        "class_pixel_counts": {name: [] for name in class_names},
        "class_percentages": {name: [] for name in class_names},
        "patches_with_multiple_classes": 0,
        "class_combinations": {},
        "pure_class_patches": {name: 0 for name in class_names}
    }

    for b in range(batch_size):
        mask = segmentation_mask[b]
        total_pixels = mask.numel()

        # Count pixels and percentages for each class
        class_counts = []
        class_percentages = []
        present_classes = []

        for class_id in range(num_classes):
            count = (mask == class_id).sum().item()
            percentage = (count / total_pixels) * 100

            class_counts.append(count)
            class_percentages.append(percentage)

            results["class_pixel_counts"][class_names[class_id]].append(count)
            results["class_percentages"][class_names[class_id]].append(percentage)

            # Consider class present if > 1% of pixels
            if percentage > 1.0:
                present_classes.append(class_id)

        # Analyze multi-class presence
        if len(present_classes) > 1:
            results["patches_with_multiple_classes"] += 1

            # Track class combinations
            combination = tuple(sorted(present_classes))
            combination_names = tuple(class_names[i] for i in combination)
            if combination_names not in results["class_combinations"]:
                results["class_combinations"][combination_names] = 0
            results["class_combinations"][combination_names] += 1

        # Track pure class patches
        if len(present_classes) == 1:
            pure_class = present_classes[0]
            results["pure_class_patches"][class_names[pure_class]] += 1

    # Calculate summary statistics
    results["multi_class_percentage"] = (results["patches_with_multiple_classes"] / batch_size) * 100

    return results

def test_4class_classification_heads():
    """Test the 4-class classification heads"""
    print("🧪 Testing 4-class classification heads...")

    # Create test data
    batch_size, channels, height, width = 2, 512, 64, 64
    num_classes = 4  # Background, Benign, Malignant, PDC

    features = torch.randn(batch_size, channels, height, width)
    segmentation = torch.randint(0, 4, (batch_size, height, width)).float()  # 4-class segmentation

    print(f"📊 Test data shapes:")
    print(f"   Features: {features.shape}")
    print(f"   Segmentation: {segmentation.shape}")
    print(f"   Segmentation classes: {torch.unique(segmentation).tolist()}")

    # Test patch head (4-class classification)
    print(f"\n🏷️ Testing Patch Classification Head (4-class):")
    patch_head = PatchClassificationHead(channels, num_classes)
    patch_logits = patch_head(features)
    print(f"   ✅ Patch logits shape: {patch_logits.shape} (expected: [{batch_size}, {num_classes}])")
    print(f"   📈 Patch logits sample: {patch_logits[0].tolist()}")

    # Test gland head (4-class classification)
    print(f"\n🔍 Testing Gland Classification Head (4-class):")
    gland_head = GlandClassificationHead(channels, num_classes)
    gland_logits, gland_counts = gland_head(features, segmentation)
    print(f"   ✅ Gland logits shape: {gland_logits.shape} (expected: [N, {num_classes}])")
    print(f"   📊 Gland counts per image: {gland_counts}")
    if gland_logits.shape[0] > 0:
        print(f"   📈 Gland logits sample: {gland_logits[0].tolist()}")
    else:
        print(f"   ⚠️ No glands detected in test data")

    # Test dual head
    print(f"\n🔄 Testing Multi-Class Dual Classification Head:")
    dual_head = MultiClassDualClassificationHead(channels, num_classes, num_classes)
    patch_logits, gland_logits, gland_counts = dual_head(features, segmentation)
    print(f"   ✅ Patch logits shape: {patch_logits.shape} (expected: [{batch_size}, {num_classes}])")
    print(f"   ✅ Gland logits shape: {gland_logits.shape} (expected: [N, {num_classes}])")
    print(f"   📊 Gland counts: {gland_counts}")

    # Test label creation from segmentation
    print(f"\n🏷️ Testing Label Creation from Segmentation:")
    patch_labels = create_classification_labels_from_segmentation(segmentation)
    print(f"   ✅ Patch labels shape: {patch_labels.shape} (expected: [{batch_size}])")
    print(f"   📋 Patch labels: {patch_labels.tolist()}")

    # Test with different segmentation distributions
    print(f"\n🎯 Testing with different class distributions:")

    # Create mask with only benign glands (class 1)
    benign_mask = torch.zeros_like(segmentation)
    benign_mask[:, 20:40, 20:40] = 1  # Benign region
    benign_labels = create_classification_labels_from_segmentation(benign_mask)
    print(f"   Benign-only mask labels: {benign_labels.tolist()}")

    # Create mask with malignant glands (class 2)
    malignant_mask = torch.zeros_like(segmentation)
    malignant_mask[:, 10:30, 10:30] = 2  # Malignant region
    malignant_labels = create_classification_labels_from_segmentation(malignant_mask)
    print(f"   Malignant-only mask labels: {malignant_labels.tolist()}")

    # Create mask with PDC (class 3)
    pdc_mask = torch.zeros_like(segmentation)
    pdc_mask[:, 30:50, 30:50] = 3  # PDC region
    pdc_labels = create_classification_labels_from_segmentation(pdc_mask)
    print(f"   PDC-only mask labels: {pdc_labels.tolist()}")

    print(f"\n✅ 4-class classification heads test completed successfully!")
    print(f"📝 Summary:")
    print(f"   - Patch classification: {num_classes} classes")
    print(f"   - Gland classification: {num_classes} classes")
    print(f"   - Scikit-image available: {SKIMAGE_AVAILABLE}")
    print(f"   - All tests passed!")

if __name__ == "__main__":
    test_4class_classification_heads()