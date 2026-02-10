#!/usr/bin/env python3
"""
4-Class Dataset Loader for Combined Gland Segmentation
Supports unified loading of any combined dataset (mixed/separate magnifications)
"""

import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, Any, Tuple, List, Optional
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Import from our config
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from configs.paths_config import get_dataset_path, EVALUATION_CONFIG
from src.models.projection_heads import create_classification_labels_from_segmentation, create_multilabel_patch_labels_from_segmentation
from src.models.loss_functions import create_4class_gland_labels_from_patch_labels

class CombinedGlandDataset(Dataset):
    """
    4-class multi-task dataset for combined gland segmentation and classification
    Supports: Background(0), Benign(1), Malignant(2), PDC(3)
    """

    def __init__(self,
                 dataset_key: str = 'mag5x',
                 split: str = 'train',
                 image_size: Tuple[int, int] = (512, 512),
                 augment: bool = True,
                 auto_generate_labels: bool = True,
                 use_multilabel_patch: bool = True):
        """
        Args:
            dataset_key: Key for dataset selection ('mixed', 'mag5x', 'mag10x', 'mag20x', 'mag40x')
            split: Data split ('train', 'val', 'test')
            image_size: Target image size
            augment: Whether to apply data augmentation
            auto_generate_labels: Whether to auto-generate classification labels from segmentation
            use_multilabel_patch: Whether to use multi-label patch classification
        """

        self.dataset_key = dataset_key
        self.split = split
        self.image_size = image_size
        self.augment = augment and split == 'train'
        self.auto_generate_labels = auto_generate_labels
        self.use_multilabel_patch = use_multilabel_patch

        # Get dataset path
        self.data_root = Path(get_dataset_path(dataset_key))
        print(f"📁 Loading dataset: {dataset_key}")
        print(f"   Path: {self.data_root}")
        print(f"   Split: {split}")

        # Validate dataset exists
        if not self.data_root.exists():
            raise FileNotFoundError(f"Dataset not found: {self.data_root}")

        # Load dataset configuration
        dataset_json_path = self.data_root / "dataset.json"
        if not dataset_json_path.exists():
            raise FileNotFoundError(f"dataset.json not found: {dataset_json_path}")

        with open(dataset_json_path, 'r') as f:
            self.dataset_config = json.load(f)

        # Load classification labels if available
        self.patch_labels = {}
        classification_labels_path = self.data_root / "classification_labels.json"
        if classification_labels_path.exists():
            with open(classification_labels_path, 'r') as f:
                classification_data = json.load(f)
                self.patch_labels = classification_data.get('patch_labels', {})
                print(f"   ✅ Loaded {len(self.patch_labels)} classification labels")
        else:
            print(f"   ⚠️ No classification labels found, will auto-generate from segmentation")

        # Get file list for the split
        self.samples = self._get_split_samples()

        # Setup augmentations
        self.setup_augmentations()

        print(f"   📊 Loaded {len(self.samples)} {split} samples")

        # Verify class distribution in first few samples
        self._analyze_dataset()

    def _get_split_samples(self) -> List[Dict[str, str]]:
        """Get samples for the specified split"""
        split_map = {
            'train': 'training',
            'val': 'validation',
            'test': 'test'
        }

        nnunet_split = split_map.get(self.split, self.split)

        # Warwick: use test set for validation
        if self.dataset_key == 'warwick' and self.split == 'val':
            if 'test' in self.dataset_config:
                print(f"   🔄 Warwick: Using test set ({len(self.dataset_config['test'])} samples) for validation")
                return self.dataset_config['test']
            else:
                print(f"   ⚠️ Warwick: Test split not found, falling back to validation split")

        if nnunet_split in self.dataset_config:
            return self.dataset_config[nnunet_split]
        else:
            # Fallback: if validation split doesn't exist, use test for validation
            if self.split == 'val' and 'test' in self.dataset_config:
                print(f"   ⚠️ No validation split found, using test split for validation")
                return self.dataset_config['test']
            else:
                available_splits = list(self.dataset_config.keys())
                raise ValueError(f"Split '{nnunet_split}' not found. Available splits: {available_splits}")

    def setup_augmentations(self):
        """Setup data augmentations for 4-class segmentation"""
        if self.augment:
            self.transform = A.Compose([
                A.Resize(*self.image_size),
                # Spatial augmentations
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.ShiftScaleRotate(
                    shift_limit=0.1,
                    scale_limit=0.1,
                    rotate_limit=15,
                    border_mode=cv2.BORDER_CONSTANT,
                    p=0.5
                ),
                # Color augmentations (help with stain variation)
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
                A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.3),
                A.GaussianBlur(blur_limit=3, p=0.3),
                # Add some noise to make model more robust
                A.GaussNoise(noise_scale_factor=0.1, p=0.2),
                # Normalize using ImageNet statistics
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ])
        else:
            self.transform = A.Compose([
                A.Resize(*self.image_size),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ])

    def _analyze_dataset(self):
        """Analyze dataset to understand class distribution"""
        print(f"   🔍 Analyzing dataset...")

        # Check a few samples to understand the data
        sample_indices = list(range(min(5, len(self.samples))))
        class_counts = {0: 0, 1: 0, 2: 0, 3: 0}

        for idx in sample_indices:
            try:
                sample = self.samples[idx]

                # Get paths
                if self.split == 'train':
                    label_path = self.data_root / sample['label'].replace('./labelsTr/', 'labelsTr/')
                elif self.split == 'val':
                    label_path = self.data_root / sample['label'].replace('./labelsVal/', 'labelsVal/').replace('./labelsTs/', 'labelsVal/')
                else:  # test
                    label_path = self.data_root / sample['label'].replace('./labelsTs/', 'labelsTs/')

                if label_path.exists():
                    mask = cv2.imread(str(label_path), cv2.IMREAD_GRAYSCALE)
                    if mask is not None:
                        unique_classes = np.unique(mask)
                        for cls in unique_classes:
                            if cls in class_counts:
                                class_counts[cls] += 1

            except Exception as e:
                print(f"   ⚠️ Error analyzing sample {idx}: {e}")
                continue

        print(f"   📈 Class distribution in sample:")
        class_names = EVALUATION_CONFIG["class_names"]
        for cls_id, count in class_counts.items():
            if count > 0:
                print(f"      Class {cls_id} ({class_names[cls_id]}): {count} samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]

        # Get file paths based on split
        if self.split == 'train':
            image_path = self.data_root / sample['image'].replace('./imagesTr/', 'imagesTr/')
            label_path = self.data_root / sample['label'].replace('./labelsTr/', 'labelsTr/')
        elif self.split == 'val':
            # Warwick: use test paths for validation
            if self.dataset_key == 'warwick':
                image_path = self.data_root / sample['image'].replace('./imagesTs/', 'imagesTs/')
                label_path = self.data_root / sample['label'].replace('./labelsTs/', 'labelsTs/')
            else:
                # Handle both validation and test paths for validation split
                image_path = self.data_root / sample['image'].replace('./imagesVal/', 'imagesVal/').replace('./imagesTs/', 'imagesVal/')
                label_path = self.data_root / sample['label'].replace('./labelsVal/', 'labelsVal/').replace('./labelsTs/', 'labelsVal/')
        else:  # test
            image_path = self.data_root / sample['image'].replace('./imagesTs/', 'imagesTs/')
            label_path = self.data_root / sample['label'].replace('./labelsTs/', 'labelsTs/')

        # Extract case ID (remove nnU-Net naming convention)
        case_id = Path(sample['image']).stem.replace('_0000', '')

        # Extract original filename (with _0000 and extension)
        original_filename = Path(sample['image']).name

        # Load image and mask
        try:
            image = cv2.imread(str(image_path))
            if image is None:
                raise ValueError(f"Could not load image: {image_path}")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            mask = cv2.imread(str(label_path), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                raise ValueError(f"Could not load mask: {label_path}")

        except Exception as e:
            print(f"Error loading sample {idx} ({case_id}): {e}")
            # Return a dummy sample to avoid breaking the batch
            image = np.zeros((self.image_size[0], self.image_size[1], 3), dtype=np.uint8)
            mask = np.zeros((self.image_size[0], self.image_size[1]), dtype=np.uint8)

        # Apply augmentations
        augmented = self.transform(image=image, mask=mask)
        image = augmented['image']
        mask = augmented['mask']

        # Ensure mask is long tensor and contains valid class indices (0-3)
        mask = torch.clamp(mask.long(), 0, 3)

        # Get or generate patch label(s)
        if case_id in self.patch_labels:
            patch_label = self.patch_labels[case_id]
            if self.use_multilabel_patch:
                # Convert single label to multi-label format if needed
                if isinstance(patch_label, int):
                    patch_label_tensor = torch.zeros(4, dtype=torch.float32)
                    patch_label_tensor[patch_label] = 1.0
                else:
                    patch_label_tensor = torch.tensor(patch_label, dtype=torch.float32)
            else:
                patch_label_tensor = torch.tensor(patch_label, dtype=torch.long)
        elif self.auto_generate_labels:
            # Generate patch label(s) from segmentation mask
            if self.use_multilabel_patch:
                # Multi-label: can have multiple classes present
                patch_label_tensor = self._generate_multilabel_patch_labels_from_mask(mask)
            else:
                # Single-label: dominant class only
                patch_label = self._generate_patch_label_from_mask(mask.numpy())
                patch_label_tensor = torch.tensor(patch_label, dtype=torch.long)
        else:
            # Default labels
            if self.use_multilabel_patch:
                patch_label_tensor = torch.zeros(4, dtype=torch.float32)
                patch_label_tensor[0] = 1.0  # Default to background
            else:
                patch_label_tensor = torch.tensor(0, dtype=torch.long)

        return {
            'image': image,
            'segmentation_target': mask,
            'patch_label': patch_label_tensor,
            'case_id': case_id,
            'original_filename': original_filename,
            'dataset_key': self.dataset_key
        }

    def _generate_patch_label_from_mask(self, mask: np.ndarray) -> int:
        """Generate patch-level label from segmentation mask"""
        # Count pixels for each class
        unique_classes, counts = np.unique(mask, return_counts=True)
        class_counts = dict(zip(unique_classes, counts))

        # Exclude background when determining patch label
        foreground_classes = {k: v for k, v in class_counts.items() if k > 0}

        if not foreground_classes:
            return 0  # Only background present

        # Return the dominant foreground class
        dominant_class = max(foreground_classes, key=foreground_classes.get)
        return int(dominant_class)

    def _generate_multilabel_patch_labels_from_mask(self, mask: torch.Tensor, min_pixels_threshold: int = 50) -> torch.Tensor:
        """Generate multi-label patch labels from segmentation mask"""
        # Use the function from projection_heads module
        mask_batch = mask.unsqueeze(0)  # Add batch dimension
        multilabel_batch = create_multilabel_patch_labels_from_segmentation(mask_batch, min_pixels_threshold)
        return multilabel_batch.squeeze(0)  # Remove batch dimension

    def get_original_image(self, idx: int) -> Optional[np.ndarray]:
        """
        Get original unnormalized image for visualization purposes

        Args:
            idx: Sample index

        Returns:
            Original image as numpy array [H, W, 3] in uint8 format, or None if not available
        """
        try:
            # Get sample info
            sample = self.samples[idx]

            # Determine paths
            if self.split == 'train':
                image_path = self.data_root / sample['image'].replace('./imagesTr/', 'imagesTr/')
            elif self.split == 'val':
                # Handle both validation and test paths for validation split
                image_path = self.data_root / sample['image'].replace('./imagesVal/', 'imagesVal/').replace('./imagesTs/', 'imagesVal/')
            else:  # test
                image_path = self.data_root / sample['image'].replace('./imagesTs/', 'imagesTs/')

            # Load original image without any transformations
            image = cv2.imread(str(image_path))
            if image is None:
                return None

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Resize to target size but don't apply augmentations
            image = cv2.resize(image, (self.image_size[1], self.image_size[0]))

            return image

        except Exception as e:
            print(f"⚠️ Could not load original image for index {idx}: {e}")
            return None

def create_combined_data_loaders(
    dataset_key: str = 'mag5x',
    batch_size: int = 4,
    num_workers: int = 4,
    image_size: Tuple[int, int] = (512, 512),
    auto_generate_labels: bool = True,
    use_multilabel_patch: bool = True,
    disable_augmentation: bool = False
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create training, validation, and test data loaders for combined datasets

    Args:
        dataset_key: Dataset to load ('mixed', 'mag5x', 'mag10x', 'mag20x', 'mag40x')
        batch_size: Batch size for training
        num_workers: Number of data loading workers
        image_size: Input image size
        auto_generate_labels: Whether to auto-generate classification labels
        use_multilabel_patch: Whether to use multi-label patch classification
        disable_augmentation: Whether to disable augmentation for all splits (useful for evaluation)

    Returns:
        train_loader, val_loader, test_loader
    """

    print(f"🔄 Creating data loaders for dataset: {dataset_key}")

    # Create datasets
    # For training: use augmentation unless explicitly disabled
    # For val/test: always disable augmentation unless specifically enabled during training
    train_augment = not disable_augmentation  # True for training unless disabled
    val_test_augment = False  # Always False for val/test

    train_dataset = CombinedGlandDataset(
        dataset_key=dataset_key,
        split='train',
        image_size=image_size,
        augment=train_augment,
        auto_generate_labels=auto_generate_labels,
        use_multilabel_patch=use_multilabel_patch
    )

    val_dataset = CombinedGlandDataset(
        dataset_key=dataset_key,
        split='val',
        image_size=image_size,
        augment=val_test_augment,
        auto_generate_labels=auto_generate_labels,
        use_multilabel_patch=use_multilabel_patch
    )

    test_dataset = CombinedGlandDataset(
        dataset_key=dataset_key,
        split='test',
        image_size=image_size,
        augment=val_test_augment,
        auto_generate_labels=auto_generate_labels,
        use_multilabel_patch=use_multilabel_patch
    )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_fn
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=collate_fn
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=collate_fn
    )

    print(f"✅ Data loaders created:")
    print(f"   📈 Train: {len(train_loader)} batches ({len(train_dataset)} samples)")
    print(f"   📊 Val: {len(val_loader)} batches ({len(val_dataset)} samples)")
    print(f"   📉 Test: {len(test_loader)} batches ({len(test_dataset)} samples)")

    return train_loader, val_loader, test_loader

def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Custom collate function for 4-class multi-task data
    Handles both single-label and multi-label patch classification
    """
    images = torch.stack([item['image'] for item in batch])
    seg_targets = torch.stack([item['segmentation_target'] for item in batch])
    patch_labels = torch.stack([item['patch_label'] for item in batch])
    case_ids = [item['case_id'] for item in batch]
    original_filenames = [item['original_filename'] for item in batch]
    dataset_keys = [item['dataset_key'] for item in batch]

    # Generate gland labels from patch labels (simplified approach)
    gland_counts = [1] * len(batch)  # Assume 1 gland per patch for now

    # Handle both multi-label and single-label patch labels for gland generation
    if patch_labels.dim() == 2:  # Multi-label: [B, 4]
        # Convert multi-label to single-label for gland generation (use dominant class)
        dominant_classes = torch.argmax(patch_labels, dim=1)
        gland_labels = create_4class_gland_labels_from_patch_labels(dominant_classes, gland_counts)
    else:  # Single-label: [B]
        gland_labels = create_4class_gland_labels_from_patch_labels(patch_labels, gland_counts)

    return {
        'images': images,
        'segmentation_targets': seg_targets,
        'patch_labels': patch_labels,
        'gland_labels': gland_labels,
        'gland_counts': gland_counts,
        'case_ids': case_ids,
        'original_filenames': original_filenames,
        'dataset_keys': dataset_keys
    }

def test_combined_dataset():
    """Test the combined dataset loader"""
    print("🧪 Testing Combined Gland Dataset...")

    # Test with different datasets
    for dataset_key in ['mag5x', 'mag10x']:
        try:
            print(f"\n📊 Testing dataset: {dataset_key}")

            # Create dataset
            dataset = CombinedGlandDataset(
                dataset_key=dataset_key,
                split='train',
                auto_generate_labels=True
            )

            if len(dataset) == 0:
                print(f"   ⚠️ No samples found for {dataset_key}")
                continue

            # Test single sample
            sample = dataset[0]
            print(f"   📋 Sample keys: {list(sample.keys())}")
            print(f"   🖼️ Image shape: {sample['image'].shape}")
            print(f"   🎯 Mask shape: {sample['segmentation_target'].shape}")
            print(f"   🏷️ Patch label: {sample['patch_label'].item()}")
            print(f"   🆔 Case ID: {sample['case_id']}")
            print(f"   📂 Dataset: {sample['dataset_key']}")

            # Check mask classes
            unique_classes = torch.unique(sample['segmentation_target'])
            print(f"   📈 Mask classes: {unique_classes.tolist()}")

            # Test data loader
            print(f"   🔄 Testing data loader...")
            train_loader, val_loader, test_loader = create_combined_data_loaders(
                dataset_key=dataset_key,
                batch_size=2,
                num_workers=0  # Use 0 for testing
            )

            # Test a batch
            batch = next(iter(train_loader))
            print(f"   📦 Batch keys: {list(batch.keys())}")
            print(f"   🖼️ Batch images shape: {batch['images'].shape}")
            print(f"   🎯 Batch segmentation targets shape: {batch['segmentation_targets'].shape}")
            print(f"   🏷️ Batch patch labels shape: {batch['patch_labels'].shape}")
            print(f"   🔍 Batch gland labels shape: {batch['gland_labels'].shape}")
            print(f"   📊 Gland counts: {batch['gland_counts']}")

            # Check class distribution in batch
            batch_classes = torch.unique(batch['segmentation_targets'])
            print(f"   📈 Batch segmentation classes: {batch_classes.tolist()}")
            patch_classes = torch.unique(batch['patch_labels'])
            print(f"   🏷️ Batch patch classes: {patch_classes.tolist()}")

            print(f"   ✅ {dataset_key} dataset test passed!")

        except Exception as e:
            print(f"   ❌ {dataset_key} dataset test failed: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n✅ Combined dataset testing completed!")

if __name__ == "__main__":
    test_combined_dataset()