"""
Data loading and preprocessing utilities for Ranjana Script dataset
"""
import os
import numpy as np
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from typing import Tuple, Optional

from config import (
    TRAIN_DIR, VAL_DIR, TEST_DIR,
    IMAGE_SIZE, BATCH_SIZE, NUM_WORKERS, PIN_MEMORY,
    ROTATION_RANGE, WIDTH_SHIFT_RANGE, HEIGHT_SHIFT_RANGE, ZOOM_RANGE
)
from dataset_utils import get_normalization_transform


class RanjanaDataset(Dataset):
    """
    PyTorch Dataset for Ranjana Script images
    """
    
    def __init__(self, root_dir: Path, transform: Optional[transforms.Compose] = None):
        """
        Args:
            root_dir: Path to training/validation/testing directory
            transform: Optional transform to be applied on images
        """
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.samples = []
        self.class_to_idx = {}
        
        # Collect all image paths and labels
        self._load_samples()
    
    def _load_samples(self):
        """Load all image paths and create class mapping"""
        class_folders = sorted([d for d in self.root_dir.iterdir() if d.is_dir()],
                              key=lambda x: int(x.name))
        
        for idx, class_folder in enumerate(class_folders):
            class_name = class_folder.name
            self.class_to_idx[class_name] = idx
            
            # Get all PNG images in this class folder
            for img_path in class_folder.glob("*.png"):
                self.samples.append((str(img_path), idx))
        
        print(f"Loaded {len(self.samples)} images from {len(self.class_to_idx)} classes")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Returns:
            image: Tensor of shape (1, 64, 64)
            label: Integer class label (0-61)
        """
        img_path, label = self.samples[idx]
        
        # Load image
        image = Image.open(img_path)
        
        # Convert RGBA to grayscale
        if image.mode == 'RGBA':
            # Create white background
            background = Image.new('RGB', image.size, (255, 255, 255))
            background.paste(image, mask=image.split()[3])  # Use alpha channel as mask
            image = background.convert('L')
        elif image.mode != 'L':
            image = image.convert('L')
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        return image, label


def get_transforms(augment: bool = False, use_dataset_stats: bool = True) -> transforms.Compose:
    """
    Get image transforms
    
    Args:
        augment: Whether to apply data augmentation
        use_dataset_stats: Whether to use pre-computed dataset mean/std
    
    Returns:
        Composed transforms
    """
    # Get normalization parameters
    if use_dataset_stats:
        try:
            mean, std = get_normalization_transform()
        except:
            # Fallback to default
            mean, std = 0.5, 0.5
            print(" Using default normalization (0.5, 0.5)")
    else:
        mean, std = 0.5, 0.5
    
    if augment:
        # Training transforms with augmentation
        transform = transforms.Compose([
            transforms.Resize(IMAGE_SIZE),
            transforms.RandomRotation(ROTATION_RANGE),
            transforms.RandomAffine(
                degrees=0,
                translate=(WIDTH_SHIFT_RANGE, HEIGHT_SHIFT_RANGE),
                scale=(1.0 - ZOOM_RANGE, 1.0 + ZOOM_RANGE)
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=[mean], std=[std])
        ])
    else:
        # Validation/test transforms (no augmentation)
        transform = transforms.Compose([
            transforms.Resize(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=[mean], std=[std])
        ])
    
    return transform


def get_data_loaders(batch_size: int = BATCH_SIZE, num_workers: int = NUM_WORKERS):
    """
    Create data loaders for training, validation, and testing
    
    Args:
        batch_size: Batch size for data loaders
        num_workers: Number of worker processes for data loading
    
    Returns:
        train_loader, val_loader, test_loader
    """
    # Create datasets
    train_dataset = RanjanaDataset(TRAIN_DIR, transform=get_transforms(augment=True))
    val_dataset = RanjanaDataset(VAL_DIR, transform=get_transforms(augment=False))
    test_dataset = RanjanaDataset(TEST_DIR, transform=get_transforms(augment=False))
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=PIN_MEMORY
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=PIN_MEMORY
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=PIN_MEMORY
    )
    
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # Test data loading
    print("Testing data loaders...")
    train_loader, val_loader, test_loader = get_data_loaders()
    
    # Get a batch
    images, labels = next(iter(train_loader))
    print(f"\nBatch shape: {images.shape}")
    print(f"Labels shape: {labels.shape}")
    print(f"Image range: [{images.min():.2f}, {images.max():.2f}]")
    print(f"Sample labels: {labels[:10]}")
