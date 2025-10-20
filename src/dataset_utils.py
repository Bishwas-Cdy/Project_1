"""
Utility functions for dataset operations, integrity checks, and statistics
"""
import numpy as np
import torch
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import json

from config import DATA_DIR, TRAIN_DIR, VAL_DIR, TEST_DIR, STATS_FILE, NUM_CLASSES


def check_dataset_integrity(verbose: bool = True):
    """
    Check dataset integrity: file counts, corrupted images, class balance
    
    Returns:
        dict: Statistics about the dataset
    """
    if verbose:
        print("=" * 70)
        print("DATASET INTEGRITY CHECK")
        print("=" * 70)
    
    stats = {
        'train': {'total': 0, 'per_class': {}, 'corrupted': []},
        'val': {'total': 0, 'per_class': {}, 'corrupted': []},
        'test': {'total': 0, 'per_class': {}, 'corrupted': []}
    }
    
    splits = {'train': TRAIN_DIR, 'val': VAL_DIR, 'test': TEST_DIR}
    
    for split_name, split_dir in splits.items():
        if verbose:
            print(f"\nChecking {split_name} set: {split_dir}")
        
        class_folders = sorted([d for d in split_dir.iterdir() if d.is_dir()],
                              key=lambda x: int(x.name))
        
        for class_folder in tqdm(class_folders, desc=f"Checking {split_name}", disable=not verbose):
            class_name = class_folder.name
            image_files = list(class_folder.glob("*.png"))
            
            # Count images
            count = len(image_files)
            stats[split_name]['per_class'][class_name] = count
            stats[split_name]['total'] += count
            
            # Check for corrupted images
            for img_path in image_files:
                try:
                    img = Image.open(img_path)
                    img.verify()  # Verify it's a valid image
                except Exception as e:
                    stats[split_name]['corrupted'].append(str(img_path))
                    if verbose:
                        print(f"   Corrupted: {img_path}")
        
        if verbose:
            print(f"  Total images: {stats[split_name]['total']}")
            print(f"  Classes: {len(stats[split_name]['per_class'])}")
            print(f"  Corrupted: {len(stats[split_name]['corrupted'])}")
    
    # Check class balance
    if verbose:
        print("\n" + "=" * 70)
        print("CLASS BALANCE CHECK")
        print("=" * 70)
        
        for split_name in splits.keys():
            counts = list(stats[split_name]['per_class'].values())
            if counts:
                print(f"\n{split_name.upper()}:")
                print(f"  Min images per class: {min(counts)}")
                print(f"  Max images per class: {max(counts)}")
                print(f"  Mean images per class: {np.mean(counts):.1f}")
                print(f"  Std images per class: {np.std(counts):.1f}")
                
                if min(counts) == max(counts):
                    print(f"   Perfectly balanced!")
                else:
                    print(f"   Class imbalance detected")
    
    # Summary
    total_images = sum(s['total'] for s in stats.values())
    total_corrupted = sum(len(s['corrupted']) for s in stats.values())
    
    if verbose:
        print("\n" + "=" * 70)
        print("SUMMARY")
        print("=" * 70)
        print(f"Total images: {total_images}")
        print(f"Total corrupted: {total_corrupted}")
        
        if total_corrupted == 0:
            print(" All images are valid!")
        else:
            print(f" Found {total_corrupted} corrupted images")
        
        print("=" * 70 + "\n")
    
    return stats


def compute_dataset_statistics(save: bool = True):
    """
    Compute mean and std of training set for normalization
    
    Args:
        save: Whether to save statistics to file
    
    Returns:
        dict: {'mean': float, 'std': float}
    """
    print("\n" + "=" * 70)
    print("COMPUTING DATASET STATISTICS (mean, std)")
    print("=" * 70)
    
    # Collect all training images
    image_paths = []
    class_folders = sorted([d for d in TRAIN_DIR.iterdir() if d.is_dir()],
                          key=lambda x: int(x.name))
    
    for class_folder in class_folders:
        image_paths.extend(list(class_folder.glob("*.png")))
    
    print(f"Found {len(image_paths)} training images")
    
    # Compute statistics
    pixel_sum = 0
    pixel_squared_sum = 0
    num_pixels = 0
    
    print("Computing mean and std...")
    for img_path in tqdm(image_paths):
        try:
            img = Image.open(img_path)
            
            # Convert to grayscale
            if img.mode == 'RGBA':
                background = Image.new('RGB', img.size, (255, 255, 255))
                background.paste(img, mask=img.split()[3])
                img = background.convert('L')
            elif img.mode != 'L':
                img = img.convert('L')
            
            # Convert to array and normalize to [0, 1]
            img_array = np.array(img, dtype=np.float32) / 255.0
            
            pixel_sum += img_array.sum()
            pixel_squared_sum += (img_array ** 2).sum()
            num_pixels += img_array.size
            
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            continue
    
    mean = pixel_sum / num_pixels
    std = np.sqrt((pixel_squared_sum / num_pixels) - (mean ** 2))
    
    stats = {
        'mean': float(mean),
        'std': float(std),
        'num_images': len(image_paths),
        'num_pixels': int(num_pixels)
    }
    
    print(f"\nDataset Statistics:")
    print(f"  Mean: {mean:.4f}")
    print(f"  Std:  {std:.4f}")
    
    if save:
        # Save as numpy
        STATS_FILE.parent.mkdir(parents=True, exist_ok=True)
        np.savez(STATS_FILE, mean=mean, std=std)
        
        # Also save as JSON for easy reading
        json_path = STATS_FILE.with_suffix('.json')
        with open(json_path, 'w') as f:
            json.dump(stats, f, indent=4)
        
        print(f"\n Saved statistics to:")
        print(f"  {STATS_FILE}")
        print(f"  {json_path}")
    
    print("=" * 70 + "\n")
    
    return stats


def load_dataset_statistics():
    """
    Load pre-computed dataset statistics
    
    Returns:
        dict: {'mean': float, 'std': float}
    """
    if not STATS_FILE.exists():
        print(f" Statistics file not found: {STATS_FILE}")
        print("Computing statistics now...")
        return compute_dataset_statistics(save=True)
    
    data = np.load(STATS_FILE)
    return {
        'mean': float(data['mean']),
        'std': float(data['std'])
    }


def get_normalization_transform():
    """
    Get normalization transform using dataset statistics
    
    Returns:
        tuple: (mean, std) for Normalize transform
    """
    stats = load_dataset_statistics()
    return stats['mean'], stats['std']


if __name__ == "__main__":
    # Run integrity check
    check_dataset_integrity(verbose=True)
    
    # Compute and save statistics
    compute_dataset_statistics(save=True)
