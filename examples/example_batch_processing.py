"""
Batch Processing Example for Ranjana Script Models

This example demonstrates how to efficiently process multiple images
using batch processing for better performance.

Features:
- Process entire directories
- Progress tracking
- Results export to CSV
- Error handling
- Performance metrics
"""

import sys
import os
from pathlib import Path
import time
from glob import glob
import csv

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from inference import RanjanaInference
from PIL import Image


def process_directory(model, directory_path, output_csv='results.csv'):
    """
    Process all images in a directory and save results to CSV.
    
    Args:
        model: RanjanaInference instance
        directory_path: Path to directory containing images
        output_csv: Output CSV file path
    """
    print(f"Processing directory: {directory_path}")
    
    # Find all images
    image_extensions = ['*.png', '*.jpg', '*.jpeg', '*.PNG', '*.JPG', '*.JPEG']
    image_paths = []
    
    for ext in image_extensions:
        image_paths.extend(glob(os.path.join(directory_path, '**', ext), recursive=True))
    
    if len(image_paths) == 0:
        print(f"No images found in {directory_path}")
        return
    
    print(f"Found {len(image_paths)} images")
    
    # Process in batches
    batch_size = 32
    results = []
    
    start_time = time.time()
    
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i+batch_size]
        
        print(f"Processing batch {i//batch_size + 1}/{(len(image_paths)-1)//batch_size + 1}...")
        
        # Classify batch
        batch_results = model.classify_batch(batch_paths, top_k=5, batch_size=batch_size)
        
        # Store results
        for path, (classes, probs) in zip(batch_paths, batch_results):
            results.append({
                'image_path': path,
                'predicted_class': int(classes[0]),
                'folder_name': int(classes[0]) + 1,
                'confidence': float(probs[0]),
                'top_5_classes': [int(c) for c in classes],
                'top_5_confidences': [float(p) for p in probs]
            })
        
        # Progress
        processed = min(i + batch_size, len(image_paths))
        progress = (processed / len(image_paths)) * 100
        print(f"Progress: {processed}/{len(image_paths)} ({progress:.1f}%)")
    
    end_time = time.time()
    total_time = end_time - start_time
    
    # Save to CSV
    print(f"\nSaving results to {output_csv}...")
    
    with open(output_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'image_path', 'predicted_class', 'folder_name', 'confidence',
            'top_2_class', 'top_2_conf', 'top_3_class', 'top_3_conf'
        ])
        writer.writeheader()
        
        for result in results:
            row = {
                'image_path': result['image_path'],
                'predicted_class': result['predicted_class'],
                'folder_name': result['folder_name'],
                'confidence': f"{result['confidence']:.4f}",
            }
            
            # Add top 2-3 predictions
            if len(result['top_5_classes']) > 1:
                row['top_2_class'] = result['top_5_classes'][1]
                row['top_2_conf'] = f"{result['top_5_confidences'][1]:.4f}"
            
            if len(result['top_5_classes']) > 2:
                row['top_3_class'] = result['top_5_classes'][2]
                row['top_3_conf'] = f"{result['top_5_confidences'][2]:.4f}"
            
            writer.writerow(row)
    
    # Print statistics
    print(f"\n{'='*60}")
    print("PROCESSING COMPLETE")
    print(f"{'='*60}")
    print(f"Total images: {len(results)}")
    print(f"Total time: {total_time:.2f} seconds")
    print(f"Average time per image: {total_time/len(results)*1000:.1f} ms")
    print(f"Results saved to: {output_csv}")
    
    # Confidence statistics
    confidences = [r['confidence'] for r in results]
    avg_conf = sum(confidences) / len(confidences)
    min_conf = min(confidences)
    max_conf = max(confidences)
    
    print(f"\nConfidence Statistics:")
    print(f"  Average: {avg_conf:.2%}")
    print(f"  Minimum: {min_conf:.2%}")
    print(f"  Maximum: {max_conf:.2%}")
    
    # Low confidence warnings
    low_conf_threshold = 0.5
    low_conf_images = [r for r in results if r['confidence'] < low_conf_threshold]
    
    if low_conf_images:
        print(f"\nWarning: {len(low_conf_images)} images with confidence < {low_conf_threshold:.0%}:")
        for r in low_conf_images[:5]:  # Show first 5
            print(f"  {os.path.basename(r['image_path'])}: {r['confidence']:.2%}")
        if len(low_conf_images) > 5:
            print(f"  ... and {len(low_conf_images) - 5} more")


def compare_directories(model, dir1, dir2, output_csv='comparison.csv'):
    """
    Compare all images in two directories (e.g., student vs reference).
    
    Args:
        model: RanjanaInference instance
        dir1: First directory path
        dir2: Second directory path
        output_csv: Output CSV file path
    """
    print(f"Comparing directories:")
    print(f"  Directory 1: {dir1}")
    print(f"  Directory 2: {dir2}")
    
    # Get all images from both directories
    images1 = sorted(glob(os.path.join(dir1, '*.png')))
    images2 = sorted(glob(os.path.join(dir2, '*.png')))
    
    print(f"Found {len(images1)} images in directory 1")
    print(f"Found {len(images2)} images in directory 2")
    
    # Create pairs (match by filename)
    pairs = []
    for img1_path in images1:
        img1_name = os.path.basename(img1_path)
        
        # Find matching image in dir2
        img2_path = os.path.join(dir2, img1_name)
        if os.path.exists(img2_path):
            pairs.append((img1_path, img2_path))
    
    print(f"Found {len(pairs)} matching pairs")
    
    if len(pairs) == 0:
        print("No matching pairs found!")
        return
    
    # Compute similarities
    print("\nComputing similarities...")
    start_time = time.time()
    
    similarity_results = model.compare_batch(pairs, batch_size=32)
    
    end_time = time.time()
    total_time = end_time - start_time
    
    # Save results
    print(f"Saving results to {output_csv}...")
    
    with open(output_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'image_name', 'similarity_score', 'distance', 'is_similar'
        ])
        writer.writeheader()
        
        for (img1, img2), (similarity, distance) in zip(pairs, similarity_results):
            writer.writerow({
                'image_name': os.path.basename(img1),
                'similarity_score': f"{similarity:.2f}",
                'distance': f"{distance:.4f}",
                'is_similar': 'Yes' if distance < 0.45 else 'No'
            })
    
    # Statistics
    similarities = [s for s, d in similarity_results]
    distances = [d for s, d in similarity_results]
    
    avg_similarity = sum(similarities) / len(similarities)
    avg_distance = sum(distances) / len(distances)
    
    similar_count = sum(1 for d in distances if d < 0.45)
    
    print(f"\n{'='*60}")
    print("COMPARISON COMPLETE")
    print(f"{'='*60}")
    print(f"Total pairs: {len(pairs)}")
    print(f"Total time: {total_time:.2f} seconds")
    print(f"Average time per pair: {total_time/len(pairs)*1000:.1f} ms")
    print(f"\nSimilarity Statistics:")
    print(f"  Average similarity: {avg_similarity:.1f}%")
    print(f"  Average distance: {avg_distance:.4f}")
    print(f"  Similar pairs (distance < 0.45): {similar_count} ({similar_count/len(pairs)*100:.1f}%)")
    print(f"  Different pairs: {len(pairs) - similar_count} ({(len(pairs)-similar_count)/len(pairs)*100:.1f}%)")
    print(f"\nResults saved to: {output_csv}")


def generate_gradcam_for_directory(model, directory_path, output_dir='gradcam_outputs'):
    """
    Generate Grad-CAM visualizations for all images in a directory.
    
    Args:
        model: RanjanaInference instance
        directory_path: Input directory path
        output_dir: Output directory for Grad-CAM images
    """
    print(f"Generating Grad-CAM visualizations for: {directory_path}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Find images
    image_paths = glob(os.path.join(directory_path, '*.png'))
    print(f"Found {len(image_paths)} images")
    
    if len(image_paths) == 0:
        print("No images found!")
        return
    
    # Process each image
    start_time = time.time()
    
    for i, img_path in enumerate(image_paths):
        print(f"Processing {i+1}/{len(image_paths)}: {os.path.basename(img_path)}")
        
        # Generate Grad-CAM
        output_path = os.path.join(output_dir, f'gradcam_{os.path.basename(img_path)}')
        
        result = model.generate_gradcam(img_path, save_path=output_path)
        
        print(f"  Predicted: Class {result['predicted_class']} ({result['confidence']:.2%})")
        print(f"  Saved to: {output_path}")
    
    end_time = time.time()
    total_time = end_time - start_time
    
    print(f"\n{'='*60}")
    print("GRAD-CAM GENERATION COMPLETE")
    print(f"{'='*60}")
    print(f"Total images: {len(image_paths)}")
    print(f"Total time: {total_time:.2f} seconds")
    print(f"Average time per image: {total_time/len(image_paths)*1000:.1f} ms")
    print(f"Output directory: {output_dir}")


def main():
    """Main batch processing examples."""
    
    print("Ranjana Script - Batch Processing Examples")
    print("=" * 60)
    
    # Initialize model
    print("\n1. Initializing model...")
    model = RanjanaInference(
        'efficientnet_b0',
        checkpoint_path='models/efficientnet_b0_best.pth',
        device='auto'
    )
    print(" Model loaded successfully")
    
    # Example 1: Process a directory
    print("\n" + "="*60)
    print("EXAMPLE 1: Classify All Images in Directory")
    print("="*60)
    
    test_dir = 'test_images'  # Replace with your directory
    
    if os.path.exists(test_dir):
        process_directory(model, test_dir, output_csv='classification_results.csv')
    else:
        print(f"Directory not found: {test_dir}")
        print("Skipping Example 1")
    
    # Example 2: Compare two directories
    print("\n" + "="*60)
    print("EXAMPLE 2: Compare Student vs Reference Images")
    print("="*60)
    
    student_dir = 'student_images'  # Replace with your directory
    reference_dir = 'reference_images'  # Replace with your directory
    
    if os.path.exists(student_dir) and os.path.exists(reference_dir):
        compare_directories(model, student_dir, reference_dir, output_csv='similarity_results.csv')
    else:
        print(f"Directories not found")
        print("Skipping Example 2")
    
    # Example 3: Generate Grad-CAM for directory
    print("\n" + "="*60)
    print("EXAMPLE 3: Generate Grad-CAM for All Images")
    print("="*60)
    
    if os.path.exists(test_dir):
        generate_gradcam_for_directory(model, test_dir, output_dir='gradcam_visualizations')
    else:
        print(f"Directory not found: {test_dir}")
        print("Skipping Example 3")
    
    print("\n" + "="*60)
    print("ALL EXAMPLES COMPLETE!")
    print("="*60)


if __name__ == '__main__':
    main()
