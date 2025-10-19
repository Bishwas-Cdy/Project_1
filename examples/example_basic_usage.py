"""
BASIC USAGE EXAMPLES
Demonstrates all 3 core features with simple examples

Run this to verify everything is working!
"""

import sys
sys.path.insert(0, 'src')

from inference import RanjanaInference
from PIL import Image
import numpy as np


def create_sample_image():
    """Create a dummy 64x64 grayscale image for testing"""
    # Create a simple image (you should use real Ranjana character images)
    img = np.random.randint(0, 255, (64, 64), dtype=np.uint8)
    Image.fromarray(img).save('sample_image.png')
    print(" Created sample_image.png")
    return 'sample_image.png'


def example_1_classification(model, image_path):
    """Example 1: Classify a single character"""
    print("\n" + "="*70)
    print("EXAMPLE 1: CLASSIFICATION")
    print("="*70)
    
    classes, probs = model.classify(image_path)
    
    print(f"\n Image: {image_path}")
    print(f" Predicted Class: {classes[0]}")
    print(f" Confidence: {probs[0]:.2%}")
    print(f"\n Top 5 Predictions:")
    for i, (cls, prob) in enumerate(zip(classes, probs), 1):
        print(f"   {i}. Class {cls}: {prob:.2%}")
    
    return classes[0], probs[0]


def example_2_similarity(model, image1_path, image2_path):
    """Example 2: Compare two characters"""
    print("\n" + "="*70)
    print("EXAMPLE 2: SIMILARITY COMPARISON")
    print("="*70)
    
    similarity, distance = model.compute_similarity(image1_path, image2_path)
    
    print(f"\n Image 1: {image1_path}")
    print(f" Image 2: {image2_path}")
    print(f" Similarity Score: {similarity:.1f}%")
    print(f" Distance: {distance:.4f}")
    print(f" Match: {'Yes! ' if distance < 0.45 else 'No '} (threshold: 0.45)")
    
    return similarity, distance


def example_3_gradcam(model, image_path):
    """Example 3: Generate attention heatmap"""
    print("\n" + "="*70)
    print("EXAMPLE 3: GRAD-CAM VISUALIZATION")
    print("="*70)
    
    result = model.generate_gradcam(
        image_path,
        save_path='gradcam_output.png'
    )
    
    print(f"\n Image: {image_path}")
    print(f" Predicted Class: {result['predicted_class']}")
    print(f" Confidence: {result['confidence']:.2%}")
    print(f" Heatmap saved to: gradcam_output.png")
    print(f" CAM shape: {result['cam'].shape}")
    print(f" Overlay shape: {result['overlay'].shape}")
    
    return result


def main():
    print("="*70)
    print("RANJANA SCRIPT MODEL - BASIC USAGE EXAMPLES")
    print("="*70)
    
    # Initialize model
    print("\n Loading models...")
    print("(This may take a few seconds on first run)")
    
    model = RanjanaInference(
        'efficientnet_b0',
        checkpoint_path='models/efficientnet_b0_best.pth',
        device='cpu'  # Change to 'cuda' if you have GPU
    )
    
    print(" Models loaded successfully!")
    
    # Create sample images for testing
    # NOTE: Replace these with real Ranjana character images!
    image1 = create_sample_image()
    
    # Example 1: Classification
    pred_class, confidence = example_1_classification(model, image1)
    
    # Example 2: Similarity
    # For this example, we compare the image with itself (similarity ~100%)
    # In real use, compare with a different image!
    similarity, distance = example_2_similarity(model, image1, image1)
    
    # Example 3: Grad-CAM
    gradcam_result = example_3_gradcam(model, image1)
    
    # Summary
    print("\n" + "="*70)
    print(" ALL EXAMPLES COMPLETED!")
    print("="*70)
    print(f"\n Classification: Class {pred_class} ({confidence:.2%} confidence)")
    print(f" Similarity: {similarity:.1f}%")
    print(f" Grad-CAM: Saved to gradcam_output.png")
    print("\n Everything is working! You're ready to integrate! ")
    print("\nNEXT STEPS:")
    print("1. Replace sample_image.png with real Ranjana character images")
    print("2. See example_flask_api.py for REST API example")
    print("3. Read documentation/INTEGRATION_GUIDE.txt for your platform")
    print("="*70)


if __name__ == "__main__":
    main()
