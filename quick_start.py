#!/usr/bin/env python3
"""
Quick Start Guide for Ranjana Script Recognition
Run this to verify everything works!
"""
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

try:
    print("="*70)
    print("RANJANA SCRIPT RECOGNITION - QUICK START")
    print("="*70)
    
    print("\n1. Checking dependencies...")
    import torch
    import torchvision
    import numpy as np
    from PIL import Image
    print("    All dependencies installed")
    
    print("\n2. Loading models...")
    from inference import RanjanaInference
    
    # Initialize inference
    model = RanjanaInference(model_name='efficientnet_b0', device='cpu')
    print("    Classification model loaded")
    
    print("\n3. Creating test image...")
    test_img = np.random.randint(0, 255, (64, 64), dtype=np.uint8)
    test_path = 'test_image.png'
    Image.fromarray(test_img).save(test_path)
    print(f"    Test image created: {test_path}")
    
    print("\n4. Running inference...")
    result = model.predict(test_path, top_k=3)
    print(f"    Prediction successful!")
    print(f"      - Class: {result['class']}")
    print(f"      - Confidence: {result['confidence']:.2f}%")
    
    print("\n5. Testing similarity...")
    similarity, distance = model.compute_similarity(test_path, test_path)
    print(f"    Similarity computed: {similarity:.2f}%")
    
    print("\n6. Generating Grad-CAM...")
    gradcam_result = model.generate_gradcam(test_path, save_path='test_gradcam.png')
    print(f"    Grad-CAM saved: test_gradcam.png")
    
    print("\n7. Extracting embeddings...")
    embedding = model.get_embedding(test_path)
    print(f"    Embedding shape: {embedding.shape}")
    
    print("\n" + "="*70)
    print(" SUCCESS! Everything is working perfectly!")
    print("="*70)
    print("\nNext steps:")
    print("  1. Check examples/ folder for usage examples")
    print("  2. Read documentation/API.md for detailed API reference")
    print("  3. Start using the model with your own images!")
    print("\nQuick API example:")
    print("  from inference import RanjanaInference")
    print("  model = RanjanaInference('efficientnet_b0')")
    print("  result = model.predict('your_image.png')")
    print("="*70)
    
    # Cleanup
    os.remove(test_path)
    if os.path.exists('test_gradcam.png'):
        os.remove('test_gradcam.png')

except Exception as e:
    print(f"\n Error: {e}")
    print("\nTroubleshooting:")
    print("  1. Install dependencies: pip install -r requirements.txt")
    print("  2. Ensure you're in the DEPLOYMENT_PACKAGE directory")
    print("  3. Check that models/ directory contains checkpoint files")
    sys.exit(1)
