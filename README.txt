================================================================================
                 RANJANA SCRIPT INTELLIGENT HANDWRITING TUTOR
                            Deployment Package v1.0
                         Released: October 19, 2025
================================================================================

OVERVIEW
--------

This package provides production-ready deep learning models for Ranjana Script
character recognition and analysis. It includes trained neural networks,
inference APIs, and integration examples for mobile and web applications.

The system achieves 99.5% classification accuracy and 92.7% similarity
detection accuracy on the Ranjana-64 benchmark dataset.

================================================================================
                              PACKAGE CONTENTS
================================================================================

DEPLOYMENT_PACKAGE/
 models/                              # Trained neural network weights (72 MB)
    efficientnet_b0_best.pth        # Classification model (47 MB, 99.5% acc)
    siamese_efficientnet_b0_best.pth # Similarity model (25 MB, 92.7% acc)

 src/                                 # Python source code
    inference.py                    # Main inference API
    gradcam.py                      # Gradient-weighted Class Activation Mapping
    siamese_network.py              # Siamese network architecture
    models.py                       # Model definitions
    config.py                       # Configuration utilities
    data_loader.py                  # Data loading utilities

 examples/                            # Integration examples
    example_basic_usage.py          # Basic API usage
    example_flask_api.py            # REST API server
    example_batch_processing.py     # Batch inference
    README_EXAMPLES.txt             # Examples documentation

 documentation/                       # Comprehensive documentation
    API_REFERENCE.txt               # API documentation
    INTEGRATION_GUIDE.txt           # Integration guides
    TROUBLESHOOTING.txt             # Troubleshooting guide
    CLASS_MAPPING.txt               # Class ID mappings

 config.yaml                          # Model configuration
 requirements.txt                     # Python dependencies
 README.txt                           # This file
 DEPLOYMENT_CHECKLIST.txt             # Deployment instructions

================================================================================
                              QUICK START
================================================================================

Prerequisites
-------------
- Python 3.8 or higher
- pip package manager
- 2 GB RAM minimum
- 100 MB disk space (plus 72 MB for model weights)

Installation
------------
1. Install dependencies:
   
   $ cd DEPLOYMENT_PACKAGE
   $ pip install -r requirements.txt

2. Verify installation:
   
   $ python examples/example_basic_usage.py

   Expected output: Model loads successfully and runs inference on test images.

Basic Usage
-----------
```python
from src.inference import RanjanaInference

# Initialize model
model = RanjanaInference('efficientnet_b0', device='cpu')

# Classify character
classes, probabilities = model.classify('image.png')
print(f"Predicted class: {classes[0]}")
print(f"Confidence: {probabilities[0]:.2%}")

# Compare similarity
similarity, distance = model.compute_similarity('img1.png', 'img2.png')
print(f"Similarity: {similarity:.1f}%")

# Generate visualization
result = model.generate_gradcam('image.png', save_path='heatmap.png')
```

Integration Options
-------------------
A. Backend API (Recommended for web/mobile apps)
   - Use src/inference.py on server
   - See examples/example_flask_api.py
   - Deploy to any Python-compatible hosting

B. On-Device Inference (Advanced)
   - Convert models to ONNX/TFLite
   - See documentation/INTEGRATION_GUIDE.txt
   - Suitable for offline mobile apps

C. Hybrid Approach
   - Backend API for complex operations
   - On-device for latency-sensitive tasks

================================================================================
                              CORE FEATURES
================================================================================

1. Character Classification
   
   Identifies Ranjana script characters from input images.
   
   - Architecture: EfficientNet-B0
   - Test Accuracy: 99.50%
   - Inference Time: ~50ms (CPU), ~5ms (GPU)
   - Input: 64x64 grayscale image
   - Output: Class ID (0-61) and confidence scores
   
   Usage:
   ```python
   classes, probabilities = model.classify('image.png')
   predicted_class = classes[0]  # Top prediction
   confidence = probabilities[0]  # Confidence score
   ```

2. Similarity Scoring
   
   Computes visual similarity between two character images.
   
   - Architecture: Siamese EfficientNet-B0
   - Test Accuracy: 92.71%
   - Inference Time: ~100ms (CPU), ~10ms (GPU)
   - Input: Two 64x64 grayscale images
   - Output: Similarity percentage (0-100) and Euclidean distance
   
   Usage:
   ```python
   similarity, distance = model.compute_similarity('img1.png', 'img2.png')
   is_similar = distance < 0.45  # Default threshold
   ```

3. Gradient-weighted Class Activation Mapping (Grad-CAM)
   
   Visualizes which regions of the image the model focuses on during prediction.
   
   - Method: Gradient-weighted CAM on final convolutional layer
   - Use Case: Model interpretability and debugging
   - Inference Time: ~60ms (CPU), ~8ms (GPU)
   - Input: 64x64 grayscale image
   - Output: Heatmap overlay image
   
   Usage:
   ```python
   result = model.generate_gradcam('image.png', save_path='heatmap.png')
   predicted_class = result['predicted_class']
   overlay_image = result['overlay']
   ```

================================================================================
                          SYSTEM REQUIREMENTS
================================================================================

Minimum Requirements
--------------------
- Python 3.8 or higher
- 2 GB RAM
- 100 MB disk space (plus 72 MB for model weights)
- CPU-only systems supported

Recommended Configuration
-------------------------
- Python 3.10 or higher
- 4 GB RAM
- NVIDIA GPU with CUDA support (optional, 10x speedup)
- SSD storage for faster model loading

Tested Platforms
----------------
- Ubuntu 22.04 LTS (Python 3.13)
- Windows 10/11
- macOS 12+ (Intel and Apple Silicon)
- Docker containers (CPU and GPU)

================================================================================
                         FLUTTER INTEGRATION
================================================================================

Backend API Approach (Recommended)
-----------------------------------
Advantages:
- Simple implementation
- Models stay on server, easy to update
- No model conversion required
- Centralized processing

Implementation:
1. Deploy REST API using examples/example_flask_api.py
2. Make HTTP requests from Flutter application
3. Handle JSON responses

Example (Dart):
```dart
import 'package:http/http.dart' as http;
import 'dart:convert';

Future<Map<String, dynamic>> classifyCharacter(String base64Image) async {
  final response = await http.post(
    Uri.parse('https://api.yourserver.com/classify'),
    headers: {'Content-Type': 'application/json'},
    body: jsonEncode({'image': base64Image}),
  );
  return jsonDecode(response.body);
}
```

On-Device Inference (Advanced)
-------------------------------
Advantages:
- Offline capability
- Lower latency
- Enhanced privacy

Requirements:
- Model conversion to TFLite format
- tflite_flutter package integration
- Larger app size (~75 MB)

See documentation/INTEGRATION_GUIDE.txt for detailed conversion steps.

================================================================================
                          DJANGO INTEGRATION
================================================================================

Setup Instructions
------------------
1. Add package to Django project:
   
   your_project/
    ai_models/
       models/           # Copy from this package
       src/              # Copy from this package
    your_app/
        views.py

2. Create inference view:

```python
import sys
import os
from django.conf import settings
from django.http import JsonResponse

# Add AI package to path
sys.path.insert(0, os.path.join(settings.BASE_DIR, 'ai_models/src'))
from inference import RanjanaInference

# Initialize model at startup (singleton pattern)
_classifier = None

def get_classifier():
    global _classifier
    if _classifier is None:
        model_path = os.path.join(settings.BASE_DIR, 'ai_models/models')
        _classifier = RanjanaInference(
            'efficientnet_b0',
            checkpoint_path=os.path.join(model_path, 'efficientnet_b0_best.pth'),
            device='cpu'  # Use 'cuda' if GPU available
        )
    return _classifier

def classify_view(request):
    if request.method == 'POST':
        image_file = request.FILES['image']
        
        # Process image
        temp_path = '/tmp/upload.png'
        with open(temp_path, 'wb') as f:
            for chunk in image_file.chunks():
                f.write(chunk)
        
        # Inference
        model = get_classifier()
        classes, probabilities = model.classify(temp_path)
        
        return JsonResponse({
            'predicted_class': int(classes[0]),
            'confidence': float(probabilities[0]),
            'top_5': list(zip(classes.tolist(), probabilities.tolist()))
        })
    return JsonResponse({'error': 'POST required'}, status=405)
```

3. Add URL routing and deploy to production server.

See examples/example_flask_api.py for complete REST API implementation.

Deployment Platforms
--------------------
- AWS EC2 / Lambda
- Google Cloud Platform
- Heroku
- DigitalOcean
- Self-hosted servers

================================================================================
                            CONFIGURATION
================================================================================

The config.yaml file contains model and inference parameters:

- image_size: Input image dimensions (default: 64x64)
- num_classes: Number of character classes (default: 62)
- similarity_threshold: Distance threshold for similarity matching (default: 0.45)
- device: Computation device ('cpu', 'cuda', or 'auto')
- normalization: Mean and standard deviation for preprocessing

Modifications to config.yaml require model reinitialization.

================================================================================
                          CLASS MAPPING
================================================================================

The model predicts 62 distinct Ranjana script characters:

- Classes 0-35:  Consonants (36 characters)
- Classes 36-51: Vowels (16 characters)
- Classes 52-61: Numerals (10 characters)

Important Note on Indexing
---------------------------
Dataset folders are numbered 1-62, but model outputs use 0-based indexing (0-61).

Conversion formula:
  folder_name = predicted_class + 1
  predicted_class = folder_name - 1

Example:
  Model predicts class 0  -> Corresponds to folder "1"
  Model predicts class 61 -> Corresponds to folder "62"

Complete character mappings are available in documentation/CLASS_MAPPING.txt

================================================================================
                          TROUBLESHOOTING
================================================================================

Common Issues
-------------

Issue: ModuleNotFoundError for 'torch' or other dependencies
Solution: Install all requirements
  $ pip install -r requirements.txt

Issue: Model produces incorrect predictions
Solution: Verify image preprocessing
  - Images must be 64x64 pixels (automatically resized)
  - Images must be grayscale (automatically converted)
  - Normalization applied automatically (mean=0.2611, std=0.4186)

Issue: CUDA out of memory error
Solution: Use CPU inference or reduce batch size
  model = RanjanaInference('efficientnet_b0', device='cpu')

Issue: Inconsistent similarity scores
Solution: Ensure both images use identical preprocessing pipeline

For comprehensive troubleshooting, see documentation/TROUBLESHOOTING.txt

================================================================================
                          MODEL PERFORMANCE
================================================================================

Classification Model (EfficientNet-B0)
---------------------------------------
- Test Set Accuracy: 99.50%
- Validation Accuracy: 99.48%
- Stress Test Performance: 100% (100/100 random samples)
- Strong Classes: 60/62 (96.8% achieve >90% accuracy)
- Average Confidence: 95.3%
- Inference Latency:
  * CPU: ~50ms per image
  * GPU (CUDA): ~5ms per image
  * Batch (32 images): ~20ms per image

Similarity Model (Siamese EfficientNet-B0)
-------------------------------------------
- Test Set Accuracy: 92.71%
- ROC-AUC Score: 0.9726
- Optimal Distance Threshold: 0.45
- False Positive Rate: 4%
- False Negative Rate: 2%
- Inference Latency:
  * CPU: ~100ms per pair
  * GPU (CUDA): ~10ms per pair

Grad-CAM Visualization
----------------------
- Method: Gradient-weighted Class Activation Mapping
- Target Layer: Final convolutional layer
- No separate model required (uses classification model)
- Inference Latency: ~60ms (CPU), ~8ms (GPU)

================================================================================
                         SUPPORT & RESOURCES
================================================================================

Documentation
-------------
- API Reference: documentation/API_REFERENCE.txt
- Integration Guide: documentation/INTEGRATION_GUIDE.txt
- Troubleshooting: documentation/TROUBLESHOOTING.txt
- Class Mappings: documentation/CLASS_MAPPING.txt

Examples
--------
- Basic Usage: examples/example_basic_usage.py
- REST API: examples/example_flask_api.py
- Batch Processing: examples/example_batch_processing.py

Contact
-------
Project: Ranjana Script Intelligent Handwriting Tutor
Developer: Bishwas
Release Date: October 19, 2025

For issues or questions, please contact the development team.

================================================================================
                          LICENSE & USAGE
================================================================================

This package is part of an educational group project for the Ranjana Script
Intelligent Handwriting Tutor application.

Permitted Use
-------------
- Integration into Flutter mobile application
- Integration into Django web application
- Modification for project requirements
- Deployment to production environments
- Sharing with authorized team members

Restrictions
------------
- Public redistribution of trained models is not permitted
- Commercial use requires explicit authorization
- Attribution to original authors must be maintained

The software is provided "as-is" without warranty of any kind, express or
implied. The authors are not liable for any damages arising from its use.

================================================================================
                           NEXT STEPS
================================================================================

1. Verify Installation
   $ python examples/example_basic_usage.py

2. Review Documentation
   Read documentation/INTEGRATION_GUIDE.txt for platform-specific instructions

3. Integrate into Application
   Use provided examples as templates for your implementation

4. Deploy to Production
   Follow deployment guidelines in DEPLOYMENT_CHECKLIST.txt

For detailed API documentation, refer to documentation/API_REFERENCE.txt

================================================================================

End of README

================================================================================
