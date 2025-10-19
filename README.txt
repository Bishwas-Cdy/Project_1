================================================================================
                 RANJANA SCRIPT INTELLIGENT HANDWRITING TUTOR
                        DEPLOYMENT PACKAGE v1.0
                         Created: October 19, 2025
================================================================================

👋 HEY TEAMMATES! This package contains EVERYTHING you need to integrate
   the Ranjana Script AI models into your Flutter app and Django backend!

================================================================================
                           📦 WHAT'S INSIDE
================================================================================

DEPLOYMENT_PACKAGE/
├── models/                              ← TRAINED MODELS (27 MB total)
│   ├── efficientnet_b0_best.pth        ← Classification model (99.5% accuracy)
│   └── siamese_efficientnet_b0_best.pth ← Similarity model (92.7% accuracy)
│
├── src/                                 ← SOURCE CODE (Python)
│   ├── inference.py                    ← MAIN API (USE THIS!)
│   ├── gradcam.py                      ← Visualization code
│   ├── siamese_network.py              ← Similarity network
│   ├── models.py                       ← Classification architecture
│   ├── config.py                       ← Constants
│   └── data_loader.py                  ← Data utilities
│
├── examples/                            ← EXAMPLE CODE (start here!)
│   ├── example_basic_usage.py          ← Simple examples
│   ├── example_flask_api.py            ← Flask REST API
│   └── example_batch_processing.py     ← Process multiple images
│
├── documentation/                       ← FULL DOCUMENTATION
│   ├── API_REFERENCE.txt               ← All functions explained
│   ├── INTEGRATION_GUIDE.txt           ← How to integrate
│   ├── TROUBLESHOOTING.txt             ← Common issues & fixes
│   └── CLASS_MAPPING.txt               ← Class IDs to characters
│
├── config.yaml                          ← Configuration file
├── requirements.txt                     ← Python dependencies
└── README.txt                           ← THIS FILE!

================================================================================
                        ⚡ QUICK START (5 MINUTES)
================================================================================

STEP 1: Install Python Dependencies
------------------------------------
cd DEPLOYMENT_PACKAGE
pip install -r requirements.txt

(Installs: torch, torchvision, pillow, numpy, opencv-python)


STEP 2: Test the Models
------------------------
python examples/example_basic_usage.py

This will:
✓ Load both models
✓ Test classification on a sample image
✓ Test similarity comparison
✓ Generate Grad-CAM visualization


STEP 3: Choose Your Integration Path
-------------------------------------

Option A: Python Backend (Django/Flask)
   → Use src/inference.py directly
   → See examples/example_flask_api.py
   → Models run on your server

Option B: Mobile App (Flutter)
   → Convert models to ONNX or TFLite
   → See documentation/INTEGRATION_GUIDE.txt
   → Models run on-device

Option C: Hybrid (Recommended!)
   → Backend API for heavy processing
   → On-device for simple tasks
   → Best of both worlds!

================================================================================
                        🎯 THE 3 CORE FEATURES
================================================================================

1️⃣ CLASSIFICATION (Branch 1)
   What: Identifies which Ranjana character (1-62)
   Accuracy: 99.5%
   Input: Single 64×64 grayscale image
   Output: Class ID + confidence

   Example:
   ```python
   from src.inference import RanjanaInference
   model = RanjanaInference('efficientnet_b0')
   classes, probs = model.classify('student_image.png')
   print(f"Predicted: Class {classes[0]}, Confidence: {probs[0]:.2%}")
   ```


2️⃣ SIMILARITY SCORING (Branch 2)
   What: Compares two characters, gives similarity 0-100%
   Accuracy: 92.7%
   Input: Two 64×64 grayscale images
   Output: Similarity percentage + distance

   Example:
   ```python
   similarity, distance = model.compute_similarity(
       'student_ka.png',
       'reference_ka.png'
   )
   print(f"Similarity: {similarity:.1f}%")
   ```


3️⃣ GRAD-CAM VISUALIZATION (Branch 3)
   What: Shows where the model is looking (heatmap)
   Use: Help students see mistakes
   Input: Single image
   Output: Heatmap overlay image

   Example:
   ```python
   result = model.generate_gradcam(
       'student_image.png',
       save_path='heatmap.png'
   )
   print(f"Predicted: {result['predicted_class']}")
   # Heatmap saved to heatmap.png
   ```

================================================================================
                     🔧 SYSTEM REQUIREMENTS
================================================================================

MINIMUM:
- Python 3.8+
- 2 GB RAM
- 500 MB disk space
- CPU is fine (no GPU needed for inference!)

RECOMMENDED:
- Python 3.10+
- 4 GB RAM
- GPU (optional, makes it faster)

TESTED ON:
✓ Ubuntu 22.04 (Python 3.13)
✓ Windows 10/11
✓ macOS (M1/M2 chips supported!)

================================================================================
                       📱 FOR FLUTTER DEVELOPERS
================================================================================

You have 2 options:

OPTION 1: Call Backend API
---------------------------
✓ Easiest to implement
✓ Models stay on server
✓ Always up-to-date
✓ No model conversion needed

1. Your Django friend creates REST API endpoints:
   POST /api/classify       (Branch 1)
   POST /api/similarity     (Branch 2)
   POST /api/gradcam        (Branch 3)

2. You make HTTP requests from Flutter:
   ```dart
   final response = await http.post(
     Uri.parse('https://yourserver.com/api/classify'),
     body: {'image': base64Image},
   );
   ```

3. Done! ✓


OPTION 2: On-Device Inference
------------------------------
✓ Works offline
✓ Faster (no network delay)
✓ More private
✗ Requires model conversion

1. Convert models to TFLite:
   See documentation/INTEGRATION_GUIDE.txt
   Section: "Converting to TensorFlow Lite"

2. Use tflite_flutter package:
   https://pub.dev/packages/tflite_flutter

3. Load models and run inference in Dart

================================================================================
                       🌐 FOR DJANGO DEVELOPERS
================================================================================

QUICK SETUP:
------------

1. Create Django view:
```python
import sys
sys.path.insert(0, 'DEPLOYMENT_PACKAGE/src')
from inference import RanjanaInference

# Initialize once (at startup)
classifier = RanjanaInference(
    'efficientnet_b0',
    checkpoint_path='DEPLOYMENT_PACKAGE/models/efficientnet_b0_best.pth',
    device='cpu'  # or 'cuda' if you have GPU
)

# In your view
def classify_image(request):
    if request.method == 'POST':
        image = request.FILES['image']
        
        # Save temp
        temp_path = '/tmp/temp_image.png'
        with open(temp_path, 'wb') as f:
            f.write(image.read())
        
        # Classify
        classes, probs = classifier.classify(temp_path)
        
        return JsonResponse({
            'predicted_class': int(classes[0]),
            'confidence': float(probs[0]),
            'top_5_classes': classes.tolist(),
            'top_5_confidences': probs.tolist()
        })
```

2. See examples/example_flask_api.py for complete REST API!

3. Deploy on:
   - Heroku (with buildpack)
   - AWS EC2
   - DigitalOcean
   - Your own server

================================================================================
                         ⚙️ CONFIGURATION
================================================================================

Edit config.yaml to customize:

- Image size (default: 64×64)
- Number of classes (default: 62)
- Similarity threshold (default: 0.45)
- Device preference (CPU/GPU)

================================================================================
                      🎓 CLASS MAPPING (62 CLASSES)
================================================================================

Classes 0-35:  Consonants (क, ख, ग, घ, etc.)
Classes 36-51: Vowels (अ, आ, इ, ई, etc.)
Classes 52-61: Numerals (०-९)

Folder "1" → Class 0 (first consonant)
Folder "2" → Class 1 (second consonant)
...
Folder "62" → Class 61 (last numeral)

Full mapping in: documentation/CLASS_MAPPING.txt

================================================================================
                       🐛 TROUBLESHOOTING
================================================================================

PROBLEM: "ModuleNotFoundError: No module named 'torch'"
SOLUTION: pip install -r requirements.txt

PROBLEM: Models load but give wrong predictions
SOLUTION: Check image preprocessing:
   - Must be 64×64 pixels
   - Must be grayscale
   - Must be normalized (mean=0.2611, std=0.4186)

PROBLEM: "CUDA out of memory"
SOLUTION: Use device='cpu' instead of 'cuda'

PROBLEM: Similarity always returns high/low scores
SOLUTION: Check that both images are preprocessed the same way

See documentation/TROUBLESHOOTING.txt for more!

================================================================================
                        📊 MODEL PERFORMANCE
================================================================================

CLASSIFICATION MODEL:
✓ Test Accuracy: 99.50%
✓ Stress Test: 100% (100/100 samples)
✓ Average Confidence: 95%+
✓ Inference Time: ~50ms per image (CPU)

SIMILARITY MODEL:
✓ Test Accuracy: 92.71%
✓ ROC-AUC: 0.9726
✓ Optimal Threshold: 0.45
✓ False Positive Rate: <5%
✓ Inference Time: ~100ms per pair (CPU)

GRAD-CAM:
✓ Works on any classification input
✓ No additional model needed
✓ Inference Time: ~60ms per image

================================================================================
                        📞 CONTACT & SUPPORT
================================================================================

Model Developer: Bishwas
Date: October 19, 2025
Project: Ranjana Script Intelligent Handwriting Tutor

If you run into issues:
1. Check documentation/TROUBLESHOOTING.txt
2. Check examples/ for working code
3. Contact Bishwas

================================================================================
                          📄 LICENSE & USAGE
================================================================================

This is a GROUP PROJECT for educational purposes.

Models and code are provided as-is for integration into the
Flutter mobile app and Django web application.

DO:
✓ Use in your Flutter/Django app
✓ Modify code as needed
✓ Deploy to production
✓ Share with team members

DON'T:
✗ Redistribute models publicly
✗ Claim as your own work
✗ Use for commercial purposes without permission

================================================================================
                       🎉 YOU'RE ALL SET!
================================================================================

Next Steps:
1. Run examples/example_basic_usage.py to verify everything works
2. Read documentation/INTEGRATION_GUIDE.txt for your platform
3. Start integrating into your app!

The models are PRODUCTION READY - tested and verified! 🚀

Good luck with your project! 💪

================================================================================
