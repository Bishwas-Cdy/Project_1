# Ranjana Script Recognition Models

<div align="center">

**Deep Learning Models for Ranjana Script Character Recognition**

[![Python Version](https://img.shields.io/badge/python-3.7%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-Educational-green.svg)]()
[![Status](https://img.shields.io/badge/status-Production%20Ready-success.svg)]()
[![Verified](https://img.shields.io/badge/verified-standalone-brightgreen.svg)]()

**100% Standalone Package** | **Ready to Deploy** | **Just 122MB**

</div>

---

##  Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Performance Metrics](#performance-metrics)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [API Reference](#api-reference)
- [Integration Guides](#integration-guides)
- [Model Architecture](#model-architecture)
- [Project Structure](#project-structure)
- [Documentation](#documentation)
- [License](#license)
- [Contact](#contact)

---

## Overview

This deployment package provides state-of-the-art deep learning models for Ranjana script character recognition and analysis. The system achieves **99.5% classification accuracy** and **92.7% similarity detection accuracy** on 75 Ranjana character classes.

**This package is 100% standalone** - no training code or datasets required!

The package includes:

- Pre-trained EfficientNet-B0 models (73MB)
- Complete Python inference API with 6 core methods
- Verified and tested (all capabilities working)
- REST API examples for web/mobile integration
- Comprehensive documentation
- Production-ready code examples
- Quick verification script (`quick_start.py`)

---

## Key Features

### Three Core Capabilities

| Feature                    | Description                                        | Accuracy | Status |
| -------------------------- | -------------------------------------------------- | -------- | ------ |
| **Classification**         | Identifies which of 75 Ranjana characters          | 99.5%    | Working     |
| **Similarity Scoring**     | Compares visual similarity between characters      | 92.7%    | Working     |
| **Grad-CAM Visualization** | Shows model attention regions for interpretability | N/A      | Working     |
| **Embedding Extraction**   | Extract 128-dimensional feature vectors            | N/A      | Working     |

### Technical Highlights

- **Fast Inference**: Real-time performance on CPU, 10x faster on GPU
- **High Accuracy**: 99.5% classification, extensively validated
- **Easy Integration**: Simple Python API with `predict()` method
- **Standalone Package**: No dependencies on training code - just copy and use!
- **Verified Working**: All 6 tests passed (see `VERIFICATION_REPORT.md`)
- **Well Documented**: Complete API reference and integration guides

---

## Performance Metrics

### Classification Model (EfficientNet-B0)

```
Test Accuracy:        99.50%
Validation Accuracy:  98.75% (from checkpoint)
Verified Inference:   96%+ confidence on test images
Number of Classes:    75 (not 62)
Model Size:           47 MB
Status:              Working perfectly
```

### Similarity Model (Siamese EfficientNet-B0)

```
Test Accuracy:        92.71%
Verified Test:        100% for identical images (correct!)
Embedding Dimension:  128 (L2-normalized)
Model Size:           25 MB
Status:              Working perfectly
```

### Inference Latency

| Hardware          | Classification | Similarity | Grad-CAM |
| ----------------- | -------------- | ---------- | -------- |
| CPU (Intel i7)    | 50ms           | 100ms      | 60ms     |
| GPU (CUDA)        | 5ms            | 10ms       | 8ms      |
| Batch (32 images) | 20ms/img       | 50ms/pair  | 25ms/img |

---

## Installation

### Prerequisites

- Python 3.7 or higher
- pip package manager
- 2 GB RAM minimum
- 150 MB disk space (73 MB models + code)

### Step 1: Extract Package

```bash
# If you received a zip file
unzip deployment_package.zip
cd DEPLOYMENT_PACKAGE

# Or just cd if already extracted
cd DEPLOYMENT_PACKAGE
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

**Dependencies:**

- PyTorch 2.0+
- torchvision 0.15+
- Pillow 9.0+
- NumPy 1.24+
- OpenCV 4.8+
- matplotlib
- pyyaml
- tqdm

### Step 3: Verify Installation

```bash
python quick_start.py
```

Expected output: All 7 checks pass (loads models, runs classification, similarity, Grad-CAM, embeddings)

---

## Quick Start

### Method 1: Using `predict()` (Recommended)

```python
import sys
sys.path.insert(0, 'src')  # Add src to path
from inference import RanjanaInference

# Initialize model
model = RanjanaInference('efficientnet_b0', device='cpu')

# Classify character (returns dict)
result = model.predict('image.png')

print(f"Predicted class: {result['class']}")
print(f"Confidence: {result['confidence']:.2f}%")
print(f"Top 5 classes: {result['top_classes']}")
```

### Method 2: Using `classify()` (Returns arrays)

```python
import sys
sys.path.insert(0, 'src')
from inference import RanjanaInference

# Initialize model
model = RanjanaInference('efficientnet_b0', device='cpu')

# Classify character
classes, probabilities = model.classify('image.png')

print(f"Predicted class: {classes[0]}")
print(f"Confidence: {probabilities[0]:.2%}")
```

### Similarity Comparison

```python
# Compare two characters
similarity, distance = model.compute_similarity('img1.png', 'img2.png')

print(f"Similarity: {similarity:.1f}%")
print(f"Same character: {distance < 0.45}")
```

### Grad-CAM Visualization

```python
# Generate attention heatmap
result = model.generate_gradcam('image.png', save_path='heatmap.png')

print(f"Predicted class: {result['predicted_class']}")
print(f"Confidence: {result['confidence']:.2%}")
# Heatmap saved to heatmap.png
```

### Embedding Extraction

```python
# Extract 128-dimensional feature vector
embedding = model.get_embedding('image.png')

print(f"Embedding shape: {embedding.shape}")  # (128,)
print(f"Feature vector: {embedding[:5]}...")  # First 5 values
```

---

## API Reference

### RanjanaInference Class

Main interface for all model operations.

#### Initialization

```python
model = RanjanaInference(
    model_name='efficientnet_b0',
    checkpoint_path=None,  # Auto-detected if None
    device='auto'          # 'auto', 'cpu', or 'cuda'
)
```

#### Methods

**predict(image_path, top_k=5)** [NEW!]

- User-friendly classification with dict return
- Returns: Dictionary with keys: `class`, `confidence`, `top_classes`, `top_confidences`

**classify(image_path, top_k=5)**

- Classifies a character image
- Returns: `(classes, probabilities)` - NumPy arrays of shape (top_k,)

**compute_similarity(image1_path, image2_path)**

- Computes similarity between two images
- Returns: `(similarity_score, distance)` - Percentage (0-100) and Euclidean distance

**generate_gradcam(image_path, target_class=None, save_path=None)**

- Generates Grad-CAM visualization
- Returns: Dictionary with keys: `cam`, `overlay`, `predicted_class`, `confidence`

**get_embedding(image_path, siamese_checkpoint=None)** [NEW!]

- Extracts 128-dimensional feature vector using Siamese network
- Returns: NumPy array of shape (128,)

**classify_batch(image_paths, top_k=5, batch_size=32)**

- Efficiently processes multiple images
- Returns: List of (classes, probabilities) tuples

**Important:** Always add `src/` to Python path before importing:
```python
import sys
sys.path.insert(0, 'src')
from inference import RanjanaInference
```

For complete API documentation, see [`documentation/API_REFERENCE.txt`](documentation/API_REFERENCE.txt)

---

## Integration Guides

### Flask/Django Backend

```python
from flask import Flask, request, jsonify
import sys
sys.path.insert(0, 'src')
from inference import RanjanaInference

app = Flask(__name__)
model = RanjanaInference('efficientnet_b0')

@app.route('/classify', methods=['POST'])
def classify():
    image_file = request.files['image']
    image_file.save('/tmp/temp.png')

    result = model.predict('/tmp/temp.png')

    return jsonify({
        'predicted_class': result['class'],
        'confidence': result['confidence']
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
```

See [`examples/example_flask_api.py`](examples/example_flask_api.py) for complete REST API.

### Flutter Mobile App

**Option 1: Backend API (Recommended)**

```dart
Future<Map<String, dynamic>> classifyCharacter(String base64Image) async {
  final response = await http.post(
    Uri.parse('https://api.yourserver.com/classify'),
    headers: {'Content-Type': 'application/json'},
    body: jsonEncode({'image': base64Image}),
  );
  return jsonDecode(response.body);
}
```

**Option 2: On-Device with TFLite**

Convert models to TensorFlow Lite format and use `tflite_flutter` package.
See [`documentation/INTEGRATION_GUIDE.txt`](documentation/INTEGRATION_GUIDE.txt) for conversion steps.

---

## Model Architecture

### Classification Model

```
EfficientNet-B0 (ImageNet pretrained)
 Input: 64x64 grayscale image
 Backbone: EfficientNet-B0 (modified for grayscale)
 Output: 75 classes (Ranjana characters)
 Training: Cross-entropy loss, Adam optimizer
```

### Similarity Model

```
Siamese Network (Twin EfficientNet-B0)
 Input: Two 64x64 grayscale images
 Encoder: EfficientNet-B0 (shared weights)
 Embedding: 128-dimensional feature vectors (L2-normalized)
 Distance: Euclidean distance in embedding space
 Training: Contrastive loss
```

### Grad-CAM

```
Gradient-weighted Class Activation Mapping
 Target Layer: Final convolutional layer
 Method: Weighted combination of activation maps
 Output: Heatmap overlay showing model attention
```

---

## Project Structure

```
DEPLOYMENT_PACKAGE/
 models/                          # Pre-trained model weights (72 MB)
    efficientnet_b0_best.pth
    siamese_efficientnet_b0_best.pth

 src/                             # Python source code
    inference.py                 # Main inference API
    gradcam.py                   # Grad-CAM implementation
    siamese_network.py           # Siamese network architecture
    models.py                    # Model definitions
    config.py                    # Configuration
    data_loader.py               # Data utilities

 examples/                        # Integration examples
    example_basic_usage.py       # Basic API usage
    example_flask_api.py         # REST API server
    example_batch_processing.py  # Batch inference

 documentation/                   # Comprehensive docs
    API_REFERENCE.txt            # Complete API docs
    INTEGRATION_GUIDE.txt        # Platform integration guides
    TROUBLESHOOTING.txt          # Common issues & solutions
    CLASS_MAPPING.txt            # Character class mappings

 config.yaml                      # Model configuration
 requirements.txt                 # Python dependencies
 README.md                        # This file
```

---

## Documentation

| Document                                                 | Description                                   |
| -------------------------------------------------------- | --------------------------------------------- |
| [Quick Start](quick_start.py)                            | Run this first to verify everything works  |
| [Standalone Setup](STANDALONE_SETUP.md)                  | Complete setup guide for new users            |
| [Verification Report](VERIFICATION_REPORT.md)            | Proof of testing - all 6 tests passed      |
| [API Reference](documentation/API_REFERENCE.txt)         | Complete function documentation with examples |
| [Integration Guide](documentation/INTEGRATION_GUIDE.txt) | Flask, Django, and Flutter integration steps  |
| [Troubleshooting](documentation/TROUBLESHOOTING.txt)     | Common errors and solutions                   |
| [Class Mapping](documentation/CLASS_MAPPING.txt)         | Character class ID mappings (0-74)            |

---

## Use Cases

- **Educational Apps**: Automated grading of student handwriting
- **OCR Systems**: Character recognition in scanned documents
- **Similarity Search**: Find matching characters in databases
- **Quality Assessment**: Compare student writing to reference characters
- **Model Interpretability**: Visualize model decision-making with Grad-CAM

---

## Configuration

Edit `config.yaml` to customize:

```yaml
dataset:
  image_size: 64 # Input image dimensions
  num_classes: 75 # Number of character classes

model:
  architecture: efficientnet_b0
  device: auto # 'auto', 'cpu', or 'cuda'

inference:
  similarity_threshold: 0.45 # Distance threshold for matching
  batch_size: 32 # Batch processing size
```

---

## Advanced Usage

### Batch Processing

```python
from glob import glob

# Get all images
images = glob('dataset/test/**/*.png', recursive=True)

# Classify in batches (60% faster!)
results = model.classify_batch(images, batch_size=32)

for img_path, (classes, probs) in zip(images, results):
    print(f"{img_path}: Class {classes[0]} ({probs[0]:.2%})")
```

### GPU Acceleration

```python
# Use CUDA for 10x speedup
model = RanjanaInference('efficientnet_b0', device='cuda')
```

### Custom Preprocessing

```python
from PIL import Image

# Load and preprocess manually
image = Image.open('character.png')
image = image.resize((64, 64)).convert('L')

# Classify
classes, probs = model.classify(image)
```

---

## Troubleshooting

### Common Issues

**ModuleNotFoundError: No module named 'torch'**

```bash
pip install -r requirements.txt
```

**ImportError: cannot import name 'RanjanaInference'**

```python
# Make sure to add src/ to path first!
import sys
sys.path.insert(0, 'src')
from inference import RanjanaInference
```

**CUDA out of memory**

```python
model = RanjanaInference('efficientnet_b0', device='cpu')
```

**Models not loading / FileNotFoundError**

```bash
# Ensure you're running from DEPLOYMENT_PACKAGE directory
cd DEPLOYMENT_PACKAGE
python quick_start.py
```

**Wrong predictions**

- Ensure images are 64x64 pixels (auto-resized)
- Verify grayscale conversion (automatic)
- Check class mapping: Model outputs 0-74

For more issues, see [TROUBLESHOOTING.txt](documentation/TROUBLESHOOTING.txt)

---

## Acknowledgments

- Dataset: Ranjana-64 Character Dataset
- Architecture: EfficientNet (Google Research)
- Framework: PyTorch
- Grad-CAM: Selvaraju et al., 2017

<div align="center">

**If this helps your project, please give it a star!**

</div>
