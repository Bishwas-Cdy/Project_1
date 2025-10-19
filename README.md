# Ranjana Script Recognition Models

<div align="center">

**Production-Ready Deep Learning Models for Ranjana Script Character Recognition**

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-Educational-green.svg)]()
[![Status](https://img.shields.io/badge/status-Production%20Ready-success.svg)]()

</div>

---

## 📋 Table of Contents

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

## 🎯 Overview

This deployment package provides state-of-the-art deep learning models for Ranjana script character recognition and analysis. The system achieves **99.5% classification accuracy** and **92.7% similarity detection accuracy** on the Ranjana-64 benchmark dataset.

The package includes:
- Pre-trained EfficientNet-B0 models
- Complete Python inference API
- REST API examples for web/mobile integration
- Comprehensive documentation
- Production-ready code examples

---

## ✨ Key Features

### Three Core Capabilities

| Feature | Description | Accuracy | Speed (CPU) |
|---------|-------------|----------|-------------|
| **Classification** | Identifies which of 62 Ranjana characters | 99.5% | ~50ms |
| **Similarity Scoring** | Compares visual similarity between characters | 92.7% | ~100ms |
| **Grad-CAM Visualization** | Shows model attention regions for interpretability | N/A | ~60ms |

### Technical Highlights

- ⚡ **Fast Inference**: Real-time performance on CPU, 10x faster on GPU
- 🎯 **High Accuracy**: Extensively validated with stress testing
- 📦 **Easy Integration**: Simple Python API, REST endpoints included
- 🔧 **Flexible Deployment**: Backend API or on-device (ONNX/TFLite)
- 📚 **Well Documented**: Complete API reference and integration guides

---

## 📊 Performance Metrics

### Classification Model (EfficientNet-B0)

```
Test Accuracy:        99.50%
Validation Accuracy:  99.48%
Stress Test:          100% (100/100 samples)
Strong Classes:       96.8% (60/62 classes >90% accuracy)
Avg Confidence:       95.3%
Model Size:           47 MB
```

### Similarity Model (Siamese EfficientNet-B0)

```
Test Accuracy:        92.71%
ROC-AUC Score:        0.9726
False Positive Rate:  4%
False Negative Rate:  2%
Model Size:           25 MB
```

### Inference Latency

| Hardware | Classification | Similarity | Grad-CAM |
|----------|---------------|------------|----------|
| CPU (Intel i7) | 50ms | 100ms | 60ms |
| GPU (CUDA) | 5ms | 10ms | 8ms |
| Batch (32 images) | 20ms/img | 50ms/pair | 25ms/img |

---

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager
- 2 GB RAM minimum
- 100 MB disk space (plus 72 MB for models)

### Install Dependencies

```bash
cd DEPLOYMENT_PACKAGE
pip install -r requirements.txt
```

**Dependencies:**
- PyTorch 2.0+
- torchvision 0.15+
- Pillow 9.0+
- NumPy 1.24+
- OpenCV 4.8+

### Verify Installation

```bash
python examples/example_basic_usage.py
```

Expected output: Model loads successfully and runs inference on test images.

---

## 🎓 Quick Start

### Basic Classification

```python
from src.inference import RanjanaInference

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

---

## 📖 API Reference

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

**classify(image_path, top_k=5)**
- Classifies a character image
- Returns: `(classes, probabilities)` - NumPy arrays of shape (top_k,)

**compute_similarity(image1_path, image2_path)**
- Computes similarity between two images
- Returns: `(similarity_score, distance)` - Percentage (0-100) and Euclidean distance

**generate_gradcam(image_path, target_class=None, save_path=None)**
- Generates Grad-CAM visualization
- Returns: Dictionary with keys: `cam`, `overlay`, `predicted_class`, `confidence`

**classify_batch(image_paths, top_k=5, batch_size=32)**
- Efficiently processes multiple images
- Returns: List of (classes, probabilities) tuples

For complete API documentation, see [`documentation/API_REFERENCE.txt`](documentation/API_REFERENCE.txt)

---

## 🔗 Integration Guides

### Flask/Django Backend

```python
from flask import Flask, request, jsonify
from src.inference import RanjanaInference

app = Flask(__name__)
model = RanjanaInference('efficientnet_b0')

@app.route('/classify', methods=['POST'])
def classify():
    image_file = request.files['image']
    image_file.save('/tmp/temp.png')
    
    classes, probs = model.classify('/tmp/temp.png')
    
    return jsonify({
        'predicted_class': int(classes[0]),
        'confidence': float(probs[0])
    })
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

## 🏗️ Model Architecture

### Classification Model

```
EfficientNet-B0 (ImageNet pretrained)
├── Input: 64x64 grayscale image
├── Backbone: EfficientNet-B0 (modified for grayscale)
├── Output: 62 classes (Ranjana characters)
└── Training: Cross-entropy loss, Adam optimizer
```

### Similarity Model

```
Siamese Network (Twin EfficientNet-B0)
├── Input: Two 64x64 grayscale images
├── Encoder: EfficientNet-B0 (shared weights)
├── Embedding: 512-dimensional feature vectors
├── Distance: Euclidean distance in embedding space
└── Training: Contrastive loss
```

### Grad-CAM

```
Gradient-weighted Class Activation Mapping
├── Target Layer: Final convolutional layer
├── Method: Weighted combination of activation maps
└── Output: Heatmap overlay showing model attention
```

---

## 📁 Project Structure

```
DEPLOYMENT_PACKAGE/
├── models/                          # Pre-trained model weights (72 MB)
│   ├── efficientnet_b0_best.pth
│   └── siamese_efficientnet_b0_best.pth
│
├── src/                             # Python source code
│   ├── inference.py                 # Main inference API
│   ├── gradcam.py                   # Grad-CAM implementation
│   ├── siamese_network.py           # Siamese network architecture
│   ├── models.py                    # Model definitions
│   ├── config.py                    # Configuration
│   └── data_loader.py               # Data utilities
│
├── examples/                        # Integration examples
│   ├── example_basic_usage.py       # Basic API usage
│   ├── example_flask_api.py         # REST API server
│   └── example_batch_processing.py  # Batch inference
│
├── documentation/                   # Comprehensive docs
│   ├── API_REFERENCE.txt            # Complete API docs
│   ├── INTEGRATION_GUIDE.txt        # Platform integration guides
│   ├── TROUBLESHOOTING.txt          # Common issues & solutions
│   └── CLASS_MAPPING.txt            # Character class mappings
│
├── config.yaml                      # Model configuration
├── requirements.txt                 # Python dependencies
└── README.md                        # This file
```

---

## 📚 Documentation

| Document | Description |
|----------|-------------|
| [API Reference](documentation/API_REFERENCE.txt) | Complete function documentation with examples |
| [Integration Guide](documentation/INTEGRATION_GUIDE.txt) | Flask, Django, and Flutter integration steps |
| [Troubleshooting](documentation/TROUBLESHOOTING.txt) | Common errors and solutions |
| [Class Mapping](documentation/CLASS_MAPPING.txt) | Character class ID mappings (0-61) |

---

## 🎯 Use Cases

- **Educational Apps**: Automated grading of student handwriting
- **OCR Systems**: Character recognition in scanned documents
- **Similarity Search**: Find matching characters in databases
- **Quality Assessment**: Compare student writing to reference characters
- **Model Interpretability**: Visualize model decision-making with Grad-CAM

---

## ⚙️ Configuration

Edit `config.yaml` to customize:

```yaml
dataset:
  image_size: 64        # Input image dimensions
  num_classes: 62       # Number of character classes

model:
  architecture: efficientnet_b0
  device: auto          # 'auto', 'cpu', or 'cuda'

inference:
  similarity_threshold: 0.45  # Distance threshold for matching
  batch_size: 32             # Batch processing size
```

---

## 🔧 Advanced Usage

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

## 🐛 Troubleshooting

### Common Issues

**ModuleNotFoundError: No module named 'torch'**
```bash
pip install -r requirements.txt
```

**CUDA out of memory**
```python
model = RanjanaInference('efficientnet_b0', device='cpu')
```

**Wrong predictions**
- Ensure images are 64x64 pixels (auto-resized)
- Verify grayscale conversion (automatic)
- Check class mapping: Model outputs 0-61, folders are 1-62

For more issues, see [TROUBLESHOOTING.txt](documentation/TROUBLESHOOTING.txt)

---

## 📜 License

This project is part of an educational initiative for the Ranjana Script Intelligent Handwriting Tutor.

**Permitted Use:**
- Integration into educational applications
- Modification for research purposes
- Deployment in non-commercial projects

**Restrictions:**
- Public redistribution of trained models requires authorization
- Commercial use requires explicit permission
- Attribution must be maintained

---

## 👥 Contact

**Project:** Ranjana Script Intelligent Handwriting Tutor  
**Developer:** Bishwas  
**Release Date:** October 19, 2025

For questions, issues, or collaboration:
- Open an issue in the repository
- Contact the development team
- Check documentation first

---

## 🙏 Acknowledgments

- Dataset: Ranjana-64 Character Dataset
- Architecture: EfficientNet (Google Research)
- Framework: PyTorch
- Grad-CAM: Selvaraju et al., 2017

---

## 📈 Changelog

### Version 1.0.0 (October 19, 2025)
- Initial production release
- Classification accuracy: 99.5%
- Similarity accuracy: 92.7%
- Complete API documentation
- Integration examples for Flask, Django, Flutter
- Grad-CAM visualization support

---

<div align="center">

**⭐ If this helps your project, please give it a star! ⭐**

Made with ❤️ for the Ranjana Script Community

</div>
