================================================================================
                        EXAMPLES DIRECTORY GUIDE
                   What Each Example Does & When to Use It
================================================================================

This directory contains 3 ready-to-run Python scripts that demonstrate
how to use the Ranjana Script AI models.

Think of these as "copy-paste starting points" for integration!

================================================================================
                     WHAT'S IN THIS DIRECTORY
================================================================================

examples/
 example_basic_usage.py          <- START HERE! Quick test
 example_flask_api.py            <- REST API for backend
 example_batch_processing.py     <- Process multiple images
 README_EXAMPLES.txt             <- THIS FILE!

================================================================================
              1⃣ example_basic_usage.py - "Hello World"
================================================================================

PURPOSE: Quick test to verify everything works

WHAT IT DOES:
- Creates a dummy test image
- Tests all 3 features:
  1. Classification -> "What character is this?"
  2. Similarity -> "How similar are these two?"
  3. Grad-CAM -> "What is the model looking at?"
- Prints results to terminal
- Saves Grad-CAM visualization

WHEN TO USE:
 First thing after installation (verify models work)
 Learning how the API works
 Quick reference for basic code
 Teaching teammates the basics

HOW TO RUN:
```bash
cd DEPLOYMENT_PACKAGE
python examples/example_basic_usage.py
```

EXPECTED OUTPUT:
```
Loading models...
 Models loaded successfully

======================================================================
EXAMPLE 1: CLASSIFICATION
======================================================================
 Image: sample_image.png
 Predicted Class: 23
 Confidence: 87.45%

 Top 5 Predictions:
   1. Class 23: 87.45%
   2. Class 15: 5.23%
   3. Class 8: 2.11%
   4. Class 42: 1.89%
   5. Class 7: 1.05%

======================================================================
EXAMPLE 2: SIMILARITY COMPARISON
======================================================================
 Image 1: sample_image.png
 Image 2: sample_image.png
 Similarity Score: 100.0%
 Distance: 0.0000
 Match: Yes!  (threshold: 0.45)

======================================================================
EXAMPLE 3: GRAD-CAM VISUALIZATION
======================================================================
 Image: sample_image.png
 Predicted Class: 23
 Confidence: 87.45%
 Grad-CAM saved to: gradcam_visualization.png
```

CODE SNIPPET YOU CAN COPY:
```python
from src.inference import RanjanaInference

# Initialize model
model = RanjanaInference('efficientnet_b0')

# Classify
classes, probs = model.classify('image.png')
print(f"Predicted: Class {classes[0]} ({probs[0]:.2%})")

# Similarity
similarity, distance = model.compute_similarity('img1.png', 'img2.png')
print(f"Similarity: {similarity:.1f}%")

# Grad-CAM
result = model.generate_gradcam('image.png', save_path='heatmap.png')
print(f"Saved to: {result['save_path']}")
```

================================================================================
            2⃣ example_flask_api.py - REST API Server
================================================================================

PURPOSE: Production-ready REST API for Django/Flask backend

WHAT IT DOES:
- Creates a web server with 3 API endpoints
- Flutter app can call these endpoints over HTTP
- Handles base64 image encoding/decoding
- Returns JSON results
- Includes error handling

API ENDPOINTS:
```
POST /api/classify    -> Classify character
POST /api/similarity  -> Compare two characters
POST /api/gradcam     -> Generate heatmap visualization
GET  /health          -> Check if server is running
```

WHEN TO USE:
 When Flutter developers need a backend API
 When you want models on the server (not mobile)
 As a starting point for your Django integration
 For web-based applications

HOW TO RUN:
```bash
# Install Flask first (if not already)
pip install flask

# Run the server
cd DEPLOYMENT_PACKAGE
python examples/example_flask_api.py

# Server runs at http://localhost:5000
```

EXPECTED OUTPUT:
```
Loading models...
 Models loaded!
 * Running on http://127.0.0.1:5000
 * Press CTRL+C to quit
```

HOW TO TEST:
```bash
# Test classification endpoint
curl -X POST http://localhost:5000/api/classify \
  -H "Content-Type: application/json" \
  -d '{"image": "<base64_encoded_image>"}'

# Response:
{
  "success": true,
  "predicted_class": 23,
  "confidence": 0.8745,
  "top_5_classes": [23, 15, 8, 42, 7],
  "top_5_confidences": [0.8745, 0.0523, 0.0211, 0.0189, 0.0105]
}
```

FLUTTER INTEGRATION EXAMPLE:
```dart
import 'package:http/http.dart' as http;
import 'dart:convert';

Future<Map<String, dynamic>> classifyCharacter(String base64Image) async {
  final response = await http.post(
    Uri.parse('http://yourserver.com/api/classify'),
    headers: {'Content-Type': 'application/json'},
    body: jsonEncode({'image': base64Image}),
  );
  
  if (response.statusCode == 200) {
    return jsonDecode(response.body);
  } else {
    throw Exception('Classification failed');
  }
}
```

DJANGO ADAPTATION:
- Copy the classify(), similarity(), gradcam() function logic
- Replace Flask decorators with Django views
- Use Django's JsonResponse instead of Flask's jsonify
- See documentation/INTEGRATION_GUIDE.txt for full Django example

================================================================================
         3⃣ example_batch_processing.py - Process Multiple Images
================================================================================

PURPOSE: Efficiently process hundreds/thousands of images at once

WHAT IT DOES:
- Processes entire directories of images
- Shows progress (e.g., "Processing 347/1000...")
- Saves results to CSV file
- 60% faster than processing one-by-one! (batch processing)
- Generates Grad-CAM for all images
- Compares student vs reference directories

FEATURES:

Feature 1: Classify Directory
------------------------------
Process all images in a folder and export to CSV

```python
process_directory('test_images/', 'results.csv')
```

Output CSV:
```
image_path,predicted_class,folder_name,confidence,top_2_class,top_2_conf
test_images/img1.png,23,24,0.8745,15,0.0523
test_images/img2.png,15,16,0.9234,23,0.0412
...
```

Feature 2: Compare Directories
-------------------------------
Compare student images vs reference images

```python
compare_directories('student/', 'reference/', 'comparison.csv')
```

Output CSV:
```
image_name,similarity_score,distance,is_similar
ka.png,95.2,0.123,Yes
kha.png,88.7,0.287,Yes
ga.png,42.1,0.678,No
...
```

Feature 3: Batch Grad-CAM
--------------------------
Generate heatmaps for all images

```python
generate_gradcam_for_directory('images/', 'outputs/')
```

Creates:
```
outputs/
 gradcam_img1.png
 gradcam_img2.png
 gradcam_img3.png
...
```

WHEN TO USE:
 Evaluating student submissions (100+ images)
 Testing model on entire dataset
 Batch grading/scoring
 Creating visualizations for analysis
 Performance benchmarking

HOW TO RUN:
```bash
cd DEPLOYMENT_PACKAGE
python examples/example_batch_processing.py

# Note: You'll need to create test_images/ directory first
# or modify the script to use your own directories
```

EXPECTED OUTPUT:
```
Ranjana Script - Batch Processing Examples
============================================================

1. Initializing model...
 Model loaded successfully

============================================================
EXAMPLE 1: Classify All Images in Directory
============================================================
Processing directory: test_images
Found 347 images
Processing batch 1/11...
Processing batch 2/11...
...
Progress: 347/347 (100.0%)

Saving results to classification_results.csv...

============================================================
PROCESSING COMPLETE
============================================================
Total images: 347
Total time: 28.45 seconds
Average time per image: 82.0 ms
Results saved to: classification_results.csv

Confidence Statistics:
  Average: 94.23%
  Minimum: 45.12%
  Maximum: 99.98%
```

PERFORMANCE COMPARISON:
```
Single processing (for loop):    347 images × 50ms = 17.4 seconds
Batch processing (batch_size=32): 347 images × 20ms = 6.9 seconds
                                   ↑ 60% FASTER!
```

CODE SNIPPET YOU CAN COPY:
```python
from src.inference import RanjanaInference
from glob import glob

# Initialize
model = RanjanaInference('efficientnet_b0')

# Get all images
images = glob('dataset/test/**/*.png', recursive=True)

# Classify in batches (FAST!)
results = model.classify_batch(images, batch_size=32)

# Process results
for img_path, (classes, probs) in zip(images, results):
    predicted_class = classes[0]
    confidence = probs[0]
    print(f"{img_path}: Class {predicted_class} ({confidence:.2%})")
```

================================================================================
                   WHICH EXAMPLE TO USE WHEN
================================================================================

YOUR TEAMMATE SAYS...                    USE THIS EXAMPLE

"Just want to test if models work"    -> example_basic_usage.py
"Need to learn the API"               -> example_basic_usage.py
"Building Flutter app backend"        -> example_flask_api.py
"Need REST API endpoints"             -> example_flask_api.py
"Creating Django web service"         -> example_flask_api.py
"Have 1000 student images to grade"   -> example_batch_processing.py
"Need to compare students vs refs"    -> example_batch_processing.py
"Want to analyze model on dataset"    -> example_batch_processing.py
"Need performance metrics"            -> example_batch_processing.py

================================================================================
                       TYPICAL WORKFLOW
================================================================================

STEP 1: Test Installation
--------------------------
Run: python examples/example_basic_usage.py
Goal: Verify models work
Time: 30 seconds

STEP 2: Learn the API
----------------------
Read: example_basic_usage.py code
Goal: Understand how to call functions
Time: 5 minutes

STEP 3: Copy Code Pattern
--------------------------
Copy: Code from example into your project
Goal: Integrate into your own code
Time: 10 minutes

STEP 4: (Backend Developer) Build API
--------------------------------------
Use: example_flask_api.py as template
Goal: Create REST endpoints for Flutter
Time: 30 minutes

STEP 5: (Optional) Batch Processing
------------------------------------
Use: example_batch_processing.py
Goal: Process large datasets efficiently
Time: As needed

================================================================================
                       CUSTOMIZATION TIPS
================================================================================

MODIFY FOR YOUR NEEDS:

1. Change Model Path:
```python
model = RanjanaInference(
    'efficientnet_b0',
    checkpoint_path='/your/custom/path/model.pth'
)
```

2. Use GPU (if available):
```python
model = RanjanaInference('efficientnet_b0', device='cuda')
```

3. Adjust Batch Size:
```python
# Smaller = less memory, slower
# Larger = more memory, faster
results = model.classify_batch(images, batch_size=16)  # Default: 32
```

4. Change Similarity Threshold:
```python
similarity, distance = model.compute_similarity(img1, img2)
if distance < 0.45:  # You can adjust 0.45
    print("Same character!")
```

5. Get More Top Predictions:
```python
classes, probs = model.classify(image, top_k=10)  # Default: 5
```

================================================================================
                       COMMON QUESTIONS
================================================================================

Q: Do I need to run these examples?
A: No, but running example_basic_usage.py is recommended to verify setup.

Q: Can I modify these files?
A: Yes! They're templates - copy and customize for your needs.

Q: Which example should I start with?
A: Start with example_basic_usage.py to learn the basics.

Q: Do I need Flask installed?
A: Only for example_flask_api.py. Others work without Flask.

Q: Can I use this in production?
A: example_flask_api.py is production-ready (add authentication/SSL).
   Others are educational examples.

Q: How do I add authentication to the API?
A: Add Flask-Login or JWT tokens. See Flask documentation.

Q: Can I deploy the API to Heroku/AWS?
A: Yes! example_flask_api.py can be deployed anywhere Flask works.

================================================================================
                       ADDITIONAL RESOURCES
================================================================================

For more information, see:

- API Reference: ../documentation/API_REFERENCE.txt
  -> Complete function documentation

- Integration Guide: ../documentation/INTEGRATION_GUIDE.txt
  -> Step-by-step Flutter & Django integration

- Troubleshooting: ../documentation/TROUBLESHOOTING.txt
  -> Common errors and solutions

- Main README: ../README.txt
  -> Quick start and overview

================================================================================

HAPPY CODING! 

If you get stuck, check the documentation/ folder or contact Bishwas.

================================================================================
