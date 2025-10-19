"""
FLASK REST API EXAMPLE
Complete REST API server for Django/Flask backend

This provides 3 endpoints:
- POST /api/classify       (Branch 1: Classification)
- POST /api/similarity     (Branch 2: Similarity scoring)
- POST /api/gradcam        (Branch 3: Visualization)
"""

import sys
sys.path.insert(0, 'src')

from flask import Flask, request, jsonify, send_file
from inference import RanjanaInference
import base64
import io
from PIL import Image
import os

app = Flask(__name__)

# Initialize model once at startup (not per request!)
print("Loading models...")
classifier = RanjanaInference(
    'efficientnet_b0',
    checkpoint_path='models/efficientnet_b0_best.pth',
    device='cpu'  # Change to 'cuda' if GPU available
)
print(" Models loaded!")


def decode_base64_image(base64_string):
    """Convert base64 string to PIL Image"""
    image_data = base64.b64decode(base64_string)
    image = Image.open(io.BytesIO(image_data))
    return image


@app.route('/api/classify', methods=['POST'])
def classify():
    """
    Classify a Ranjana character
    
    Request:
        {
            "image": "<base64_encoded_image>"
        }
    
    Response:
        {
            "predicted_class": 23,
            "confidence": 0.987,
            "top_5_classes": [23, 15, 8, 42, 7],
            "top_5_confidences": [0.987, 0.008, ...]
        }
    """
    try:
        data = request.get_json()
        base64_image = data['image']
        
        # Decode image
        image = decode_base64_image(base64_image)
        
        # Save temp file
        temp_path = '/tmp/classify_temp.png'
        image.save(temp_path)
        
        # Classify
        classes, probs = classifier.classify(temp_path)
        
        # Clean up
        os.remove(temp_path)
        
        return jsonify({
            'success': True,
            'predicted_class': int(classes[0]),
            'confidence': float(probs[0]),
            'top_5_classes': [int(c) for c in classes],
            'top_5_confidences': [float(p) for p in probs]
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400


@app.route('/api/similarity', methods=['POST'])
def similarity():
    """
    Compare two Ranjana characters
    
    Request:
        {
            "image1": "<base64_encoded_image>",
            "image2": "<base64_encoded_image>"
        }
    
    Response:
        {
            "similarity_score": 87.5,  // 0-100%
            "distance": 0.234,
            "is_similar": true,        // based on threshold 0.45
            "threshold": 0.45
        }
    """
    try:
        data = request.get_json()
        base64_image1 = data['image1']
        base64_image2 = data['image2']
        
        # Decode images
        image1 = decode_base64_image(base64_image1)
        image2 = decode_base64_image(base64_image2)
        
        # Save temp files
        temp_path1 = '/tmp/similarity_temp1.png'
        temp_path2 = '/tmp/similarity_temp2.png'
        image1.save(temp_path1)
        image2.save(temp_path2)
        
        # Compute similarity
        similarity_score, distance = classifier.compute_similarity(
            temp_path1,
            temp_path2
        )
        
        # Clean up
        os.remove(temp_path1)
        os.remove(temp_path2)
        
        return jsonify({
            'success': True,
            'similarity_score': float(similarity_score),
            'distance': float(distance),
            'is_similar': distance < 0.45,
            'threshold': 0.45
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400


@app.route('/api/gradcam', methods=['POST'])
def gradcam():
    """
    Generate Grad-CAM visualization
    
    Request:
        {
            "image": "<base64_encoded_image>",
            "return_base64": true  // optional, default false
        }
    
    Response (if return_base64=true):
        {
            "predicted_class": 23,
            "confidence": 0.987,
            "heatmap_base64": "<base64_encoded_heatmap>"
        }
    
    Response (if return_base64=false):
        Returns the heatmap image file directly
    """
    try:
        data = request.get_json()
        base64_image = data['image']
        return_base64 = data.get('return_base64', False)
        
        # Decode image
        image = decode_base64_image(base64_image)
        
        # Save temp file
        temp_path = '/tmp/gradcam_temp.png'
        heatmap_path = '/tmp/gradcam_output.png'
        image.save(temp_path)
        
        # Generate Grad-CAM
        result = classifier.generate_gradcam(
            temp_path,
            save_path=heatmap_path
        )
        
        # Clean up input
        os.remove(temp_path)
        
        if return_base64:
            # Return as base64
            with open(heatmap_path, 'rb') as f:
                heatmap_base64 = base64.b64encode(f.read()).decode('utf-8')
            
            os.remove(heatmap_path)
            
            return jsonify({
                'success': True,
                'predicted_class': int(result['predicted_class']),
                'confidence': float(result['confidence']),
                'heatmap_base64': heatmap_base64
            })
        else:
            # Return as image file
            response = send_file(heatmap_path, mimetype='image/png')
            os.remove(heatmap_path)
            return response
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'models_loaded': True,
        'version': '1.0'
    })


if __name__ == '__main__':
    print("\n" + "="*70)
    print("RANJANA SCRIPT API SERVER")
    print("="*70)
    print("\nEndpoints:")
    print("  POST   /api/classify      - Classify character")
    print("  POST   /api/similarity    - Compare characters")
    print("  POST   /api/gradcam       - Generate heatmap")
    print("  GET    /health            - Health check")
    print("\n" + "="*70)
    print("Starting server on http://localhost:5000")
    print("="*70 + "\n")
    
    app.run(host='0.0.0.0', port=5000, debug=True)
