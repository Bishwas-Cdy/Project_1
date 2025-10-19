"""
Inference utilities for classification, similarity, and Grad-CAM
"""
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2

from config import MODELS_DIR, IMAGE_SIZE
from models import get_model
from data_loader import get_transforms
from gradcam import GradCAM


class RanjanaInference:
    """
    Inference pipeline for Ranjana Script recognition
    """
    
    def __init__(self, model_name: str, device: str = 'cuda', checkpoint_path: str = None):
        """
        Args:
            model_name: Name of trained model to load
            device: Device to run inference on
            checkpoint_path: Optional custom checkpoint path
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model_name = model_name
        self.transform = get_transforms(augment=False)
        
        # Load model
        self.model = get_model(model_name, pretrained=False)
        
        # Determine checkpoint path
        if checkpoint_path is None:
            checkpoint_path = MODELS_DIR / f"{model_name}_best.pth"
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.to(self.device)
        self.model.eval()
        
        val_acc = checkpoint.get('val_acc', checkpoint.get('val_accuracy', 0))
        print(f"Loaded {model_name} model (Val Acc: {val_acc:.2f}%)")
    
    def preprocess_image(self, image_path: str):
        """Preprocess image for inference"""
        image = Image.open(image_path)
        
        # Convert to grayscale
        if image.mode == 'RGBA':
            background = Image.new('RGB', image.size, (255, 255, 255))
            background.paste(image, mask=image.split()[3])
            image = background.convert('L')
        elif image.mode != 'L':
            image = image.convert('L')
        
        # Apply transforms
        image_tensor = self.transform(image).unsqueeze(0)  # Add batch dimension
        return image_tensor, image
    
    def classify(self, image_path: str, top_k: int = 5):
        """
        Classify an image
        
        Args:
            image_path: Path to image
            top_k: Number of top predictions to return
        
        Returns:
            top_classes: List of top k class indices
            top_probs: List of top k probabilities
        """
        image_tensor, _ = self.preprocess_image(image_path)
        image_tensor = image_tensor.to(self.device)
        
        with torch.no_grad():
            outputs = self.model(image_tensor)
            probs = F.softmax(outputs, dim=1)
        
        # Get top k predictions
        top_probs, top_classes = torch.topk(probs, top_k)
        top_probs = top_probs.cpu().numpy()[0]
        top_classes = top_classes.cpu().numpy()[0]
        
        return top_classes, top_probs
    
    def compute_similarity(self, image1_path: str, image2_path: str, 
                          siamese_checkpoint: str = None):
        """
        Compute similarity between two images using Siamese Network
        
        Args:
            image1_path: Path to first image (e.g., student's handwriting)
            image2_path: Path to second image (e.g., reference character)
            siamese_checkpoint: Path to Siamese model checkpoint (optional)
        
        Returns:
            similarity_score: Similarity percentage [0, 100]
            distance: Euclidean distance between embeddings
        """
        from siamese_network import SiameseNetwork
        
        # Load Siamese model if not already loaded
        if not hasattr(self, 'siamese_model'):
            # Auto-detect checkpoint if not provided
            if siamese_checkpoint is None:
                import glob
                siamese_runs = sorted(glob.glob(str(MODELS_DIR / "*siamese*efficientnet*")))
                if not siamese_runs:
                    raise FileNotFoundError("No Siamese model checkpoint found!")
                siamese_checkpoint = f"{siamese_runs[-1]}/siamese_efficientnet_b0_best.pth"
            
            print(f"Loading Siamese model from: {siamese_checkpoint}")
            checkpoint = torch.load(siamese_checkpoint, map_location=self.device)
            
            self.siamese_model = SiameseNetwork(
                backbone=checkpoint['backbone'],
                embedding_dim=checkpoint['embedding_dim'],
                pretrained_path=None
            )
            self.siamese_model.load_state_dict(checkpoint['model_state_dict'])
            self.siamese_model = self.siamese_model.to(self.device)
            self.siamese_model.eval()
            self.optimal_threshold = 0.45  # From evaluation results
            print("✓ Siamese model loaded")
        
        # Preprocess both images
        img1_tensor, _ = self.preprocess_image(image1_path)
        img2_tensor, _ = self.preprocess_image(image2_path)
        img1_tensor = img1_tensor.unsqueeze(0).to(self.device)
        img2_tensor = img2_tensor.unsqueeze(0).to(self.device)
        
        # Get embeddings
        with torch.no_grad():
            emb1, emb2 = self.siamese_model(img1_tensor, img2_tensor)
            
            # Calculate Euclidean distance
            distance = F.pairwise_distance(emb1, emb2).item()
            
            # Convert distance to similarity percentage
            # Lower distance = higher similarity
            # Use optimal threshold (0.45) as reference
            # similarity = 100% when distance = 0
            # similarity = 0% when distance >= threshold * 2
            max_distance = self.optimal_threshold * 2
            similarity_score = max(0, 100 * (1 - distance / max_distance))
        
        return similarity_score, distance
    
    def generate_gradcam(self, image_path: str, target_class: int = None, save_path: str = None):
        """
        Generate Grad-CAM heatmap to visualize model's attention
        
        Args:
            image_path: Path to image
            target_class: Target class for Grad-CAM (if None, uses predicted class)
            save_path: Optional path to save visualization
        
        Returns:
            dict: {
                'predicted_class': int,
                'confidence': float,
                'cam': np.ndarray (H, W) - normalized heatmap,
                'overlay': np.ndarray (H, W, 3) - RGB visualization,
                'save_path': str (if saved)
            }
        """
        # Load image
        image = Image.open(image_path).convert('L')  # Grayscale
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # Get prediction
        self.model.eval()
        with torch.no_grad():
            output = self.model(input_tensor)
            probabilities = F.softmax(output, dim=1)
            predicted_class = output.argmax(dim=1).item()
            confidence = probabilities[0, predicted_class].item()
        
        # Initialize Grad-CAM
        gradcam = GradCAM(self.model)
        
        # Generate CAM
        cam, overlay = gradcam(input_tensor, target_class)
        
        result = {
            'predicted_class': predicted_class,
            'confidence': confidence,
            'cam': cam,
            'overlay': overlay
        }
        
        # Save if requested
        if save_path:
            Image.fromarray(overlay).save(save_path)
            result['save_path'] = save_path
        
        return result


def visualize_predictions(image_path: str, top_classes, top_probs, save_path: str = None):
    """
    Visualize top predictions
    
    Args:
        image_path: Path to input image
        top_classes: Top predicted classes
        top_probs: Top prediction probabilities
        save_path: Optional path to save visualization
    """
    # Load original image
    image = Image.open(image_path).convert('L')
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Display image
    ax1.imshow(image, cmap='gray')
    ax1.set_title('Input Image', fontsize=14)
    ax1.axis('off')
    
    # Display predictions
    y_pos = np.arange(len(top_classes))
    ax2.barh(y_pos, top_probs, color='steelblue')
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels([f"Class {c+1}" for c in top_classes])
    ax2.invert_yaxis()
    ax2.set_xlabel('Probability', fontsize=12)
    ax2.set_title('Top Predictions', fontsize=14)
    ax2.grid(axis='x', alpha=0.3)
    
    # Add percentage labels
    for i, (cls, prob) in enumerate(zip(top_classes, top_probs)):
        ax2.text(prob + 0.01, i, f'{prob*100:.1f}%', va='center')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved visualization to {save_path}")
    else:
        plt.show()
    
    plt.close()


if __name__ == "__main__":
    # Example usage (requires trained model)
    pass
