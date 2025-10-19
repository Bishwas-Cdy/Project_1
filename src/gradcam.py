"""
Grad-CAM (Gradient-weighted Class Activation Mapping) Implementation
For visualizing which parts of Ranjana character images the model focuses on.

Author: Bishwas
Date: October 19, 2025
Branch 3: Explainability & Visualization
"""

import torch
import torch.nn.functional as F
import numpy as np
import cv2
from typing import Optional, Tuple
from PIL import Image


class GradCAM:
    """
    Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization
    
    Generates heatmaps showing which regions of the input image are important
    for the model's prediction.
    
    Args:
        model: PyTorch model (EfficientNet-B0 in our case)
        target_layer: The convolutional layer to visualize (default: last conv layer)
    """
    
    def __init__(self, model: torch.nn.Module, target_layer: Optional[torch.nn.Module] = None):
        self.model = model
        self.model.eval()
        
        # Storage for activations and gradients
        self.activations = None
        self.gradients = None
        
        # Find target layer (last conv layer if not specified)
        if target_layer is None:
            target_layer = self._find_target_layer()
        
        self.target_layer = target_layer
        
        # Register hooks
        self._register_hooks()
    
    def _find_target_layer(self) -> torch.nn.Module:
        """
        Automatically find the last convolutional layer in the model.
        For EfficientNet-B0, this is typically the final conv layer before pooling.
        """
        # For our wrapped EfficientNet models
        if hasattr(self.model, 'efficientnet'):
            # Model is wrapped (EfficientNetModel class)
            base_model = self.model.efficientnet
        else:
            base_model = self.model
        
        # For EfficientNet, the last conv layer is in features
        if hasattr(base_model, 'features'):
            # Find the last Conv2d layer
            last_conv = None
            for module in base_model.features.modules():
                if isinstance(module, torch.nn.Conv2d):
                    last_conv = module
            if last_conv is not None:
                return last_conv
        
        # For ResNet models
        if hasattr(self.model, 'resnet'):
            base_model = self.model.resnet
            if hasattr(base_model, 'layer4'):
                # Return the last layer
                return base_model.layer4[-1].conv2
        
        # For VGG models
        if hasattr(self.model, 'vgg'):
            base_model = self.model.vgg
            if hasattr(base_model, 'features'):
                last_conv = None
                for module in base_model.features.modules():
                    if isinstance(module, torch.nn.Conv2d):
                        last_conv = module
                if last_conv is not None:
                    return last_conv
        
        # For CustomCNN
        if hasattr(self.model, 'conv3'):
            return self.model.conv3
        
        raise ValueError("Could not find convolutional layers in model")
    
    def _register_hooks(self):
        """Register forward and backward hooks on the target layer."""
        
        def forward_hook(module, input, output):
            """Save the activations from the forward pass."""
            self.activations = output.detach()
        
        def backward_hook(module, grad_input, grad_output):
            """Save the gradients from the backward pass."""
            self.gradients = grad_output[0].detach()
        
        # Register hooks
        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)
    
    def generate_cam(
        self, 
        input_tensor: torch.Tensor, 
        target_class: Optional[int] = None
    ) -> np.ndarray:
        """
        Generate Class Activation Map (CAM) for the input image.
        
        Args:
            input_tensor: Input image tensor (1, 1, 64, 64) for grayscale
            target_class: Target class index. If None, uses the predicted class.
        
        Returns:
            cam: Class activation map as numpy array (H, W) with values in [0, 1]
        """
        # Forward pass
        self.model.zero_grad()
        output = self.model(input_tensor)
        
        # Get target class (use predicted class if not specified)
        if target_class is None:
            target_class = output.argmax(dim=1).item()
        
        # Backward pass for target class
        output[:, target_class].backward()
        
        # Get activations and gradients
        activations = self.activations  # (1, C, H, W)
        gradients = self.gradients      # (1, C, H, W)
        
        # Global average pooling of gradients (weights for each channel)
        weights = gradients.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)
        
        # Weighted combination of activation maps
        cam = (weights * activations).sum(dim=1, keepdim=True)  # (1, 1, H, W)
        
        # Apply ReLU (only positive contributions)
        cam = F.relu(cam)
        
        # Normalize to [0, 1]
        cam = cam.squeeze().cpu().numpy()  # (H, W)
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)  # Avoid division by zero
        
        return cam
    
    def overlay_heatmap(
        self,
        image: np.ndarray,
        cam: np.ndarray,
        alpha: float = 0.5,
        colormap: int = cv2.COLORMAP_JET
    ) -> np.ndarray:
        """
        Overlay the CAM heatmap on the original image.
        
        Args:
            image: Original image as numpy array (H, W) or (H, W, 3)
            cam: Class activation map (H, W) with values in [0, 1]
            alpha: Blending factor (0 = only image, 1 = only heatmap)
            colormap: OpenCV colormap to use for heatmap
        
        Returns:
            overlay: RGB image with heatmap overlay (H, W, 3)
        """
        # Resize CAM to match input image size
        cam_resized = cv2.resize(cam, (image.shape[1], image.shape[0]))
        
        # Convert CAM to heatmap (0-255)
        heatmap = np.uint8(255 * cam_resized)
        heatmap = cv2.applyColorMap(heatmap, colormap)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        
        # Convert grayscale image to RGB if needed
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        
        # Normalize image to [0, 255] and convert to uint8
        if image.dtype != np.uint8:
            if image.max() <= 1.0:
                image = np.uint8(255 * image)
            else:
                image = np.uint8(image)
        
        # Ensure heatmap is uint8
        heatmap = heatmap.astype(np.uint8)
        
        # Blend heatmap with original image
        overlay = cv2.addWeighted(image, 1 - alpha, heatmap, alpha, 0)
        
        return overlay
    
    def __call__(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None,
        return_cam_only: bool = False
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate Grad-CAM visualization.
        
        Args:
            input_tensor: Input image tensor (1, 1, 64, 64)
            target_class: Target class index
            return_cam_only: If True, only return CAM without overlay
        
        Returns:
            cam: Class activation map (H, W)
            overlay: RGB image with heatmap overlay (H, W, 3) or None if return_cam_only
        """
        cam = self.generate_cam(input_tensor, target_class)
        
        if return_cam_only:
            return cam, None
        
        # Convert input tensor to numpy image
        image = input_tensor.squeeze().cpu().numpy()  # (64, 64)
        
        # Create overlay
        overlay = self.overlay_heatmap(image, cam)
        
        return cam, overlay


def visualize_prediction(
    model: torch.nn.Module,
    image_path: str,
    device: str = 'cuda',
    target_class: Optional[int] = None,
    save_path: Optional[str] = None,
    colormap: int = cv2.COLORMAP_JET
) -> Tuple[int, float, np.ndarray, np.ndarray]:
    """
    Convenient function to visualize a single prediction with Grad-CAM.
    
    Args:
        model: PyTorch classification model
        image_path: Path to input image
        device: Device to run on ('cuda' or 'cpu')
        target_class: Target class for visualization (None = predicted class)
        save_path: Path to save visualization (None = don't save)
        colormap: OpenCV colormap for heatmap
    
    Returns:
        predicted_class: Predicted class index
        confidence: Prediction confidence
        cam: Class activation map
        overlay: RGB visualization with heatmap
    """
    from PIL import Image
    import torchvision.transforms as transforms
    from data_loader import get_transforms
    
    # Load and preprocess image
    image = Image.open(image_path).convert('L')  # Grayscale
    transform = get_transforms('test')
    input_tensor = transform(image).unsqueeze(0).to(device)
    
    # Get prediction
    model.eval()
    with torch.no_grad():
        output = model(input_tensor)
        probabilities = F.softmax(output, dim=1)
        predicted_class = output.argmax(dim=1).item()
        confidence = probabilities[0, predicted_class].item()
    
    # Generate Grad-CAM
    gradcam = GradCAM(model)
    cam, overlay = gradcam(input_tensor, target_class)
    
    # Save if requested
    if save_path:
        Image.fromarray(overlay).save(save_path)
    
    return predicted_class, confidence, cam, overlay


def compare_predictions(
    model: torch.nn.Module,
    image_paths: list,
    labels: list,
    save_dir: str,
    device: str = 'cuda'
):
    """
    Generate Grad-CAM visualizations for multiple images (e.g., correct vs incorrect).
    
    Args:
        model: PyTorch classification model
        image_paths: List of image paths
        labels: List of descriptive labels for each image
        save_dir: Directory to save visualizations
        device: Device to run on
    """
    import os
    from pathlib import Path
    
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    
    gradcam = GradCAM(model)
    
    for img_path, label in zip(image_paths, labels):
        # Load image
        from PIL import Image
        from data_loader import get_transforms
        
        image = Image.open(img_path).convert('L')
        transform = get_transforms('test')
        input_tensor = transform(image).unsqueeze(0).to(device)
        
        # Predict
        model.eval()
        with torch.no_grad():
            output = model(input_tensor)
            pred_class = output.argmax(dim=1).item()
            confidence = F.softmax(output, dim=1)[0, pred_class].item()
        
        # Generate Grad-CAM
        cam, overlay = gradcam(input_tensor)
        
        # Save
        save_path = os.path.join(save_dir, f"{label}_pred{pred_class}_conf{confidence:.2f}.png")
        Image.fromarray(overlay).save(save_path)
        
        print(f"Saved: {save_path}")


if __name__ == "__main__":
    """
    Test Grad-CAM implementation on a sample image.
    """
    import sys
    
    # Example usage
    print("Grad-CAM module loaded successfully!")
    print("Use visualize_prediction() or GradCAM class for generating heatmaps.")
    print("\nExample:")
    print("  from gradcam import GradCAM, visualize_prediction")
    print("  pred, conf, cam, overlay = visualize_prediction(model, 'image.png')")
