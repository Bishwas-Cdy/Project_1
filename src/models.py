"""
Model architectures for Ranjana Script classification
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

from config import NUM_CLASSES


class CustomCNN(nn.Module):
    """
    Custom CNN baseline model (LeNet-style)
    """
    
    def __init__(self, num_classes: int = NUM_CLASSES):
        super(CustomCNN, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.5)
        
        # Fully connected layers
        # After 3 pooling layers: 64 -> 32 -> 16 -> 8
        self.fc1 = nn.Linear(128 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)
    
    def forward(self, x):
        # Conv block 1
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        
        # Conv block 2
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        
        # Conv block 3
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        
        return x


class VGG16Model(nn.Module):
    """
    VGG16 model adapted for grayscale images
    """
    
    def __init__(self, num_classes: int = NUM_CLASSES, pretrained: bool = True):
        super(VGG16Model, self).__init__()
        
        # Load pretrained VGG16
        self.vgg = models.vgg16(pretrained=pretrained)
        
        # Modify first conv layer for grayscale input (1 channel)
        self.vgg.features[0] = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        
        # Modify classifier for our num_classes
        self.vgg.classifier[6] = nn.Linear(4096, num_classes)
    
    def forward(self, x):
        return self.vgg(x)


class ResNetModel(nn.Module):
    """
    ResNet model (ResNet18/34/50) adapted for grayscale images
    """
    
    def __init__(self, num_classes: int = NUM_CLASSES, model_name: str = 'resnet18', pretrained: bool = True):
        super(ResNetModel, self).__init__()
        
        # Load ResNet
        if model_name == 'resnet18':
            self.resnet = models.resnet18(pretrained=pretrained)
        elif model_name == 'resnet34':
            self.resnet = models.resnet34(pretrained=pretrained)
        elif model_name == 'resnet50':
            self.resnet = models.resnet50(pretrained=pretrained)
        else:
            raise ValueError(f"Unknown model: {model_name}")
        
        # Modify first conv layer for grayscale input
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # Modify final FC layer for our num_classes
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_features, num_classes)
    
    def forward(self, x):
        return self.resnet(x)
    
    def get_features(self, x):
        """Extract features before final FC layer (for Siamese network)"""
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)
        
        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)
        
        x = self.resnet.avgpool(x)
        x = torch.flatten(x, 1)
        
        return x


class EfficientNetModel(nn.Module):
    """
    EfficientNet model (B0/B1) adapted for grayscale images
    """
    
    def __init__(self, num_classes: int = NUM_CLASSES, model_name: str = 'efficientnet_b0', pretrained: bool = True):
        super(EfficientNetModel, self).__init__()
        
        # Load EfficientNet
        if model_name == 'efficientnet_b0':
            self.efficientnet = models.efficientnet_b0(pretrained=pretrained)
        elif model_name == 'efficientnet_b1':
            self.efficientnet = models.efficientnet_b1(pretrained=pretrained)
        else:
            raise ValueError(f"Unknown model: {model_name}")
        
        # Modify first conv layer for grayscale input
        self.efficientnet.features[0][0] = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, bias=False)
        
        # Modify classifier for our num_classes
        num_features = self.efficientnet.classifier[1].in_features
        self.efficientnet.classifier[1] = nn.Linear(num_features, num_classes)
    
    def forward(self, x):
        return self.efficientnet(x)


def get_model(model_name: str, num_classes: int = NUM_CLASSES, pretrained: bool = True):
    """
    Factory function to get model by name
    
    Args:
        model_name: One of ['custom_cnn', 'vgg16', 'resnet18', 'resnet34', 'resnet50', 'efficientnet_b0', 'efficientnet_b1']
        num_classes: Number of output classes
        pretrained: Whether to use pretrained weights
    
    Returns:
        Model instance
    """
    if model_name == 'custom_cnn':
        return CustomCNN(num_classes)
    elif model_name == 'vgg16':
        return VGG16Model(num_classes, pretrained)
    elif model_name in ['resnet18', 'resnet34', 'resnet50']:
        return ResNetModel(num_classes, model_name, pretrained)
    elif model_name in ['efficientnet_b0', 'efficientnet_b1']:
        return EfficientNetModel(num_classes, model_name, pretrained)
    else:
        raise ValueError(f"Unknown model: {model_name}")


if __name__ == "__main__":
    # Test models
    batch_size = 4
    x = torch.randn(batch_size, 1, 64, 64)
    
    models_to_test = ['custom_cnn', 'resnet18', 'efficientnet_b0']
    
    for model_name in models_to_test:
        print(f"\nTesting {model_name}...")
        model = get_model(model_name, pretrained=False)
        output = model(x)
        print(f"Input shape: {x.shape}")
        print(f"Output shape: {output.shape}")
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
