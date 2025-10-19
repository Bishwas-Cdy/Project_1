"""
Siamese Network for similarity scoring
Branch 2: Handwriting similarity measurement
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from models import get_model


class SiameseNetwork(nn.Module):
    """
    Siamese Network with shared encoder
    
    Architecture:
        Input: Two images (img1, img2)
        -> Shared Encoder -> Feature vectors (embedding1, embedding2)
        -> Distance computation -> Similarity score
    """
    
    def __init__(self, backbone='efficientnet_b0', embedding_dim=128, pretrained_path=None):
        """
        Args:
            backbone: Base model architecture ('efficientnet_b0', 'resnet18', 'custom_cnn')
            embedding_dim: Size of the embedding vector
            pretrained_path: Path to pretrained classification model checkpoint
        """
        super(SiameseNetwork, self).__init__()
        
        self.backbone_name = backbone
        self.embedding_dim = embedding_dim
        
        # Load base model (without classification head)
        base_model = get_model(backbone, pretrained=False)
        
        # Load pretrained weights if provided
        if pretrained_path:
            print(f"Loading pretrained weights from {pretrained_path}")
            checkpoint = torch.load(pretrained_path, map_location='cpu')
            base_model.load_state_dict(checkpoint['model_state_dict'])
            print(" Loaded pretrained classification model")
        
        # Remove classification head and extract feature extractor
        if 'efficientnet' in backbone:
            # EfficientNet: features + avgpool, remove classifier
            self.encoder = nn.Sequential(*list(base_model.children())[:-1])  # Remove classifier
            # Get feature dimension from the base model
            with torch.no_grad():
                dummy_input = torch.randn(1, 1, 64, 64)
                features = self.encoder(dummy_input)
                feature_dim = features.view(features.size(0), -1).size(1)
        
        elif 'resnet' in backbone:
            # ResNet: all layers except fc
            self.encoder = nn.Sequential(*list(base_model.children())[:-1])  # Remove fc layer
            with torch.no_grad():
                dummy_input = torch.randn(1, 1, 64, 64)
                features = self.encoder(dummy_input)
                feature_dim = features.view(features.size(0), -1).size(1)
        
        elif 'custom_cnn' in backbone:
            # Custom CNN: features part only
            self.encoder = base_model.features
            with torch.no_grad():
                dummy_input = torch.randn(1, 1, 64, 64)
                features = self.encoder(dummy_input)
                feature_dim = features.view(features.size(0), -1).size(1)
        
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        
        # Projection head: maps features to embedding space
        self.projection_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(feature_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, embedding_dim),
            nn.LayerNorm(embedding_dim)  # Normalize embeddings
        )
        
        print(f" Siamese Network initialized:")
        print(f"  - Backbone: {backbone}")
        print(f"  - Feature dim: {feature_dim}")
        print(f"  - Embedding dim: {embedding_dim}")
    
    def forward_once(self, x):
        """
        Forward pass for one image
        Returns normalized embedding vector
        """
        features = self.encoder(x)
        embedding = self.projection_head(features)
        # L2 normalize embeddings for cosine similarity
        embedding = F.normalize(embedding, p=2, dim=1)
        return embedding
    
    def forward(self, img1, img2):
        """
        Forward pass for image pair
        Returns embeddings for both images
        """
        embedding1 = self.forward_once(img1)
        embedding2 = self.forward_once(img2)
        return embedding1, embedding2
    
    def compute_similarity(self, img1, img2, metric='cosine'):
        """
        Compute similarity between two images
        
        Args:
            img1, img2: Input images (batch_size, 1, 64, 64)
            metric: 'cosine' or 'euclidean'
        
        Returns:
            similarity: Similarity score [0, 1] (higher = more similar)
        """
        embedding1, embedding2 = self.forward(img1, img2)
        
        if metric == 'cosine':
            # Cosine similarity (already normalized, so just dot product)
            similarity = (embedding1 * embedding2).sum(dim=1)
            # Convert from [-1, 1] to [0, 1]
            similarity = (similarity + 1) / 2
        
        elif metric == 'euclidean':
            # Euclidean distance -> similarity
            distance = F.pairwise_distance(embedding1, embedding2)
            # Convert distance to similarity (inverse)
            similarity = 1 / (1 + distance)
        
        else:
            raise ValueError(f"Unknown metric: {metric}")
        
        return similarity


class ContrastiveLoss(nn.Module):
    """
    Contrastive Loss for Siamese Networks
    
    Loss = (1 - label) * 0.5 * distance^2 + label * 0.5 * max(0, margin - distance)^2
    
    Where:
        - label = 0 for similar pairs (same class)
        - label = 1 for dissimilar pairs (different class)
        - margin = threshold for dissimilar pairs
    """
    
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
    
    def forward(self, embedding1, embedding2, label):
        """
        Args:
            embedding1, embedding2: Normalized embeddings (batch_size, embedding_dim)
            label: 0 for similar, 1 for dissimilar (batch_size,)
        
        Returns:
            loss: Scalar loss value
        """
        # Euclidean distance
        distance = F.pairwise_distance(embedding1, embedding2)
        
        # Contrastive loss
        loss_similar = (1 - label) * torch.pow(distance, 2)
        loss_dissimilar = label * torch.pow(torch.clamp(self.margin - distance, min=0.0), 2)
        
        loss = torch.mean(loss_similar + loss_dissimilar) / 2.0
        
        return loss


def count_parameters(model):
    """Count trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test the Siamese Network
    print("Testing Siamese Network...")
    
    # Create model
    model = SiameseNetwork(
        backbone='efficientnet_b0',
        embedding_dim=128,
        pretrained_path=None  # Set to your checkpoint path for testing
    )
    
    print(f"\nTotal parameters: {count_parameters(model):,}")
    
    # Test forward pass
    batch_size = 4
    img1 = torch.randn(batch_size, 1, 64, 64)
    img2 = torch.randn(batch_size, 1, 64, 64)
    
    # Get embeddings
    emb1, emb2 = model(img1, img2)
    print(f"\nEmbedding 1 shape: {emb1.shape}")
    print(f"Embedding 2 shape: {emb2.shape}")
    
    # Compute similarity
    similarity = model.compute_similarity(img1, img2, metric='cosine')
    print(f"Similarity scores: {similarity}")
    
    # Test contrastive loss
    criterion = ContrastiveLoss(margin=1.0)
    labels = torch.tensor([0, 1, 0, 1])  # 0=similar, 1=dissimilar
    loss = criterion(emb1, emb2, labels)
    print(f"\nContrastive Loss: {loss.item():.4f}")
    
    print("\n Siamese Network test passed!")
