"""
Classification heads for deepfake detection models.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List


class BinaryClassificationHead(nn.Module):
    """Binary classification head for real/fake detection."""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 512,
        dropout_rate: float = 0.5,
        num_classes: int = 2
    ):
        super().__init__()
        
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.dropout1 = nn.Dropout(dropout_rate)
        
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.bn2 = nn.BatchNorm1d(hidden_dim // 2)
        self.dropout2 = nn.Dropout(dropout_rate)
        
        self.classifier = nn.Linear(hidden_dim // 2, num_classes)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout1(x)
        
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout2(x)
        
        x = self.classifier(x)
        return x


class LateFusionHead(nn.Module):
    """Late fusion head for multi-modal inputs."""
    
    def __init__(
        self,
        visual_dim: int,
        audio_dim: int,
        hidden_dim: int = 512,
        dropout_rate: float = 0.5,
        num_classes: int = 2
    ):
        super().__init__()
        
        # Individual encoders
        self.visual_encoder = nn.Sequential(
            nn.Linear(visual_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        self.audio_encoder = nn.Sequential(
            nn.Linear(audio_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        # Fusion layers
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        # Classifier
        self.classifier = nn.Linear(hidden_dim // 2, num_classes)
        
    def forward(self, visual_features: torch.Tensor, audio_features: torch.Tensor) -> torch.Tensor:
        # Encode individual modalities
        visual_encoded = self.visual_encoder(visual_features)
        audio_encoded = self.audio_encoder(audio_features)
        
        # Concatenate and fuse
        fused = torch.cat([visual_encoded, audio_encoded], dim=1)
        fused = self.fusion(fused)
        
        # Classify
        output = self.classifier(fused)
        return output


class AttentionHead(nn.Module):
    """Attention-based classification head."""
    
    def __init__(
        self,
        input_dim: int,
        num_heads: int = 8,
        hidden_dim: int = 512,
        dropout_rate: float = 0.1,
        num_classes: int = 2
    ):
        super().__init__()
        
        self.attention = nn.MultiheadAttention(
            embed_dim=input_dim,
            num_heads=num_heads,
            dropout=dropout_rate,
            batch_first=True
        )
        
        self.norm = nn.LayerNorm(input_dim)
        self.dropout = nn.Dropout(dropout_rate)
        
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, num_classes)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Self-attention
        attn_output, _ = self.attention(x, x, x)
        
        # Residual connection and normalization
        x = self.norm(x + self.dropout(attn_output))
        
        # Global average pooling
        x = x.mean(dim=1)
        
        # Classification
        output = self.classifier(x)
        return output