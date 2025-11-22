"""
ResNet with frequency domain analysis for deepfake detection.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class FrequencyBranch(nn.Module):
    """Frequency domain analysis branch."""
    
    def __init__(self, input_channels: int = 3):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, 3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, 128)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Apply FFT
        x_fft = torch.fft.fft2(x, dim=(-2, -1))
        x_fft = torch.abs(x_fft)
        
        # Process frequency domain
        x = F.relu(self.conv1(x_fft))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        x = x.flatten(1)
        x = self.fc(x)
        
        return x


class ResNetFreq(nn.Module):
    """ResNet with frequency domain analysis."""
    
    def __init__(
        self,
        num_classes: int = 2,
        input_size: int = 224,
        dropout_rate: float = 0.5
    ):
        super().__init__()
        
        # Spatial branch (ResNet-like)
        self.spatial_conv1 = nn.Conv2d(3, 64, 7, stride=2, padding=3)
        self.spatial_bn1 = nn.BatchNorm2d(64)
        self.spatial_pool = nn.MaxPool2d(3, stride=2, padding=1)
        
        # ResNet blocks
        self.spatial_layer1 = self._make_layer(64, 64, 2)
        self.spatial_layer2 = self._make_layer(64, 128, 2, stride=2)
        self.spatial_layer3 = self._make_layer(128, 256, 2, stride=2)
        self.spatial_layer4 = self._make_layer(256, 512, 2, stride=2)
        
        self.spatial_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.spatial_fc = nn.Linear(512, 256)
        
        # Frequency branch
        self.freq_branch = FrequencyBranch(3)
        
        # Fusion
        self.fusion_fc = nn.Linear(256 + 128, 512)
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(512, num_classes)
        
    def _make_layer(self, in_channels: int, out_channels: int, blocks: int, stride: int = 1):
        """Make ResNet layer."""
        layers = []
        
        # First block with potential stride
        layers.append(nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1))
        layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU(inplace=True))
        
        # Remaining blocks
        for _ in range(1, blocks):
            layers.append(nn.Conv2d(out_channels, out_channels, 3, padding=1))
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU(inplace=True))
        
        return nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Spatial branch
        spatial = F.relu(self.spatial_bn1(self.spatial_conv1(x)))
        spatial = self.spatial_pool(spatial)
        spatial = self.spatial_layer1(spatial)
        spatial = self.spatial_layer2(spatial)
        spatial = self.spatial_layer3(spatial)
        spatial = self.spatial_layer4(spatial)
        spatial = self.spatial_pool(spatial)
        spatial = spatial.flatten(1)
        spatial = self.spatial_fc(spatial)
        
        # Frequency branch
        freq = self.freq_branch(x)
        
        # Fusion
        fused = torch.cat([spatial, freq], dim=1)
        fused = F.relu(self.fusion_fc(fused))
        fused = self.dropout(fused)
        output = self.classifier(fused)
        
        return output