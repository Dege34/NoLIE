"""
Xception-based deepfake detection model.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class XceptionBlock(nn.Module):
    """Xception block with depthwise separable convolutions."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        activation: str = "relu"
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        
        # Pointwise convolution
        self.pointwise_conv1 = nn.Conv2d(in_channels, in_channels, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(in_channels)
        
        # Depthwise convolution
        self.depthwise_conv = nn.Conv2d(
            in_channels, in_channels, 3, stride, 1, 
            groups=in_channels, bias=False
        )
        self.bn2 = nn.BatchNorm2d(in_channels)
        
        # Pointwise convolution
        self.pointwise_conv2 = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        
        # Skip connection
        self.skip = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        
        self.activation = getattr(F, activation)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.skip(x)
        
        out = self.pointwise_conv1(x)
        out = self.bn1(out)
        out = self.activation(out)
        
        out = self.depthwise_conv(out)
        out = self.bn2(out)
        out = self.activation(out)
        
        out = self.pointwise_conv2(out)
        out = self.bn3(out)
        
        out += residual
        out = self.activation(out)
        
        return out


class XceptionDeepfakeDetector(nn.Module):
    """
    Xception-based deepfake detection model.
    
    This model uses the Xception architecture adapted for deepfake detection,
    with modifications for better performance on face images.
    """
    
    def __init__(
        self,
        num_classes: int = 2,
        input_size: int = 224,
        dropout_rate: float = 0.5
    ):
        super().__init__()
        
        self.num_classes = num_classes
        self.input_size = input_size
        
        # Initial convolution
        self.conv1 = nn.Conv2d(3, 32, 3, 2, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        
        # Xception blocks
        self.block1 = XceptionBlock(32, 64, 1)
        self.block2 = XceptionBlock(64, 128, 2)
        self.block3 = XceptionBlock(128, 256, 2)
        self.block4 = XceptionBlock(256, 512, 2)
        self.block5 = XceptionBlock(512, 728, 2)
        
        # Middle flow (repeated blocks)
        self.middle_flow = nn.ModuleList([
            XceptionBlock(728, 728, 1) for _ in range(8)
        ])
        
        # Exit flow
        self.exit_block1 = XceptionBlock(728, 1024, 2)
        self.exit_block2 = XceptionBlock(1024, 1536, 1)
        
        # Global average pooling
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Classifier
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(1536, num_classes)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize model weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network."""
        # Initial convolution
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        
        # Entry flow
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        
        # Middle flow
        for block in self.middle_flow:
            x = block(x)
        
        # Exit flow
        x = self.exit_block1(x)
        x = self.exit_block2(x)
        
        # Global pooling and classification
        x = self.avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        
        return x
    
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Get prediction probabilities."""
        with torch.no_grad():
            logits = self.forward(x)
            probs = F.softmax(logits, dim=1)
        return probs
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Get prediction class."""
        with torch.no_grad():
            logits = self.forward(x)
            pred = torch.argmax(logits, dim=1)
        return pred


def create_xception_model(
    num_classes: int = 2,
    input_size: int = 224,
    dropout_rate: float = 0.5,
    pretrained: bool = False
) -> XceptionDeepfakeDetector:
    """
    Create an Xception deepfake detection model.
    
    Args:
        num_classes: Number of output classes (default: 2 for real/fake)
        input_size: Input image size (default: 224)
        dropout_rate: Dropout rate for regularization (default: 0.5)
        pretrained: Whether to load pretrained weights (not implemented yet)
    
    Returns:
        XceptionDeepfakeDetector model
    """
    model = XceptionDeepfakeDetector(
        num_classes=num_classes,
        input_size=input_size,
        dropout_rate=dropout_rate
    )
    
    if pretrained:
        # TODO: Load pretrained weights
        print("Warning: Pretrained weights not available yet")
    
    return model


# Example usage and testing
if __name__ == "__main__":
    # Create model
    model = create_xception_model()
    
    # Test with dummy input
    dummy_input = torch.randn(1, 3, 224, 224)
    
    # Forward pass
    output = model(dummy_input)
    print(f"Model output shape: {output.shape}")
    
    # Get probabilities
    probs = model.predict_proba(dummy_input)
    print(f"Probabilities: {probs}")
    
    # Get prediction
    pred = model.predict(dummy_input)
    print(f"Prediction: {pred}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")