"""
Tests for model architectures.

Tests the various model architectures including Xception, ViT,
ResNet with frequency analysis, and audio-visual models.
"""

import pytest
import torch
import numpy as np
from pathlib import Path

from deepfake_forensics.models import (
    XceptionForensics, ViTForensics, ResNetFrequencyForensics,
    AudioVisualModel, ClassificationHead, TemporalAggregator
)


class TestXceptionForensics:
    """Test Xception model."""
    
    def test_xception_creation(self):
        """Test Xception model creation."""
        model = XceptionForensics(num_classes=2)
        assert model.num_classes == 2
        assert model.input_channels == 3
    
    def test_xception_forward(self):
        """Test Xception forward pass."""
        model = XceptionForensics(num_classes=2)
        x = torch.randn(2, 3, 224, 224)
        output = model(x)
        assert output.shape == (2, 2)
    
    def test_xception_get_features(self):
        """Test Xception feature extraction."""
        model = XceptionForensics(num_classes=2)
        x = torch.randn(2, 3, 224, 224)
        features = model.get_features(x)
        assert features.shape[0] == 2
        assert features.shape[1] == model.feature_dim
    
    def test_xception_get_attention_maps(self):
        """Test Xception attention maps."""
        model = XceptionForensics(num_classes=2)
        x = torch.randn(2, 3, 224, 224)
        attention = model.get_attention_maps(x)
        assert attention.shape[0] == 2
        assert attention.shape[1] == 1
        assert attention.shape[2] == x.shape[2]
        assert attention.shape[3] == x.shape[3]


class TestViTForensics:
    """Test ViT model."""
    
    def test_vit_creation(self):
        """Test ViT model creation."""
        model = ViTForensics(num_classes=2)
        assert model.num_classes == 2
        assert model.input_channels == 3
    
    def test_vit_forward(self):
        """Test ViT forward pass."""
        model = ViTForensics(num_classes=2)
        x = torch.randn(2, 3, 224, 224)
        output = model(x)
        assert output.shape == (2, 2)
    
    def test_vit_get_features(self):
        """Test ViT feature extraction."""
        model = ViTForensics(num_classes=2)
        x = torch.randn(2, 3, 224, 224)
        features = model.get_features(x)
        assert features.shape[0] == 2
        assert features.shape[1] == model.embed_dim
    
    def test_vit_get_attention_maps(self):
        """Test ViT attention maps."""
        model = ViTForensics(num_classes=2)
        x = torch.randn(2, 3, 224, 224)
        attention_maps = model.get_attention_maps(x)
        assert len(attention_maps) == model.depth
        for attention in attention_maps:
            assert attention.shape[0] == 2  # batch size
            assert attention.shape[1] == model.num_heads


class TestResNetFrequencyForensics:
    """Test ResNet frequency model."""
    
    def test_resnet_freq_creation(self):
        """Test ResNet frequency model creation."""
        model = ResNetFrequencyForensics(num_classes=2)
        assert model.num_classes == 2
        assert model.input_channels == 3
    
    def test_resnet_freq_forward(self):
        """Test ResNet frequency forward pass."""
        model = ResNetFrequencyForensics(num_classes=2)
        x = torch.randn(2, 3, 224, 224)
        output = model(x)
        assert output.shape == (2, 2)
    
    def test_resnet_freq_get_features(self):
        """Test ResNet frequency feature extraction."""
        model = ResNetFrequencyForensics(num_classes=2)
        x = torch.randn(2, 3, 224, 224)
        backbone_features, freq_features = model.get_features(x)
        assert backbone_features.shape[0] == 2
        assert freq_features.shape[0] == 2
        assert freq_features.shape[1] == model.freq_feature_dim
    
    def test_resnet_freq_get_attention_maps(self):
        """Test ResNet frequency attention maps."""
        model = ResNetFrequencyForensics(num_classes=2)
        x = torch.randn(2, 3, 224, 224)
        attention = model.get_attention_maps(x)
        assert attention.shape[0] == 2
        assert attention.shape[1] == 1
        assert attention.shape[2] == x.shape[2]
        assert attention.shape[3] == x.shape[3]


class TestAudioVisualModel:
    """Test audio-visual model."""
    
    def test_audio_visual_creation(self):
        """Test audio-visual model creation."""
        model = AudioVisualModel(num_classes=2)
        assert model.num_classes == 2
        assert model.visual_channels == 3
    
    def test_audio_visual_forward(self):
        """Test audio-visual forward pass."""
        model = AudioVisualModel(num_classes=2)
        visual_input = torch.randn(2, 16, 3, 224, 224)  # (B, T, C, H, W)
        audio_input = torch.randn(2, 16, 13)  # (B, T, audio_dim)
        output = model(visual_input, audio_input)
        assert output.shape == (2, 2)
    
    def test_audio_visual_get_visual_features(self):
        """Test audio-visual visual feature extraction."""
        model = AudioVisualModel(num_classes=2)
        visual_input = torch.randn(2, 16, 3, 224, 224)
        features = model.get_visual_features(visual_input)
        assert features.shape[0] == 2
        assert features.shape[1] == 16  # temporal dimension
        assert features.shape[2] == model.visual_dim
    
    def test_audio_visual_get_audio_features(self):
        """Test audio-visual audio feature extraction."""
        model = AudioVisualModel(num_classes=2)
        audio_input = torch.randn(2, 16, 13)
        features = model.get_audio_features(audio_input)
        assert features.shape[0] == 2
        assert features.shape[1] == 16  # temporal dimension
        assert features.shape[2] == model.visual_dim
    
    def test_audio_visual_get_sync_score(self):
        """Test audio-visual sync score."""
        model = AudioVisualModel(num_classes=2)
        visual_input = torch.randn(2, 16, 3, 224, 224)
        audio_input = torch.randn(2, 16, 13)
        sync_score = model.get_sync_score(visual_input, audio_input)
        assert sync_score.shape[0] == 2
        assert sync_score.shape[1] == 1


class TestClassificationHead:
    """Test classification head."""
    
    def test_classification_head_creation(self):
        """Test classification head creation."""
        head = ClassificationHead(input_dim=512, num_classes=2)
        assert head.input_dim == 512
        assert head.num_classes == 2
    
    def test_classification_head_forward(self):
        """Test classification head forward pass."""
        head = ClassificationHead(input_dim=512, num_classes=2)
        x = torch.randn(2, 512)
        output = head(x)
        assert output.shape == (2, 2)
    
    def test_classification_head_with_hidden(self):
        """Test classification head with hidden layer."""
        head = ClassificationHead(input_dim=512, num_classes=2, hidden_dim=256)
        x = torch.randn(2, 512)
        output = head(x)
        assert output.shape == (2, 2)


class TestTemporalAggregator:
    """Test temporal aggregator."""
    
    def test_temporal_aggregator_creation(self):
        """Test temporal aggregator creation."""
        aggregator = TemporalAggregator(input_dim=512, method="mean")
        assert aggregator.input_dim == 512
        assert aggregator.method == "mean"
    
    def test_temporal_aggregator_mean(self):
        """Test temporal aggregator with mean method."""
        aggregator = TemporalAggregator(input_dim=512, method="mean")
        x = torch.randn(2, 16, 512)  # (B, T, D)
        output = aggregator(x)
        assert output.shape == (2, 512)
    
    def test_temporal_aggregator_max(self):
        """Test temporal aggregator with max method."""
        aggregator = TemporalAggregator(input_dim=512, method="max")
        x = torch.randn(2, 16, 512)
        output = aggregator(x)
        assert output.shape == (2, 512)
    
    def test_temporal_aggregator_attention(self):
        """Test temporal aggregator with attention method."""
        aggregator = TemporalAggregator(input_dim=512, method="attention")
        x = torch.randn(2, 16, 512)
        output = aggregator(x)
        assert output.shape == (2, 512)
    
    def test_temporal_aggregator_lstm(self):
        """Test temporal aggregator with LSTM method."""
        aggregator = TemporalAggregator(input_dim=512, method="lstm")
        x = torch.randn(2, 16, 512)
        output = aggregator(x)
        assert output.shape == (2, 512)


class TestModelIntegration:
    """Test model integration."""
    
    def test_model_device_placement(self):
        """Test model device placement."""
        if torch.cuda.is_available():
            model = XceptionForensics(num_classes=2)
            model = model.cuda()
            x = torch.randn(2, 3, 224, 224).cuda()
            output = model(x)
            assert output.device.type == "cuda"
    
    def test_model_gradient_flow(self):
        """Test model gradient flow."""
        model = XceptionForensics(num_classes=2)
        x = torch.randn(2, 3, 224, 224, requires_grad=True)
        output = model(x)
        loss = output.sum()
        loss.backward()
        assert x.grad is not None
    
    def test_model_eval_mode(self):
        """Test model evaluation mode."""
        model = XceptionForensics(num_classes=2)
        model.eval()
        x = torch.randn(2, 3, 224, 224)
        with torch.no_grad():
            output = model(x)
        assert output.requires_grad == False


@pytest.mark.parametrize("model_class", [
    XceptionForensics,
    ViTForensics,
    ResNetFrequencyForensics,
])
def test_model_consistency(model_class):
    """Test model consistency across different inputs."""
    model = model_class(num_classes=2)
    model.eval()
    
    # Test with different batch sizes
    for batch_size in [1, 2, 4]:
        x = torch.randn(batch_size, 3, 224, 224)
        output = model(x)
        assert output.shape == (batch_size, 2)
        assert torch.isfinite(output).all()


@pytest.mark.parametrize("input_size", [(224, 224), (256, 256), (192, 192)])
def test_model_input_sizes(input_size):
    """Test model with different input sizes."""
    model = XceptionForensics(num_classes=2)
    x = torch.randn(2, 3, *input_size)
    output = model(x)
    assert output.shape == (2, 2)


def test_model_memory_usage():
    """Test model memory usage."""
    model = XceptionForensics(num_classes=2)
    
    # Test memory usage with different batch sizes
    for batch_size in [1, 2, 4, 8]:
        x = torch.randn(batch_size, 3, 224, 224)
        output = model(x)
        
        # Check that output is finite
        assert torch.isfinite(output).all()
        
        # Clear cache
        del x, output
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
