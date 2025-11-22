"""
Model loading utilities for inference.
"""

import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, Any, Optional, Union
import logging

from ..models.xception import create_xception_model

logger = logging.getLogger(__name__)

# Available model configurations
AVAILABLE_MODELS = {
    "xception": {
        "class": "XceptionDeepfakeDetector",
        "module": "deepfake_forensics.models.xception",
        "function": "create_xception_model",
        "default_params": {
            "num_classes": 2,
            "input_size": 224,
            "dropout_rate": 0.5
        }
    }
}


def get_available_models() -> Dict[str, Dict[str, Any]]:
    """Get list of available models."""
    return AVAILABLE_MODELS.copy()


def load_model(
    model_name: str = "xception",
    model_path: Optional[Union[str, Path]] = None,
    device: str = "cpu",
    **kwargs
) -> nn.Module:
    """
    Load a deepfake detection model.
    
    Args:
        model_name: Name of the model to load
        model_path: Path to saved model weights (optional)
        device: Device to load model on ('cpu' or 'cuda')
        **kwargs: Additional model parameters
    
    Returns:
        Loaded model
    """
    if model_name not in AVAILABLE_MODELS:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(AVAILABLE_MODELS.keys())}")
    
    model_config = AVAILABLE_MODELS[model_name]
    
    # Get model parameters
    params = model_config["default_params"].copy()
    params.update(kwargs)
    
    # Create model
    if model_name == "xception":
        model = create_xception_model(**params)
    else:
        raise ValueError(f"Model creation not implemented for: {model_name}")
    
    # Load weights if provided
    if model_path:
        model_path = Path(model_path)
        if model_path.exists():
            logger.info(f"Loading model weights from {model_path}")
            checkpoint = torch.load(model_path, map_location=device)
            
            # Handle different checkpoint formats
            if isinstance(checkpoint, dict):
                if "model_state_dict" in checkpoint:
                    state_dict = checkpoint["model_state_dict"]
                elif "state_dict" in checkpoint:
                    state_dict = checkpoint["state_dict"]
                else:
                    state_dict = checkpoint
            else:
                state_dict = checkpoint
            
            model.load_state_dict(state_dict, strict=False)
            logger.info("Model weights loaded successfully")
        else:
            logger.warning(f"Model weights not found at {model_path}, using random weights")
    
    # Move to device
    model = model.to(device)
    model.eval()
    
    logger.info(f"Model {model_name} loaded on {device}")
    return model


def create_mock_model(device: str = "cpu") -> nn.Module:
    """
    Create a mock model for testing purposes.
    
    This model always returns random predictions for demonstration.
    """
    class MockModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 1, 1)
            self.fc = nn.Linear(224 * 224, 2)
        
        def forward(self, x):
            # Random prediction based on image hash
            batch_size = x.shape[0]
            # Use image statistics to create deterministic "random" predictions
            img_mean = x.mean(dim=[1, 2, 3])
            img_std = x.std(dim=[1, 2, 3])
            
            # Create pseudo-random scores based on image content
            fake_score = torch.sigmoid(img_mean.mean(dim=1) * 10 + img_std.mean(dim=1) * 5)
            real_score = 1 - fake_score
            
            # Add some noise
            noise = torch.randn_like(fake_score) * 0.1
            fake_score = torch.clamp(fake_score + noise, 0, 1)
            real_score = 1 - fake_score
            
            # Stack scores
            scores = torch.stack([real_score, fake_score], dim=1)
            return scores * 10  # Scale to logits
        
        def predict_proba(self, x):
            with torch.no_grad():
                logits = self.forward(x)
                return torch.softmax(logits, dim=1)
        
        def predict(self, x):
            with torch.no_grad():
                logits = self.forward(x)
                return torch.argmax(logits, dim=1)
    
    model = MockModel().to(device)
    model.eval()
    return model
