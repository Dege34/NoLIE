"""
Inference module for deepfake detection.
"""

from .predictor import DeepfakePredictor
from .models import load_model, get_available_models

__all__ = [
    "DeepfakePredictor",
    "load_model", 
    "get_available_models"
]
