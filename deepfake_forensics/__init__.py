"""
Deepfake Forensics: Production-grade deepfake detection with explainability and robustness.

This package provides a comprehensive toolkit for detecting deepfake content in images and videos,
with support for multiple model architectures, explainability methods, and robustness testing.
"""

__version__ = "0.1.0"
__author__ = "Deepfake Forensics Team"
__email__ = "team@deepfake-forensics.com"

# Import main classes for easy access
from .models import (
    XceptionDeepfakeDetector,
    VisionTransformer,
    ResNetFreq,
    AudioVisualModel,
)

# Import CLI and API
from .cli import app as cli_app

__all__ = [
    # Models
    "XceptionDeepfakeDetector",
    "VisionTransformer", 
    "ResNetFreq",
    "AudioVisualModel",
    # CLI
    "cli_app",
]