"""
Model architectures for deepfake detection.
"""

from .xception import XceptionDeepfakeDetector
from .vit import VisionTransformer
from .resnet_freq import ResNetFreq
from .audio_visual import AudioVisualModel
from .heads import BinaryClassificationHead, LateFusionHead

__all__ = [
    "XceptionDeepfakeDetector",
    "VisionTransformer",
    "ResNetFreq", 
    "AudioVisualModel",
    "BinaryClassificationHead",
    "LateFusionHead",
]