"""
Utility functions and helpers for the deepfake forensics package.
"""

from .logging import setup_logging
from .seeds import set_all_seeds
from .io import load_json, save_json, load_yaml, save_yaml, create_dir_if_not_exists
from .metrics import compute_binary_metrics, plot_roc_curve, plot_pr_curve, plot_confusion_matrix
from .av_sync import extract_audio, compute_mfccs, detect_lip_landmarks, compute_sync_error
from .fft_features import compute_fft_features, extract_jpeg_artifacts, extract_prnu_noise

__all__ = [
    # Logging
    "setup_logging",
    # Seeds
    "set_all_seeds",
    # I/O
    "load_json",
    "save_json", 
    "load_yaml",
    "save_yaml",
    "create_dir_if_not_exists",
    # Metrics
    "compute_binary_metrics",
    "plot_roc_curve",
    "plot_pr_curve",
    "plot_confusion_matrix",
    # Audio-visual sync
    "extract_audio",
    "compute_mfccs",
    "detect_lip_landmarks",
    "compute_sync_error",
    # FFT features
    "compute_fft_features",
    "extract_jpeg_artifacts",
    "extract_prnu_noise",
]