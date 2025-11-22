"""
Frequency domain feature extraction utilities for deepfake detection.

Provides functions to extract frequency domain features that can help detect
deepfake artifacts and inconsistencies.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Union
import cv2
import logging
from scipy import fft, fftpack
from scipy.stats import kurtosis, skew
import torchvision.transforms as transforms

logger = logging.getLogger(__name__)


class FFTFeatureExtractor:
    """
    Extract frequency domain features for deepfake detection.
    """
    
    def __init__(
        self,
        image_size: int = 224,
        fft_size: Optional[int] = None,
        overlap_ratio: float = 0.5,
    ):
        """
        Initialize FFT feature extractor.
        
        Args:
            image_size: Input image size
            fft_size: FFT size (default: image_size)
            overlap_ratio: Overlap ratio for windowed FFT
        """
        self.image_size = image_size
        self.fft_size = fft_size or image_size
        self.overlap_ratio = overlap_ratio
        
        # Precompute window function
        self.window = self._create_window()
    
    def extract_features(
        self,
        image: Union[np.ndarray, torch.Tensor],
        features: List[str] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Extract frequency domain features from image.
        
        Args:
            image: Input image (H, W, C) or (C, H, W)
            features: List of features to extract (None for all)
            
        Returns:
            Dictionary containing extracted features
        """
        if features is None:
            features = [
                "magnitude_spectrum",
                "phase_spectrum", 
                "log_magnitude",
                "spectral_centroid",
                "spectral_bandwidth",
                "spectral_rolloff",
                "spectral_flux",
                "mfcc",
                "chroma",
                "zero_crossing_rate",
                "spectral_contrast",
                "spectral_flatness",
                "jpeg_artifacts",
                "prnu_pattern",
            ]
        
        # Convert to numpy if needed
        if isinstance(image, torch.Tensor):
            image = image.detach().cpu().numpy()
        
        # Ensure image is in (H, W, C) format
        if image.ndim == 3 and image.shape[0] in [1, 3]:
            image = np.transpose(image, (1, 2, 0))
        
        # Convert to grayscale if needed
        if image.shape[2] == 3:
            gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray_image = image.squeeze()
        
        # Normalize to [0, 1]
        gray_image = gray_image.astype(np.float32) / 255.0
        
        extracted_features = {}
        
        for feature_name in features:
            try:
                if feature_name == "magnitude_spectrum":
                    extracted_features[feature_name] = self._extract_magnitude_spectrum(gray_image)
                elif feature_name == "phase_spectrum":
                    extracted_features[feature_name] = self._extract_phase_spectrum(gray_image)
                elif feature_name == "log_magnitude":
                    extracted_features[feature_name] = self._extract_log_magnitude(gray_image)
                elif feature_name == "spectral_centroid":
                    extracted_features[feature_name] = self._extract_spectral_centroid(gray_image)
                elif feature_name == "spectral_bandwidth":
                    extracted_features[feature_name] = self._extract_spectral_bandwidth(gray_image)
                elif feature_name == "spectral_rolloff":
                    extracted_features[feature_name] = self._extract_spectral_rolloff(gray_image)
                elif feature_name == "spectral_flux":
                    extracted_features[feature_name] = self._extract_spectral_flux(gray_image)
                elif feature_name == "mfcc":
                    extracted_features[feature_name] = self._extract_mfcc(gray_image)
                elif feature_name == "chroma":
                    extracted_features[feature_name] = self._extract_chroma(gray_image)
                elif feature_name == "zero_crossing_rate":
                    extracted_features[feature_name] = self._extract_zero_crossing_rate(gray_image)
                elif feature_name == "spectral_contrast":
                    extracted_features[feature_name] = self._extract_spectral_contrast(gray_image)
                elif feature_name == "spectral_flatness":
                    extracted_features[feature_name] = self._extract_spectral_flatness(gray_image)
                elif feature_name == "jpeg_artifacts":
                    extracted_features[feature_name] = self._extract_jpeg_artifacts(gray_image)
                elif feature_name == "prnu_pattern":
                    extracted_features[feature_name] = self._extract_prnu_pattern(gray_image)
                else:
                    logger.warning(f"Unknown feature: {feature_name}")
                    
            except Exception as e:
                logger.warning(f"Failed to extract {feature_name}: {e}")
                extracted_features[feature_name] = np.array([])
        
        return extracted_features
    
    def _create_window(self) -> np.ndarray:
        """Create window function for FFT."""
        return np.hanning(self.fft_size)
    
    def _extract_magnitude_spectrum(self, image: np.ndarray) -> np.ndarray:
        """Extract magnitude spectrum."""
        # Apply window
        windowed = image * self.window
        
        # Compute 2D FFT
        fft_result = fft.fft2(windowed)
        magnitude = np.abs(fft_result)
        
        # Shift to center low frequencies
        magnitude = np.fft.fftshift(magnitude)
        
        return magnitude
    
    def _extract_phase_spectrum(self, image: np.ndarray) -> np.ndarray:
        """Extract phase spectrum."""
        # Apply window
        windowed = image * self.window
        
        # Compute 2D FFT
        fft_result = fft.fft2(windowed)
        phase = np.angle(fft_result)
        
        # Shift to center low frequencies
        phase = np.fft.fftshift(phase)
        
        return phase
    
    def _extract_log_magnitude(self, image: np.ndarray) -> np.ndarray:
        """Extract log magnitude spectrum."""
        magnitude = self._extract_magnitude_spectrum(image)
        
        # Add small epsilon to avoid log(0)
        log_magnitude = np.log(magnitude + 1e-8)
        
        return log_magnitude
    
    def _extract_spectral_centroid(self, image: np.ndarray) -> float:
        """Extract spectral centroid."""
        magnitude = self._extract_magnitude_spectrum(image)
        
        # Compute frequency bins
        freqs = np.fft.fftfreq(self.fft_size)
        freqs = np.fft.fftshift(freqs)
        
        # Compute spectral centroid
        centroid = np.sum(freqs * magnitude) / np.sum(magnitude)
        
        return float(centroid)
    
    def _extract_spectral_bandwidth(self, image: np.ndarray) -> float:
        """Extract spectral bandwidth."""
        magnitude = self._extract_magnitude_spectrum(image)
        centroid = self._extract_spectral_centroid(image)
        
        # Compute frequency bins
        freqs = np.fft.fftfreq(self.fft_size)
        freqs = np.fft.fftshift(freqs)
        
        # Compute spectral bandwidth
        bandwidth = np.sqrt(np.sum(((freqs - centroid) ** 2) * magnitude) / np.sum(magnitude))
        
        return float(bandwidth)
    
    def _extract_spectral_rolloff(self, image: np.ndarray, rolloff_threshold: float = 0.85) -> float:
        """Extract spectral rolloff."""
        magnitude = self._extract_magnitude_spectrum(image)
        
        # Compute cumulative sum
        cumsum = np.cumsum(magnitude)
        total_energy = cumsum[-1]
        
        # Find rolloff point
        rolloff_energy = rolloff_threshold * total_energy
        rolloff_idx = np.where(cumsum >= rolloff_energy)[0]
        
        if len(rolloff_idx) > 0:
            rolloff_freq = rolloff_idx[0] / self.fft_size
        else:
            rolloff_freq = 1.0
        
        return float(rolloff_freq)
    
    def _extract_spectral_flux(self, image: np.ndarray) -> float:
        """Extract spectral flux."""
        magnitude = self._extract_magnitude_spectrum(image)
        
        # Compute difference between consecutive frames
        # For single image, compute difference with shifted version
        shifted = np.roll(magnitude, 1, axis=0)
        flux = np.sum(np.abs(magnitude - shifted))
        
        return float(flux)
    
    def _extract_mfcc(self, image: np.ndarray, n_mfcc: int = 13) -> np.ndarray:
        """Extract MFCC features."""
        magnitude = self._extract_magnitude_spectrum(image)
        
        # Apply mel filter bank
        mel_filters = self._create_mel_filter_bank(n_mfcc)
        mel_spectrum = np.dot(mel_filters, magnitude)
        
        # Apply log
        log_mel = np.log(mel_spectrum + 1e-8)
        
        # Apply DCT
        mfcc = fftpack.dct(log_mel, norm='ortho')
        
        return mfcc[:n_mfcc]
    
    def _extract_chroma(self, image: np.ndarray, n_chroma: int = 12) -> np.ndarray:
        """Extract chroma features."""
        magnitude = self._extract_magnitude_spectrum(image)
        
        # Create chroma filter bank
        chroma_filters = self._create_chroma_filter_bank(n_chroma)
        chroma = np.dot(chroma_filters, magnitude)
        
        return chroma
    
    def _extract_zero_crossing_rate(self, image: np.ndarray) -> float:
        """Extract zero crossing rate."""
        # Convert to 1D signal
        signal = image.flatten()
        
        # Compute zero crossings
        zero_crossings = np.sum(np.diff(np.sign(signal)) != 0)
        zcr = zero_crossings / len(signal)
        
        return float(zcr)
    
    def _extract_spectral_contrast(self, image: np.ndarray, n_bands: int = 6) -> np.ndarray:
        """Extract spectral contrast."""
        magnitude = self._extract_magnitude_spectrum(image)
        
        # Divide spectrum into bands
        band_size = len(magnitude) // n_bands
        contrasts = []
        
        for i in range(n_bands):
            start_idx = i * band_size
            end_idx = (i + 1) * band_size if i < n_bands - 1 else len(magnitude)
            
            band = magnitude[start_idx:end_idx]
            
            # Compute contrast as difference between peak and valley
            peak = np.max(band)
            valley = np.min(band)
            contrast = peak - valley
            
            contrasts.append(contrast)
        
        return np.array(contrasts)
    
    def _extract_spectral_flatness(self, image: np.ndarray) -> float:
        """Extract spectral flatness."""
        magnitude = self._extract_magnitude_spectrum(image)
        
        # Compute geometric mean
        geometric_mean = np.exp(np.mean(np.log(magnitude + 1e-8)))
        
        # Compute arithmetic mean
        arithmetic_mean = np.mean(magnitude)
        
        # Compute flatness
        flatness = geometric_mean / arithmetic_mean if arithmetic_mean > 0 else 0.0
        
        return float(flatness)
    
    def _extract_jpeg_artifacts(self, image: np.ndarray) -> np.ndarray:
        """Extract JPEG compression artifacts."""
        # Convert to uint8 for JPEG simulation
        image_uint8 = (image * 255).astype(np.uint8)
        
        # Simulate JPEG compression
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 50]
        _, encoded_img = cv2.imencode('.jpg', image_uint8, encode_param)
        decoded_img = cv2.imdecode(encoded_img, cv2.IMREAD_GRAYSCALE)
        
        # Compute difference
        difference = np.abs(image_uint8.astype(float) - decoded_img.astype(float))
        
        # Extract features from difference
        features = [
            np.mean(difference),
            np.std(difference),
            np.max(difference),
            np.sum(difference > 0.1) / difference.size,  # Percentage of significant differences
        ]
        
        return np.array(features)
    
    def _extract_prnu_pattern(self, image: np.ndarray) -> np.ndarray:
        """Extract PRNU (Photo Response Non-Uniformity) pattern."""
        # Convert to float
        img_float = image.astype(np.float64)
        
        # Apply denoising to get noise pattern
        denoised = cv2.bilateralFilter(img_float, 9, 75, 75)
        noise = img_float - denoised
        
        # Extract PRNU features
        features = [
            np.mean(noise),
            np.std(noise),
            kurtosis(noise.flatten()),
            skew(noise.flatten()),
        ]
        
        return np.array(features)
    
    def _create_mel_filter_bank(self, n_mels: int) -> np.ndarray:
        """Create mel filter bank."""
        # This is a simplified implementation
        # In practice, you would use librosa or similar library
        
        n_fft = self.fft_size
        mel_filters = np.zeros((n_mels, n_fft))
        
        # Create triangular filters
        for i in range(n_mels):
            start = int(i * n_fft / n_mels)
            center = int((i + 1) * n_fft / n_mels)
            end = int((i + 2) * n_fft / n_mels)
            
            if start < n_fft:
                mel_filters[i, start:center] = np.linspace(0, 1, center - start)
            if center < n_fft and end <= n_fft:
                mel_filters[i, center:end] = np.linspace(1, 0, end - center)
        
        return mel_filters
    
    def _create_chroma_filter_bank(self, n_chroma: int) -> np.ndarray:
        """Create chroma filter bank."""
        # This is a simplified implementation
        n_fft = self.fft_size
        chroma_filters = np.zeros((n_chroma, n_fft))
        
        # Create chroma filters
        for i in range(n_chroma):
            start = int(i * n_fft / n_chroma)
            end = int((i + 1) * n_fft / n_chroma)
            
            if start < n_fft:
                chroma_filters[i, start:end] = 1.0
        
        return chroma_filters


def extract_frequency_features(
    image: Union[np.ndarray, torch.Tensor],
    features: Optional[List[str]] = None,
    image_size: int = 224,
) -> Dict[str, np.ndarray]:
    """
    Extract frequency domain features from an image.
    
    Args:
        image: Input image
        features: List of features to extract
        image_size: Image size for processing
        
    Returns:
        Dictionary containing extracted features
    """
    extractor = FFTFeatureExtractor(image_size=image_size)
    return extractor.extract_features(image, features)


def batch_extract_frequency_features(
    images: List[Union[np.ndarray, torch.Tensor]],
    features: Optional[List[str]] = None,
    image_size: int = 224,
) -> List[Dict[str, np.ndarray]]:
    """
    Extract frequency domain features from a batch of images.
    
    Args:
        images: List of input images
        features: List of features to extract
        image_size: Image size for processing
        
    Returns:
        List of dictionaries containing extracted features
    """
    extractor = FFTFeatureExtractor(image_size=image_size)
    results = []
    
    for image in images:
        features_dict = extractor.extract_features(image, features)
        results.append(features_dict)
    
    return results


class FrequencyBranch(nn.Module):
    """
    Neural network branch for processing frequency domain features.
    """
    
    def __init__(
        self,
        input_size: int = 224,
        feature_dim: int = 64,
        hidden_dim: int = 128,
    ):
        """
        Initialize frequency branch.
        
        Args:
            input_size: Input image size
            feature_dim: Output feature dimension
            hidden_dim: Hidden layer dimension
        """
        super().__init__()
        
        self.input_size = input_size
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        
        # Feature extractor
        self.feature_extractor = FFTFeatureExtractor(input_size)
        
        # Neural network layers
        self.fc1 = nn.Linear(input_size * input_size, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, feature_dim)
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through frequency branch.
        
        Args:
            x: Input tensor (B, C, H, W)
            
        Returns:
            Frequency features tensor (B, feature_dim)
        """
        batch_size = x.shape[0]
        
        # Extract frequency features for each image in batch
        features = []
        for i in range(batch_size):
            img = x[i].cpu().numpy()
            img_features = self.feature_extractor.extract_features(img)
            
            # Combine features into a single vector
            combined_features = []
            for feature_name, feature_data in img_features.items():
                if feature_data.size > 0:
                    # Flatten and take mean if multi-dimensional
                    if feature_data.ndim > 1:
                        feature_data = feature_data.flatten()
                    combined_features.extend(feature_data)
            
            features.append(combined_features)
        
        # Pad features to same length
        max_len = max(len(f) for f in features)
        padded_features = []
        for f in features:
            padded = np.pad(f, (0, max_len - len(f)), mode='constant')
            padded_features.append(padded)
        
        # Convert to tensor
        x = torch.tensor(padded_features, dtype=torch.float32, device=x.device)
        
        # Pass through neural network
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        x = self.fc3(x)
        
        return x
