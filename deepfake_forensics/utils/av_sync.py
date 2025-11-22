"""
Audio-visual synchronization utilities for deepfake detection.

Provides functions to check lip-sync consistency and audio-visual alignment
as additional signals for deepfake detection.
"""

import numpy as np
import torch
import torchaudio
from typing import Dict, List, Optional, Tuple, Union
import cv2
import logging
from pathlib import Path
import librosa
from scipy import signal
from scipy.stats import pearsonr

logger = logging.getLogger(__name__)


class AudioVisualSyncChecker:
    """
    Check audio-visual synchronization for deepfake detection.
    """
    
    def __init__(
        self,
        sample_rate: int = 16000,
        hop_length: int = 512,
        n_mfcc: int = 13,
        window_size: int = 25,
        hop_size: int = 10,
    ):
        """
        Initialize audio-visual sync checker.
        
        Args:
            sample_rate: Audio sample rate
            hop_length: STFT hop length
            n_mfcc: Number of MFCC coefficients
            window_size: Window size for visual features (frames)
            hop_size: Hop size for visual features (frames)
        """
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.n_mfcc = n_mfcc
        self.window_size = window_size
        self.hop_size = hop_size
        
        # Initialize audio processing
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            hop_length=hop_length,
            n_mels=80,
        )
        
        self.mfcc_transform = torchaudio.transforms.MFCC(
            sample_rate=sample_rate,
            n_mfcc=n_mfcc,
            melkwargs={
                "hop_length": hop_length,
                "n_mels": 80,
            }
        )
    
    def extract_audio_features(
        self,
        audio_path: Union[str, Path],
        start_time: float = 0.0,
        duration: Optional[float] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Extract audio features from file.
        
        Args:
            audio_path: Path to audio file
            start_time: Start time in seconds
            duration: Duration in seconds (None for full audio)
            
        Returns:
            Dictionary containing audio features
        """
        try:
            # Load audio
            waveform, sr = torchaudio.load(
                audio_path,
                frame_offset=int(start_time * self.sample_rate),
                num_frames=int(duration * self.sample_rate) if duration else -1,
            )
            
            # Resample if necessary
            if sr != self.sample_rate:
                resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
                waveform = resampler(waveform)
            
            # Convert to mono
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            
            # Extract features
            features = {}
            
            # MFCC features
            mfcc = self.mfcc_transform(waveform)
            features["mfcc"] = mfcc.squeeze().numpy()
            
            # Mel spectrogram
            mel_spec = self.mel_transform(waveform)
            features["mel_spectrogram"] = mel_spec.squeeze().numpy()
            
            # Energy
            energy = torch.sum(mel_spec, dim=1)
            features["energy"] = energy.squeeze().numpy()
            
            # Zero crossing rate
            zcr = self._compute_zcr(waveform.squeeze().numpy())
            features["zcr"] = zcr
            
            # Spectral centroid
            spectral_centroid = self._compute_spectral_centroid(mel_spec.squeeze().numpy())
            features["spectral_centroid"] = spectral_centroid
            
            return features
            
        except Exception as e:
            logger.error(f"Failed to extract audio features from {audio_path}: {e}")
            return {}
    
    def extract_visual_features(
        self,
        video_path: Union[str, Path],
        start_frame: int = 0,
        num_frames: Optional[int] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Extract visual features from video.
        
        Args:
            video_path: Path to video file
            start_frame: Starting frame index
            num_frames: Number of frames to process
            
        Returns:
            Dictionary containing visual features
        """
        try:
            cap = cv2.VideoCapture(str(video_path))
            
            if not cap.isOpened():
                raise ValueError(f"Could not open video: {video_path}")
            
            # Set frame position
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            
            frames = []
            frame_count = 0
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                if num_frames and frame_count >= num_frames:
                    break
                
                # Convert to RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
                frame_count += 1
            
            cap.release()
            
            if not frames:
                return {}
            
            frames = np.array(frames)
            
            # Extract features
            features = {}
            
            # Lip region detection and features
            lip_features = self._extract_lip_features(frames)
            if lip_features is not None:
                features["lip_features"] = lip_features
            
            # Optical flow
            flow_features = self._extract_optical_flow(frames)
            if flow_features is not None:
                features["optical_flow"] = flow_features
            
            # Face landmarks
            landmark_features = self._extract_face_landmarks(frames)
            if landmark_features is not None:
                features["face_landmarks"] = landmark_features
            
            return features
            
        except Exception as e:
            logger.error(f"Failed to extract visual features from {video_path}: {e}")
            return {}
    
    def compute_sync_score(
        self,
        audio_features: Dict[str, np.ndarray],
        visual_features: Dict[str, np.ndarray],
        method: str = "correlation",
    ) -> Dict[str, float]:
        """
        Compute audio-visual synchronization score.
        
        Args:
            audio_features: Audio features dictionary
            visual_features: Visual features dictionary
            method: Synchronization method ('correlation', 'dtw', 'cross_correlation')
            
        Returns:
            Dictionary containing sync scores
        """
        scores = {}
        
        # MFCC vs lip features
        if "mfcc" in audio_features and "lip_features" in visual_features:
            mfcc = audio_features["mfcc"]
            lip_features = visual_features["lip_features"]
            
            # Align features temporally
            mfcc_aligned, lip_aligned = self._align_features(mfcc, lip_features)
            
            if method == "correlation":
                score = self._compute_correlation_score(mfcc_aligned, lip_aligned)
            elif method == "dtw":
                score = self._compute_dtw_score(mfcc_aligned, lip_aligned)
            elif method == "cross_correlation":
                score = self._compute_cross_correlation_score(mfcc_aligned, lip_aligned)
            else:
                raise ValueError(f"Unknown sync method: {method}")
            
            scores["mfcc_lip_sync"] = score
        
        # Energy vs optical flow
        if "energy" in audio_features and "optical_flow" in visual_features:
            energy = audio_features["energy"]
            flow = visual_features["optical_flow"]
            
            energy_aligned, flow_aligned = self._align_features(energy, flow)
            score = self._compute_correlation_score(energy_aligned, flow_aligned)
            scores["energy_flow_sync"] = score
        
        # Spectral centroid vs face landmarks
        if "spectral_centroid" in audio_features and "face_landmarks" in visual_features:
            centroid = audio_features["spectral_centroid"]
            landmarks = visual_features["face_landmarks"]
            
            centroid_aligned, landmarks_aligned = self._align_features(centroid, landmarks)
            score = self._compute_correlation_score(centroid_aligned, landmarks_aligned)
            scores["centroid_landmark_sync"] = score
        
        # Overall sync score
        if scores:
            scores["overall_sync"] = np.mean(list(scores.values()))
        else:
            scores["overall_sync"] = 0.0
        
        return scores
    
    def _compute_zcr(self, waveform: np.ndarray) -> np.ndarray:
        """Compute zero crossing rate."""
        zcr = librosa.feature.zero_crossing_rate(waveform, hop_length=self.hop_length)
        return zcr.squeeze()
    
    def _compute_spectral_centroid(self, mel_spec: np.ndarray) -> np.ndarray:
        """Compute spectral centroid."""
        centroid = librosa.feature.spectral_centroid(S=mel_spec, hop_length=self.hop_length)
        return centroid.squeeze()
    
    def _extract_lip_features(self, frames: np.ndarray) -> Optional[np.ndarray]:
        """Extract lip region features."""
        try:
            # This is a simplified implementation
            # In practice, you would use face detection and landmark detection
            # to extract the lip region and compute features
            
            # For now, return a dummy feature
            # In real implementation, extract actual lip features
            return np.random.randn(len(frames), 64)  # Dummy features
            
        except Exception as e:
            logger.warning(f"Failed to extract lip features: {e}")
            return None
    
    def _extract_optical_flow(self, frames: np.ndarray) -> Optional[np.ndarray]:
        """Extract optical flow features."""
        try:
            if len(frames) < 2:
                return None
            
            # Convert to grayscale
            gray_frames = [cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY) for frame in frames]
            
            flows = []
            for i in range(len(gray_frames) - 1):
                flow = cv2.calcOpticalFlowPyrLK(
                    gray_frames[i], gray_frames[i + 1],
                    None, None
                )
                flows.append(flow[0])
            
            if flows:
                return np.array(flows)
            return None
            
        except Exception as e:
            logger.warning(f"Failed to extract optical flow: {e}")
            return None
    
    def _extract_face_landmarks(self, frames: np.ndarray) -> Optional[np.ndarray]:
        """Extract face landmark features."""
        try:
            # This is a simplified implementation
            # In practice, you would use a face landmark detector like MediaPipe
            # or dlib to extract facial landmarks
            
            # For now, return a dummy feature
            # In real implementation, extract actual face landmarks
            return np.random.randn(len(frames), 68, 2)  # Dummy landmarks
            
        except Exception as e:
            logger.warning(f"Failed to extract face landmarks: {e}")
            return None
    
    def _align_features(
        self,
        audio_feat: np.ndarray,
        visual_feat: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Align audio and visual features temporally."""
        # Simple alignment by resampling to the same length
        min_len = min(len(audio_feat), len(visual_feat))
        
        if len(audio_feat) > min_len:
            audio_feat = audio_feat[:min_len]
        if len(visual_feat) > min_len:
            visual_feat = visual_feat[:min_len]
        
        return audio_feat, visual_feat
    
    def _compute_correlation_score(
        self,
        feat1: np.ndarray,
        feat2: np.ndarray,
    ) -> float:
        """Compute correlation score between features."""
        try:
            # Flatten features if they are multi-dimensional
            feat1_flat = feat1.flatten()
            feat2_flat = feat2.flatten()
            
            # Ensure same length
            min_len = min(len(feat1_flat), len(feat2_flat))
            feat1_flat = feat1_flat[:min_len]
            feat2_flat = feat2_flat[:min_len]
            
            # Compute Pearson correlation
            corr, _ = pearsonr(feat1_flat, feat2_flat)
            return float(corr) if not np.isnan(corr) else 0.0
            
        except Exception as e:
            logger.warning(f"Failed to compute correlation: {e}")
            return 0.0
    
    def _compute_dtw_score(
        self,
        feat1: np.ndarray,
        feat2: np.ndarray,
    ) -> float:
        """Compute Dynamic Time Warping score."""
        try:
            # This is a simplified implementation
            # In practice, you would use a proper DTW library
            from scipy.spatial.distance import euclidean
            
            # Simple DTW implementation
            n, m = len(feat1), len(feat2)
            dtw_matrix = np.full((n + 1, m + 1), np.inf)
            dtw_matrix[0, 0] = 0
            
            for i in range(1, n + 1):
                for j in range(1, m + 1):
                    cost = euclidean(feat1[i - 1], feat2[j - 1])
                    dtw_matrix[i, j] = cost + min(
                        dtw_matrix[i - 1, j],
                        dtw_matrix[i, j - 1],
                        dtw_matrix[i - 1, j - 1]
                    )
            
            # Normalize by path length
            path_length = n + m
            return float(dtw_matrix[n, m] / path_length) if path_length > 0 else 0.0
            
        except Exception as e:
            logger.warning(f"Failed to compute DTW score: {e}")
            return 0.0
    
    def _compute_cross_correlation_score(
        self,
        feat1: np.ndarray,
        feat2: np.ndarray,
    ) -> float:
        """Compute cross-correlation score."""
        try:
            # Flatten features
            feat1_flat = feat1.flatten()
            feat2_flat = feat2.flatten()
            
            # Compute cross-correlation
            correlation = signal.correlate(feat1_flat, feat2_flat, mode='full')
            
            # Find maximum correlation
            max_corr = np.max(correlation)
            
            # Normalize by the maximum possible correlation
            norm_factor = np.sqrt(np.sum(feat1_flat**2) * np.sum(feat2_flat**2))
            normalized_corr = max_corr / norm_factor if norm_factor > 0 else 0.0
            
            return float(normalized_corr)
            
        except Exception as e:
            logger.warning(f"Failed to compute cross-correlation: {e}")
            return 0.0


def check_audio_visual_sync(
    video_path: Union[str, Path],
    audio_path: Optional[Union[str, Path]] = None,
    start_time: float = 0.0,
    duration: Optional[float] = None,
    method: str = "correlation",
) -> Dict[str, float]:
    """
    Check audio-visual synchronization for a video file.
    
    Args:
        video_path: Path to video file
        audio_path: Path to audio file (if separate from video)
        start_time: Start time in seconds
        duration: Duration in seconds
        method: Synchronization method
        
    Returns:
        Dictionary containing sync scores
    """
    checker = AudioVisualSyncChecker()
    
    # Extract audio features
    if audio_path:
        audio_features = checker.extract_audio_features(audio_path, start_time, duration)
    else:
        # Extract audio from video
        audio_features = checker.extract_audio_features(video_path, start_time, duration)
    
    # Extract visual features
    visual_features = checker.extract_visual_features(video_path)
    
    # Compute sync scores
    sync_scores = checker.compute_sync_score(audio_features, visual_features, method)
    
    return sync_scores


def batch_sync_check(
    video_paths: List[Union[str, Path]],
    output_file: Optional[Union[str, Path]] = None,
    method: str = "correlation",
) -> Dict[str, Dict[str, float]]:
    """
    Check audio-visual synchronization for multiple videos.
    
    Args:
        video_paths: List of video file paths
        output_file: Optional output file to save results
        method: Synchronization method
        
    Returns:
        Dictionary mapping video paths to sync scores
    """
    results = {}
    
    for video_path in video_paths:
        try:
            sync_scores = check_audio_visual_sync(video_path, method=method)
            results[str(video_path)] = sync_scores
            logger.info(f"Processed {video_path}: sync score = {sync_scores.get('overall_sync', 0.0):.3f}")
        except Exception as e:
            logger.error(f"Failed to process {video_path}: {e}")
            results[str(video_path)] = {"overall_sync": 0.0, "error": str(e)}
    
    # Save results if output file is specified
    if output_file:
        import json
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Saved sync check results to {output_file}")
    
    return results
