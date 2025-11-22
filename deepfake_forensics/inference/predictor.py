"""
Main predictor class for deepfake detection.
"""

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import cv2
from pathlib import Path
from typing import Union, Dict, Any, List, Optional, Tuple
import logging
from dataclasses import dataclass

from .models import load_model, create_mock_model
from ..data.transforms import get_inference_transforms

logger = logging.getLogger(__name__)


@dataclass
class PredictionResult:
    """Result of a deepfake prediction."""
    score: float  # Deepfake probability (0-1)
    label: str    # "Real" or "Deepfake"
    confidence: float  # Confidence in prediction
    per_frame_scores: Optional[List[float]] = None  # For video
    explanation_assets: Optional[Dict[str, Any]] = None  # Heatmaps, etc.


class DeepfakePredictor:
    """
    Main class for deepfake detection inference.
    """
    
    def __init__(
        self,
        model_name: str = "xception",
        model_path: Optional[Union[str, Path]] = None,
        device: str = "cpu",
        use_mock: bool = False,
        **model_kwargs
    ):
        """
        Initialize the predictor.
        
        Args:
            model_name: Name of the model to use
            model_path: Path to model weights
            device: Device to run inference on
            use_mock: Whether to use mock model for testing
            **model_kwargs: Additional model parameters
        """
        self.device = device
        self.model_name = model_name
        self.use_mock = use_mock
        
        # Load model
        if use_mock:
            logger.info("Using mock model for testing")
            self.model = create_mock_model(device)
        else:
            self.model = load_model(
                model_name=model_name,
                model_path=model_path,
                device=device,
                **model_kwargs
            )
        
        # Get transforms
        self.transforms = get_inference_transforms()
        
        logger.info(f"Predictor initialized with {model_name} on {device}")
    
    def preprocess_image(self, image: Union[str, Path, Image.Image, np.ndarray]) -> torch.Tensor:
        """
        Preprocess an image for inference.
        
        Args:
            image: Input image (path, PIL Image, or numpy array)
        
        Returns:
            Preprocessed tensor
        """
        # Load image if path provided
        if isinstance(image, (str, Path)):
            image = Image.open(image).convert('RGB')
        elif isinstance(image, np.ndarray):
            if image.shape[2] == 3:  # BGR to RGB
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(image)
        
        # Apply transforms
        tensor = self.transforms(image)
        
        # Add batch dimension
        if tensor.dim() == 3:
            tensor = tensor.unsqueeze(0)
        
        return tensor.to(self.device)
    
    def predict_image(self, image: Union[str, Path, Image.Image, np.ndarray]) -> PredictionResult:
        """
        Predict if an image is a deepfake.
        
        Args:
            image: Input image
        
        Returns:
            Prediction result
        """
        # Preprocess
        tensor = self.preprocess_image(image)
        
        # Predict
        with torch.no_grad():
            if hasattr(self.model, 'predict_proba'):
                probs = self.model.predict_proba(tensor)
            else:
                logits = self.model(tensor)
                probs = F.softmax(logits, dim=1)
        
        # Extract scores
        probs_np = probs.cpu().numpy()[0]
        real_prob = probs_np[0]
        fake_prob = probs_np[1]
        
        # Determine label and confidence
        if fake_prob > 0.6:
            label = "Deepfake"
            confidence = fake_prob
        elif fake_prob < 0.4:
            label = "Real"
            confidence = real_prob
        else:
            label = "Uncertain"
            confidence = max(real_prob, fake_prob)
        
        return PredictionResult(
            score=fake_prob,
            label=label,
            confidence=confidence
        )
    
    def predict_video(
        self, 
        video_path: Union[str, Path],
        max_frames: int = 30,
        frame_interval: int = 1
    ) -> PredictionResult:
        """
        Predict if a video contains deepfakes.
        
        Args:
            video_path: Path to video file
            max_frames: Maximum number of frames to process
            frame_interval: Interval between frames to process
        
        Returns:
            Prediction result with per-frame scores
        """
        video_path = Path(video_path)
        
        # Extract frames
        cap = cv2.VideoCapture(str(video_path))
        frames = []
        frame_count = 0
        
        while len(frames) < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_count % frame_interval == 0:
                frames.append(frame)
            frame_count += 1
        
        cap.release()
        
        if not frames:
            raise ValueError(f"Could not extract frames from {video_path}")
        
        # Predict each frame
        frame_scores = []
        frame_predictions = []
        
        for frame in frames:
            pred = self.predict_image(frame)
            frame_scores.append(pred.score)
            frame_predictions.append(pred)
        
        # Aggregate results
        avg_score = np.mean(frame_scores)
        
        # Determine overall label
        if avg_score > 0.6:
            label = "Deepfake"
            confidence = avg_score
        elif avg_score < 0.4:
            label = "Real"
            confidence = 1 - avg_score
        else:
            label = "Uncertain"
            confidence = max(avg_score, 1 - avg_score)
        
        return PredictionResult(
            score=avg_score,
            label=label,
            confidence=confidence,
            per_frame_scores=frame_scores
        )
    
    def predict(
        self, 
        input_path: Union[str, Path],
        **kwargs
    ) -> PredictionResult:
        """
        Predict deepfake for image or video.
        
        Args:
            input_path: Path to input file
            **kwargs: Additional arguments
        
        Returns:
            Prediction result
        """
        input_path = Path(input_path)
        
        # Check file type
        video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.webm'}
        
        if input_path.suffix.lower() in video_extensions:
            return self.predict_video(input_path, **kwargs)
        else:
            return self.predict_image(input_path)


# Example usage
if __name__ == "__main__":
    # Create predictor
    predictor = DeepfakePredictor(use_mock=True)
    
    # Test with dummy image
    dummy_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    result = predictor.predict_image(dummy_image)
    
    print(f"Prediction: {result.label}")
    print(f"Score: {result.score:.3f}")
    print(f"Confidence: {result.confidence:.3f}")
