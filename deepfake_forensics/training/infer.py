"""
Inference pipeline for deepfake detection.

Provides comprehensive inference functionality for both images and videos
with temporal aggregation, explainability, and robustness testing.
"""

import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, List, Union, Tuple
import cv2
import json
import logging
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns

from .module import DeepfakeDetector, create_detector
from ..data import get_transforms, get_video_transforms
from ..data.video_reader import VideoReader, create_video_reader
from ..utils.logging import get_logger
from ..utils.io import load_config, save_json
from ..explain.gradcam import GradCAM
from ..explain.attribution import IntegratedGradients

logger = get_logger(__name__)


class InferencePipeline:
    """
    Comprehensive inference pipeline for deepfake detection.
    """
    
    def __init__(
        self,
        model_path: str,
        model_name: str = "xception",
        device: str = "auto",
        config: Optional[Dict[str, Any]] = None,
        **model_kwargs,
    ):
        """
        Initialize inference pipeline.
        
        Args:
            model_path: Path to saved model
            model_name: Model architecture name
            device: Device to run on
            config: Configuration dictionary
            **model_kwargs: Model arguments
        """
        self.model_path = model_path
        self.model_name = model_name
        self.device = device
        self.config = config or {}
        self.model_kwargs = model_kwargs
        
        # Load model
        self.detector = create_detector(
            model_path=model_path,
            model_name=model_name,
            device=device,
            **model_kwargs,
        )
        
        # Get transforms
        self.image_transform, _ = get_transforms(
            image_size=self.config.get("data", {}).get("image_size", 224),
        )
        
        self.video_transform, _ = get_video_transforms(
            image_size=self.config.get("data", {}).get("image_size", 224),
            max_frames=self.config.get("data", {}).get("max_frames", 16),
        )
        
        # Video reader
        self.video_reader = create_video_reader(
            max_frames=self.config.get("data", {}).get("max_frames", 16),
            fps=self.config.get("data", {}).get("fps", 8),
            image_size=self.config.get("data", {}).get("image_size", 224),
        )
        
        # Explainability
        self.explain_config = self.config.get("explain", {})
        if self.explain_config.get("enabled", False):
            self._setup_explainability()
    
    def _setup_explainability(self) -> None:
        """Set up explainability methods."""
        method = self.explain_config.get("method", "gradcam")
        
        if method == "gradcam":
            self.explainer = GradCAM(
                model=self.detector.model,
                target_layer=self.explain_config.get("target_layer", "auto"),
            )
        elif method == "integrated_gradients":
            self.explainer = IntegratedGradients(
                model=self.detector.model,
                steps=self.explain_config.get("steps", 50),
            )
        else:
            logger.warning(f"Unknown explainability method: {method}")
            self.explainer = None
    
    def predict_image(
        self,
        image_path: Union[str, Path],
        return_attention: bool = False,
        return_explanation: bool = False,
    ) -> Dict[str, Any]:
        """
        Predict deepfake probability for a single image.
        
        Args:
            image_path: Path to image file
            return_attention: Whether to return attention maps
            return_explanation: Whether to return explanations
            
        Returns:
            Dictionary containing predictions and metadata
        """
        # Load image
        image = self._load_image(image_path)
        
        # Preprocess
        image_tensor = self._preprocess_image(image)
        
        # Predict
        results = self.detector.predict(
            image_tensor,
            return_probabilities=True,
            return_attention=return_attention,
        )
        
        # Get explanation if requested
        explanation = None
        if return_explanation and self.explainer is not None:
            explanation = self._get_explanation(image_tensor)
        
        # Prepare output
        output = {
            "image_path": str(image_path),
            "prediction": results["predictions"].item(),
            "probability": results["probabilities"].item(),
            "confidence": float(torch.max(results["probabilities"]).item()),
        }
        
        if return_attention and "attention" in results:
            output["attention"] = results["attention"].cpu().numpy()
        
        if explanation is not None:
            output["explanation"] = explanation
        
        return output
    
    def predict_video(
        self,
        video_path: Union[str, Path],
        temporal_aggregation: str = "mean",
        return_frame_predictions: bool = False,
        return_attention: bool = False,
        return_explanation: bool = False,
    ) -> Dict[str, Any]:
        """
        Predict deepfake probability for a video.
        
        Args:
            video_path: Path to video file
            temporal_aggregation: Temporal aggregation method
            return_frame_predictions: Whether to return frame-level predictions
            return_attention: Whether to return attention maps
            return_explanation: Whether to return explanations
            
        Returns:
            Dictionary containing predictions and metadata
        """
        # Load video
        video_frames = self._load_video(video_path)
        
        # Preprocess
        video_tensor = self._preprocess_video(video_frames)
        
        # Predict
        results = self.detector.predict(
            video_tensor,
            return_probabilities=True,
            return_attention=return_attention,
        )
        
        # Temporal aggregation
        if temporal_aggregation == "mean":
            final_prediction = torch.mean(results["predictions"].float()).item()
            final_probability = torch.mean(results["probabilities"]).item()
        elif temporal_aggregation == "majority":
            final_prediction = torch.mode(results["predictions"]).values.item()
            final_probability = torch.mean(results["probabilities"]).item()
        elif temporal_aggregation == "attention":
            # Use attention-weighted aggregation
            attention_weights = F.softmax(results["probabilities"], dim=0)
            final_prediction = torch.sum(attention_weights * results["predictions"].float()).item()
            final_probability = torch.sum(attention_weights * results["probabilities"]).item()
        else:
            raise ValueError(f"Unknown temporal aggregation: {temporal_aggregation}")
        
        # Get explanation if requested
        explanation = None
        if return_explanation and self.explainer is not None:
            explanation = self._get_explanation(video_tensor)
        
        # Prepare output
        output = {
            "video_path": str(video_path),
            "prediction": final_prediction,
            "probability": final_probability,
            "confidence": float(torch.max(results["probabilities"]).item()),
            "temporal_aggregation": temporal_aggregation,
        }
        
        if return_frame_predictions:
            output["frame_predictions"] = results["predictions"].cpu().numpy().tolist()
            output["frame_probabilities"] = results["probabilities"].cpu().numpy().tolist()
        
        if return_attention and "attention" in results:
            output["attention"] = results["attention"].cpu().numpy()
        
        if explanation is not None:
            output["explanation"] = explanation
        
        return output
    
    def predict_batch(
        self,
        input_paths: List[Union[str, Path]],
        batch_size: int = 32,
        return_attention: bool = False,
        return_explanation: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Predict deepfake probability for a batch of inputs.
        
        Args:
            input_paths: List of input file paths
            batch_size: Batch size for processing
            return_attention: Whether to return attention maps
            return_explanation: Whether to return explanations
            
        Returns:
            List of prediction dictionaries
        """
        results = []
        
        for i in range(0, len(input_paths), batch_size):
            batch_paths = input_paths[i:i + batch_size]
            
            # Load and preprocess batch
            batch_tensors = []
            for path in batch_paths:
                if self._is_video_file(path):
                    video_frames = self._load_video(path)
                    tensor = self._preprocess_video(video_frames)
                else:
                    image = self._load_image(path)
                    tensor = self._preprocess_image(image)
                batch_tensors.append(tensor)
            
            # Stack tensors
            batch_tensor = torch.stack(batch_tensors, dim=0)
            
            # Predict
            batch_results = self.detector.predict(
                batch_tensor,
                return_probabilities=True,
                return_attention=return_attention,
            )
            
            # Process results
            for j, path in enumerate(batch_paths):
                result = {
                    "input_path": str(path),
                    "prediction": batch_results["predictions"][j].item(),
                    "probability": batch_results["probabilities"][j].item(),
                    "confidence": float(torch.max(batch_results["probabilities"][j]).item()),
                }
                
                if return_attention and "attention" in batch_results:
                    result["attention"] = batch_results["attention"][j].cpu().numpy()
                
                if return_explanation and self.explainer is not None:
                    result["explanation"] = self._get_explanation(batch_tensor[j:j+1])
                
                results.append(result)
        
        return results
    
    def _load_image(self, image_path: Union[str, Path]) -> np.ndarray:
        """Load image from file."""
        image_path = Path(image_path)
        
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        # Load image
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        return image
    
    def _load_video(self, video_path: Union[str, Path]) -> np.ndarray:
        """Load video from file."""
        video_path = Path(video_path)
        
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")
        
        # Load video using video reader
        video_frames = self.video_reader.read_video(video_path)
        
        return video_frames
    
    def _preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        """Preprocess image for inference."""
        # Apply transforms
        transformed = self.image_transform(image=image)
        image_tensor = transformed["image"]
        
        # Add batch dimension
        image_tensor = image_tensor.unsqueeze(0)
        
        return image_tensor
    
    def _preprocess_video(self, video_frames: np.ndarray) -> torch.Tensor:
        """Preprocess video for inference."""
        # Apply transforms
        transformed = self.video_transform(video_frames)
        
        # Add batch dimension
        video_tensor = transformed.unsqueeze(0)
        
        return video_tensor
    
    def _get_explanation(self, x: torch.Tensor) -> Dict[str, Any]:
        """Get explanation for input."""
        if self.explainer is None:
            return None
        
        try:
            explanation = self.explainer.explain(x)
            return explanation
        except Exception as e:
            logger.warning(f"Failed to get explanation: {e}")
            return None
    
    def _is_video_file(self, path: Union[str, Path]) -> bool:
        """Check if file is a video."""
        video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm'}
        return Path(path).suffix.lower() in video_extensions
    
    def save_results(
        self,
        results: List[Dict[str, Any]],
        output_path: Union[str, Path],
        format: str = "json",
    ) -> None:
        """Save prediction results."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if format == "json":
            save_json(results, output_path)
        elif format == "csv":
            import pandas as pd
            df = pd.DataFrame(results)
            df.to_csv(output_path, index=False)
        else:
            raise ValueError(f"Unknown format: {format}")
        
        logger.info(f"Results saved to {output_path}")


def predict_single(
    input_path: Union[str, Path],
    model_path: str,
    model_name: str = "xception",
    device: str = "auto",
    config: Optional[Dict[str, Any]] = None,
    **kwargs,
) -> Dict[str, Any]:
    """
    Predict deepfake probability for a single input.
    
    Args:
        input_path: Path to input file
        model_path: Path to saved model
        model_name: Model architecture name
        device: Device to run on
        config: Configuration dictionary
        **kwargs: Additional arguments
        
    Returns:
        Prediction dictionary
    """
    # Create inference pipeline
    pipeline = InferencePipeline(
        model_path=model_path,
        model_name=model_name,
        device=device,
        config=config,
        **kwargs,
    )
    
    # Predict
    if pipeline._is_video_file(input_path):
        return pipeline.predict_video(input_path)
    else:
        return pipeline.predict_image(input_path)


def predict_batch(
    input_paths: List[Union[str, Path]],
    model_path: str,
    model_name: str = "xception",
    device: str = "auto",
    config: Optional[Dict[str, Any]] = None,
    batch_size: int = 32,
    **kwargs,
) -> List[Dict[str, Any]]:
    """
    Predict deepfake probability for a batch of inputs.
    
    Args:
        input_paths: List of input file paths
        model_path: Path to saved model
        model_name: Model architecture name
        device: Device to run on
        config: Configuration dictionary
        batch_size: Batch size for processing
        **kwargs: Additional arguments
        
    Returns:
        List of prediction dictionaries
    """
    # Create inference pipeline
    pipeline = InferencePipeline(
        model_path=model_path,
        model_name=model_name,
        device=device,
        config=config,
        **kwargs,
    )
    
    # Predict
    return pipeline.predict_batch(
        input_paths=input_paths,
        batch_size=batch_size,
    )


def main():
    """Main inference function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run deepfake detection inference")
    parser.add_argument("--input", type=str, required=True, help="Path to input file or directory")
    parser.add_argument("--model", type=str, required=True, help="Path to saved model")
    parser.add_argument("--output", type=str, help="Path to output file")
    parser.add_argument("--config", type=str, help="Path to config file")
    parser.add_argument("--model_name", type=str, default="xception", help="Model architecture name")
    parser.add_argument("--device", type=str, default="auto", help="Device to run on")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--format", type=str, default="json", help="Output format")
    
    args = parser.parse_args()
    
    # Load configuration
    config = None
    if args.config:
        config = load_config(args.config)
    
    # Get input paths
    input_path = Path(args.input)
    if input_path.is_file():
        input_paths = [input_path]
    else:
        # Get all files in directory
        input_paths = []
        for ext in ['.jpg', '.jpeg', '.png', '.mp4', '.avi', '.mov']:
            input_paths.extend(input_path.glob(f"**/*{ext}"))
    
    # Run inference
    if len(input_paths) == 1:
        results = predict_single(
            input_paths[0],
            model_path=args.model,
            model_name=args.model_name,
            device=args.device,
            config=config,
        )
        results = [results]
    else:
        results = predict_batch(
            input_paths,
            model_path=args.model,
            model_name=args.model_name,
            device=args.device,
            config=config,
            batch_size=args.batch_size,
        )
    
    # Save results
    if args.output:
        pipeline = InferencePipeline(
            model_path=args.model,
            model_name=args.model_name,
            device=args.device,
            config=config,
        )
        pipeline.save_results(results, args.output, args.format)
    
    # Print results
    for result in results:
        print(f"Input: {result['input_path']}")
        print(f"Prediction: {result['prediction']}")
        print(f"Probability: {result['probability']:.3f}")
        print(f"Confidence: {result['confidence']:.3f}")
        print("-" * 50)


if __name__ == "__main__":
    main()
