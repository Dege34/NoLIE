"""
Pydantic schemas for API requests and responses.

Defines the data models for API endpoints including request/response
schemas for deepfake detection predictions.
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any, Union
from enum import Enum


class ModelType(str, Enum):
    """Model type enumeration."""
    XCEPTION = "xception"
    VIT = "vit"
    RESNET_FREQ = "resnet_freq"
    AUDIO_VISUAL = "audio_visual"


class PredictionRequest(BaseModel):
    """Request schema for prediction endpoint."""
    
    # Input data
    input_path: Optional[str] = Field(None, description="Path to input file")
    input_data: Optional[bytes] = Field(None, description="Input data as bytes")
    
    # Model configuration
    model_name: str = Field("xception", description="Model architecture name")
    confidence_threshold: float = Field(0.5, description="Confidence threshold for prediction")
    
    # Processing options
    return_attention: bool = Field(False, description="Whether to return attention maps")
    return_explanation: bool = Field(False, description="Whether to return explanations")
    temporal_aggregation: str = Field("mean", description="Temporal aggregation method for videos")
    
    # Explanation options
    explanation_method: str = Field("gradcam", description="Explanation method")
    explanation_class_idx: Optional[int] = Field(None, description="Class index for explanation")
    
    class Config:
        schema_extra = {
            "example": {
                "input_path": "/path/to/video.mp4",
                "model_name": "xception",
                "confidence_threshold": 0.5,
                "return_attention": True,
                "return_explanation": True,
                "temporal_aggregation": "mean",
                "explanation_method": "gradcam",
            }
        }


class PredictionResponse(BaseModel):
    """Response schema for prediction endpoint."""
    
    # Prediction results
    prediction: int = Field(..., description="Predicted class (0: real, 1: fake)")
    probability: float = Field(..., description="Prediction probability")
    confidence: float = Field(..., description="Confidence score")
    
    # Metadata
    model_name: str = Field(..., description="Model architecture name")
    processing_time: float = Field(..., description="Processing time in seconds")
    input_type: str = Field(..., description="Input type (image/video)")
    
    # Additional information
    frame_predictions: Optional[List[float]] = Field(None, description="Frame-level predictions for videos")
    frame_probabilities: Optional[List[float]] = Field(None, description="Frame-level probabilities for videos")
    temporal_aggregation: Optional[str] = Field(None, description="Temporal aggregation method used")
    
    # Attention and explanations
    attention_map: Optional[List[List[float]]] = Field(None, description="Attention map as 2D array")
    explanation: Optional[Dict[str, Any]] = Field(None, description="Explanation data")
    
    # Error information
    error: Optional[str] = Field(None, description="Error message if any")
    
    class Config:
        schema_extra = {
            "example": {
                "prediction": 1,
                "probability": 0.85,
                "confidence": 0.85,
                "model_name": "xception",
                "processing_time": 0.123,
                "input_type": "video",
                "frame_predictions": [0.8, 0.9, 0.7, 0.85],
                "frame_probabilities": [0.8, 0.9, 0.7, 0.85],
                "temporal_aggregation": "mean",
                "attention_map": [[0.1, 0.2], [0.3, 0.4]],
                "explanation": {"method": "gradcam", "confidence": 0.8},
            }
        }


class HealthResponse(BaseModel):
    """Response schema for health check endpoint."""
    
    status: str = Field(..., description="Service status")
    version: str = Field(..., description="API version")
    model_loaded: bool = Field(..., description="Whether model is loaded")
    model_name: Optional[str] = Field(None, description="Loaded model name")
    device: Optional[str] = Field(None, description="Device being used")
    timestamp: str = Field(..., description="Current timestamp")
    
    class Config:
        schema_extra = {
            "example": {
                "status": "healthy",
                "version": "0.1.0",
                "model_loaded": True,
                "model_name": "xception",
                "device": "cuda",
                "timestamp": "2024-01-01T00:00:00Z",
            }
        }


class ModelInfo(BaseModel):
    """Response schema for model information endpoint."""
    
    model_name: str = Field(..., description="Model architecture name")
    model_path: str = Field(..., description="Path to model file")
    model_size: int = Field(..., description="Model size in bytes")
    input_shape: List[int] = Field(..., description="Expected input shape")
    output_shape: List[int] = Field(..., description="Expected output shape")
    num_classes: int = Field(..., description="Number of output classes")
    device: str = Field(..., description="Device being used")
    precision: str = Field(..., description="Model precision")
    
    class Config:
        schema_extra = {
            "example": {
                "model_name": "xception",
                "model_path": "/path/to/model.ckpt",
                "model_size": 1024000,
                "input_shape": [1, 3, 224, 224],
                "output_shape": [1, 2],
                "num_classes": 2,
                "device": "cuda",
                "precision": "fp16",
            }
        }


class BatchPredictionRequest(BaseModel):
    """Request schema for batch prediction endpoint."""
    
    # Input data
    input_paths: List[str] = Field(..., description="List of input file paths")
    
    # Model configuration
    model_name: str = Field("xception", description="Model architecture name")
    confidence_threshold: float = Field(0.5, description="Confidence threshold for prediction")
    
    # Processing options
    batch_size: int = Field(32, description="Batch size for processing")
    return_attention: bool = Field(False, description="Whether to return attention maps")
    return_explanation: bool = Field(False, description="Whether to return explanations")
    
    # Explanation options
    explanation_method: str = Field("gradcam", description="Explanation method")
    
    class Config:
        schema_extra = {
            "example": {
                "input_paths": ["/path/to/video1.mp4", "/path/to/image1.jpg"],
                "model_name": "xception",
                "confidence_threshold": 0.5,
                "batch_size": 32,
                "return_attention": True,
                "return_explanation": True,
                "explanation_method": "gradcam",
            }
        }


class BatchPredictionResponse(BaseModel):
    """Response schema for batch prediction endpoint."""
    
    # Batch results
    predictions: List[PredictionResponse] = Field(..., description="List of prediction results")
    total_processed: int = Field(..., description="Total number of inputs processed")
    processing_time: float = Field(..., description="Total processing time in seconds")
    
    # Error information
    errors: List[str] = Field(default_factory=list, description="List of error messages")
    
    class Config:
        schema_extra = {
            "example": {
                "predictions": [
                    {
                        "prediction": 1,
                        "probability": 0.85,
                        "confidence": 0.85,
                        "model_name": "xception",
                        "processing_time": 0.123,
                        "input_type": "video",
                    }
                ],
                "total_processed": 1,
                "processing_time": 0.123,
                "errors": [],
            }
        }


class ExplanationRequest(BaseModel):
    """Request schema for explanation endpoint."""
    
    # Input data
    input_path: str = Field(..., description="Path to input file")
    
    # Model configuration
    model_name: str = Field("xception", description="Model architecture name")
    
    # Explanation options
    explanation_method: str = Field("gradcam", description="Explanation method")
    class_idx: Optional[int] = Field(None, description="Class index for explanation")
    return_overlay: bool = Field(True, description="Whether to return overlay image")
    alpha: float = Field(0.4, description="Overlay transparency")
    colormap: str = Field("jet", description="Colormap for visualization")
    
    class Config:
        schema_extra = {
            "example": {
                "input_path": "/path/to/video.mp4",
                "model_name": "xception",
                "explanation_method": "gradcam",
                "class_idx": 1,
                "return_overlay": True,
                "alpha": 0.4,
                "colormap": "jet",
            }
        }


class ExplanationResponse(BaseModel):
    """Response schema for explanation endpoint."""
    
    # Explanation results
    explanation_method: str = Field(..., description="Explanation method used")
    class_idx: int = Field(..., description="Class index explained")
    attribution_map: List[List[float]] = Field(..., description="Attribution map as 2D array")
    
    # Visualization
    overlay_image: Optional[bytes] = Field(None, description="Overlay image as bytes")
    overlay_format: Optional[str] = Field(None, description="Overlay image format")
    
    # Metadata
    processing_time: float = Field(..., description="Processing time in seconds")
    input_type: str = Field(..., description="Input type (image/video)")
    
    class Config:
        schema_extra = {
            "example": {
                "explanation_method": "gradcam",
                "class_idx": 1,
                "attribution_map": [[0.1, 0.2], [0.3, 0.4]],
                "overlay_image": b"base64_encoded_image",
                "overlay_format": "png",
                "processing_time": 0.123,
                "input_type": "video",
            }
        }


class ErrorResponse(BaseModel):
    """Response schema for error endpoints."""
    
    error: str = Field(..., description="Error message")
    error_code: str = Field(..., description="Error code")
    timestamp: str = Field(..., description="Error timestamp")
    request_id: Optional[str] = Field(None, description="Request ID for tracking")
    
    class Config:
        schema_extra = {
            "example": {
                "error": "Model not found",
                "error_code": "MODEL_NOT_FOUND",
                "timestamp": "2024-01-01T00:00:00Z",
                "request_id": "req_123456",
            }
        }
