"""
FastAPI server for deepfake detection API.

Provides comprehensive REST API endpoints for deepfake detection
including prediction, explanation, and model management.
"""

import os
import time
import uuid
from pathlib import Path
from typing import Optional, Dict, Any, List
import logging

import torch
import numpy as np
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Depends
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import uvicorn

from .schemas import (
    PredictionRequest, PredictionResponse, HealthResponse, ModelInfo,
    BatchPredictionRequest, BatchPredictionResponse, ExplanationRequest,
    ExplanationResponse, ErrorResponse
)
from ..inference.predictor import DeepfakePredictor
from ..utils.logging import get_logger
from ..utils.io import load_config

logger = get_logger(__name__)

# Global variables
app: Optional[FastAPI] = None
predictor: Optional[DeepfakePredictor] = None
model_info: Optional[ModelInfo] = None


def create_app(
    model_path: Optional[str] = None,
    model_name: str = "xception",
    device: str = "cpu",
    use_mock: bool = True,
    config: Optional[Dict[str, Any]] = None,
    title: str = "Deepfake Forensics API",
    description: str = "API for deepfake detection and analysis",
    version: str = "0.1.0",
) -> FastAPI:
    """
    Create FastAPI application.
    
    Args:
        model_path: Path to saved model (optional)
        model_name: Model architecture name
        device: Device to run on
        use_mock: Whether to use mock model for testing
        config: Configuration dictionary
        title: API title
        description: API description
        version: API version
        
    Returns:
        FastAPI application instance
    """
    global app, predictor, model_info
    
    # Create FastAPI app
    app = FastAPI(
        title=title,
        description=description,
        version=version,
        docs_url="/docs",
        redoc_url="/redoc",
    )
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Load model
    try:
        predictor = DeepfakePredictor(
            model_name=model_name,
            model_path=model_path,
            device=device,
            use_mock=use_mock,
        )
        
        # Get model info
        model_info = ModelInfo(
            model_name=model_name,
            model_path=model_path or "mock",
            model_size=0,
            input_shape=[1, 3, 224, 224],
            output_shape=[1, 2],
            num_classes=2,
            device=device,
            precision="fp32",
        )
        
        logger.info(f"Model loaded successfully: {model_name} (mock={use_mock})")
        
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        predictor = None
        model_info = None
    
    # Define routes
    setup_routes()
    
    return app


def setup_routes():
    """Set up API routes."""
    
    @app.get("/health", response_model=HealthResponse)
    async def health_check():
        """Health check endpoint."""
        return HealthResponse(
            status="healthy" if predictor is not None else "unhealthy",
            version="0.1.0",
            model_loaded=predictor is not None,
            model_name=model_info.model_name if model_info else None,
            device=model_info.device if model_info else None,
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        )
    
    @app.get("/models", response_model=ModelInfo)
    async def get_model_info():
        """Get model information."""
        if model_info is None:
            raise HTTPException(status_code=503, detail="Model not loaded")
        
        return model_info
    
    @app.post("/predict", response_model=PredictionResponse)
    async def predict(
        request: PredictionRequest,
        request_id: str = Depends(lambda: str(uuid.uuid4()))
    ):
        """Predict deepfake probability."""
        if detector is None:
            raise HTTPException(status_code=503, detail="Model not loaded")
        
        start_time = time.time()
        
        try:
            # Determine input type
            if request.input_path:
                input_path = Path(request.input_path)
                if not input_path.exists():
                    raise HTTPException(status_code=404, detail="Input file not found")
                
                # Predict based on file type
                if input_path.suffix.lower() in ['.mp4', '.avi', '.mov', '.mkv']:
                    # Video prediction
                    result = inference_pipeline.predict_video(
                        video_path=input_path,
                        temporal_aggregation=request.temporal_aggregation,
                        return_frame_predictions=True,
                        return_attention=request.return_attention,
                        return_explanation=request.return_explanation,
                    )
                    input_type = "video"
                else:
                    # Image prediction
                    result = inference_pipeline.predict_image(
                        image_path=input_path,
                        return_attention=request.return_attention,
                        return_explanation=request.return_explanation,
                    )
                    input_type = "image"
            
            elif request.input_data:
                # Handle uploaded data
                # This would require additional implementation for handling bytes
                raise HTTPException(status_code=501, detail="Direct data upload not implemented yet")
            
            else:
                raise HTTPException(status_code=400, detail="Either input_path or input_data must be provided")
            
            # Process results
            processing_time = time.time() - start_time
            
            response = PredictionResponse(
                prediction=int(result["prediction"]),
                probability=float(result["probability"]),
                confidence=float(result["confidence"]),
                model_name=request.model_name,
                processing_time=processing_time,
                input_type=input_type,
                frame_predictions=result.get("frame_predictions"),
                frame_probabilities=result.get("frame_probabilities"),
                temporal_aggregation=result.get("temporal_aggregation"),
                attention_map=result.get("attention", {}).get("attention_map") if result.get("attention") else None,
                explanation=result.get("explanation"),
            )
            
            return response
            
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.post("/predict/batch", response_model=BatchPredictionResponse)
    async def predict_batch(
        request: BatchPredictionRequest,
        request_id: str = Depends(lambda: str(uuid.uuid4()))
    ):
        """Predict deepfake probability for batch of inputs."""
        if detector is None:
            raise HTTPException(status_code=503, detail="Model not loaded")
        
        start_time = time.time()
        
        try:
            # Convert paths to Path objects
            input_paths = [Path(p) for p in request.input_paths]
            
            # Validate paths
            valid_paths = []
            errors = []
            
            for path in input_paths:
                if path.exists():
                    valid_paths.append(path)
                else:
                    errors.append(f"File not found: {path}")
            
            if not valid_paths:
                raise HTTPException(status_code=404, detail="No valid input files found")
            
            # Run batch prediction
            results = inference_pipeline.predict_batch(
                input_paths=valid_paths,
                batch_size=request.batch_size,
                return_attention=request.return_attention,
                return_explanation=request.return_explanation,
            )
            
            # Convert to response format
            predictions = []
            for result in results:
                pred = PredictionResponse(
                    prediction=int(result["prediction"]),
                    probability=float(result["probability"]),
                    confidence=float(result["confidence"]),
                    model_name=request.model_name,
                    processing_time=0.0,  # Individual processing time not available
                    input_type="image" if result["input_path"].endswith(('.jpg', '.jpeg', '.png')) else "video",
                    attention_map=result.get("attention"),
                    explanation=result.get("explanation"),
                )
                predictions.append(pred)
            
            processing_time = time.time() - start_time
            
            response = BatchPredictionResponse(
                predictions=predictions,
                total_processed=len(predictions),
                processing_time=processing_time,
                errors=errors,
            )
            
            return response
            
        except Exception as e:
            logger.error(f"Batch prediction failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.post("/explain", response_model=ExplanationResponse)
    async def explain(
        request: ExplanationRequest,
        request_id: str = Depends(lambda: str(uuid.uuid4()))
    ):
        """Generate explanation for model prediction."""
        if detector is None:
            raise HTTPException(status_code=503, detail="Model not loaded")
        
        start_time = time.time()
        
        try:
            # Check if input file exists
            input_path = Path(request.input_path)
            if not input_path.exists():
                raise HTTPException(status_code=404, detail="Input file not found")
            
            # Determine input type
            if input_path.suffix.lower() in ['.mp4', '.avi', '.mov', '.mkv']:
                input_type = "video"
            else:
                input_type = "image"
            
            # Generate explanation
            if input_type == "video":
                result = inference_pipeline.predict_video(
                    video_path=input_path,
                    return_explanation=True,
                )
            else:
                result = inference_pipeline.predict_image(
                    image_path=input_path,
                    return_explanation=True,
                )
            
            # Process explanation
            explanation = result.get("explanation", {})
            if not explanation:
                raise HTTPException(status_code=500, detail="Failed to generate explanation")
            
            # Convert attribution map to 2D array
            attribution_map = explanation.get("attribution", [])
            if isinstance(attribution_map, np.ndarray):
                attribution_map = attribution_map.tolist()
            
            # Get overlay image if available
            overlay_image = None
            overlay_format = None
            if explanation.get("overlay") is not None:
                # Convert overlay to bytes
                overlay_image = explanation["overlay"].tobytes()
                overlay_format = "png"
            
            processing_time = time.time() - start_time
            
            response = ExplanationResponse(
                explanation_method=request.explanation_method,
                class_idx=explanation.get("class_idx", 0),
                attribution_map=attribution_map,
                overlay_image=overlay_image,
                overlay_format=overlay_format,
                processing_time=processing_time,
                input_type=input_type,
            )
            
            return response
            
        except Exception as e:
            logger.error(f"Explanation failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.post("/predict")
    async def predict_upload(
        file: UploadFile = File(...),
    ):
        """Predict deepfake probability for uploaded file."""
        if predictor is None:
            raise HTTPException(status_code=503, detail="Model not loaded")
        
        start_time = time.time()
        
        try:
            # Save uploaded file temporarily
            import tempfile
            with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as temp_file:
                content = await file.read()
                temp_file.write(content)
                temp_path = temp_file.name
            
            # Predict using our predictor
            result = predictor.predict(temp_path)
            
            # Clean up temporary file
            Path(temp_path).unlink()
            
            # Process results
            processing_time = time.time() - start_time
            
            # Determine input type
            input_type = "video" if file.filename.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')) else "image"
            
            response = {
                "score": float(result.score),
                "label": result.label,
                "confidence": float(result.confidence),
                "per_frame_scores": result.per_frame_scores,
                "explanation_assets": result.explanation_assets,
                "meta": {
                    "model_name": "xception",
                    "processing_time": processing_time,
                    "input_type": input_type,
                    "filename": file.filename,
                    "file_size": len(content)
                }
            }
            
            return response
            
        except Exception as e:
            logger.error(f"Upload prediction failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/")
    async def root():
        """Root endpoint."""
        return {
            "message": "Deepfake Forensics API",
            "version": "0.1.0",
            "docs": "/docs",
            "health": "/health",
        }
    
    # Error handlers
    @app.exception_handler(404)
    async def not_found_handler(request, exc):
        return JSONResponse(
            status_code=404,
            content=ErrorResponse(
                error="Not found",
                error_code="NOT_FOUND",
                timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            ).dict()
        )
    
    @app.exception_handler(500)
    async def internal_error_handler(request, exc):
        return JSONResponse(
            status_code=500,
            content=ErrorResponse(
                error="Internal server error",
                error_code="INTERNAL_ERROR",
                timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            ).dict()
        )


def run_server(
    model_path: str,
    host: str = "0.0.0.0",
    port: int = 8000,
    workers: int = 1,
    model_name: str = "xception",
    device: str = "auto",
    config: Optional[Dict[str, Any]] = None,
):
    """
    Run the API server.
    
    Args:
        model_path: Path to saved model
        host: Host to bind to
        port: Port to bind to
        workers: Number of worker processes
        model_name: Model architecture name
        device: Device to run on
        config: Configuration dictionary
    """
    # Create app
    app_instance = create_app(
        model_path=model_path,
        model_name=model_name,
        device=device,
        config=config,
    )
    
    # Run server
    uvicorn.run(
        app_instance,
        host=host,
        port=port,
        workers=workers,
    )


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run Deepfake Forensics API server")
    parser.add_argument("--model", type=str, required=True, help="Path to saved model")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--workers", type=int, default=1, help="Number of worker processes")
    parser.add_argument("--model-name", type=str, default="xception", help="Model architecture name")
    parser.add_argument("--device", type=str, default="auto", help="Device to run on")
    parser.add_argument("--config", type=str, help="Path to config file")
    
    args = parser.parse_args()
    
    # Load configuration
    config = None
    if args.config:
        config = load_config(args.config)
    
    # Run server
    run_server(
        model_path=args.model,
        host=args.host,
        port=args.port,
        workers=args.workers,
        model_name=args.model_name,
        device=args.device,
        config=config,
    )
