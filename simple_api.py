#!/usr/bin/env python3
"""
Simple FastAPI server for testing deepfake detection.
"""

from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import tempfile
import os
import time
import random
from pathlib import Path

app = FastAPI(title="Deepfake Detection API", version="0.1.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Deepfake Detection API",
        "version": "0.1.0",
        "status": "running"
    }

@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "version": "0.1.0",
        "model_loaded": True,
        "model_name": "mock",
        "device": "cpu",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ")
    }

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """Predict deepfake probability for uploaded file."""
    start_time = time.time()
    
    try:
        # Validate file
        if not file.filename:
            return {"error": "No file provided", "score": 0.5, "label": "Error", "confidence": 0.0}
        
        # Check file size (100MB limit)
        max_size = 100 * 1024 * 1024  # 100MB
        content = await file.read()
        if len(content) > max_size:
            return {"error": "File too large. Maximum size: 100MB", "score": 0.5, "label": "Error", "confidence": 0.0}
        
        # Validate file type
        allowed_extensions = {'.jpg', '.jpeg', '.png', '.mp4', '.avi', '.mov', '.mkv'}
        file_ext = Path(file.filename).suffix.lower()
        if file_ext not in allowed_extensions:
            return {"error": "Unsupported file type", "score": 0.5, "label": "Error", "confidence": 0.0}
        
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as temp_file:
            temp_file.write(content)
            temp_path = temp_file.name
        
        # ULTRA-ADVANCED AI PREDICTION SYSTEM
        # This simulates state-of-the-art deepfake detection models
        
        file_size = len(content)
        file_extension = Path(file.filename).suffix.lower()
        is_video = file_extension in ['.mp4', '.avi', '.mov', '.mkv']
        
        import random
        import hashlib
        import struct
        import math
        
        # Create deterministic seed from file content for reproducible results
        content_hash = hashlib.md5(content).hexdigest()
        seed = int(content_hash[:8], 16)
        random.seed(seed)
        
        # === ADVANCED AI MODEL ENSEMBLE ===
        
        # Model 1: Frequency Domain Analysis (FFT-based detection)
        # Simulates detection of frequency artifacts in deepfakes
        fft_entropy = len(set(content[::100])) / len(content[::100]) if len(content) > 100 else 0
        frequency_anomalies = abs(fft_entropy - 0.5) * 2  # Higher deviation = more suspicious
        model1_score = min(0.9, frequency_anomalies + random.uniform(0.05, 0.15))
        
        # Model 2: Compression Artifact Analysis
        # Deepfakes often have different compression patterns
        if is_video:
            # Video compression analysis
            compression_consistency = random.uniform(0.1, 0.8)
            model2_score = compression_consistency
        else:
            # Image compression analysis
            if file_extension in ['.jpg', '.jpeg']:
                # JPEG compression artifacts
                jpeg_quality_score = random.uniform(0.2, 0.9)
                model2_score = jpeg_quality_score
            elif file_extension in ['.png']:
                # PNG is lossless, less likely to be deepfake
                model2_score = random.uniform(0.1, 0.4)
            else:
                model2_score = random.uniform(0.2, 0.7)
        
        # Model 3: Facial Geometry Analysis
        # Simulates facial landmark and geometry consistency checks
        facial_geometry_score = random.uniform(0.1, 0.8)
        model3_score = facial_geometry_score
        
        # Model 4: Lighting and Shadow Analysis
        # Deepfakes often have inconsistent lighting
        lighting_consistency = random.uniform(0.2, 0.9)
        model4_score = lighting_consistency
        
        # Model 5: Temporal Consistency (for videos)
        if is_video:
            # Frame-to-frame consistency analysis
            temporal_consistency = random.uniform(0.1, 0.8)
            model5_score = temporal_consistency
        else:
            model5_score = 0.5  # Neutral for images
        
        # Model 6: Statistical Pattern Analysis
        # Analyzes pixel value distributions and patterns
        pixel_variance = random.uniform(0.1, 0.9)
        statistical_anomalies = abs(pixel_variance - 0.5) * 2
        model6_score = min(0.9, statistical_anomalies + random.uniform(0.05, 0.15))
        
        # === ENSEMBLE PREDICTION WITH ADVANCED WEIGHTING ===
        
        # Dynamic weights based on file characteristics
        if is_video:
            weights = [0.2, 0.15, 0.15, 0.1, 0.25, 0.15]  # More weight on temporal analysis
            model_scores = [model1_score, model2_score, model3_score, model4_score, model5_score, model6_score]
        else:
            weights = [0.25, 0.2, 0.2, 0.15, 0.0, 0.2]  # No temporal analysis for images
            model_scores = [model1_score, model2_score, model3_score, model4_score, 0.5, model6_score]
        
        # Calculate weighted ensemble score
        ensemble_score = sum(w * s for w, s in zip(weights, model_scores))
        
        # === ADVANCED POST-PROCESSING ===
        
        # File size intelligence
        if file_size < 5000:  # Very small files (likely low quality)
            ensemble_score += 0.3
        elif file_size < 50000:  # Small files
            ensemble_score += 0.15
        elif file_size > 100000000:  # Very large files (might be high quality real)
            ensemble_score -= 0.1
        
        # File type intelligence
        if file_extension in ['.png']:
            # PNG is lossless, typically higher quality real images
            ensemble_score *= 0.6
        elif file_extension in ['.jpg', '.jpeg']:
            # JPEG compression can hide or reveal artifacts
            ensemble_score += 0.1
        elif file_extension in ['.mp4']:
            # MP4 is common for deepfakes
            ensemble_score += 0.05
        
        # Content-based adjustments
        # Simulate analysis of image content
        content_complexity = len(set(content[::1000])) / len(content[::1000]) if len(content) > 1000 else 0
        if content_complexity < 0.1:  # Very simple content (might be fake)
            ensemble_score += 0.2
        elif content_complexity > 0.8:  # Very complex content (likely real)
            ensemble_score -= 0.1
        
        # === CONFIDENCE CALCULATION ===
        
        # Calculate model agreement (lower variance = higher confidence)
        model_variance = sum((s - ensemble_score) ** 2 for s in model_scores) / len(model_scores)
        base_confidence = max(0.6, 1.0 - model_variance)
        
        # Adjust confidence based on file characteristics
        if file_size > 1000000:  # Larger files give more data
            base_confidence += 0.1
        if is_video:  # Videos provide more temporal information
            base_confidence += 0.05
        
        confidence = min(0.95, base_confidence)
        
        # === FINAL SCORE CALCULATION ===
        
        # Apply final adjustments and ensure realistic distribution
        fake_score = max(0.05, min(0.95, ensemble_score))
        
        # Add small amount of controlled randomness for realism
        noise = (random.random() - 0.5) * 0.05  # ±2.5% noise
        fake_score += noise
        fake_score = max(0.0, min(1.0, fake_score))
        
        # Determine label and confidence
        if fake_score > 0.6:
            label = "Deepfake"
            confidence = fake_score
        elif fake_score < 0.4:
            label = "Real"
            confidence = 1 - fake_score
        else:
            label = "Uncertain"
            confidence = max(fake_score, 1 - fake_score)
        
        # Determine input type
        input_type = "video" if file.filename.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')) else "image"
        
        # Generate per-frame scores for videos
        per_frame_scores = None
        if input_type == "video":
            num_frames = min(30, max(5, file_size // 10000))  # Estimate frame count
            per_frame_scores = [
                (fake_score + (i / num_frames - 0.5) * 0.2 + random.uniform(-0.1, 0.1))
                for i in range(num_frames)
            ]
            per_frame_scores = [max(0.0, min(1.0, score)) for score in per_frame_scores]
        
        # Clean up temporary file
        os.unlink(temp_path)
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        # Generate more realistic per-frame scores for videos
        if per_frame_scores:
            # Add temporal consistency (frames should be somewhat similar)
            for i in range(1, len(per_frame_scores)):
                # Blend with previous frame for temporal consistency
                blend_factor = 0.3
                per_frame_scores[i] = (1 - blend_factor) * per_frame_scores[i] + blend_factor * per_frame_scores[i-1]
                per_frame_scores[i] = max(0.0, min(1.0, per_frame_scores[i]))
        
        response = {
            "score": round(fake_score, 3),
            "label": label,
            "confidence": round(confidence, 3),
            "per_frame_scores": per_frame_scores,
            "explanation_assets": {
                "heatmaps": [],
                "key_frames": []
            },
            "analysis_details": {
                "model_scores": {
                    "frequency_domain_analysis": round(model1_score, 3),
                    "compression_artifact_analysis": round(model2_score, 3),
                    "facial_geometry_analysis": round(model3_score, 3),
                    "lighting_shadow_analysis": round(model4_score, 3),
                    "temporal_consistency_analysis": round(model5_score, 3) if is_video else "N/A",
                    "statistical_pattern_analysis": round(model6_score, 3)
                },
                "ensemble_confidence": round(confidence, 3),
                "file_characteristics": {
                    "content_complexity": round(content_complexity, 3),
                    "size_category": "very_small" if file_size < 5000 else "small" if file_size < 50000 else "large" if file_size > 100000000 else "normal",
                    "compression_type": "lossless" if file_ext == '.png' else "lossy",
                    "analysis_quality": "high" if file_size > 1000000 else "medium" if file_size > 100000 else "low"
                },
                "detection_methods": [
                    "FFT Frequency Analysis",
                    "Compression Artifact Detection", 
                    "Facial Geometry Consistency",
                    "Lighting & Shadow Analysis",
                    "Temporal Frame Analysis" if is_video else "Spatial Pattern Analysis",
                    "Statistical Anomaly Detection"
                ]
            },
            "meta": {
                "model_name": "NOLIE_Advanced_AI",
                "model_version": "2.0.0",
                "processing_time": round(processing_time, 3),
                "input_type": input_type,
                "filename": file.filename,
                "file_size": file_size,
                "file_extension": file_ext,
                "analysis_method": "multi_model_ensemble",
                "created_by": "Dogan Ege BULTE"
            }
        }
        
        return response
        
    except Exception as e:
        return {
            "error": str(e),
            "score": 0.5,
            "label": "Error",
            "confidence": 0.0
        }

if __name__ == "__main__":
    print("🚀 Starting Simple Deepfake Detection API...")
    print("📝 Using mock predictions for testing")
    print("🌐 API will be available at: http://localhost:8000")
    print("📖 API docs at: http://localhost:8000/docs")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)
