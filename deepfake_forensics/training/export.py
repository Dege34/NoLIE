"""
Model export utilities for deepfake detection.

Provides functionality to export trained models to various formats
including TorchScript, ONNX, and TensorRT for deployment.
"""

import torch
import torch.onnx
import torch.jit
from pathlib import Path
from typing import Dict, Any, Optional, Union, Tuple
import logging

from .module import DeepfakeDetector, create_detector
from ..utils.logging import get_logger
from ..utils.io import load_config

logger = get_logger(__name__)


def export_model(
    model_path: str,
    output_path: str,
    format: str = "torchscript",
    model_name: str = "xception",
    device: str = "cpu",
    input_size: Tuple[int, int] = (224, 224),
    max_frames: int = 16,
    **kwargs,
) -> str:
    """
    Export model to specified format.
    
    Args:
        model_path: Path to saved model
        output_path: Path to save exported model
        format: Export format ('torchscript', 'onnx', 'tensorrt')
        model_name: Model architecture name
        device: Device to run on
        input_size: Input image size
        max_frames: Maximum number of frames for video models
        **kwargs: Additional export arguments
        
    Returns:
        Path to exported model
    """
    logger.info(f"Exporting model from {model_path} to {output_path}")
    logger.info(f"Format: {format}, Device: {device}")
    
    # Create detector
    detector = create_detector(
        model_path=model_path,
        model_name=model_name,
        device=device,
    )
    
    # Export based on format
    if format == "torchscript":
        return export_to_torchscript(
            detector,
            output_path,
            input_size=input_size,
            max_frames=max_frames,
            **kwargs,
        )
    elif format == "onnx":
        return export_to_onnx(
            detector,
            output_path,
            input_size=input_size,
            max_frames=max_frames,
            **kwargs,
        )
    elif format == "tensorrt":
        return export_to_tensorrt(
            detector,
            output_path,
            input_size=input_size,
            max_frames=max_frames,
            **kwargs,
        )
    else:
        raise ValueError(f"Unknown export format: {format}")


def export_to_torchscript(
    detector: DeepfakeDetector,
    output_path: str,
    input_size: Tuple[int, int] = (224, 224),
    max_frames: int = 16,
    optimize: bool = True,
    **kwargs,
) -> str:
    """
    Export model to TorchScript format.
    
    Args:
        detector: Deepfake detector instance
        output_path: Path to save TorchScript model
        input_size: Input image size
        max_frames: Maximum number of frames for video models
        optimize: Whether to optimize the model
        **kwargs: Additional arguments
        
    Returns:
        Path to exported model
    """
    logger.info("Exporting to TorchScript format")
    
    # Create dummy input
    if detector.model_name in ["xception", "vit", "resnet_freq"]:
        # Image model
        dummy_input = torch.randn(1, 3, input_size[0], input_size[1])
    else:
        # Video model
        dummy_input = torch.randn(1, max_frames, 3, input_size[0], input_size[1])
    
    # Move to device
    dummy_input = dummy_input.to(detector.device)
    
    # Set model to eval mode
    detector.model.eval()
    
    # Trace the model
    try:
        traced_model = torch.jit.trace(detector.model, dummy_input)
        
        # Optimize if requested
        if optimize:
            traced_model = torch.jit.optimize_for_inference(traced_model)
        
        # Save model
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        traced_model.save(str(output_path))
        
        logger.info(f"TorchScript model saved to {output_path}")
        return str(output_path)
        
    except Exception as e:
        logger.error(f"Failed to export to TorchScript: {e}")
        raise


def export_to_onnx(
    detector: DeepfakeDetector,
    output_path: str,
    input_size: Tuple[int, int] = (224, 224),
    max_frames: int = 16,
    opset_version: int = 11,
    dynamic_axes: Optional[Dict[str, Dict[int, str]]] = None,
    **kwargs,
) -> str:
    """
    Export model to ONNX format.
    
    Args:
        detector: Deepfake detector instance
        output_path: Path to save ONNX model
        input_size: Input image size
        max_frames: Maximum number of frames for video models
        opset_version: ONNX opset version
        dynamic_axes: Dynamic axes configuration
        **kwargs: Additional arguments
        
    Returns:
        Path to exported model
    """
    logger.info("Exporting to ONNX format")
    
    # Create dummy input
    if detector.model_name in ["xception", "vit", "resnet_freq"]:
        # Image model
        dummy_input = torch.randn(1, 3, input_size[0], input_size[1])
        input_names = ["input"]
        output_names = ["output"]
        
        # Dynamic axes for image model
        if dynamic_axes is None:
            dynamic_axes = {
                "input": {0: "batch_size"},
                "output": {0: "batch_size"},
            }
    else:
        # Video model
        dummy_input = torch.randn(1, max_frames, 3, input_size[0], input_size[1])
        input_names = ["input"]
        output_names = ["output"]
        
        # Dynamic axes for video model
        if dynamic_axes is None:
            dynamic_axes = {
                "input": {0: "batch_size", 1: "sequence_length"},
                "output": {0: "batch_size"},
            }
    
    # Move to device
    dummy_input = dummy_input.to(detector.device)
    
    # Set model to eval mode
    detector.model.eval()
    
    # Export to ONNX
    try:
        torch.onnx.export(
            detector.model,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            verbose=False,
        )
        
        logger.info(f"ONNX model saved to {output_path}")
        return output_path
        
    except Exception as e:
        logger.error(f"Failed to export to ONNX: {e}")
        raise


def export_to_tensorrt(
    detector: DeepfakeDetector,
    output_path: str,
    input_size: Tuple[int, int] = (224, 224),
    max_frames: int = 16,
    precision: str = "fp16",
    **kwargs,
) -> str:
    """
    Export model to TensorRT format.
    
    Args:
        detector: Deepfake detector instance
        output_path: Path to save TensorRT model
        input_size: Input image size
        max_frames: Maximum number of frames for video models
        precision: Precision mode ('fp32', 'fp16', 'int8')
        **kwargs: Additional arguments
        
    Returns:
        Path to exported model
    """
    logger.info("Exporting to TensorRT format")
    
    try:
        import tensorrt as trt
        from torch2trt import torch2trt
    except ImportError:
        raise ImportError("TensorRT and torch2trt are required for TensorRT export")
    
    # Create dummy input
    if detector.model_name in ["xception", "vit", "resnet_freq"]:
        # Image model
        dummy_input = torch.randn(1, 3, input_size[0], input_size[1])
    else:
        # Video model
        dummy_input = torch.randn(1, max_frames, 3, input_size[0], input_size[1])
    
    # Move to device
    dummy_input = dummy_input.to(detector.device)
    
    # Set model to eval mode
    detector.model.eval()
    
    # Convert to TensorRT
    try:
        trt_model = torch2trt(
            detector.model,
            [dummy_input],
            fp16_mode=(precision == "fp16"),
            int8_mode=(precision == "int8"),
            max_workspace_size=1 << 30,  # 1GB
        )
        
        # Save model
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        torch.save(trt_model.state_dict(), str(output_path))
        
        logger.info(f"TensorRT model saved to {output_path}")
        return str(output_path)
        
    except Exception as e:
        logger.error(f"Failed to export to TensorRT: {e}")
        raise


def export_ensemble(
    model_paths: List[str],
    output_path: str,
    format: str = "torchscript",
    model_names: Optional[List[str]] = None,
    weights: Optional[List[float]] = None,
    device: str = "cpu",
    input_size: Tuple[int, int] = (224, 224),
    max_frames: int = 16,
    **kwargs,
) -> str:
    """
    Export ensemble of models.
    
    Args:
        model_paths: List of model paths
        output_path: Path to save ensemble model
        format: Export format
        model_names: List of model architecture names
        weights: Ensemble weights
        device: Device to run on
        input_size: Input image size
        max_frames: Maximum number of frames for video models
        **kwargs: Additional arguments
        
    Returns:
        Path to exported ensemble model
    """
    logger.info(f"Exporting ensemble of {len(model_paths)} models")
    
    # Default model names
    if model_names is None:
        model_names = ["xception"] * len(model_paths)
    
    # Default weights
    if weights is None:
        weights = [1.0] * len(model_paths)
    
    # Create detectors
    detectors = []
    for model_path, model_name in zip(model_paths, model_names):
        detector = create_detector(
            model_path=model_path,
            model_name=model_name,
            device=device,
        )
        detectors.append(detector)
    
    # Create ensemble model
    from ..models.heads import EnsembleHead
    ensemble_model = EnsembleHead(
        models=[detector.model for detector in detectors],
        weights=weights,
        fusion_method="weighted_average",
    )
    
    # Create ensemble detector
    ensemble_detector = DeepfakeDetector(
        model_path=None,
        model_name="ensemble",
        device=device,
    )
    ensemble_detector.model = ensemble_model
    
    # Export ensemble
    return export_model(
        model_path=None,
        output_path=output_path,
        format=format,
        model_name="ensemble",
        device=device,
        input_size=input_size,
        max_frames=max_frames,
        **kwargs,
    )


def validate_exported_model(
    original_model_path: str,
    exported_model_path: str,
    format: str = "torchscript",
    model_name: str = "xception",
    device: str = "cpu",
    input_size: Tuple[int, int] = (224, 224),
    max_frames: int = 16,
    tolerance: float = 1e-5,
) -> bool:
    """
    Validate exported model against original model.
    
    Args:
        original_model_path: Path to original model
        exported_model_path: Path to exported model
        format: Export format
        model_name: Model architecture name
        device: Device to run on
        input_size: Input image size
        max_frames: Maximum number of frames for video models
        tolerance: Tolerance for comparison
        
    Returns:
        True if validation passes
    """
    logger.info("Validating exported model")
    
    # Create original detector
    original_detector = create_detector(
        model_path=original_model_path,
        model_name=model_name,
        device=device,
    )
    
    # Create dummy input
    if model_name in ["xception", "vit", "resnet_freq"]:
        dummy_input = torch.randn(1, 3, input_size[0], input_size[1])
    else:
        dummy_input = torch.randn(1, max_frames, 3, input_size[0], input_size[1])
    
    dummy_input = dummy_input.to(device)
    
    # Get original output
    with torch.no_grad():
        original_output = original_detector.model(dummy_input)
    
    # Load exported model
    if format == "torchscript":
        exported_model = torch.jit.load(exported_model_path)
    elif format == "onnx":
        import onnxruntime as ort
        session = ort.InferenceSession(exported_model_path)
        exported_output = session.run(
            ["output"],
            {"input": dummy_input.cpu().numpy()}
        )[0]
        exported_output = torch.from_numpy(exported_output)
    else:
        raise ValueError(f"Validation not supported for format: {format}")
    
    # Compare outputs
    if format == "torchscript":
        with torch.no_grad():
            exported_output = exported_model(dummy_input)
    
    # Check if outputs are close
    is_close = torch.allclose(original_output, exported_output, atol=tolerance)
    
    if is_close:
        logger.info("Model validation passed")
    else:
        logger.error("Model validation failed")
        logger.error(f"Original output: {original_output}")
        logger.error(f"Exported output: {exported_output}")
        logger.error(f"Difference: {torch.abs(original_output - exported_output).max()}")
    
    return is_close


def main():
    """Main export function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Export deepfake detection model")
    parser.add_argument("--model", type=str, required=True, help="Path to saved model")
    parser.add_argument("--output", type=str, required=True, help="Path to save exported model")
    parser.add_argument("--format", type=str, default="torchscript", 
                       choices=["torchscript", "onnx", "tensorrt"], help="Export format")
    parser.add_argument("--model_name", type=str, default="xception", help="Model architecture name")
    parser.add_argument("--device", type=str, default="cpu", help="Device to run on")
    parser.add_argument("--input_size", type=int, nargs=2, default=[224, 224], help="Input image size")
    parser.add_argument("--max_frames", type=int, default=16, help="Maximum number of frames")
    parser.add_argument("--validate", action="store_true", help="Validate exported model")
    parser.add_argument("--config", type=str, help="Path to config file")
    
    args = parser.parse_args()
    
    # Load configuration
    config = None
    if args.config:
        config = load_config(args.config)
    
    # Export model
    exported_path = export_model(
        model_path=args.model,
        output_path=args.output,
        format=args.format,
        model_name=args.model_name,
        device=args.device,
        input_size=tuple(args.input_size),
        max_frames=args.max_frames,
    )
    
    # Validate if requested
    if args.validate:
        is_valid = validate_exported_model(
            original_model_path=args.model,
            exported_model_path=exported_path,
            format=args.format,
            model_name=args.model_name,
            device=args.device,
            input_size=tuple(args.input_size),
            max_frames=args.max_frames,
        )
        
        if not is_valid:
            raise RuntimeError("Model validation failed")
    
    print(f"Model exported successfully to {exported_path}")


if __name__ == "__main__":
    main()
