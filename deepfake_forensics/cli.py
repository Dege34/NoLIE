"""
Command-line interface for deepfake forensics.

Provides a comprehensive CLI for training, inference, and evaluation
of deepfake detection models.
"""

import typer
from pathlib import Path
from typing import Optional, List
import yaml
import json
import logging

from .training.train import train_model
from .training.infer import predict_single, predict_batch
from .training.export import export_model
from .utils.logging import setup_logging, get_logger
from .utils.io import load_config, save_json
from .data import create_data_loaders

app = typer.Typer(help="Deepfake Forensics CLI")
logger = get_logger(__name__)


@app.command()
def train(
    config: Path = typer.Option(..., "--config", "-c", help="Path to config file"),
    data_dir: Path = typer.Option(..., "--data-dir", "-d", help="Path to data directory"),
    output_dir: Path = typer.Option(..., "--output-dir", "-o", help="Path to output directory"),
    resume_from: Optional[Path] = typer.Option(None, "--resume-from", help="Path to checkpoint to resume from"),
    fast_dev_run: bool = typer.Option(False, "--fast-dev-run", help="Run fast dev run"),
    gpus: Optional[int] = typer.Option(None, "--gpus", help="Number of GPUs to use"),
    batch_size: Optional[int] = typer.Option(None, "--batch-size", help="Batch size"),
    learning_rate: Optional[float] = typer.Option(None, "--learning-rate", help="Learning rate"),
    max_epochs: Optional[int] = typer.Option(None, "--max-epochs", help="Maximum number of epochs"),
):
    """Train a deepfake detection model."""
    # Load configuration
    config_dict = load_config(config)
    
    # Override config with command line arguments
    if gpus is not None:
        config_dict["num_gpus"] = gpus
    if batch_size is not None:
        config_dict["data"]["batch_size"] = batch_size
    if learning_rate is not None:
        config_dict["training"]["learning_rate"] = learning_rate
    if max_epochs is not None:
        config_dict["training"]["max_epochs"] = max_epochs
    
    # Train model
    best_checkpoint = train_model(
        config=config_dict,
        data_dir=str(data_dir),
        output_dir=str(output_dir),
        resume_from_checkpoint=str(resume_from) if resume_from else None,
        fast_dev_run=fast_dev_run,
    )
    
    typer.echo(f"Training completed. Best checkpoint: {best_checkpoint}")


@app.command()
def predict(
    input_path: Path = typer.Option(..., "--input", "-i", help="Path to input file or directory"),
    model_path: Path = typer.Option(..., "--model", "-m", help="Path to saved model"),
    output_path: Optional[Path] = typer.Option(None, "--output", "-o", help="Path to output file"),
    config: Optional[Path] = typer.Option(None, "--config", "-c", help="Path to config file"),
    model_name: str = typer.Option("xception", "--model-name", help="Model architecture name"),
    device: str = typer.Option("auto", "--device", help="Device to run on"),
    batch_size: int = typer.Option(32, "--batch-size", help="Batch size"),
    format: str = typer.Option("json", "--format", help="Output format"),
    return_attention: bool = typer.Option(False, "--attention", help="Return attention maps"),
    return_explanation: bool = typer.Option(False, "--explanation", help="Return explanations"),
):
    """Run deepfake detection inference."""
    # Load configuration
    config_dict = None
    if config:
        config_dict = load_config(config)
    
    # Get input paths
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
            model_path=str(model_path),
            model_name=model_name,
            device=device,
            config=config_dict,
        )
        results = [results]
    else:
        results = predict_batch(
            input_paths,
            model_path=str(model_path),
            model_name=model_name,
            device=device,
            config=config_dict,
            batch_size=batch_size,
        )
    
    # Save results
    if output_path:
        if format == "json":
            save_json(results, output_path)
        elif format == "csv":
            import pandas as pd
            df = pd.DataFrame(results)
            df.to_csv(output_path, index=False)
        else:
            raise ValueError(f"Unknown format: {format}")
        
        typer.echo(f"Results saved to {output_path}")
    
    # Print results
    for result in results:
        typer.echo(f"Input: {result['input_path']}")
        typer.echo(f"Prediction: {result['prediction']}")
        typer.echo(f"Probability: {result['probability']:.3f}")
        typer.echo(f"Confidence: {result['confidence']:.3f}")
        typer.echo("-" * 50)


@app.command()
def export(
    model_path: Path = typer.Option(..., "--model", "-m", help="Path to saved model"),
    output_path: Path = typer.Option(..., "--output", "-o", help="Path to save exported model"),
    format: str = typer.Option("torchscript", "--format", help="Export format"),
    model_name: str = typer.Option("xception", "--model-name", help="Model architecture name"),
    device: str = typer.Option("cpu", "--device", help="Device to run on"),
    input_size: List[int] = typer.Option([224, 224], "--input-size", help="Input image size"),
    max_frames: int = typer.Option(16, "--max-frames", help="Maximum number of frames"),
    validate: bool = typer.Option(False, "--validate", help="Validate exported model"),
):
    """Export model to specified format."""
    # Export model
    exported_path = export_model(
        model_path=str(model_path),
        output_path=str(output_path),
        format=format,
        model_name=model_name,
        device=device,
        input_size=tuple(input_size),
        max_frames=max_frames,
    )
    
    # Validate if requested
    if validate:
        from .training.export import validate_exported_model
        is_valid = validate_exported_model(
            original_model_path=str(model_path),
            exported_model_path=exported_path,
            format=format,
            model_name=model_name,
            device=device,
            input_size=tuple(input_size),
            max_frames=max_frames,
        )
        
        if not is_valid:
            typer.echo("Model validation failed!", err=True)
            raise typer.Exit(1)
    
    typer.echo(f"Model exported successfully to {exported_path}")


@app.command()
def prepare_data(
    src_dir: Path = typer.Option(..., "--src", "-s", help="Source data directory"),
    out_dir: Path = typer.Option(..., "--out", "-o", help="Output data directory"),
    fps: int = typer.Option(8, "--fps", help="Frames per second for video extraction"),
    image_size: int = typer.Option(224, "--image-size", help="Image size for processing"),
    max_frames: int = typer.Option(16, "--max-frames", help="Maximum number of frames per video"),
    subjects_file: Optional[Path] = typer.Option(None, "--subjects-file", help="Path to subjects file"),
):
    """Prepare data for training."""
    from .scripts.prepare_data import prepare_dataset
    
    # Prepare dataset
    prepare_dataset(
        src_dir=str(src_dir),
        out_dir=str(out_dir),
        fps=fps,
        image_size=image_size,
        max_frames=max_frames,
        subjects_file=str(subjects_file) if subjects_file else None,
    )
    
    typer.echo(f"Data prepared successfully in {out_dir}")


@app.command()
def extract_frames(
    video_path: Path = typer.Option(..., "--video", "-v", help="Path to video file"),
    output_dir: Path = typer.Option(..., "--output", "-o", help="Output directory for frames"),
    fps: int = typer.Option(8, "--fps", help="Frames per second for extraction"),
    image_size: int = typer.Option(224, "--image-size", help="Image size for processing"),
    prefix: str = typer.Option("frame", "--prefix", help="Prefix for frame filenames"),
):
    """Extract frames from video."""
    from .scripts.extract_frames import extract_video_frames
    
    # Extract frames
    frame_paths = extract_video_frames(
        video_path=str(video_path),
        output_dir=str(output_dir),
        fps=fps,
        image_size=image_size,
        prefix=prefix,
    )
    
    typer.echo(f"Extracted {len(frame_paths)} frames to {output_dir}")


@app.command()
def split_folds(
    data_dir: Path = typer.Option(..., "--data-dir", "-d", help="Data directory"),
    output_file: Path = typer.Option(..., "--output", "-o", help="Output file for splits"),
    train_ratio: float = typer.Option(0.7, "--train-ratio", help="Training split ratio"),
    val_ratio: float = typer.Option(0.15, "--val-ratio", help="Validation split ratio"),
    test_ratio: float = typer.Option(0.15, "--test-ratio", help="Test split ratio"),
    random_seed: int = typer.Option(42, "--random-seed", help="Random seed"),
):
    """Split data into train/val/test folds."""
    from .scripts.split_folds import create_data_splits
    
    # Create splits
    splits = create_data_splits(
        data_dir=str(data_dir),
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        random_seed=random_seed,
    )
    
    # Save splits
    save_json(splits, output_file)
    
    typer.echo(f"Data splits saved to {output_file}")


@app.command()
def evaluate(
    model_path: Path = typer.Option(..., "--model", "-m", help="Path to saved model"),
    data_dir: Path = typer.Option(..., "--data-dir", "-d", help="Data directory"),
    output_dir: Path = typer.Option(..., "--output-dir", "-o", help="Output directory"),
    config: Optional[Path] = typer.Option(None, "--config", "-c", help="Path to config file"),
    model_name: str = typer.Option("xception", "--model-name", help="Model architecture name"),
    device: str = typer.Option("auto", "--device", help="Device to run on"),
    batch_size: int = typer.Option(32, "--batch-size", help="Batch size"),
):
    """Evaluate model on test data."""
    from .training.evaluate import evaluate_model
    
    # Load configuration
    config_dict = None
    if config:
        config_dict = load_config(config)
    
    # Evaluate model
    results = evaluate_model(
        model_path=str(model_path),
        data_dir=str(data_dir),
        output_dir=str(output_dir),
        config=config_dict,
        model_name=model_name,
        device=device,
        batch_size=batch_size,
    )
    
    typer.echo(f"Evaluation completed. Results saved to {output_dir}")


@app.command()
def explain(
    input_path: Path = typer.Option(..., "--input", "-i", help="Path to input file"),
    model_path: Path = typer.Option(..., "--model", "-m", help="Path to saved model"),
    output_dir: Path = typer.Option(..., "--output-dir", "-o", help="Output directory"),
    model_name: str = typer.Option("xception", "--model-name", help="Model architecture name"),
    device: str = typer.Option("auto", "--device", help="Device to run on"),
    method: str = typer.Option("gradcam", "--method", help="Explanation method"),
    class_idx: Optional[int] = typer.Option(None, "--class-idx", help="Class index to explain"),
):
    """Generate explanations for model predictions."""
    from .training.explain import generate_explanation
    
    # Generate explanation
    explanation = generate_explanation(
        input_path=str(input_path),
        model_path=str(model_path),
        output_dir=str(output_dir),
        model_name=model_name,
        device=device,
        method=method,
        class_idx=class_idx,
    )
    
    typer.echo(f"Explanation generated and saved to {output_dir}")


@app.command()
def serve(
    model_path: Optional[Path] = typer.Option(None, "--model", "-m", help="Path to saved model"),
    host: str = typer.Option("0.0.0.0", "--host", help="Host to bind to"),
    port: int = typer.Option(8000, "--port", help="Port to bind to"),
    model_name: str = typer.Option("xception", "--model-name", help="Model architecture name"),
    device: str = typer.Option("cpu", "--device", help="Device to run on"),
    use_mock: bool = typer.Option(True, "--use-mock", help="Use mock model for testing"),
    workers: int = typer.Option(1, "--workers", help="Number of worker processes"),
):
    """Start API server."""
    import uvicorn
    from .api.server import create_app
    
    # Create FastAPI app
    app_instance = create_app(
        model_path=str(model_path) if model_path else None,
        model_name=model_name,
        device=device,
        use_mock=use_mock,
    )
    
    # Start server
    uvicorn.run(
        app_instance,
        host=host,
        port=port,
        workers=workers,
    )


@app.command()
def test(
    test_dir: Optional[Path] = typer.Option(None, "--test-dir", help="Test directory"),
    coverage: bool = typer.Option(False, "--coverage", help="Run with coverage"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output"),
):
    """Run tests."""
    import subprocess
    import sys
    
    # Build test command
    cmd = [sys.executable, "-m", "pytest"]
    
    if test_dir:
        cmd.append(str(test_dir))
    else:
        cmd.append("tests/")
    
    if coverage:
        cmd.extend(["--cov=deepfake_forensics", "--cov-report=html", "--cov-report=term"])
    
    if verbose:
        cmd.append("-v")
    
    # Run tests
    result = subprocess.run(cmd)
    
    if result.returncode != 0:
        typer.echo("Tests failed!", err=True)
        raise typer.Exit(1)
    
    typer.echo("All tests passed!")


@app.command()
def lint(
    check: bool = typer.Option(False, "--check", help="Check only, don't fix"),
    files: Optional[List[Path]] = typer.Option(None, "--files", help="Files to lint"),
):
    """Run linting."""
    import subprocess
    import sys
    
    # Build lint command
    cmd = [sys.executable, "-m", "ruff"]
    
    if check:
        cmd.append("--check")
    else:
        cmd.append("--fix")
    
    if files:
        cmd.extend([str(f) for f in files])
    else:
        cmd.append("deepfake_forensics/")
    
    # Run linting
    result = subprocess.run(cmd)
    
    if result.returncode != 0:
        typer.echo("Linting failed!", err=True)
        raise typer.Exit(1)
    
    typer.echo("Linting completed!")


@app.command()
def format_code(
    files: Optional[List[Path]] = typer.Option(None, "--files", help="Files to format"),
):
    """Format code."""
    import subprocess
    import sys
    
    # Build format command
    cmd = [sys.executable, "-m", "black"]
    
    if files:
        cmd.extend([str(f) for f in files])
    else:
        cmd.append("deepfake_forensics/")
    
    # Run formatting
    result = subprocess.run(cmd)
    
    if result.returncode != 0:
        typer.echo("Formatting failed!", err=True)
        raise typer.Exit(1)
    
    typer.echo("Code formatted!")


if __name__ == "__main__":
    app()
