"""
Training script for deepfake detection models.

Provides comprehensive training functionality with PyTorch Lightning,
including data loading, model training, validation, and testing.
"""

import os
import yaml
import argparse
from pathlib import Path
from typing import Dict, Any, Optional, List
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from pytorch_lightning.strategies import DDPStrategy
import logging

from .module import DeepfakeLightningModule, create_lightning_module
from ..data import create_data_loaders
from ..utils.logging import setup_logging, get_logger
from ..utils.seeds import set_seed
from ..utils.io import load_config

logger = get_logger(__name__)


def train_model(
    config: Dict[str, Any],
    data_dir: str,
    output_dir: str,
    resume_from_checkpoint: Optional[str] = None,
    fast_dev_run: bool = False,
) -> str:
    """
    Train a deepfake detection model.
    
    Args:
        config: Training configuration
        data_dir: Data directory
        output_dir: Output directory
        resume_from_checkpoint: Path to checkpoint to resume from
        fast_dev_run: Whether to run fast dev run
        
    Returns:
        Path to best checkpoint
    """
    # Set up logging
    setup_logging(
        level=config.get("logging", {}).get("level", "INFO"),
        log_file=Path(output_dir) / "training.log",
    )
    
    # Set seed for reproducibility
    seed = config.get("seed", 42)
    set_seed(seed, deterministic=True)
    
    logger.info("Starting training...")
    logger.info(f"Configuration: {config}")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create data loaders
    data_config = config.get("data", {})
    data_loaders = create_data_loaders(
        data_dir=data_dir,
        batch_size=data_config.get("batch_size", 32),
        num_workers=data_config.get("num_workers", 4),
        image_size=data_config.get("image_size", 224),
        max_frames=data_config.get("max_frames", 16),
        fps=data_config.get("fps", 8),
        train_split=data_config.get("train_split", 0.7),
        val_split=data_config.get("val_split", 0.15),
        test_split=data_config.get("test_split", 0.15),
        subjects_file=data_config.get("subjects_file"),
        use_balanced_sampling=data_config.get("use_balanced_sampling", True),
    )
    
    # Create model
    model_config = config.get("model", {})
    model = create_lightning_module(
        model_name=model_config.get("name", "xception"),
        num_classes=model_config.get("num_classes", 2),
        learning_rate=config.get("training", {}).get("learning_rate", 1e-4),
        weight_decay=config.get("training", {}).get("weight_decay", 1e-4),
        loss_name=config.get("training", {}).get("loss", {}).get("name", "bce"),
        focal_alpha=config.get("training", {}).get("loss", {}).get("focal_alpha", 0.25),
        focal_gamma=config.get("training", {}).get("loss", {}).get("focal_gamma", 2.0),
        label_smoothing=config.get("training", {}).get("loss", {}).get("label_smoothing", 0.1),
        dropout=model_config.get("dropout", 0.5),
        **{k: v for k, v in model_config.items() if k not in ["name", "num_classes", "dropout"]},
    )
    
    # Create callbacks
    callbacks = create_callbacks(config, output_path)
    
    # Create logger
    logger_obj = create_logger(config, output_path)
    
    # Create trainer
    trainer = create_trainer(config, callbacks, logger_obj, output_path)
    
    # Train model
    trainer.fit(
        model,
        train_dataloaders=data_loaders["train"],
        val_dataloaders=data_loaders["val"],
        ckpt_path=resume_from_checkpoint,
    )
    
    # Test model
    if data_loaders["test"] is not None:
        trainer.test(
            model,
            dataloaders=data_loaders["test"],
            ckpt_path="best",
        )
    
    # Get best checkpoint path
    best_checkpoint = trainer.checkpoint_callback.best_model_path
    
    logger.info(f"Training completed. Best checkpoint: {best_checkpoint}")
    
    return best_checkpoint


def create_callbacks(config: Dict[str, Any], output_path: Path) -> List:
    """Create training callbacks."""
    callbacks = []
    
    # Model checkpoint
    checkpoint_callback = ModelCheckpoint(
        dirpath=output_path / "checkpoints",
        filename="best-{epoch:02d}-{val_auroc:.3f}",
        monitor="val_auroc",
        mode="max",
        save_top_k=3,
        save_last=True,
        verbose=True,
    )
    callbacks.append(checkpoint_callback)
    
    # Early stopping
    early_stopping_config = config.get("training", {}).get("early_stopping", {})
    if early_stopping_config.get("enabled", True):
        early_stopping = EarlyStopping(
            monitor=early_stopping_config.get("monitor", "val_auroc"),
            mode=early_stopping_config.get("mode", "max"),
            patience=early_stopping_config.get("patience", 10),
            min_delta=early_stopping_config.get("min_delta", 0.001),
            verbose=True,
        )
        callbacks.append(early_stopping)
    
    # Learning rate monitor
    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    callbacks.append(lr_monitor)
    
    return callbacks


def create_logger(config: Dict[str, Any], output_path: Path) -> Optional[pl.loggers.Logger]:
    """Create training logger."""
    logging_config = config.get("logging", {})
    experiment_tracker = logging_config.get("experiment_tracker", "tensorboard")
    
    if experiment_tracker == "tensorboard":
        return TensorBoardLogger(
            save_dir=str(output_path / "logs"),
            name=logging_config.get("project_name", "deepfake-forensics"),
            version=logging_config.get("run_name", "run"),
        )
    elif experiment_tracker == "wandb":
        return WandbLogger(
            project=logging_config.get("project_name", "deepfake-forensics"),
            name=logging_config.get("run_name", "run"),
            save_dir=str(output_path / "logs"),
        )
    else:
        return None


def create_trainer(
    config: Dict[str, Any],
    callbacks: List,
    logger: Optional[pl.loggers.Logger],
    output_path: Path,
) -> pl.Trainer:
    """Create PyTorch Lightning trainer."""
    training_config = config.get("training", {})
    
    # Trainer arguments
    trainer_args = {
        "max_epochs": training_config.get("max_epochs", 100),
        "callbacks": callbacks,
        "logger": logger,
        "precision": training_config.get("precision", 16),
        "gradient_clip_val": training_config.get("gradient_clip_val", 1.0),
        "accumulate_grad_batches": training_config.get("accumulate_grad_batches", 1),
        "val_check_interval": config.get("validation", {}).get("val_check_interval", 1.0),
        "check_val_every_n_epoch": config.get("validation", {}).get("check_val_every_n_epoch", 1),
        "log_every_n_steps": config.get("logging", {}).get("log_every_n_steps", 10),
        "enable_checkpointing": True,
        "enable_progress_bar": True,
        "enable_model_summary": True,
    }
    
    # Device configuration
    device_config = config.get("device", "auto")
    if device_config == "auto":
        if torch.cuda.is_available():
            trainer_args["devices"] = "auto"
            trainer_args["accelerator"] = "gpu"
        else:
            trainer_args["devices"] = 1
            trainer_args["accelerator"] = "cpu"
    elif device_config == "cpu":
        trainer_args["devices"] = 1
        trainer_args["accelerator"] = "cpu"
    else:
        trainer_args["devices"] = device_config
        trainer_args["accelerator"] = "gpu"
    
    # Multi-GPU configuration
    num_gpus = config.get("num_gpus", 1)
    if num_gpus > 1 and torch.cuda.is_available():
        trainer_args["strategy"] = DDPStrategy(find_unused_parameters=False)
        trainer_args["devices"] = num_gpus
    
    # Fast dev run
    if config.get("fast_dev_run", False):
        trainer_args["fast_dev_run"] = True
        trainer_args["max_epochs"] = 1
        trainer_args["limit_train_batches"] = 2
        trainer_args["limit_val_batches"] = 2
        trainer_args["limit_test_batches"] = 2
    
    return pl.Trainer(**trainer_args)


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train deepfake detection model")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to data directory")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to output directory")
    parser.add_argument("--resume_from", type=str, help="Path to checkpoint to resume from")
    parser.add_argument("--fast_dev_run", action="store_true", help="Run fast dev run")
    parser.add_argument("--gpus", type=int, help="Number of GPUs to use")
    parser.add_argument("--batch_size", type=int, help="Batch size")
    parser.add_argument("--learning_rate", type=float, help="Learning rate")
    parser.add_argument("--max_epochs", type=int, help="Maximum number of epochs")
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Override config with command line arguments
    if args.gpus is not None:
        config["num_gpus"] = args.gpus
    if args.batch_size is not None:
        config["data"]["batch_size"] = args.batch_size
    if args.learning_rate is not None:
        config["training"]["learning_rate"] = args.learning_rate
    if args.max_epochs is not None:
        config["training"]["max_epochs"] = args.max_epochs
    
    # Train model
    best_checkpoint = train_model(
        config=config,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        resume_from_checkpoint=args.resume_from,
        fast_dev_run=args.fast_dev_run,
    )
    
    print(f"Training completed. Best checkpoint: {best_checkpoint}")


if __name__ == "__main__":
    main()
