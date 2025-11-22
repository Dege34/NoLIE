"""
PyTorch Lightning module for deepfake detection.

Implements the main training module with comprehensive metrics,
loss functions, and optimization strategies.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from typing import Dict, Any, Optional, Tuple, List, Union
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import logging

from ..models import XceptionForensics, ViTForensics, ResNetFrequencyForensics, AudioVisualModel
from ..utils.metrics import compute_metrics, MetricsTracker
from ..utils.logging import get_logger

logger = get_logger(__name__)


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance.
    """
    
    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        reduction: str = "mean",
    ):
        """
        Initialize focal loss.
        
        Args:
            alpha: Weighting factor for rare class
            gamma: Focusing parameter
            reduction: Reduction method
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            inputs: Predicted logits (B, num_classes)
            targets: Ground truth labels (B,)
            
        Returns:
            Loss value
        """
        # Convert to probabilities
        probs = F.softmax(inputs, dim=1)
        
        # Get probabilities for true class
        pt = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        
        # Compute focal loss
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        focal_weight = alpha_t * (1 - pt) ** self.gamma
        
        # Cross entropy loss
        ce_loss = F.cross_entropy(inputs, targets, reduction="none")
        
        # Focal loss
        focal_loss = focal_weight * ce_loss
        
        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss


class LabelSmoothingLoss(nn.Module):
    """
    Label smoothing loss for regularization.
    """
    
    def __init__(
        self,
        num_classes: int,
        smoothing: float = 0.1,
        reduction: str = "mean",
    ):
        """
        Initialize label smoothing loss.
        
        Args:
            num_classes: Number of classes
            smoothing: Smoothing factor
            reduction: Reduction method
        """
        super().__init__()
        self.num_classes = num_classes
        self.smoothing = smoothing
        self.reduction = reduction
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            inputs: Predicted logits (B, num_classes)
            targets: Ground truth labels (B,)
            
        Returns:
            Loss value
        """
        # Convert targets to one-hot
        targets_one_hot = torch.zeros_like(inputs)
        targets_one_hot.scatter_(1, targets.unsqueeze(1), 1)
        
        # Apply label smoothing
        smoothed_targets = (1 - self.smoothing) * targets_one_hot + \
                          self.smoothing / self.num_classes
        
        # Compute loss
        log_probs = F.log_softmax(inputs, dim=1)
        loss = -torch.sum(smoothed_targets * log_probs, dim=1)
        
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


class DeepfakeLightningModule(pl.LightningModule):
    """
    PyTorch Lightning module for deepfake detection.
    """
    
    def __init__(
        self,
        model_name: str = "xception",
        num_classes: int = 2,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-4,
        loss_name: str = "bce",
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        label_smoothing: float = 0.1,
        dropout: float = 0.5,
        **model_kwargs,
    ):
        """
        Initialize deepfake detection module.
        
        Args:
            model_name: Name of the model architecture
            num_classes: Number of output classes
            learning_rate: Learning rate
            weight_decay: Weight decay
            loss_name: Loss function name
            focal_alpha: Focal loss alpha parameter
            focal_gamma: Focal loss gamma parameter
            label_smoothing: Label smoothing factor
            dropout: Dropout rate
            **model_kwargs: Additional model arguments
        """
        super().__init__()
        
        self.save_hyperparameters()
        
        self.model_name = model_name
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.loss_name = loss_name
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.label_smoothing = label_smoothing
        self.dropout = dropout
        
        # Build model
        self.model = self._build_model(**model_kwargs)
        
        # Build loss function
        self.criterion = self._build_loss()
        
        # Metrics tracker
        self.train_metrics = MetricsTracker("train")
        self.val_metrics = MetricsTracker("val")
        self.test_metrics = MetricsTracker("test")
    
    def _build_model(self, **kwargs) -> nn.Module:
        """Build the model architecture."""
        if self.model_name == "xception":
            return XceptionForensics(
                num_classes=self.num_classes,
                dropout=self.dropout,
                **kwargs
            )
        elif self.model_name == "vit":
            return ViTForensics(
                num_classes=self.num_classes,
                dropout=self.dropout,
                **kwargs
            )
        elif self.model_name == "resnet_freq":
            return ResNetFrequencyForensics(
                num_classes=self.num_classes,
                dropout=self.dropout,
                **kwargs
            )
        elif self.model_name == "audio_visual":
            return AudioVisualModel(
                num_classes=self.num_classes,
                **kwargs
            )
        else:
            raise ValueError(f"Unknown model: {self.model_name}")
    
    def _build_loss(self) -> nn.Module:
        """Build the loss function."""
        if self.loss_name == "bce":
            return nn.BCEWithLogitsLoss()
        elif self.loss_name == "ce":
            return nn.CrossEntropyLoss()
        elif self.loss_name == "focal":
            return FocalLoss(
                alpha=self.focal_alpha,
                gamma=self.focal_gamma,
            )
        elif self.loss_name == "label_smoothing":
            return LabelSmoothingLoss(
                num_classes=self.num_classes,
                smoothing=self.label_smoothing,
            )
        else:
            raise ValueError(f"Unknown loss: {self.loss_name}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.model(x)
    
    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Training step."""
        x, y = batch
        
        # Forward pass
        logits = self.forward(x)
        
        # Compute loss
        if self.loss_name == "bce":
            loss = self.criterion(logits, y.float())
        else:
            loss = self.criterion(logits, y)
        
        # Compute metrics
        with torch.no_grad():
            if self.loss_name == "bce":
                probs = torch.sigmoid(logits)
                preds = (probs > 0.5).long()
            else:
                probs = F.softmax(logits, dim=1)
                preds = torch.argmax(logits, dim=1)
            
            # Update metrics
            self.train_metrics.update(preds, y, probs, loss.item())
        
        # Log metrics
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        
        return loss
    
    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Validation step."""
        x, y = batch
        
        # Forward pass
        logits = self.forward(x)
        
        # Compute loss
        if self.loss_name == "bce":
            loss = self.criterion(logits, y.float())
        else:
            loss = self.criterion(logits, y)
        
        # Compute metrics
        with torch.no_grad():
            if self.loss_name == "bce":
                probs = torch.sigmoid(logits)
                preds = (probs > 0.5).long()
            else:
                probs = F.softmax(logits, dim=1)
                preds = torch.argmax(logits, dim=1)
            
            # Update metrics
            self.val_metrics.update(preds, y, probs, loss.item())
        
        # Log metrics
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        
        return loss
    
    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Test step."""
        x, y = batch
        
        # Forward pass
        logits = self.forward(x)
        
        # Compute loss
        if self.loss_name == "bce":
            loss = self.criterion(logits, y.float())
        else:
            loss = self.criterion(logits, y)
        
        # Compute metrics
        with torch.no_grad():
            if self.loss_name == "bce":
                probs = torch.sigmoid(logits)
                preds = (probs > 0.5).long()
            else:
                probs = F.softmax(logits, dim=1)
                preds = torch.argmax(logits, dim=1)
            
            # Update metrics
            self.test_metrics.update(preds, y, probs, loss.item())
        
        # Log metrics
        self.log("test/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        
        return loss
    
    def on_validation_epoch_end(self) -> None:
        """Validation epoch end."""
        # Compute metrics
        metrics = self.val_metrics.compute_metrics()
        
        # Log metrics
        for key, value in metrics.items():
            self.log(f"val/{key}", value, on_epoch=True, prog_bar=True)
        
        # Reset metrics
        self.val_metrics.reset()
    
    def on_test_epoch_end(self) -> None:
        """Test epoch end."""
        # Compute metrics
        metrics = self.test_metrics.compute_metrics()
        
        # Log metrics
        for key, value in metrics.items():
            self.log(f"test/{key}", value, on_epoch=True, prog_bar=True)
        
        # Reset metrics
        self.test_metrics.reset()
    
    def configure_optimizers(self) -> Dict[str, Any]:
        """Configure optimizers and schedulers."""
        # Optimizer
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        
        # Scheduler
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.trainer.max_epochs,
            eta_min=1e-6,
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
            },
        }
    
    def get_attention_maps(self, x: torch.Tensor) -> torch.Tensor:
        """Get attention maps for visualization."""
        if hasattr(self.model, 'get_attention_maps'):
            return self.model.get_attention_maps(x)
        else:
            # Return dummy attention map
            return torch.ones(x.size(0), 1, x.size(2), x.size(3), device=x.device)


class DeepfakeDetector:
    """
    High-level deepfake detector interface.
    """
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        model_name: str = "xception",
        device: str = "auto",
        **model_kwargs,
    ):
        """
        Initialize deepfake detector.
        
        Args:
            model_path: Path to saved model
            model_name: Model architecture name
            device: Device to run on
            **model_kwargs: Model arguments
        """
        self.model_name = model_name
        self.device = self._get_device(device)
        self.model_kwargs = model_kwargs
        
        # Load model
        if model_path:
            self.model = self._load_model(model_path)
        else:
            self.model = self._build_model()
        
        self.model.to(self.device)
        self.model.eval()
    
    def _get_device(self, device: str) -> torch.device:
        """Get device."""
        if device == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            else:
                return torch.device("cpu")
        else:
            return torch.device(device)
    
    def _build_model(self) -> nn.Module:
        """Build model."""
        if self.model_name == "xception":
            return XceptionForensics(**self.model_kwargs)
        elif self.model_name == "vit":
            return ViTForensics(**self.model_kwargs)
        elif self.model_name == "resnet_freq":
            return ResNetFrequencyForensics(**self.model_kwargs)
        elif self.model_name == "audio_visual":
            return AudioVisualModel(**self.model_kwargs)
        else:
            raise ValueError(f"Unknown model: {self.model_name}")
    
    def _load_model(self, model_path: str) -> nn.Module:
        """Load model from checkpoint."""
        checkpoint = torch.load(model_path, map_location=self.device)
        
        if "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint
        
        # Build model
        model = self._build_model()
        
        # Load state dict
        model.load_state_dict(state_dict)
        
        return model
    
    def predict(
        self,
        x: torch.Tensor,
        return_probabilities: bool = True,
        return_attention: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Predict deepfake probability.
        
        Args:
            x: Input tensor
            return_probabilities: Whether to return probabilities
            return_attention: Whether to return attention maps
            
        Returns:
            Dictionary containing predictions
        """
        with torch.no_grad():
            # Move to device
            x = x.to(self.device)
            
            # Forward pass
            logits = self.model(x)
            
            # Convert to probabilities
            if return_probabilities:
                if logits.size(1) == 1:
                    probs = torch.sigmoid(logits)
                else:
                    probs = F.softmax(logits, dim=1)
            else:
                probs = None
            
            # Get predictions
            if logits.size(1) == 1:
                preds = (logits > 0).long()
            else:
                preds = torch.argmax(logits, dim=1)
            
            # Get attention maps if requested
            attention = None
            if return_attention and hasattr(self.model, 'get_attention_maps'):
                attention = self.model.get_attention_maps(x)
            
            # Prepare results
            results = {
                "logits": logits,
                "predictions": preds,
            }
            
            if probs is not None:
                results["probabilities"] = probs
            
            if attention is not None:
                results["attention"] = attention
            
            return results
    
    def predict_batch(
        self,
        x: torch.Tensor,
        batch_size: int = 32,
        return_probabilities: bool = True,
        return_attention: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Predict on batch of data.
        
        Args:
            x: Input tensor
            batch_size: Batch size for processing
            return_probabilities: Whether to return probabilities
            return_attention: Whether to return attention maps
            
        Returns:
            Dictionary containing predictions
        """
        results = {
            "logits": [],
            "predictions": [],
        }
        
        if return_probabilities:
            results["probabilities"] = []
        
        if return_attention:
            results["attention"] = []
        
        # Process in batches
        for i in range(0, x.size(0), batch_size):
            batch = x[i:i + batch_size]
            batch_results = self.predict(
                batch,
                return_probabilities=return_probabilities,
                return_attention=return_attention,
            )
            
            # Collect results
            for key, value in batch_results.items():
                results[key].append(value)
        
        # Concatenate results
        for key in results:
            results[key] = torch.cat(results[key], dim=0)
        
        return results


def create_lightning_module(
    model_name: str = "xception",
    num_classes: int = 2,
    learning_rate: float = 1e-4,
    weight_decay: float = 1e-4,
    loss_name: str = "bce",
    **kwargs,
) -> DeepfakeLightningModule:
    """
    Create PyTorch Lightning module.
    
    Args:
        model_name: Model architecture name
        num_classes: Number of output classes
        learning_rate: Learning rate
        weight_decay: Weight decay
        loss_name: Loss function name
        **kwargs: Additional arguments
        
    Returns:
        DeepfakeLightningModule instance
    """
    return DeepfakeLightningModule(
        model_name=model_name,
        num_classes=num_classes,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        loss_name=loss_name,
        **kwargs,
    )


def create_detector(
    model_path: Optional[str] = None,
    model_name: str = "xception",
    device: str = "auto",
    **kwargs,
) -> DeepfakeDetector:
    """
    Create deepfake detector.
    
    Args:
        model_path: Path to saved model
        model_name: Model architecture name
        device: Device to run on
        **kwargs: Additional arguments
        
    Returns:
        DeepfakeDetector instance
    """
    return DeepfakeDetector(
        model_path=model_path,
        model_name=model_name,
        device=device,
        **kwargs,
    )
