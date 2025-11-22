"""
Metrics computation utilities for deepfake forensics.

Provides comprehensive metrics for binary classification including accuracy,
precision, recall, F1-score, AUROC, AUPRC, and confusion matrix.
"""

import numpy as np
import torch
from typing import Dict, List, Optional, Tuple, Union, Any
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
    precision_recall_curve,
    roc_curve,
)
import logging

logger = logging.getLogger(__name__)


def compute_metrics(
    y_true: Union[np.ndarray, torch.Tensor, List],
    y_pred: Union[np.ndarray, torch.Tensor, List],
    y_prob: Optional[Union[np.ndarray, torch.Tensor, List]] = None,
    threshold: float = 0.5,
    average: str = "binary",
    zero_division: str = "warn",
) -> Dict[str, float]:
    """
    Compute comprehensive classification metrics.
    
    Args:
        y_true: Ground truth labels (0 or 1)
        y_pred: Predicted labels (0 or 1)
        y_prob: Predicted probabilities (optional)
        threshold: Classification threshold for probabilities
        average: Averaging method for multi-class metrics
        zero_division: How to handle zero division in metrics
        
    Returns:
        Dictionary containing computed metrics
    """
    # Convert to numpy arrays
    y_true = _to_numpy(y_true)
    y_pred = _to_numpy(y_pred)
    
    if y_prob is not None:
        y_prob = _to_numpy(y_prob)
        # Convert probabilities to predictions if not provided
        if y_pred is None:
            y_pred = (y_prob >= threshold).astype(int)
    
    # Basic metrics
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average=average, zero_division=zero_division),
        "recall": recall_score(y_true, y_pred, average=average, zero_division=zero_division),
        "f1": f1_score(y_true, y_pred, average=average, zero_division=zero_division),
    }
    
    # Probability-based metrics
    if y_prob is not None:
        try:
            metrics["auroc"] = roc_auc_score(y_true, y_prob)
        except ValueError as e:
            logger.warning(f"Could not compute AUROC: {e}")
            metrics["auroc"] = 0.0
        
        try:
            metrics["auprc"] = average_precision_score(y_true, y_prob)
        except ValueError as e:
            logger.warning(f"Could not compute AUPRC: {e}")
            metrics["auprc"] = 0.0
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        metrics.update({
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_positives": int(tp),
            "specificity": tn / (tn + fp) if (tn + fp) > 0 else 0.0,
            "sensitivity": tp / (tp + fn) if (tp + fn) > 0 else 0.0,
        })
    
    return metrics


def compute_detailed_metrics(
    y_true: Union[np.ndarray, torch.Tensor, List],
    y_prob: Union[np.ndarray, torch.Tensor, List],
    thresholds: Optional[List[float]] = None,
) -> Dict[str, Any]:
    """
    Compute detailed metrics including curves and per-threshold metrics.
    
    Args:
        y_true: Ground truth labels
        y_prob: Predicted probabilities
        thresholds: List of thresholds to evaluate
        
    Returns:
        Dictionary containing detailed metrics
    """
    y_true = _to_numpy(y_true)
    y_prob = _to_numpy(y_prob)
    
    # ROC curve
    fpr, tpr, roc_thresholds = roc_curve(y_true, y_prob)
    roc_auc = roc_auc_score(y_true, y_prob)
    
    # Precision-Recall curve
    precision, recall, pr_thresholds = precision_recall_curve(y_true, y_prob)
    pr_auc = average_precision_score(y_true, y_prob)
    
    # Per-threshold metrics
    if thresholds is None:
        thresholds = np.linspace(0.0, 1.0, 101)
    
    threshold_metrics = []
    for threshold in thresholds:
        y_pred = (y_prob >= threshold).astype(int)
        metrics = compute_metrics(y_true, y_pred, threshold=threshold)
        metrics["threshold"] = threshold
        threshold_metrics.append(metrics)
    
    return {
        "roc_curve": {
            "fpr": fpr.tolist(),
            "tpr": tpr.tolist(),
            "thresholds": roc_thresholds.tolist(),
            "auc": roc_auc,
        },
        "pr_curve": {
            "precision": precision.tolist(),
            "recall": recall.tolist(),
            "thresholds": pr_thresholds.tolist(),
            "auc": pr_auc,
        },
        "threshold_metrics": threshold_metrics,
        "best_threshold": _find_best_threshold(threshold_metrics),
    }


def _find_best_threshold(threshold_metrics: List[Dict[str, float]]) -> float:
    """
    Find the best threshold based on F1 score.
    
    Args:
        threshold_metrics: List of metrics for different thresholds
        
    Returns:
        Best threshold value
    """
    best_f1 = -1
    best_threshold = 0.5
    
    for metrics in threshold_metrics:
        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            best_threshold = metrics["threshold"]
    
    return best_threshold


def compute_per_class_metrics(
    y_true: Union[np.ndarray, torch.Tensor, List],
    y_pred: Union[np.ndarray, torch.Tensor, List],
    class_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Compute per-class metrics.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        class_names: Names for classes
        
    Returns:
        Dictionary containing per-class metrics
    """
    y_true = _to_numpy(y_true)
    y_pred = _to_numpy(y_pred)
    
    if class_names is None:
        class_names = ["Class 0", "Class 1"]
    
    # Classification report
    report = classification_report(
        y_true, y_pred, 
        target_names=class_names, 
        output_dict=True,
        zero_division=0
    )
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    return {
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
        "class_names": class_names,
    }


def compute_confidence_metrics(
    y_true: Union[np.ndarray, torch.Tensor, List],
    y_prob: Union[np.ndarray, torch.Tensor, List],
    confidence_bins: int = 10,
) -> Dict[str, Any]:
    """
    Compute confidence-based metrics.
    
    Args:
        y_true: Ground truth labels
        y_prob: Predicted probabilities
        confidence_bins: Number of confidence bins
        
    Returns:
        Dictionary containing confidence metrics
    """
    y_true = _to_numpy(y_true)
    y_prob = _to_numpy(y_prob)
    
    # Create confidence bins
    bin_edges = np.linspace(0, 1, confidence_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    bin_metrics = []
    for i in range(confidence_bins):
        mask = (y_prob >= bin_edges[i]) & (y_prob < bin_edges[i + 1])
        if i == confidence_bins - 1:  # Include upper bound for last bin
            mask = (y_prob >= bin_edges[i]) & (y_prob <= bin_edges[i + 1])
        
        if np.sum(mask) > 0:
            bin_y_true = y_true[mask]
            bin_y_prob = y_prob[mask]
            
            # Compute metrics for this bin
            accuracy = np.mean((bin_y_prob >= 0.5) == bin_y_true)
            avg_confidence = np.mean(bin_y_prob)
            count = len(bin_y_true)
            
            bin_metrics.append({
                "bin_center": bin_centers[i],
                "bin_range": [bin_edges[i], bin_edges[i + 1]],
                "accuracy": accuracy,
                "avg_confidence": avg_confidence,
                "count": count,
            })
    
    return {
        "bin_metrics": bin_metrics,
        "calibration_error": _compute_calibration_error(bin_metrics),
    }


def _compute_calibration_error(bin_metrics: List[Dict[str, float]]) -> float:
    """
    Compute expected calibration error.
    
    Args:
        bin_metrics: List of metrics for each confidence bin
        
    Returns:
        Expected calibration error
    """
    total_samples = sum(bin["count"] for bin in bin_metrics)
    if total_samples == 0:
        return 0.0
    
    ece = 0.0
    for bin_metric in bin_metrics:
        count = bin_metric["count"]
        accuracy = bin_metric["accuracy"]
        confidence = bin_metric["avg_confidence"]
        
        ece += (count / total_samples) * abs(accuracy - confidence)
    
    return ece


def _to_numpy(data: Union[np.ndarray, torch.Tensor, List]) -> np.ndarray:
    """
    Convert data to numpy array.
    
    Args:
        data: Input data
        
    Returns:
        Numpy array
    """
    if isinstance(data, torch.Tensor):
        return data.detach().cpu().numpy()
    elif isinstance(data, list):
        return np.array(data)
    else:
        return data


class MetricsTracker:
    """
    Track metrics during training and evaluation.
    """
    
    def __init__(self, name: str = "metrics"):
        """
        Initialize metrics tracker.
        
        Args:
            name: Name for the tracker
        """
        self.name = name
        self.reset()
    
    def reset(self) -> None:
        """Reset all tracked metrics."""
        self.predictions = []
        self.targets = []
        self.probabilities = []
        self.losses = []
        self.metrics_history = []
    
    def update(
        self,
        predictions: Union[np.ndarray, torch.Tensor, List],
        targets: Union[np.ndarray, torch.Tensor, List],
        probabilities: Optional[Union[np.ndarray, torch.Tensor, List]] = None,
        loss: Optional[float] = None,
    ) -> None:
        """
        Update tracker with new batch of data.
        
        Args:
            predictions: Predicted labels
            targets: Ground truth labels
            probabilities: Predicted probabilities
            loss: Loss value
        """
        self.predictions.extend(_to_numpy(predictions).tolist())
        self.targets.extend(_to_numpy(targets).tolist())
        
        if probabilities is not None:
            self.probabilities.extend(_to_numpy(probabilities).tolist())
        
        if loss is not None:
            self.losses.append(loss)
    
    def compute_metrics(self, threshold: float = 0.5) -> Dict[str, float]:
        """
        Compute metrics from tracked data.
        
        Args:
            threshold: Classification threshold
            
        Returns:
            Dictionary of computed metrics
        """
        if not self.predictions:
            return {}
        
        y_true = np.array(self.targets)
        y_pred = np.array(self.predictions)
        y_prob = np.array(self.probabilities) if self.probabilities else None
        
        metrics = compute_metrics(y_true, y_pred, y_prob, threshold)
        
        if self.losses:
            metrics["loss"] = np.mean(self.losses)
        
        return metrics
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get summary of tracked metrics.
        
        Returns:
            Dictionary containing summary information
        """
        return {
            "name": self.name,
            "num_samples": len(self.predictions),
            "num_batches": len(self.losses) if self.losses else 0,
            "avg_loss": np.mean(self.losses) if self.losses else None,
            "metrics": self.compute_metrics(),
        }
    
    def save_metrics(self, filepath: str) -> None:
        """
        Save metrics to file.
        
        Args:
            filepath: Path to save metrics
        """
        import json
        from pathlib import Path
        
        metrics_data = {
            "predictions": self.predictions,
            "targets": self.targets,
            "probabilities": self.probabilities,
            "losses": self.losses,
            "summary": self.get_summary(),
        }
        
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(metrics_data, f, indent=2)
        
        logger.info(f"Saved metrics to {filepath}")


def compute_ensemble_metrics(
    predictions_list: List[Union[np.ndarray, torch.Tensor, List]],
    targets: Union[np.ndarray, torch.Tensor, List],
    weights: Optional[List[float]] = None,
    method: str = "average",
) -> Dict[str, Any]:
    """
    Compute metrics for ensemble predictions.
    
    Args:
        predictions_list: List of predictions from different models
        targets: Ground truth labels
        weights: Weights for each model (optional)
        method: Ensemble method ('average', 'voting', 'weighted_average')
        
    Returns:
        Dictionary containing ensemble metrics
    """
    targets = _to_numpy(targets)
    
    # Convert all predictions to numpy
    predictions_list = [_to_numpy(pred) for pred in predictions_list]
    
    if method == "average":
        # Simple average
        ensemble_pred = np.mean(predictions_list, axis=0)
    elif method == "voting":
        # Majority voting
        ensemble_pred = np.round(np.mean(predictions_list, axis=0))
    elif method == "weighted_average":
        # Weighted average
        if weights is None:
            weights = [1.0] * len(predictions_list)
        weights = np.array(weights)
        weights = weights / np.sum(weights)  # Normalize
        
        ensemble_pred = np.average(predictions_list, axis=0, weights=weights)
    else:
        raise ValueError(f"Unknown ensemble method: {method}")
    
    # Compute metrics
    metrics = compute_metrics(targets, ensemble_pred, ensemble_pred)
    
    # Add ensemble-specific metrics
    metrics["ensemble_method"] = method
    metrics["num_models"] = len(predictions_list)
    
    if weights is not None:
        metrics["model_weights"] = weights.tolist()
    
    return metrics
