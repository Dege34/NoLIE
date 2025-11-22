"""
Attribution methods for deepfake detection.

Provides various attribution methods including integrated gradients,
saliency maps, and smooth gradients for explaining model predictions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from typing import Optional, List, Dict, Any, Union, Tuple, Callable
import logging

logger = logging.getLogger(__name__)


class IntegratedGradients:
    """
    Integrated Gradients implementation for deepfake detection.
    """
    
    def __init__(
        self,
        model: nn.Module,
        steps: int = 50,
        device: str = "cpu",
    ):
        """
        Initialize Integrated Gradients.
        
        Args:
            model: PyTorch model
            steps: Number of integration steps
            device: Device to run on
        """
        self.model = model
        self.steps = steps
        self.device = device
    
    def generate_attribution(
        self,
        input_tensor: torch.Tensor,
        class_idx: Optional[int] = None,
        baseline: Optional[torch.Tensor] = None,
    ) -> np.ndarray:
        """
        Generate integrated gradients attribution.
        
        Args:
            input_tensor: Input tensor (B, C, H, W)
            class_idx: Class index to generate attribution for
            baseline: Baseline tensor (None for zeros)
            
        Returns:
            Attribution map (H, W)
        """
        # Move to device
        input_tensor = input_tensor.to(self.device)
        
        # Set baseline
        if baseline is None:
            baseline = torch.zeros_like(input_tensor)
        else:
            baseline = baseline.to(self.device)
        
        # Get class index
        if class_idx is None:
            with torch.no_grad():
                output = self.model(input_tensor)
                class_idx = torch.argmax(output, dim=1).item()
        
        # Generate integrated gradients
        integrated_gradients = torch.zeros_like(input_tensor)
        
        for i in range(self.steps):
            # Create interpolated input
            alpha = i / (self.steps - 1)
            interpolated_input = baseline + alpha * (input_tensor - baseline)
            interpolated_input.requires_grad_(True)
            
            # Forward pass
            output = self.model(interpolated_input)
            
            # Backward pass
            self.model.zero_grad()
            output[0, class_idx].backward(retain_graph=True)
            
            # Accumulate gradients
            integrated_gradients += interpolated_input.grad / self.steps
        
        # Compute attribution
        attribution = (input_tensor - baseline) * integrated_gradients
        
        # Sum over channels
        attribution = torch.sum(attribution, dim=1).squeeze(0)  # (H, W)
        
        # Convert to numpy
        attribution = attribution.detach().cpu().numpy()
        
        return attribution
    
    def explain(
        self,
        input_tensor: torch.Tensor,
        class_idx: Optional[int] = None,
        baseline: Optional[torch.Tensor] = None,
        return_attribution: bool = True,
        return_overlay: bool = True,
        alpha: float = 0.4,
        colormap: str = "jet",
    ) -> Dict[str, Any]:
        """
        Generate explanation for input.
        
        Args:
            input_tensor: Input tensor (B, C, H, W)
            class_idx: Class index to explain
            baseline: Baseline tensor
            return_attribution: Whether to return attribution map
            return_overlay: Whether to return overlay image
            alpha: Overlay transparency
            colormap: Colormap for visualization
            
        Returns:
            Dictionary containing explanation
        """
        # Generate attribution
        attribution = self.generate_attribution(input_tensor, class_idx, baseline)
        
        # Prepare results
        results = {
            "class_idx": class_idx,
            "attribution": attribution,
        }
        
        # Create overlay if requested
        if return_overlay:
            # Get original image
            if input_tensor.size(0) > 1:
                image = input_tensor[0].detach().cpu().numpy()
            else:
                image = input_tensor[0].detach().cpu().numpy()
            
            # Convert to RGB if needed
            if image.shape[0] == 3:
                image = np.transpose(image, (1, 2, 0))
                image = (image - image.min()) / (image.max() - image.min())
                image = (image * 255).astype(np.uint8)
            else:
                image = (image - image.min()) / (image.max() - image.min())
                image = (image * 255).astype(np.uint8)
            
            # Resize attribution to match image
            attribution_resized = cv2.resize(attribution, (image.shape[1], image.shape[0]))
            
            # Normalize attribution
            attribution_resized = (attribution_resized - attribution_resized.min()) / \
                                (attribution_resized.max() - attribution_resized.min())
            
            # Apply colormap
            attribution_colored = cv2.applyColorMap(
                (attribution_resized * 255).astype(np.uint8),
                getattr(cv2, f"COLORMAP_{colormap.upper()}")
            )
            
            # Create overlay
            overlay = cv2.addWeighted(image, 1 - alpha, attribution_colored, alpha, 0)
            
            results["overlay"] = overlay
        
        return results


class SaliencyMap:
    """
    Saliency map implementation for deepfake detection.
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: str = "cpu",
    ):
        """
        Initialize Saliency Map.
        
        Args:
            model: PyTorch model
            device: Device to run on
        """
        self.model = model
        self.device = device
    
    def generate_attribution(
        self,
        input_tensor: torch.Tensor,
        class_idx: Optional[int] = None,
    ) -> np.ndarray:
        """
        Generate saliency map attribution.
        
        Args:
            input_tensor: Input tensor (B, C, H, W)
            class_idx: Class index to generate attribution for
            
        Returns:
            Attribution map (H, W)
        """
        # Move to device
        input_tensor = input_tensor.to(self.device)
        input_tensor.requires_grad_(True)
        
        # Get class index
        if class_idx is None:
            with torch.no_grad():
                output = self.model(input_tensor)
                class_idx = torch.argmax(output, dim=1).item()
        
        # Forward pass
        output = self.model(input_tensor)
        
        # Backward pass
        self.model.zero_grad()
        output[0, class_idx].backward(retain_graph=True)
        
        # Get gradients
        gradients = input_tensor.grad[0]  # (C, H, W)
        
        # Compute saliency map
        saliency = torch.abs(gradients)
        saliency = torch.sum(saliency, dim=0)  # (H, W)
        
        # Convert to numpy
        saliency = saliency.detach().cpu().numpy()
        
        return saliency
    
    def explain(
        self,
        input_tensor: torch.Tensor,
        class_idx: Optional[int] = None,
        return_attribution: bool = True,
        return_overlay: bool = True,
        alpha: float = 0.4,
        colormap: str = "jet",
    ) -> Dict[str, Any]:
        """
        Generate explanation for input.
        
        Args:
            input_tensor: Input tensor (B, C, H, W)
            class_idx: Class index to explain
            return_attribution: Whether to return attribution map
            return_overlay: Whether to return overlay image
            alpha: Overlay transparency
            colormap: Colormap for visualization
            
        Returns:
            Dictionary containing explanation
        """
        # Generate attribution
        attribution = self.generate_attribution(input_tensor, class_idx)
        
        # Prepare results
        results = {
            "class_idx": class_idx,
            "attribution": attribution,
        }
        
        # Create overlay if requested
        if return_overlay:
            # Get original image
            if input_tensor.size(0) > 1:
                image = input_tensor[0].detach().cpu().numpy()
            else:
                image = input_tensor[0].detach().cpu().numpy()
            
            # Convert to RGB if needed
            if image.shape[0] == 3:
                image = np.transpose(image, (1, 2, 0))
                image = (image - image.min()) / (image.max() - image.min())
                image = (image * 255).astype(np.uint8)
            else:
                image = (image - image.min()) / (image.max() - image.min())
                image = (image * 255).astype(np.uint8)
            
            # Resize attribution to match image
            attribution_resized = cv2.resize(attribution, (image.shape[1], image.shape[0]))
            
            # Normalize attribution
            attribution_resized = (attribution_resized - attribution_resized.min()) / \
                                (attribution_resized.max() - attribution_resized.min())
            
            # Apply colormap
            attribution_colored = cv2.applyColorMap(
                (attribution_resized * 255).astype(np.uint8),
                getattr(cv2, f"COLORMAP_{colormap.upper()}")
            )
            
            # Create overlay
            overlay = cv2.addWeighted(image, 1 - alpha, attribution_colored, alpha, 0)
            
            results["overlay"] = overlay
        
        return results


class SmoothGrad:
    """
    SmoothGrad implementation for deepfake detection.
    """
    
    def __init__(
        self,
        model: nn.Module,
        noise_level: float = 0.1,
        num_samples: int = 50,
        device: str = "cpu",
    ):
        """
        Initialize SmoothGrad.
        
        Args:
            model: PyTorch model
            noise_level: Noise level for smoothing
            num_samples: Number of samples for smoothing
            device: Device to run on
        """
        self.model = model
        self.noise_level = noise_level
        self.num_samples = num_samples
        self.device = device
    
    def generate_attribution(
        self,
        input_tensor: torch.Tensor,
        class_idx: Optional[int] = None,
    ) -> np.ndarray:
        """
        Generate SmoothGrad attribution.
        
        Args:
            input_tensor: Input tensor (B, C, H, W)
            class_idx: Class index to generate attribution for
            
        Returns:
            Attribution map (H, W)
        """
        # Move to device
        input_tensor = input_tensor.to(self.device)
        
        # Get class index
        if class_idx is None:
            with torch.no_grad():
                output = self.model(input_tensor)
                class_idx = torch.argmax(output, dim=1).item()
        
        # Generate smooth gradients
        smooth_gradients = torch.zeros_like(input_tensor)
        
        for _ in range(self.num_samples):
            # Add noise
            noise = torch.randn_like(input_tensor) * self.noise_level
            noisy_input = input_tensor + noise
            noisy_input.requires_grad_(True)
            
            # Forward pass
            output = self.model(noisy_input)
            
            # Backward pass
            self.model.zero_grad()
            output[0, class_idx].backward(retain_graph=True)
            
            # Accumulate gradients
            smooth_gradients += noisy_input.grad
        
        # Average gradients
        smooth_gradients /= self.num_samples
        
        # Compute attribution
        attribution = torch.abs(smooth_gradients)
        attribution = torch.sum(attribution, dim=1).squeeze(0)  # (H, W)
        
        # Convert to numpy
        attribution = attribution.detach().cpu().numpy()
        
        return attribution
    
    def explain(
        self,
        input_tensor: torch.Tensor,
        class_idx: Optional[int] = None,
        return_attribution: bool = True,
        return_overlay: bool = True,
        alpha: float = 0.4,
        colormap: str = "jet",
    ) -> Dict[str, Any]:
        """
        Generate explanation for input.
        
        Args:
            input_tensor: Input tensor (B, C, H, W)
            class_idx: Class index to explain
            return_attribution: Whether to return attribution map
            return_overlay: Whether to return overlay image
            alpha: Overlay transparency
            colormap: Colormap for visualization
            
        Returns:
            Dictionary containing explanation
        """
        # Generate attribution
        attribution = self.generate_attribution(input_tensor, class_idx)
        
        # Prepare results
        results = {
            "class_idx": class_idx,
            "attribution": attribution,
        }
        
        # Create overlay if requested
        if return_overlay:
            # Get original image
            if input_tensor.size(0) > 1:
                image = input_tensor[0].detach().cpu().numpy()
            else:
                image = input_tensor[0].detach().cpu().numpy()
            
            # Convert to RGB if needed
            if image.shape[0] == 3:
                image = np.transpose(image, (1, 2, 0))
                image = (image - image.min()) / (image.max() - image.min())
                image = (image * 255).astype(np.uint8)
            else:
                image = (image - image.min()) / (image.max() - image.min())
                image = (image * 255).astype(np.uint8)
            
            # Resize attribution to match image
            attribution_resized = cv2.resize(attribution, (image.shape[1], image.shape[0]))
            
            # Normalize attribution
            attribution_resized = (attribution_resized - attribution_resized.min()) / \
                                (attribution_resized.max() - attribution_resized.min())
            
            # Apply colormap
            attribution_colored = cv2.applyColorMap(
                (attribution_resized * 255).astype(np.uint8),
                getattr(cv2, f"COLORMAP_{colormap.upper()}")
            )
            
            # Create overlay
            overlay = cv2.addWeighted(image, 1 - alpha, attribution_colored, alpha, 0)
            
            results["overlay"] = overlay
        
        return results


class LIME:
    """
    LIME (Local Interpretable Model-agnostic Explanations) implementation.
    """
    
    def __init__(
        self,
        model: nn.Module,
        num_samples: int = 1000,
        device: str = "cpu",
    ):
        """
        Initialize LIME.
        
        Args:
            model: PyTorch model
            num_samples: Number of samples for LIME
            device: Device to run on
        """
        self.model = model
        self.num_samples = num_samples
        self.device = device
    
    def generate_attribution(
        self,
        input_tensor: torch.Tensor,
        class_idx: Optional[int] = None,
        segment_size: int = 8,
    ) -> np.ndarray:
        """
        Generate LIME attribution.
        
        Args:
            input_tensor: Input tensor (B, C, H, W)
            class_idx: Class index to generate attribution for
            segment_size: Size of segments for LIME
            
        Returns:
            Attribution map (H, W)
        """
        # Move to device
        input_tensor = input_tensor.to(self.device)
        
        # Get class index
        if class_idx is None:
            with torch.no_grad():
                output = self.model(input_tensor)
                class_idx = torch.argmax(output, dim=1).item()
        
        # Get original prediction
        with torch.no_grad():
            original_output = self.model(input_tensor)
            original_score = original_output[0, class_idx].item()
        
        # Create segments
        h, w = input_tensor.shape[2], input_tensor.shape[3]
        num_segments_h = h // segment_size
        num_segments_w = w // segment_size
        
        # Generate random samples
        samples = []
        labels = []
        
        for _ in range(self.num_samples):
            # Create random mask
            mask = torch.zeros(1, 1, h, w, device=self.device)
            
            # Randomly select segments to keep
            for i in range(num_segments_h):
                for j in range(num_segments_w):
                    if torch.rand(1) > 0.5:
                        start_h = i * segment_size
                        end_h = min((i + 1) * segment_size, h)
                        start_w = j * segment_size
                        end_w = min((j + 1) * segment_size, w)
                        mask[0, 0, start_h:end_h, start_w:end_w] = 1
            
            # Create masked input
            masked_input = input_tensor * mask
            
            # Get prediction
            with torch.no_grad():
                masked_output = self.model(masked_input)
                masked_score = masked_output[0, class_idx].item()
            
            samples.append(mask.flatten())
            labels.append(masked_score)
        
        # Convert to tensors
        samples = torch.stack(samples)
        labels = torch.tensor(labels, device=self.device)
        
        # Fit linear model
        from sklearn.linear_model import Ridge
        from sklearn.preprocessing import StandardScaler
        
        # Normalize features
        scaler = StandardScaler()
        samples_scaled = scaler.fit_transform(samples.cpu().numpy())
        
        # Fit Ridge regression
        ridge = Ridge(alpha=1.0)
        ridge.fit(samples_scaled, labels.cpu().numpy())
        
        # Get coefficients
        coefficients = ridge.coef_
        
        # Reshape to image size
        attribution = coefficients.reshape(1, 1, h, w)
        attribution = torch.from_numpy(attribution).to(self.device)
        
        # Convert to numpy
        attribution = attribution.squeeze().cpu().numpy()
        
        return attribution
    
    def explain(
        self,
        input_tensor: torch.Tensor,
        class_idx: Optional[int] = None,
        return_attribution: bool = True,
        return_overlay: bool = True,
        alpha: float = 0.4,
        colormap: str = "jet",
    ) -> Dict[str, Any]:
        """
        Generate explanation for input.
        
        Args:
            input_tensor: Input tensor (B, C, H, W)
            class_idx: Class index to explain
            return_attribution: Whether to return attribution map
            return_overlay: Whether to return overlay image
            alpha: Overlay transparency
            colormap: Colormap for visualization
            
        Returns:
            Dictionary containing explanation
        """
        # Generate attribution
        attribution = self.generate_attribution(input_tensor, class_idx)
        
        # Prepare results
        results = {
            "class_idx": class_idx,
            "attribution": attribution,
        }
        
        # Create overlay if requested
        if return_overlay:
            # Get original image
            if input_tensor.size(0) > 1:
                image = input_tensor[0].detach().cpu().numpy()
            else:
                image = input_tensor[0].detach().cpu().numpy()
            
            # Convert to RGB if needed
            if image.shape[0] == 3:
                image = np.transpose(image, (1, 2, 0))
                image = (image - image.min()) / (image.max() - image.min())
                image = (image * 255).astype(np.uint8)
            else:
                image = (image - image.min()) / (image.max() - image.min())
                image = (image * 255).astype(np.uint8)
            
            # Resize attribution to match image
            attribution_resized = cv2.resize(attribution, (image.shape[1], image.shape[0]))
            
            # Normalize attribution
            attribution_resized = (attribution_resized - attribution_resized.min()) / \
                                (attribution_resized.max() - attribution_resized.min())
            
            # Apply colormap
            attribution_colored = cv2.applyColorMap(
                (attribution_resized * 255).astype(np.uint8),
                getattr(cv2, f"COLORMAP_{colormap.upper()}")
            )
            
            # Create overlay
            overlay = cv2.addWeighted(image, 1 - alpha, attribution_colored, alpha, 0)
            
            results["overlay"] = overlay
        
        return results
