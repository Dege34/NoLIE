"""
Grad-CAM implementation for deepfake detection.

Provides Grad-CAM, Score-CAM, and Grad-CAM++ implementations for
explaining model predictions and visualizing important regions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from typing import Optional, List, Dict, Any, Union, Tuple
import logging

logger = logging.getLogger(__name__)


class GradCAM:
    """
    Grad-CAM implementation for deepfake detection.
    """
    
    def __init__(
        self,
        model: nn.Module,
        target_layer: Optional[str] = None,
        device: str = "cpu",
    ):
        """
        Initialize Grad-CAM.
        
        Args:
            model: PyTorch model
            target_layer: Target layer name (None for auto-detection)
            device: Device to run on
        """
        self.model = model
        self.device = device
        self.target_layer = target_layer
        
        # Find target layer
        if target_layer is None:
            self.target_layer = self._find_target_layer()
        
        # Register hooks
        self.gradients = None
        self.activations = None
        self._register_hooks()
    
    def _find_target_layer(self) -> str:
        """Find the best target layer for Grad-CAM."""
        # Look for common layer names
        layer_names = []
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.AdaptiveAvgPool2d, nn.AdaptiveMaxPool2d)):
                layer_names.append(name)
        
        if not layer_names:
            raise ValueError("No suitable target layer found")
        
        # Return the last convolutional layer
        return layer_names[-1]
    
    def _register_hooks(self) -> None:
        """Register forward and backward hooks."""
        def forward_hook(module, input, output):
            self.activations = output
        
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]
        
        # Get target layer
        target_module = None
        for name, module in self.model.named_modules():
            if name == self.target_layer:
                target_module = module
                break
        
        if target_module is None:
            raise ValueError(f"Target layer {self.target_layer} not found")
        
        # Register hooks
        self.forward_handle = target_module.register_forward_hook(forward_hook)
        self.backward_handle = target_module.register_backward_hook(backward_hook)
    
    def __del__(self):
        """Clean up hooks."""
        if hasattr(self, 'forward_handle'):
            self.forward_handle.remove()
        if hasattr(self, 'backward_handle'):
            self.backward_handle.remove()
    
    def generate_cam(
        self,
        input_tensor: torch.Tensor,
        class_idx: Optional[int] = None,
        retain_graph: bool = False,
    ) -> np.ndarray:
        """
        Generate Grad-CAM heatmap.
        
        Args:
            input_tensor: Input tensor (B, C, H, W)
            class_idx: Class index to generate CAM for (None for predicted class)
            retain_graph: Whether to retain computational graph
            
        Returns:
            Grad-CAM heatmap (H, W)
        """
        # Move to device
        input_tensor = input_tensor.to(self.device)
        
        # Forward pass
        self.model.eval()
        output = self.model(input_tensor)
        
        # Get class index
        if class_idx is None:
            class_idx = torch.argmax(output, dim=1).item()
        
        # Backward pass
        self.model.zero_grad()
        output[0, class_idx].backward(retain_graph=retain_graph)
        
        # Get gradients and activations
        gradients = self.gradients[0]  # (C, H, W)
        activations = self.activations[0]  # (C, H, W)
        
        # Compute weights
        weights = torch.mean(gradients, dim=(1, 2))  # (C,)
        
        # Generate CAM
        cam = torch.zeros(activations.shape[1:], dtype=torch.float32, device=self.device)
        for i, w in enumerate(weights):
            cam += w * activations[i]
        
        # Apply ReLU
        cam = F.relu(cam)
        
        # Normalize
        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()
        
        # Convert to numpy
        cam = cam.detach().cpu().numpy()
        
        return cam
    
    def explain(
        self,
        input_tensor: torch.Tensor,
        class_idx: Optional[int] = None,
        return_cam: bool = True,
        return_overlay: bool = True,
        alpha: float = 0.4,
        colormap: str = "jet",
    ) -> Dict[str, Any]:
        """
        Generate explanation for input.
        
        Args:
            input_tensor: Input tensor (B, C, H, W)
            class_idx: Class index to explain
            return_cam: Whether to return CAM heatmap
            return_overlay: Whether to return overlay image
            alpha: Overlay transparency
            colormap: Colormap for visualization
            
        Returns:
            Dictionary containing explanation
        """
        # Generate CAM
        cam = self.generate_cam(input_tensor, class_idx)
        
        # Prepare results
        results = {
            "class_idx": class_idx,
            "cam": cam,
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
            
            # Resize CAM to match image
            cam_resized = cv2.resize(cam, (image.shape[1], image.shape[0]))
            
            # Apply colormap
            cam_colored = cv2.applyColorMap(
                (cam_resized * 255).astype(np.uint8),
                getattr(cv2, f"COLORMAP_{colormap.upper()}")
            )
            
            # Create overlay
            overlay = cv2.addWeighted(image, 1 - alpha, cam_colored, alpha, 0)
            
            results["overlay"] = overlay
        
        return results


class ScoreCAM:
    """
    Score-CAM implementation for deepfake detection.
    """
    
    def __init__(
        self,
        model: nn.Module,
        target_layer: Optional[str] = None,
        device: str = "cpu",
    ):
        """
        Initialize Score-CAM.
        
        Args:
            model: PyTorch model
            target_layer: Target layer name
            device: Device to run on
        """
        self.model = model
        self.device = device
        self.target_layer = target_layer
        
        # Find target layer
        if target_layer is None:
            self.target_layer = self._find_target_layer()
    
    def _find_target_layer(self) -> str:
        """Find the best target layer for Score-CAM."""
        layer_names = []
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.AdaptiveAvgPool2d, nn.AdaptiveMaxPool2d)):
                layer_names.append(name)
        
        if not layer_names:
            raise ValueError("No suitable target layer found")
        
        return layer_names[-1]
    
    def generate_cam(
        self,
        input_tensor: torch.Tensor,
        class_idx: Optional[int] = None,
    ) -> np.ndarray:
        """
        Generate Score-CAM heatmap.
        
        Args:
            input_tensor: Input tensor (B, C, H, W)
            class_idx: Class index to generate CAM for
            
        Returns:
            Score-CAM heatmap (H, W)
        """
        # Move to device
        input_tensor = input_tensor.to(self.device)
        
        # Get target layer
        target_module = None
        for name, module in self.model.named_modules():
            if name == self.target_layer:
                target_module = module
                break
        
        if target_module is None:
            raise ValueError(f"Target layer {self.target_layer} not found")
        
        # Forward pass to get activations
        self.model.eval()
        with torch.no_grad():
            # Hook to get activations
            activations = None
            def hook(module, input, output):
                nonlocal activations
                activations = output
            
            handle = target_module.register_forward_hook(hook)
            _ = self.model(input_tensor)
            handle.remove()
            
            # Get activations
            activations = activations[0]  # (C, H, W)
            
            # Get class index
            if class_idx is None:
                output = self.model(input_tensor)
                class_idx = torch.argmax(output, dim=1).item()
            
            # Generate CAM
            cam = torch.zeros(activations.shape[1:], dtype=torch.float32, device=self.device)
            
            for i in range(activations.size(0)):
                # Create mask from activation
                mask = activations[i:i+1]  # (1, H, W)
                
                # Upsample mask to input size
                mask_upsampled = F.interpolate(
                    mask.unsqueeze(0),
                    size=input_tensor.shape[2:],
                    mode='bilinear',
                    align_corners=False
                ).squeeze(0)  # (1, H, W)
                
                # Apply mask to input
                masked_input = input_tensor * mask_upsampled
                
                # Get prediction for masked input
                with torch.no_grad():
                    masked_output = self.model(masked_input)
                    score = masked_output[0, class_idx].item()
                
                # Add to CAM
                cam += score * activations[i]
            
            # Apply ReLU
            cam = F.relu(cam)
            
            # Normalize
            cam = cam - cam.min()
            if cam.max() > 0:
                cam = cam / cam.max()
            
            # Convert to numpy
            cam = cam.detach().cpu().numpy()
            
            return cam
    
    def explain(
        self,
        input_tensor: torch.Tensor,
        class_idx: Optional[int] = None,
        return_cam: bool = True,
        return_overlay: bool = True,
        alpha: float = 0.4,
        colormap: str = "jet",
    ) -> Dict[str, Any]:
        """
        Generate explanation for input.
        
        Args:
            input_tensor: Input tensor (B, C, H, W)
            class_idx: Class index to explain
            return_cam: Whether to return CAM heatmap
            return_overlay: Whether to return overlay image
            alpha: Overlay transparency
            colormap: Colormap for visualization
            
        Returns:
            Dictionary containing explanation
        """
        # Generate CAM
        cam = self.generate_cam(input_tensor, class_idx)
        
        # Prepare results
        results = {
            "class_idx": class_idx,
            "cam": cam,
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
            
            # Resize CAM to match image
            cam_resized = cv2.resize(cam, (image.shape[1], image.shape[0]))
            
            # Apply colormap
            cam_colored = cv2.applyColorMap(
                (cam_resized * 255).astype(np.uint8),
                getattr(cv2, f"COLORMAP_{colormap.upper()}")
            )
            
            # Create overlay
            overlay = cv2.addWeighted(image, 1 - alpha, cam_colored, alpha, 0)
            
            results["overlay"] = overlay
        
        return results


class GradCAMPlusPlus:
    """
    Grad-CAM++ implementation for deepfake detection.
    """
    
    def __init__(
        self,
        model: nn.Module,
        target_layer: Optional[str] = None,
        device: str = "cpu",
    ):
        """
        Initialize Grad-CAM++.
        
        Args:
            model: PyTorch model
            target_layer: Target layer name
            device: Device to run on
        """
        self.model = model
        self.device = device
        self.target_layer = target_layer
        
        # Find target layer
        if target_layer is None:
            self.target_layer = self._find_target_layer()
        
        # Register hooks
        self.gradients = None
        self.activations = None
        self._register_hooks()
    
    def _find_target_layer(self) -> str:
        """Find the best target layer for Grad-CAM++."""
        layer_names = []
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.AdaptiveAvgPool2d, nn.AdaptiveMaxPool2d)):
                layer_names.append(name)
        
        if not layer_names:
            raise ValueError("No suitable target layer found")
        
        return layer_names[-1]
    
    def _register_hooks(self) -> None:
        """Register forward and backward hooks."""
        def forward_hook(module, input, output):
            self.activations = output
        
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]
        
        # Get target layer
        target_module = None
        for name, module in self.model.named_modules():
            if name == self.target_layer:
                target_module = module
                break
        
        if target_module is None:
            raise ValueError(f"Target layer {self.target_layer} not found")
        
        # Register hooks
        self.forward_handle = target_module.register_forward_hook(forward_hook)
        self.backward_handle = target_module.register_backward_hook(backward_hook)
    
    def __del__(self):
        """Clean up hooks."""
        if hasattr(self, 'forward_handle'):
            self.forward_handle.remove()
        if hasattr(self, 'backward_handle'):
            self.backward_handle.remove()
    
    def generate_cam(
        self,
        input_tensor: torch.Tensor,
        class_idx: Optional[int] = None,
        retain_graph: bool = False,
    ) -> np.ndarray:
        """
        Generate Grad-CAM++ heatmap.
        
        Args:
            input_tensor: Input tensor (B, C, H, W)
            class_idx: Class index to generate CAM for
            retain_graph: Whether to retain computational graph
            
        Returns:
            Grad-CAM++ heatmap (H, W)
        """
        # Move to device
        input_tensor = input_tensor.to(self.device)
        
        # Forward pass
        self.model.eval()
        output = self.model(input_tensor)
        
        # Get class index
        if class_idx is None:
            class_idx = torch.argmax(output, dim=1).item()
        
        # Backward pass
        self.model.zero_grad()
        output[0, class_idx].backward(retain_graph=retain_graph)
        
        # Get gradients and activations
        gradients = self.gradients[0]  # (C, H, W)
        activations = self.activations[0]  # (C, H, W)
        
        # Compute weights using Grad-CAM++ formula
        # First, compute the gradients of the output with respect to the activations
        gradients_power = gradients ** 2
        gradients_power_3 = gradients ** 3
        
        # Compute the weights
        alpha = gradients_power / (2 * gradients_power + 
                                 torch.sum(activations * gradients_power_3, dim=(1, 2), keepdim=True))
        
        # Compute the weights
        weights = torch.sum(alpha * F.relu(gradients), dim=(1, 2))  # (C,)
        
        # Generate CAM
        cam = torch.zeros(activations.shape[1:], dtype=torch.float32, device=self.device)
        for i, w in enumerate(weights):
            cam += w * activations[i]
        
        # Apply ReLU
        cam = F.relu(cam)
        
        # Normalize
        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()
        
        # Convert to numpy
        cam = cam.detach().cpu().numpy()
        
        return cam
    
    def explain(
        self,
        input_tensor: torch.Tensor,
        class_idx: Optional[int] = None,
        return_cam: bool = True,
        return_overlay: bool = True,
        alpha: float = 0.4,
        colormap: str = "jet",
    ) -> Dict[str, Any]:
        """
        Generate explanation for input.
        
        Args:
            input_tensor: Input tensor (B, C, H, W)
            class_idx: Class index to explain
            return_cam: Whether to return CAM heatmap
            return_overlay: Whether to return overlay image
            alpha: Overlay transparency
            colormap: Colormap for visualization
            
        Returns:
            Dictionary containing explanation
        """
        # Generate CAM
        cam = self.generate_cam(input_tensor, class_idx)
        
        # Prepare results
        results = {
            "class_idx": class_idx,
            "cam": cam,
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
            
            # Resize CAM to match image
            cam_resized = cv2.resize(cam, (image.shape[1], image.shape[0]))
            
            # Apply colormap
            cam_colored = cv2.applyColorMap(
                (cam_resized * 255).astype(np.uint8),
                getattr(cv2, f"COLORMAP_{colormap.upper()}")
            )
            
            # Create overlay
            overlay = cv2.addWeighted(image, 1 - alpha, cam_colored, alpha, 0)
            
            results["overlay"] = overlay
        
        return results
