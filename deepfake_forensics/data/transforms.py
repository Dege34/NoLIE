"""
Data transforms for deepfake detection.
"""

import torch
import torchvision.transforms as transforms
from torchvision.transforms import functional as F
from PIL import Image
import numpy as np
from typing import Tuple, Optional


def get_inference_transforms(
    input_size: int = 224,
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225)
) -> transforms.Compose:
    """
    Get transforms for inference.
    
    Args:
        input_size: Target image size
        mean: Normalization mean
        std: Normalization std
    
    Returns:
        Composed transforms
    """
    return transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])


def get_training_transforms(
    input_size: int = 224,
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
    augmentation: bool = True
) -> transforms.Compose:
    """
    Get transforms for training.
    
    Args:
        input_size: Target image size
        mean: Normalization mean
        std: Normalization std
        augmentation: Whether to apply data augmentation
    
    Returns:
        Composed transforms
    """
    transform_list = [
        transforms.Resize((input_size, input_size)),
    ]
    
    if augmentation:
        transform_list.extend([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        ])
    
    transform_list.extend([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    
    return transforms.Compose(transform_list)


class FaceCropTransform:
    """Transform to crop face region from image."""
    
    def __init__(self, face_cascade_path: Optional[str] = None):
        """
        Initialize face crop transform.
        
        Args:
            face_cascade_path: Path to Haar cascade file
        """
        try:
            import cv2
            if face_cascade_path:
                self.face_cascade = cv2.CascadeClassifier(face_cascade_path)
            else:
                self.face_cascade = cv2.CascadeClassifier(
                    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
                )
        except ImportError:
            self.face_cascade = None
    
    def __call__(self, image: Image.Image) -> Image.Image:
        """
        Crop face from image.
        
        Args:
            image: Input PIL image
        
        Returns:
            Cropped face image or original if no face detected
        """
        if self.face_cascade is None:
            return image
        
        try:
            import cv2
            
            # Convert PIL to OpenCV format
            img_array = np.array(image)
            if len(img_array.shape) == 3:
                gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            else:
                gray = img_array
            
            # Detect faces
            faces = self.face_cascade.detectMultiScale(
                gray, 
                scaleFactor=1.1, 
                minNeighbors=5, 
                minSize=(30, 30)
            )
            
            if len(faces) > 0:
                # Get largest face
                x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
                
                # Add some padding
                padding = 0.2
                x_pad = int(w * padding)
                y_pad = int(h * padding)
                
                x = max(0, x - x_pad)
                y = max(0, y - y_pad)
                w = min(img_array.shape[1] - x, w + 2 * x_pad)
                h = min(img_array.shape[0] - y, h + 2 * y_pad)
                
                # Crop face
                face_img = img_array[y:y+h, x:x+w]
                return Image.fromarray(face_img)
            
        except Exception as e:
            print(f"Face detection failed: {e}")
        
        return image


def get_face_crop_transforms(
    input_size: int = 224,
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225)
) -> transforms.Compose:
    """
    Get transforms with face cropping.
    
    Args:
        input_size: Target image size
        mean: Normalization mean
        std: Normalization std
    
    Returns:
        Composed transforms with face cropping
    """
    return transforms.Compose([
        FaceCropTransform(),
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])