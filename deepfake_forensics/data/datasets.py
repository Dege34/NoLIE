"""
Dataset classes for deepfake forensics.

Provides PyTorch datasets for loading images and videos with proper
labeling and preprocessing for deepfake detection tasks.
"""

import os
import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import cv2
import logging
from PIL import Image
import pandas as pd

logger = logging.getLogger(__name__)


class DeepfakeDataset(Dataset):
    """
    Main dataset class for deepfake detection.
    
    Supports both image and video data with flexible folder structures.
    """
    
    def __init__(
        self,
        data_dir: Union[str, Path],
        split: str = "train",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        image_size: int = 224,
        max_frames: int = 16,
        fps: int = 8,
        cache_frames: bool = False,
        subjects_file: Optional[Union[str, Path]] = None,
    ):
        """
        Initialize deepfake dataset.
        
        Args:
            data_dir: Root directory containing data
            split: Dataset split ('train', 'val', 'test')
            transform: Transform to apply to images
            target_transform: Transform to apply to labels
            image_size: Target image size
            max_frames: Maximum number of frames per video
            fps: Frames per second for video sampling
            cache_frames: Whether to cache extracted frames
            subjects_file: Optional file containing subject information
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.transform = transform
        self.target_transform = target_transform
        self.image_size = image_size
        self.max_frames = max_frames
        self.fps = fps
        self.cache_frames = cache_frames
        
        # Load subject information if provided
        self.subjects = self._load_subjects(subjects_file)
        
        # Load dataset samples
        self.samples = self._load_samples()
        
        # Initialize frame cache
        self.frame_cache = {} if cache_frames else None
        
        logger.info(f"Loaded {len(self.samples)} samples for {split} split")
    
    def _load_subjects(self, subjects_file: Optional[Union[str, Path]]) -> Dict[str, str]:
        """Load subject information from file."""
        if subjects_file is None:
            return {}
        
        subjects_file = Path(subjects_file)
        if not subjects_file.exists():
            logger.warning(f"Subjects file not found: {subjects_file}")
            return {}
        
        try:
            with open(subjects_file, 'r') as f:
                subjects = json.load(f)
            logger.info(f"Loaded {len(subjects)} subjects")
            return subjects
        except Exception as e:
            logger.error(f"Failed to load subjects file: {e}")
            return {}
    
    def _load_samples(self) -> List[Dict[str, Any]]:
        """Load dataset samples from directory structure."""
        samples = []
        
        # Look for different folder structures
        possible_structures = [
            # Structure 1: real/fake folders
            ("real", "fake"),
            # Structure 2: 0/1 folders
            ("0", "1"),
            # Structure 3: authentic/manipulated folders
            ("authentic", "manipulated"),
            # Structure 4: genuine/fake folders
            ("genuine", "fake"),
        ]
        
        for real_folder, fake_folder in possible_structures:
            real_path = self.data_dir / real_folder
            fake_path = self.data_dir / fake_folder
            
            if real_path.exists() and fake_path.exists():
                logger.info(f"Found data structure: {real_folder}/{fake_folder}")
                
                # Load real samples
                real_samples = self._load_samples_from_folder(real_path, label=0)
                samples.extend(real_samples)
                
                # Load fake samples
                fake_samples = self._load_samples_from_folder(fake_path, label=1)
                samples.extend(fake_samples)
                
                break
        else:
            # Try to load from metadata file
            metadata_file = self.data_dir / "metadata.json"
            if metadata_file.exists():
                samples = self._load_samples_from_metadata(metadata_file)
            else:
                raise ValueError(f"No valid data structure found in {self.data_dir}")
        
        # Filter by split if subjects file is provided
        if self.subjects:
            samples = self._filter_by_split(samples)
        
        # Shuffle samples
        random.shuffle(samples)
        
        return samples
    
    def _load_samples_from_folder(
        self,
        folder_path: Path,
        label: int,
    ) -> List[Dict[str, Any]]:
        """Load samples from a folder."""
        samples = []
        
        # Supported file extensions
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv'}
        
        for file_path in folder_path.rglob('*'):
            if file_path.is_file():
                ext = file_path.suffix.lower()
                
                if ext in image_extensions:
                    samples.append({
                        'path': str(file_path),
                        'label': label,
                        'type': 'image',
                        'subject': self._extract_subject_id(file_path),
                    })
                elif ext in video_extensions:
                    samples.append({
                        'path': str(file_path),
                        'label': label,
                        'type': 'video',
                        'subject': self._extract_subject_id(file_path),
                    })
        
        return samples
    
    def _load_samples_from_metadata(self, metadata_file: Path) -> List[Dict[str, Any]]:
        """Load samples from metadata file."""
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        
        samples = []
        for item in metadata:
            if item.get('split') == self.split:
                samples.append({
                    'path': str(self.data_dir / item['path']),
                    'label': item['label'],
                    'type': item.get('type', 'image'),
                    'subject': item.get('subject', 'unknown'),
                })
        
        return samples
    
    def _extract_subject_id(self, file_path: Path) -> str:
        """Extract subject ID from file path."""
        # Try to extract from folder structure
        parts = file_path.parts
        
        # Look for common patterns
        for i, part in enumerate(parts):
            if part in ['real', 'fake', '0', '1', 'authentic', 'manipulated', 'genuine']:
                if i > 0:
                    return parts[i - 1]
        
        # Use filename as subject ID
        return file_path.stem
    
    def _filter_by_split(self, samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Filter samples by split using subject information."""
        if not self.subjects:
            return samples
        
        # Get subjects for this split
        split_subjects = set()
        for subject, split_name in self.subjects.items():
            if split_name == self.split:
                split_subjects.add(subject)
        
        # Filter samples
        filtered_samples = []
        for sample in samples:
            if sample['subject'] in split_subjects:
                filtered_samples.append(sample)
        
        return filtered_samples
    
    def __len__(self) -> int:
        """Return number of samples."""
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """Get item by index."""
        sample = self.samples[idx]
        
        # Load data
        if sample['type'] == 'image':
            data = self._load_image(sample['path'])
        else:  # video
            data = self._load_video(sample['path'])
        
        # Apply transforms
        if self.transform:
            data = self.transform(data)
        
        # Get label
        label = sample['label']
        if self.target_transform:
            label = self.target_transform(label)
        
        return data, label
    
    def _load_image(self, image_path: str) -> np.ndarray:
        """Load image from file."""
        try:
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Could not load image: {image_path}")
            
            # Convert BGR to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Resize if needed
            if image.shape[:2] != (self.image_size, self.image_size):
                image = cv2.resize(image, (self.image_size, self.image_size))
            
            return image
            
        except Exception as e:
            logger.error(f"Failed to load image {image_path}: {e}")
            # Return a dummy image
            return np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)
    
    def _load_video(self, video_path: str) -> np.ndarray:
        """Load video frames from file."""
        try:
            # Check cache first
            if self.cache_frames and video_path in self.frame_cache:
                return self.frame_cache[video_path]
            
            # Load video
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValueError(f"Could not open video: {video_path}")
            
            # Get video properties
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            
            # Calculate frame indices to sample
            if fps > 0:
                frame_interval = max(1, int(fps / self.fps))
            else:
                frame_interval = 1
            
            frame_indices = list(range(0, total_frames, frame_interval))[:self.max_frames]
            
            # Extract frames
            frames = []
            for frame_idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                
                if ret:
                    # Convert BGR to RGB
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    
                    # Resize if needed
                    if frame.shape[:2] != (self.image_size, self.image_size):
                        frame = cv2.resize(frame, (self.image_size, self.image_size))
                    
                    frames.append(frame)
                else:
                    # Pad with last frame if needed
                    if frames:
                        frames.append(frames[-1])
                    else:
                        # Create dummy frame
                        frames.append(np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8))
            
            cap.release()
            
            # Ensure we have the right number of frames
            while len(frames) < self.max_frames:
                if frames:
                    frames.append(frames[-1])
                else:
                    frames.append(np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8))
            
            # Convert to numpy array
            video_data = np.array(frames[:self.max_frames])
            
            # Cache if enabled
            if self.cache_frames:
                self.frame_cache[video_path] = video_data
            
            return video_data
            
        except Exception as e:
            logger.error(f"Failed to load video {video_path}: {e}")
            # Return dummy video
            return np.zeros((self.max_frames, self.image_size, self.image_size, 3), dtype=np.uint8)
    
    def get_class_weights(self) -> torch.Tensor:
        """Get class weights for imbalanced dataset."""
        labels = [sample['label'] for sample in self.samples]
        class_counts = np.bincount(labels)
        
        if len(class_counts) < 2:
            return torch.ones(2)
        
        total_samples = len(labels)
        class_weights = total_samples / (2 * class_counts)
        
        return torch.tensor(class_weights, dtype=torch.float32)
    
    def get_subject_splits(self, train_ratio: float = 0.7, val_ratio: float = 0.15) -> Dict[str, List[str]]:
        """Get subject-based splits for cross-validation."""
        subjects = list(set(sample['subject'] for sample in self.samples))
        random.shuffle(subjects)
        
        n_subjects = len(subjects)
        n_train = int(n_subjects * train_ratio)
        n_val = int(n_subjects * val_ratio)
        
        splits = {
            'train': subjects[:n_train],
            'val': subjects[n_train:n_train + n_val],
            'test': subjects[n_train + n_val:],
        }
        
        return splits


class VideoDataset(Dataset):
    """
    Dataset specifically for video data.
    """
    
    def __init__(
        self,
        video_paths: List[str],
        labels: List[int],
        transform: Optional[Callable] = None,
        image_size: int = 224,
        max_frames: int = 16,
        fps: int = 8,
    ):
        """
        Initialize video dataset.
        
        Args:
            video_paths: List of video file paths
            labels: List of corresponding labels
            transform: Transform to apply to frames
            image_size: Target image size
            max_frames: Maximum number of frames per video
            fps: Frames per second for video sampling
        """
        self.video_paths = video_paths
        self.labels = labels
        self.transform = transform
        self.image_size = image_size
        self.max_frames = max_frames
        self.fps = fps
        
        assert len(video_paths) == len(labels), "Number of videos and labels must match"
    
    def __len__(self) -> int:
        """Return number of videos."""
        return len(self.video_paths)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """Get video by index."""
        video_path = self.video_paths[idx]
        label = self.labels[idx]
        
        # Load video frames
        frames = self._load_video_frames(video_path)
        
        # Apply transforms
        if self.transform:
            frames = self.transform(frames)
        
        return frames, label
    
    def _load_video_frames(self, video_path: str) -> np.ndarray:
        """Load frames from video file."""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        # Get video properties
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # Calculate frame indices to sample
        if fps > 0:
            frame_interval = max(1, int(fps / self.fps))
        else:
            frame_interval = 1
        
        frame_indices = list(range(0, total_frames, frame_interval))[:self.max_frames]
        
        # Extract frames
        frames = []
        for frame_idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            
            if ret:
                # Convert BGR to RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Resize if needed
                if frame.shape[:2] != (self.image_size, self.image_size):
                    frame = cv2.resize(frame, (self.image_size, self.image_size))
                
                frames.append(frame)
            else:
                # Pad with last frame if needed
                if frames:
                    frames.append(frames[-1])
                else:
                    # Create dummy frame
                    frames.append(np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8))
        
        cap.release()
        
        # Ensure we have the right number of frames
        while len(frames) < self.max_frames:
            if frames:
                frames.append(frames[-1])
            else:
                frames.append(np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8))
        
        return np.array(frames[:self.max_frames])


class ImageDataset(Dataset):
    """
    Dataset specifically for image data.
    """
    
    def __init__(
        self,
        image_paths: List[str],
        labels: List[int],
        transform: Optional[Callable] = None,
        image_size: int = 224,
    ):
        """
        Initialize image dataset.
        
        Args:
            image_paths: List of image file paths
            labels: List of corresponding labels
            transform: Transform to apply to images
            image_size: Target image size
        """
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.image_size = image_size
        
        assert len(image_paths) == len(labels), "Number of images and labels must match"
    
    def __len__(self) -> int:
        """Return number of images."""
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """Get image by index."""
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        
        # Load image
        image = self._load_image(image_path)
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        return image, label
    
    def _load_image(self, image_path: str) -> np.ndarray:
        """Load image from file."""
        try:
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Could not load image: {image_path}")
            
            # Convert BGR to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Resize if needed
            if image.shape[:2] != (self.image_size, self.image_size):
                image = cv2.resize(image, (self.image_size, self.image_size))
            
            return image
            
        except Exception as e:
            logger.error(f"Failed to load image {image_path}: {e}")
            # Return a dummy image
            return np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)


def create_data_loaders(
    data_dir: Union[str, Path],
    batch_size: int = 32,
    num_workers: int = 4,
    image_size: int = 224,
    max_frames: int = 16,
    fps: int = 8,
    train_split: float = 0.7,
    val_split: float = 0.15,
    test_split: float = 0.15,
    subjects_file: Optional[Union[str, Path]] = None,
    use_balanced_sampling: bool = True,
) -> Dict[str, DataLoader]:
    """
    Create data loaders for training, validation, and testing.
    
    Args:
        data_dir: Root directory containing data
        batch_size: Batch size for data loaders
        num_workers: Number of worker processes
        image_size: Target image size
        max_frames: Maximum number of frames per video
        fps: Frames per second for video sampling
        train_split: Training split ratio
        val_split: Validation split ratio
        test_split: Test split ratio
        subjects_file: Optional file containing subject information
        use_balanced_sampling: Whether to use balanced sampling
        
    Returns:
        Dictionary containing data loaders
    """
    from .transforms import get_transforms
    from .samplers import BalancedSampler
    
    # Get transforms
    train_transform, val_transform = get_transforms(image_size)
    
    # Create datasets
    train_dataset = DeepfakeDataset(
        data_dir=data_dir,
        split="train",
        transform=train_transform,
        image_size=image_size,
        max_frames=max_frames,
        fps=fps,
        subjects_file=subjects_file,
    )
    
    val_dataset = DeepfakeDataset(
        data_dir=data_dir,
        split="val",
        transform=val_transform,
        image_size=image_size,
        max_frames=max_frames,
        fps=fps,
        subjects_file=subjects_file,
    )
    
    test_dataset = DeepfakeDataset(
        data_dir=data_dir,
        split="test",
        transform=val_transform,
        image_size=image_size,
        max_frames=max_frames,
        fps=fps,
        subjects_file=subjects_file,
    )
    
    # Create samplers
    train_sampler = None
    if use_balanced_sampling:
        train_sampler = BalancedSampler(train_dataset)
    
    # Create data loaders
    data_loaders = {
        'train': DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=train_sampler is None,
            sampler=train_sampler,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True,
        ),
        'val': DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=False,
        ),
        'test': DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=False,
        ),
    }
    
    return data_loaders
