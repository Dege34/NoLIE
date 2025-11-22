"""
Custom samplers for deepfake forensics datasets.

Provides balanced and stratified samplers to handle imbalanced datasets
and ensure proper subject-level splits for cross-validation.
"""

import random
import numpy as np
from typing import Dict, List, Optional, Union, Iterator
from torch.utils.data import Sampler, Dataset
import logging

logger = logging.getLogger(__name__)


class BalancedSampler(Sampler):
    """
    Balanced sampler that ensures equal representation of classes.
    """
    
    def __init__(
        self,
        dataset: Dataset,
        replacement: bool = True,
        num_samples: Optional[int] = None,
    ):
        """
        Initialize balanced sampler.
        
        Args:
            dataset: Dataset to sample from
            replacement: Whether to sample with replacement
            num_samples: Number of samples to draw (None for dataset size)
        """
        self.dataset = dataset
        self.replacement = replacement
        self.num_samples = num_samples or len(dataset)
        
        # Get class labels
        self.labels = self._get_labels()
        
        # Get class indices
        self.class_indices = self._get_class_indices()
        
        # Calculate samples per class
        self.samples_per_class = self.num_samples // len(self.class_indices)
        
        logger.info(f"Balanced sampler initialized with {self.samples_per_class} samples per class")
    
    def _get_labels(self) -> List[int]:
        """Get labels from dataset."""
        labels = []
        for i in range(len(self.dataset)):
            _, label = self.dataset[i]
            labels.append(label)
        return labels
    
    def _get_class_indices(self) -> Dict[int, List[int]]:
        """Get indices for each class."""
        class_indices = {}
        for idx, label in enumerate(self.labels):
            if label not in class_indices:
                class_indices[label] = []
            class_indices[label].append(idx)
        return class_indices
    
    def __iter__(self) -> Iterator[int]:
        """Generate balanced sample indices."""
        indices = []
        
        for class_label, class_indices in self.class_indices.items():
            if self.replacement:
                # Sample with replacement
                class_samples = random.choices(
                    class_indices,
                    k=self.samples_per_class
                )
            else:
                # Sample without replacement
                class_samples = random.sample(
                    class_indices,
                    min(self.samples_per_class, len(class_indices))
                )
            
            indices.extend(class_samples)
        
        # Shuffle the combined indices
        random.shuffle(indices)
        
        return iter(indices)
    
    def __len__(self) -> int:
        """Return number of samples."""
        return self.num_samples


class StratifiedSampler(Sampler):
    """
    Stratified sampler that maintains class proportions.
    """
    
    def __init__(
        self,
        dataset: Dataset,
        labels: List[int],
        num_samples: Optional[int] = None,
    ):
        """
        Initialize stratified sampler.
        
        Args:
            dataset: Dataset to sample from
            labels: Labels for stratification
            num_samples: Number of samples to draw (None for dataset size)
        """
        self.dataset = dataset
        self.labels = labels
        self.num_samples = num_samples or len(dataset)
        
        # Get class proportions
        self.class_proportions = self._get_class_proportions()
        
        # Get class indices
        self.class_indices = self._get_class_indices()
        
        logger.info(f"Stratified sampler initialized with class proportions: {self.class_proportions}")
    
    def _get_class_proportions(self) -> Dict[int, float]:
        """Get class proportions."""
        class_counts = {}
        for label in self.labels:
            class_counts[label] = class_counts.get(label, 0) + 1
        
        total_samples = len(self.labels)
        class_proportions = {
            label: count / total_samples
            for label, count in class_counts.items()
        }
        
        return class_proportions
    
    def _get_class_indices(self) -> Dict[int, List[int]]:
        """Get indices for each class."""
        class_indices = {}
        for idx, label in enumerate(self.labels):
            if label not in class_indices:
                class_indices[label] = []
            class_indices[label].append(idx)
        return class_indices
    
    def __iter__(self) -> Iterator[int]:
        """Generate stratified sample indices."""
        indices = []
        
        for class_label, proportion in self.class_proportions.items():
            class_num_samples = int(self.num_samples * proportion)
            class_indices = self.class_indices[class_label]
            
            # Sample from this class
            class_samples = random.choices(
                class_indices,
                k=class_num_samples
            )
            
            indices.extend(class_samples)
        
        # Shuffle the combined indices
        random.shuffle(indices)
        
        return iter(indices)
    
    def __len__(self) -> int:
        """Return number of samples."""
        return self.num_samples


class SubjectStratifiedSampler(Sampler):
    """
    Subject-stratified sampler for cross-validation.
    
    Ensures that samples from the same subject are not split across
    train/val/test sets.
    """
    
    def __init__(
        self,
        dataset: Dataset,
        subjects: List[str],
        split: str = "train",
        subject_splits: Optional[Dict[str, List[str]]] = None,
    ):
        """
        Initialize subject-stratified sampler.
        
        Args:
            dataset: Dataset to sample from
            subjects: Subject IDs for each sample
            split: Dataset split ('train', 'val', 'test')
            subject_splits: Pre-computed subject splits
        """
        self.dataset = dataset
        self.subjects = subjects
        self.split = split
        self.subject_splits = subject_splits
        
        # Get subject indices
        self.subject_indices = self._get_subject_indices()
        
        # Get indices for this split
        self.split_indices = self._get_split_indices()
        
        logger.info(f"Subject-stratified sampler initialized for {split} split with {len(self.split_indices)} samples")
    
    def _get_subject_indices(self) -> Dict[str, List[int]]:
        """Get indices for each subject."""
        subject_indices = {}
        for idx, subject in enumerate(self.subjects):
            if subject not in subject_indices:
                subject_indices[subject] = []
            subject_indices[subject].append(idx)
        return subject_indices
    
    def _get_split_indices(self) -> List[int]:
        """Get indices for this split."""
        if self.subject_splits is None:
            # If no subject splits provided, use all samples
            return list(range(len(self.dataset)))
        
        # Get subjects for this split
        split_subjects = set(self.subject_splits.get(self.split, []))
        
        # Get indices for these subjects
        split_indices = []
        for subject, indices in self.subject_indices.items():
            if subject in split_subjects:
                split_indices.extend(indices)
        
        return split_indices
    
    def __iter__(self) -> Iterator[int]:
        """Generate sample indices for this split."""
        # Shuffle indices
        random.shuffle(self.split_indices)
        return iter(self.split_indices)
    
    def __len__(self) -> int:
        """Return number of samples in this split."""
        return len(self.split_indices)


class WeightedSampler(Sampler):
    """
    Weighted sampler based on class weights.
    """
    
    def __init__(
        self,
        dataset: Dataset,
        class_weights: List[float],
        num_samples: Optional[int] = None,
    ):
        """
        Initialize weighted sampler.
        
        Args:
            dataset: Dataset to sample from
            class_weights: Weights for each class
            num_samples: Number of samples to draw (None for dataset size)
        """
        self.dataset = dataset
        self.class_weights = class_weights
        self.num_samples = num_samples or len(dataset)
        
        # Get labels and compute sample weights
        self.labels = self._get_labels()
        self.sample_weights = self._compute_sample_weights()
        
        logger.info(f"Weighted sampler initialized with {len(self.sample_weights)} samples")
    
    def _get_labels(self) -> List[int]:
        """Get labels from dataset."""
        labels = []
        for i in range(len(self.dataset)):
            _, label = self.dataset[i]
            labels.append(label)
        return labels
    
    def _compute_sample_weights(self) -> List[float]:
        """Compute weights for each sample."""
        sample_weights = []
        for label in self.labels:
            weight = self.class_weights[label]
            sample_weights.append(weight)
        return sample_weights
    
    def __iter__(self) -> Iterator[int]:
        """Generate weighted sample indices."""
        # Sample indices based on weights
        indices = random.choices(
            range(len(self.dataset)),
            weights=self.sample_weights,
            k=self.num_samples
        )
        
        return iter(indices)
    
    def __len__(self) -> int:
        """Return number of samples."""
        return self.num_samples


class TemporalSampler(Sampler):
    """
    Temporal sampler for video data.
    
    Ensures temporal consistency in video sampling.
    """
    
    def __init__(
        self,
        dataset: Dataset,
        temporal_stride: int = 1,
        num_samples: Optional[int] = None,
    ):
        """
        Initialize temporal sampler.
        
        Args:
            dataset: Dataset to sample from
            temporal_stride: Stride for temporal sampling
            num_samples: Number of samples to draw (None for dataset size)
        """
        self.dataset = dataset
        self.temporal_stride = temporal_stride
        self.num_samples = num_samples or len(dataset)
        
        # Get video indices
        self.video_indices = self._get_video_indices()
        
        logger.info(f"Temporal sampler initialized with stride {temporal_stride}")
    
    def _get_video_indices(self) -> List[int]:
        """Get indices of video samples."""
        video_indices = []
        for i in range(len(self.dataset)):
            # Check if this is a video sample
            # This is a simplified check - in practice, you'd check the dataset
            video_indices.append(i)
        return video_indices
    
    def __iter__(self) -> Iterator[int]:
        """Generate temporally consistent sample indices."""
        indices = []
        
        for video_idx in self.video_indices:
            # Sample frames with temporal stride
            frame_indices = list(range(0, self.num_samples, self.temporal_stride))
            indices.extend([video_idx] * len(frame_indices))
        
        return iter(indices)
    
    def __len__(self) -> int:
        """Return number of samples."""
        return self.num_samples


def create_sampler(
    sampler_type: str,
    dataset: Dataset,
    **kwargs
) -> Sampler:
    """
    Create a sampler based on type.
    
    Args:
        sampler_type: Type of sampler ('balanced', 'stratified', 'weighted', 'temporal')
        dataset: Dataset to sample from
        **kwargs: Additional arguments for the sampler
        
    Returns:
        Sampler instance
    """
    if sampler_type == "balanced":
        return BalancedSampler(dataset, **kwargs)
    elif sampler_type == "stratified":
        return StratifiedSampler(dataset, **kwargs)
    elif sampler_type == "weighted":
        return WeightedSampler(dataset, **kwargs)
    elif sampler_type == "temporal":
        return TemporalSampler(dataset, **kwargs)
    else:
        raise ValueError(f"Unknown sampler type: {sampler_type}")


def create_subject_splits(
    subjects: List[str],
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_seed: int = 42,
) -> Dict[str, List[str]]:
    """
    Create subject-based splits for cross-validation.
    
    Args:
        subjects: List of subject IDs
        train_ratio: Training split ratio
        val_ratio: Validation split ratio
        test_ratio: Test split ratio
        random_seed: Random seed for reproducibility
        
    Returns:
        Dictionary mapping split names to subject lists
    """
    # Set random seed
    random.seed(random_seed)
    
    # Get unique subjects
    unique_subjects = list(set(subjects))
    random.shuffle(unique_subjects)
    
    # Calculate split sizes
    n_subjects = len(unique_subjects)
    n_train = int(n_subjects * train_ratio)
    n_val = int(n_subjects * val_ratio)
    
    # Create splits
    splits = {
        'train': unique_subjects[:n_train],
        'val': unique_subjects[n_train:n_train + n_val],
        'test': unique_subjects[n_train + n_val:],
    }
    
    # Verify ratios
    actual_train_ratio = len(splits['train']) / n_subjects
    actual_val_ratio = len(splits['val']) / n_subjects
    actual_test_ratio = len(splits['test']) / n_subjects
    
    logger.info(f"Subject splits created:")
    logger.info(f"  Train: {len(splits['train'])} subjects ({actual_train_ratio:.3f})")
    logger.info(f"  Val: {len(splits['val'])} subjects ({actual_val_ratio:.3f})")
    logger.info(f"  Test: {len(splits['test'])} subjects ({actual_test_ratio:.3f})")
    
    return splits
