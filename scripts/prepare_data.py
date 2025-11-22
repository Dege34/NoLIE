"""
Data preparation script for deepfake forensics.

Prepares datasets by extracting frames, detecting faces, and creating
the necessary folder structure for training and evaluation.
"""

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np
import logging
from tqdm import tqdm

from deepfake_forensics.data.video_reader import create_video_reader
from deepfake_forensics.utils.logging import setup_logging, get_logger
from deepfake_forensics.utils.seeds import set_seed

logger = get_logger(__name__)


def prepare_dataset(
    src_dir: str,
    out_dir: str,
    fps: int = 8,
    image_size: int = 224,
    max_frames: int = 16,
    subjects_file: Optional[str] = None,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_seed: int = 42,
) -> None:
    """
    Prepare dataset for training.
    
    Args:
        src_dir: Source data directory
        out_dir: Output data directory
        fps: Frames per second for video extraction
        image_size: Image size for processing
        max_frames: Maximum number of frames per video
        subjects_file: Path to subjects file
        train_ratio: Training split ratio
        val_ratio: Validation split ratio
        test_ratio: Test split ratio
        random_seed: Random seed for reproducibility
    """
    # Set seed for reproducibility
    set_seed(random_seed)
    
    # Create output directory
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectories
    (out_path / "real").mkdir(exist_ok=True)
    (out_path / "fake").mkdir(exist_ok=True)
    
    # Create video reader
    video_reader = create_video_reader(
        max_frames=max_frames,
        fps=fps,
        image_size=image_size,
    )
    
    # Process videos
    metadata = {
        "real": [],
        "fake": [],
        "subjects": {},
        "splits": {},
    }
    
    # Process real videos
    real_dir = Path(src_dir) / "real"
    if real_dir.exists():
        logger.info("Processing real videos...")
        real_metadata = process_videos(
            real_dir, out_path / "real", video_reader, "real"
        )
        metadata["real"] = real_metadata
    
    # Process fake videos
    fake_dir = Path(src_dir) / "fake"
    if fake_dir.exists():
        logger.info("Processing fake videos...")
        fake_metadata = process_videos(
            fake_dir, out_path / "fake", video_reader, "fake"
        )
        metadata["fake"] = fake_metadata
    
    # Create subject splits
    if subjects_file:
        subjects = load_subjects(subjects_file)
        metadata["subjects"] = subjects
        metadata["splits"] = create_subject_splits(
            subjects, train_ratio, val_ratio, test_ratio, random_seed
        )
    else:
        # Create random splits
        all_videos = metadata["real"] + metadata["fake"]
        random.shuffle(all_videos)
        
        n_total = len(all_videos)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)
        
        metadata["splits"] = {
            "train": all_videos[:n_train],
            "val": all_videos[n_train:n_train + n_val],
            "test": all_videos[n_train + n_val:],
        }
    
    # Save metadata
    with open(out_path / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"Dataset prepared successfully in {out_dir}")
    logger.info(f"Total videos: {len(metadata['real']) + len(metadata['fake'])}")
    logger.info(f"Real videos: {len(metadata['real'])}")
    logger.info(f"Fake videos: {len(metadata['fake'])}")


def process_videos(
    src_dir: Path,
    out_dir: Path,
    video_reader,
    label: str,
) -> List[Dict]:
    """
    Process videos in a directory.
    
    Args:
        src_dir: Source directory
        out_dir: Output directory
        video_reader: Video reader instance
        label: Label for videos
        
    Returns:
        List of video metadata
    """
    metadata = []
    
    # Get all video files
    video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm'}
    video_files = []
    for ext in video_extensions:
        video_files.extend(src_dir.glob(f"**/*{ext}"))
    
    # Process each video
    for video_path in tqdm(video_files, desc=f"Processing {label} videos"):
        try:
            # Create output path
            relative_path = video_path.relative_to(src_dir)
            output_path = out_dir / relative_path.with_suffix("")
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Extract frames
            frames = video_reader.read_video(video_path)
            
            # Save frames
            frame_paths = []
            for i, frame in enumerate(frames):
                frame_path = output_path / f"frame_{i:04d}.jpg"
                cv2.imwrite(str(frame_path), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                frame_paths.append(str(frame_path))
            
            # Create metadata entry
            video_metadata = {
                "path": str(relative_path),
                "label": 0 if label == "real" else 1,
                "type": "video",
                "frames": frame_paths,
                "num_frames": len(frames),
                "subject": extract_subject_id(video_path),
            }
            metadata.append(video_metadata)
            
        except Exception as e:
            logger.error(f"Failed to process {video_path}: {e}")
            continue
    
    return metadata


def extract_subject_id(video_path: Path) -> str:
    """
    Extract subject ID from video path.
    
    Args:
        video_path: Path to video file
        
    Returns:
        Subject ID
    """
    # Try to extract from folder structure
    parts = video_path.parts
    
    # Look for common patterns
    for i, part in enumerate(parts):
        if part in ['real', 'fake']:
            if i > 0:
                return parts[i - 1]
    
    # Use filename as subject ID
    return video_path.stem


def load_subjects(subjects_file: str) -> Dict[str, str]:
    """
    Load subject information from file.
    
    Args:
        subjects_file: Path to subjects file
        
    Returns:
        Dictionary mapping subject IDs to splits
    """
    with open(subjects_file, 'r') as f:
        subjects = json.load(f)
    return subjects


def create_subject_splits(
    subjects: Dict[str, str],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    random_seed: int,
) -> Dict[str, List[str]]:
    """
    Create subject-based splits.
    
    Args:
        subjects: Dictionary mapping subject IDs to splits
        train_ratio: Training split ratio
        val_ratio: Validation split ratio
        test_ratio: Test split ratio
        random_seed: Random seed
        
    Returns:
        Dictionary mapping split names to subject lists
    """
    # Get subjects for each split
    train_subjects = [s for s, split in subjects.items() if split == "train"]
    val_subjects = [s for s, split in subjects.items() if split == "val"]
    test_subjects = [s for s, split in subjects.items() if split == "test"]
    
    return {
        "train": train_subjects,
        "val": val_subjects,
        "test": test_subjects,
    }


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Prepare dataset for deepfake detection")
    parser.add_argument("--src", type=str, required=True, help="Source data directory")
    parser.add_argument("--out", type=str, required=True, help="Output data directory")
    parser.add_argument("--fps", type=int, default=8, help="Frames per second for video extraction")
    parser.add_argument("--image-size", type=int, default=224, help="Image size for processing")
    parser.add_argument("--max-frames", type=int, default=16, help="Maximum number of frames per video")
    parser.add_argument("--subjects-file", type=str, help="Path to subjects file")
    parser.add_argument("--train-ratio", type=float, default=0.7, help="Training split ratio")
    parser.add_argument("--val-ratio", type=float, default=0.15, help="Validation split ratio")
    parser.add_argument("--test-ratio", type=float, default=0.15, help="Test split ratio")
    parser.add_argument("--random-seed", type=int, default=42, help="Random seed")
    parser.add_argument("--log-level", type=str, default="INFO", help="Logging level")
    
    args = parser.parse_args()
    
    # Set up logging
    setup_logging(level=args.log_level)
    
    # Prepare dataset
    prepare_dataset(
        src_dir=args.src,
        out_dir=args.out,
        fps=args.fps,
        image_size=args.image_size,
        max_frames=args.max_frames,
        subjects_file=args.subjects_file,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        random_seed=args.random_seed,
    )


if __name__ == "__main__":
    main()
