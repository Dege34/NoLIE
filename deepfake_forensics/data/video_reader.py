"""
Video reading and processing utilities for deepfake forensics.

Provides efficient video reading, frame extraction, and preprocessing
for both training and inference pipelines.
"""

import cv2
import numpy as np
import torch
from typing import Dict, List, Optional, Tuple, Union, Iterator, Any
from pathlib import Path
import logging
import imageio
from PIL import Image
import ffmpeg
import tempfile
import os

logger = logging.getLogger(__name__)


class VideoReader:
    """
    Efficient video reader with multiple backends.
    """
    
    def __init__(
        self,
        backend: str = "opencv",
        max_frames: int = 16,
        fps: int = 8,
        image_size: int = 224,
    ):
        """
        Initialize video reader.
        
        Args:
            backend: Video reading backend ('opencv', 'imageio', 'ffmpeg')
            max_frames: Maximum number of frames to read
            fps: Target frames per second
            image_size: Target image size
        """
        self.backend = backend
        self.max_frames = max_frames
        self.fps = fps
        self.image_size = image_size
        
        # Initialize backend-specific settings
        if backend == "opencv":
            self._init_opencv()
        elif backend == "imageio":
            self._init_imageio()
        elif backend == "ffmpeg":
            self._init_ffmpeg()
        else:
            raise ValueError(f"Unknown backend: {backend}")
    
    def _init_opencv(self) -> None:
        """Initialize OpenCV backend."""
        self.cap = None
        logger.info("Initialized OpenCV video reader")
    
    def _init_imageio(self) -> None:
        """Initialize imageio backend."""
        logger.info("Initialized imageio video reader")
    
    def _init_ffmpeg(self) -> None:
        """Initialize FFmpeg backend."""
        logger.info("Initialized FFmpeg video reader")
    
    def read_video(
        self,
        video_path: Union[str, Path],
        start_time: float = 0.0,
        duration: Optional[float] = None,
    ) -> np.ndarray:
        """
        Read video and return frames.
        
        Args:
            video_path: Path to video file
            start_time: Start time in seconds
            duration: Duration in seconds (None for full video)
            
        Returns:
            Array of frames (T, H, W, C)
        """
        if self.backend == "opencv":
            return self._read_opencv(video_path, start_time, duration)
        elif self.backend == "imageio":
            return self._read_imageio(video_path, start_time, duration)
        elif self.backend == "ffmpeg":
            return self._read_ffmpeg(video_path, start_time, duration)
        else:
            raise ValueError(f"Unknown backend: {self.backend}")
    
    def _read_opencv(
        self,
        video_path: Union[str, Path],
        start_time: float = 0.0,
        duration: Optional[float] = None,
    ) -> np.ndarray:
        """Read video using OpenCV."""
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        try:
            # Get video properties
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            
            if fps <= 0:
                fps = 30.0  # Default fallback
            
            # Calculate frame indices
            start_frame = int(start_time * fps)
            if duration is not None:
                end_frame = int((start_time + duration) * fps)
            else:
                end_frame = total_frames
            
            # Calculate sampling interval
            frame_interval = max(1, int(fps / self.fps))
            
            # Extract frames
            frames = []
            frame_idx = start_frame
            
            while frame_idx < end_frame and len(frames) < self.max_frames:
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
                    # If we can't read more frames, break
                    break
                
                frame_idx += frame_interval
            
            # Pad with last frame if needed
            while len(frames) < self.max_frames:
                if frames:
                    frames.append(frames[-1])
                else:
                    # Create dummy frame
                    frames.append(np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8))
            
            return np.array(frames[:self.max_frames])
            
        finally:
            cap.release()
    
    def _read_imageio(
        self,
        video_path: Union[str, Path],
        start_time: float = 0.0,
        duration: Optional[float] = None,
    ) -> np.ndarray:
        """Read video using imageio."""
        try:
            # Read video
            reader = imageio.get_reader(str(video_path))
            
            # Get video properties
            fps = reader.get_meta_data().get('fps', 30.0)
            total_frames = reader.count_frames()
            
            # Calculate frame indices
            start_frame = int(start_time * fps)
            if duration is not None:
                end_frame = int((start_time + duration) * fps)
            else:
                end_frame = total_frames
            
            # Calculate sampling interval
            frame_interval = max(1, int(fps / self.fps))
            
            # Extract frames
            frames = []
            frame_idx = start_frame
            
            while frame_idx < end_frame and len(frames) < self.max_frames:
                try:
                    frame = reader.get_data(frame_idx)
                    
                    # Convert to RGB if needed
                    if len(frame.shape) == 3 and frame.shape[2] == 3:
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    
                    # Resize if needed
                    if frame.shape[:2] != (self.image_size, self.image_size):
                        frame = cv2.resize(frame, (self.image_size, self.image_size))
                    
                    frames.append(frame)
                except IndexError:
                    # If we can't read more frames, break
                    break
                
                frame_idx += frame_interval
            
            # Pad with last frame if needed
            while len(frames) < self.max_frames:
                if frames:
                    frames.append(frames[-1])
                else:
                    # Create dummy frame
                    frames.append(np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8))
            
            return np.array(frames[:self.max_frames])
            
        except Exception as e:
            logger.error(f"Failed to read video with imageio: {e}")
            # Return dummy video
            return np.zeros((self.max_frames, self.image_size, self.image_size, 3), dtype=np.uint8)
    
    def _read_ffmpeg(
        self,
        video_path: Union[str, Path],
        start_time: float = 0.0,
        duration: Optional[float] = None,
    ) -> np.ndarray:
        """Read video using FFmpeg."""
        try:
            # Build FFmpeg command
            cmd = ffmpeg.input(str(video_path), ss=start_time)
            
            if duration is not None:
                cmd = cmd.filter('trim', duration=duration)
            
            # Set output format
            cmd = cmd.filter('fps', fps=self.fps)
            cmd = cmd.filter('scale', self.image_size, self.image_size)
            cmd = cmd.output('pipe:', format='rawvideo', pix_fmt='rgb24')
            
            # Execute command
            out, _ = cmd.run(capture_stdout=True, capture_stderr=True)
            
            # Convert bytes to numpy array
            frames = np.frombuffer(out, np.uint8)
            frames = frames.reshape(-1, self.image_size, self.image_size, 3)
            
            # Limit to max_frames
            if len(frames) > self.max_frames:
                frames = frames[:self.max_frames]
            
            # Pad if needed
            while len(frames) < self.max_frames:
                if len(frames) > 0:
                    frames = np.vstack([frames, frames[-1:1]])
                else:
                    frames = np.zeros((1, self.image_size, self.image_size, 3), dtype=np.uint8)
            
            return frames[:self.max_frames]
            
        except Exception as e:
            logger.error(f"Failed to read video with FFmpeg: {e}")
            # Return dummy video
            return np.zeros((self.max_frames, self.image_size, self.image_size, 3), dtype=np.uint8)
    
    def get_video_info(self, video_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Get video information.
        
        Args:
            video_path: Path to video file
            
        Returns:
            Dictionary containing video information
        """
        if self.backend == "opencv":
            return self._get_info_opencv(video_path)
        elif self.backend == "imageio":
            return self._get_info_imageio(video_path)
        elif self.backend == "ffmpeg":
            return self._get_info_ffmpeg(video_path)
        else:
            raise ValueError(f"Unknown backend: {self.backend}")
    
    def _get_info_opencv(self, video_path: Union[str, Path]) -> Dict[str, Any]:
        """Get video info using OpenCV."""
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        try:
            info = {
                'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                'fps': cap.get(cv2.CAP_PROP_FPS),
                'frame_count': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
                'duration': cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS),
            }
            return info
        finally:
            cap.release()
    
    def _get_info_imageio(self, video_path: Union[str, Path]) -> Dict[str, Any]:
        """Get video info using imageio."""
        try:
            reader = imageio.get_reader(str(video_path))
            meta = reader.get_meta_data()
            
            info = {
                'width': meta.get('size', [0, 0])[0],
                'height': meta.get('size', [0, 0])[1],
                'fps': meta.get('fps', 30.0),
                'frame_count': reader.count_frames(),
                'duration': reader.count_frames() / meta.get('fps', 30.0),
            }
            return info
        except Exception as e:
            logger.error(f"Failed to get video info with imageio: {e}")
            return {}
    
    def _get_info_ffmpeg(self, video_path: Union[str, Path]) -> Dict[str, Any]:
        """Get video info using FFmpeg."""
        try:
            probe = ffmpeg.probe(str(video_path))
            video_stream = next(s for s in probe['streams'] if s['codec_type'] == 'video')
            
            info = {
                'width': int(video_stream['width']),
                'height': int(video_stream['height']),
                'fps': eval(video_stream['r_frame_rate']),
                'frame_count': int(video_stream.get('nb_frames', 0)),
                'duration': float(probe['format']['duration']),
            }
            return info
        except Exception as e:
            logger.error(f"Failed to get video info with FFmpeg: {e}")
            return {}


class FrameExtractor:
    """
    Frame extraction utility for video processing.
    """
    
    def __init__(
        self,
        image_size: int = 224,
        max_frames: int = 16,
        fps: int = 8,
    ):
        """
        Initialize frame extractor.
        
        Args:
            image_size: Target image size
            max_frames: Maximum number of frames
            fps: Target frames per second
        """
        self.image_size = image_size
        self.max_frames = max_frames
        self.fps = fps
    
    def extract_frames(
        self,
        video_path: Union[str, Path],
        output_dir: Union[str, Path],
        prefix: str = "frame",
        start_time: float = 0.0,
        duration: Optional[float] = None,
    ) -> List[Path]:
        """
        Extract frames from video and save to directory.
        
        Args:
            video_path: Path to video file
            output_dir: Directory to save frames
            prefix: Prefix for frame filenames
            start_time: Start time in seconds
            duration: Duration in seconds
            
        Returns:
            List of saved frame paths
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Read video
        reader = VideoReader(
            max_frames=self.max_frames,
            fps=self.fps,
            image_size=self.image_size,
        )
        
        frames = reader.read_video(video_path, start_time, duration)
        
        # Save frames
        frame_paths = []
        for i, frame in enumerate(frames):
            frame_path = output_dir / f"{prefix}_{i:04d}.jpg"
            cv2.imwrite(str(frame_path), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            frame_paths.append(frame_path)
        
        logger.info(f"Extracted {len(frame_paths)} frames to {output_dir}")
        return frame_paths
    
    def extract_frames_batch(
        self,
        video_paths: List[Union[str, Path]],
        output_dir: Union[str, Path],
        prefix: str = "frame",
    ) -> Dict[str, List[Path]]:
        """
        Extract frames from multiple videos.
        
        Args:
            video_paths: List of video file paths
            output_dir: Base directory to save frames
            prefix: Prefix for frame filenames
            
        Returns:
            Dictionary mapping video paths to frame paths
        """
        output_dir = Path(output_dir)
        results = {}
        
        for video_path in video_paths:
            video_name = Path(video_path).stem
            video_output_dir = output_dir / video_name
            
            try:
                frame_paths = self.extract_frames(
                    video_path,
                    video_output_dir,
                    prefix,
                )
                results[str(video_path)] = frame_paths
            except Exception as e:
                logger.error(f"Failed to extract frames from {video_path}: {e}")
                results[str(video_path)] = []
        
        return results


class VideoProcessor:
    """
    High-level video processing utility.
    """
    
    def __init__(
        self,
        image_size: int = 224,
        max_frames: int = 16,
        fps: int = 8,
        backend: str = "opencv",
    ):
        """
        Initialize video processor.
        
        Args:
            image_size: Target image size
            max_frames: Maximum number of frames
            fps: Target frames per second
            backend: Video reading backend
        """
        self.image_size = image_size
        self.max_frames = max_frames
        self.fps = fps
        self.backend = backend
        
        self.reader = VideoReader(
            backend=backend,
            max_frames=max_frames,
            fps=fps,
            image_size=image_size,
        )
    
    def process_video(
        self,
        video_path: Union[str, Path],
        transforms: Optional[Any] = None,
        return_metadata: bool = False,
    ) -> Union[np.ndarray, Tuple[np.ndarray, Dict[str, Any]]]:
        """
        Process video with optional transforms.
        
        Args:
            video_path: Path to video file
            transforms: Optional transforms to apply
            return_metadata: Whether to return metadata
            
        Returns:
            Processed video array or tuple of (array, metadata)
        """
        # Read video
        frames = self.reader.read_video(video_path)
        
        # Apply transforms if provided
        if transforms is not None:
            frames = transforms(frames)
        
        # Get metadata if requested
        if return_metadata:
            metadata = self.reader.get_video_info(video_path)
            return frames, metadata
        
        return frames
    
    def process_video_batch(
        self,
        video_paths: List[Union[str, Path]],
        transforms: Optional[Any] = None,
        return_metadata: bool = False,
    ) -> Union[List[np.ndarray], Tuple[List[np.ndarray], List[Dict[str, Any]]]]:
        """
        Process multiple videos.
        
        Args:
            video_paths: List of video file paths
            transforms: Optional transforms to apply
            return_metadata: Whether to return metadata
            
        Returns:
            List of processed video arrays or tuple of (arrays, metadata)
        """
        results = []
        metadata_list = []
        
        for video_path in video_paths:
            try:
                if return_metadata:
                    frames, metadata = self.process_video(
                        video_path, transforms, return_metadata=True
                    )
                    results.append(frames)
                    metadata_list.append(metadata)
                else:
                    frames = self.process_video(video_path, transforms)
                    results.append(frames)
            except Exception as e:
                logger.error(f"Failed to process video {video_path}: {e}")
                # Add dummy data
                dummy_frames = np.zeros((self.max_frames, self.image_size, self.image_size, 3), dtype=np.uint8)
                results.append(dummy_frames)
                if return_metadata:
                    metadata_list.append({})
        
        if return_metadata:
            return results, metadata_list
        return results


def create_video_reader(
    backend: str = "opencv",
    max_frames: int = 16,
    fps: int = 8,
    image_size: int = 224,
) -> VideoReader:
    """
    Create a video reader instance.
    
    Args:
        backend: Video reading backend
        max_frames: Maximum number of frames
        fps: Target frames per second
        image_size: Target image size
        
    Returns:
        VideoReader instance
    """
    return VideoReader(
        backend=backend,
        max_frames=max_frames,
        fps=fps,
        image_size=image_size,
    )


def create_frame_extractor(
    image_size: int = 224,
    max_frames: int = 16,
    fps: int = 8,
) -> FrameExtractor:
    """
    Create a frame extractor instance.
    
    Args:
        image_size: Target image size
        max_frames: Maximum number of frames
        fps: Target frames per second
        
    Returns:
        FrameExtractor instance
    """
    return FrameExtractor(
        image_size=image_size,
        max_frames=max_frames,
        fps=fps,
    )


def create_video_processor(
    image_size: int = 224,
    max_frames: int = 16,
    fps: int = 8,
    backend: str = "opencv",
) -> VideoProcessor:
    """
    Create a video processor instance.
    
    Args:
        image_size: Target image size
        max_frames: Maximum number of frames
        fps: Target frames per second
        backend: Video reading backend
        
    Returns:
        VideoProcessor instance
    """
    return VideoProcessor(
        image_size=image_size,
        max_frames=max_frames,
        fps=fps,
        backend=backend,
    )
