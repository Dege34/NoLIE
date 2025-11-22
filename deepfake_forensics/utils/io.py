"""
I/O utilities for deepfake forensics.

Provides functions for loading and saving checkpoints, configurations,
and other data structures with proper error handling and validation.
"""

import json
import pickle
import yaml
from pathlib import Path
from typing import Any, Dict, Optional, Union, Tuple
import torch
import logging
from omegaconf import OmegaConf, DictConfig

logger = logging.getLogger(__name__)


def load_checkpoint(
    checkpoint_path: Union[str, Path],
    map_location: Optional[str] = None,
    strict: bool = True,
) -> Dict[str, Any]:
    """
    Load a PyTorch checkpoint with error handling.
    
    Args:
        checkpoint_path: Path to the checkpoint file
        map_location: Device to map tensors to
        strict: Whether to strictly enforce checkpoint loading
        
    Returns:
        Dictionary containing checkpoint data
        
    Raises:
        FileNotFoundError: If checkpoint file doesn't exist
        RuntimeError: If checkpoint loading fails
    """
    checkpoint_path = Path(checkpoint_path)
    
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location=map_location)
        logger.info(f"Loaded checkpoint from {checkpoint_path}")
        return checkpoint
    except Exception as e:
        raise RuntimeError(f"Failed to load checkpoint from {checkpoint_path}: {e}")


def save_checkpoint(
    checkpoint: Dict[str, Any],
    checkpoint_path: Union[str, Path],
    create_dirs: bool = True,
) -> None:
    """
    Save a PyTorch checkpoint with error handling.
    
    Args:
        checkpoint: Dictionary containing checkpoint data
        checkpoint_path: Path to save the checkpoint
        create_dirs: Whether to create parent directories
        
    Raises:
        RuntimeError: If checkpoint saving fails
    """
    checkpoint_path = Path(checkpoint_path)
    
    if create_dirs:
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Saved checkpoint to {checkpoint_path}")
    except Exception as e:
        raise RuntimeError(f"Failed to save checkpoint to {checkpoint_path}: {e}")


def load_config(
    config_path: Union[str, Path],
    resolve: bool = True,
) -> DictConfig:
    """
    Load configuration from YAML file using OmegaConf.
    
    Args:
        config_path: Path to the configuration file
        resolve: Whether to resolve interpolations
        
    Returns:
        OmegaConf DictConfig object
        
    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If YAML parsing fails
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    try:
        config = OmegaConf.load(config_path)
        if resolve:
            config = OmegaConf.to_container(config, resolve=True)
            config = OmegaConf.create(config)
        logger.info(f"Loaded config from {config_path}")
        return config
    except yaml.YAMLError as e:
        raise yaml.YAMLError(f"Failed to parse YAML config from {config_path}: {e}")
    except Exception as e:
        raise RuntimeError(f"Failed to load config from {config_path}: {e}")


def save_config(
    config: Union[Dict[str, Any], DictConfig],
    config_path: Union[str, Path],
    create_dirs: bool = True,
) -> None:
    """
    Save configuration to YAML file.
    
    Args:
        config: Configuration dictionary or DictConfig
        config_path: Path to save the configuration
        create_dirs: Whether to create parent directories
        
    Raises:
        RuntimeError: If config saving fails
    """
    config_path = Path(config_path)
    
    if create_dirs:
        config_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        # Convert to OmegaConf if needed
        if not isinstance(config, DictConfig):
            config = OmegaConf.create(config)
        
        # Save as YAML
        with open(config_path, 'w') as f:
            OmegaConf.save(config, f)
        
        logger.info(f"Saved config to {config_path}")
    except Exception as e:
        raise RuntimeError(f"Failed to save config to {config_path}: {e}")


def load_json(
    json_path: Union[str, Path],
    encoding: str = "utf-8",
) -> Dict[str, Any]:
    """
    Load JSON file with error handling.
    
    Args:
        json_path: Path to the JSON file
        encoding: File encoding
        
    Returns:
        Dictionary containing JSON data
        
    Raises:
        FileNotFoundError: If JSON file doesn't exist
        json.JSONDecodeError: If JSON parsing fails
    """
    json_path = Path(json_path)
    
    if not json_path.exists():
        raise FileNotFoundError(f"JSON file not found: {json_path}")
    
    try:
        with open(json_path, 'r', encoding=encoding) as f:
            data = json.load(f)
        logger.info(f"Loaded JSON from {json_path}")
        return data
    except json.JSONDecodeError as e:
        raise json.JSONDecodeError(f"Failed to parse JSON from {json_path}: {e}")
    except Exception as e:
        raise RuntimeError(f"Failed to load JSON from {json_path}: {e}")


def save_json(
    data: Dict[str, Any],
    json_path: Union[str, Path],
    indent: int = 2,
    create_dirs: bool = True,
    encoding: str = "utf-8",
) -> None:
    """
    Save data to JSON file with error handling.
    
    Args:
        data: Data to save
        json_path: Path to save the JSON file
        indent: JSON indentation
        create_dirs: Whether to create parent directories
        encoding: File encoding
        
    Raises:
        RuntimeError: If JSON saving fails
    """
    json_path = Path(json_path)
    
    if create_dirs:
        json_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        with open(json_path, 'w', encoding=encoding) as f:
            json.dump(data, f, indent=indent, ensure_ascii=False)
        logger.info(f"Saved JSON to {json_path}")
    except Exception as e:
        raise RuntimeError(f"Failed to save JSON to {json_path}: {e}")


def load_pickle(
    pickle_path: Union[str, Path],
) -> Any:
    """
    Load pickle file with error handling.
    
    Args:
        pickle_path: Path to the pickle file
        
    Returns:
        Unpickled data
        
    Raises:
        FileNotFoundError: If pickle file doesn't exist
        pickle.PickleError: If pickle loading fails
    """
    pickle_path = Path(pickle_path)
    
    if not pickle_path.exists():
        raise FileNotFoundError(f"Pickle file not found: {pickle_path}")
    
    try:
        with open(pickle_path, 'rb') as f:
            data = pickle.load(f)
        logger.info(f"Loaded pickle from {pickle_path}")
        return data
    except pickle.PickleError as e:
        raise pickle.PickleError(f"Failed to load pickle from {pickle_path}: {e}")
    except Exception as e:
        raise RuntimeError(f"Failed to load pickle from {pickle_path}: {e}")


def save_pickle(
    data: Any,
    pickle_path: Union[str, Path],
    create_dirs: bool = True,
) -> None:
    """
    Save data to pickle file with error handling.
    
    Args:
        data: Data to save
        pickle_path: Path to save the pickle file
        create_dirs: Whether to create parent directories
        
    Raises:
        RuntimeError: If pickle saving fails
    """
    pickle_path = Path(pickle_path)
    
    if create_dirs:
        pickle_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        with open(pickle_path, 'wb') as f:
            pickle.dump(data, f)
        logger.info(f"Saved pickle to {pickle_path}")
    except Exception as e:
        raise RuntimeError(f"Failed to save pickle to {pickle_path}: {e}")


def ensure_dir(path: Union[str, Path]) -> Path:
    """
    Ensure directory exists, create if necessary.
    
    Args:
        path: Directory path
        
    Returns:
        Path object
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_file_size(file_path: Union[str, Path]) -> int:
    """
    Get file size in bytes.
    
    Args:
        file_path: Path to the file
        
    Returns:
        File size in bytes
        
    Raises:
        FileNotFoundError: If file doesn't exist
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    return file_path.stat().st_size


def get_file_extension(file_path: Union[str, Path]) -> str:
    """
    Get file extension (including the dot).
    
    Args:
        file_path: Path to the file
        
    Returns:
        File extension
    """
    return Path(file_path).suffix.lower()


def is_valid_file(file_path: Union[str, Path], extensions: list) -> bool:
    """
    Check if file has a valid extension.
    
    Args:
        file_path: Path to the file
        extensions: List of valid extensions (with or without dots)
        
    Returns:
        True if file has valid extension
    """
    file_ext = get_file_extension(file_path)
    
    # Normalize extensions (ensure they start with dot)
    normalized_extensions = [
        ext if ext.startswith('.') else f'.{ext}'
        for ext in extensions
    ]
    
    return file_ext in normalized_extensions


def find_files(
    directory: Union[str, Path],
    pattern: str = "*",
    recursive: bool = True,
    extensions: Optional[list] = None,
) -> list:
    """
    Find files matching pattern and extensions.
    
    Args:
        directory: Directory to search
        pattern: File pattern to match
        recursive: Whether to search recursively
        extensions: List of valid extensions
        
    Returns:
        List of matching file paths
    """
    directory = Path(directory)
    
    if not directory.exists():
        logger.warning(f"Directory does not exist: {directory}")
        return []
    
    # Find files matching pattern
    if recursive:
        files = list(directory.rglob(pattern))
    else:
        files = list(directory.glob(pattern))
    
    # Filter by extensions if specified
    if extensions:
        files = [f for f in files if is_valid_file(f, extensions)]
    
    return sorted(files)


def copy_file(
    src: Union[str, Path],
    dst: Union[str, Path],
    create_dirs: bool = True,
) -> None:
    """
    Copy file with error handling.
    
    Args:
        src: Source file path
        dst: Destination file path
        create_dirs: Whether to create parent directories
        
    Raises:
        FileNotFoundError: If source file doesn't exist
        RuntimeError: If copying fails
    """
    import shutil
    
    src = Path(src)
    dst = Path(dst)
    
    if not src.exists():
        raise FileNotFoundError(f"Source file not found: {src}")
    
    if create_dirs:
        dst.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        shutil.copy2(src, dst)
        logger.info(f"Copied {src} to {dst}")
    except Exception as e:
        raise RuntimeError(f"Failed to copy {src} to {dst}: {e}")


def move_file(
    src: Union[str, Path],
    dst: Union[str, Path],
    create_dirs: bool = True,
) -> None:
    """
    Move file with error handling.
    
    Args:
        src: Source file path
        dst: Destination file path
        create_dirs: Whether to create parent directories
        
    Raises:
        FileNotFoundError: If source file doesn't exist
        RuntimeError: If moving fails
    """
    import shutil
    
    src = Path(src)
    dst = Path(dst)
    
    if not src.exists():
        raise FileNotFoundError(f"Source file not found: {src}")
    
    if create_dirs:
        dst.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        shutil.move(str(src), str(dst))
        logger.info(f"Moved {src} to {dst}")
    except Exception as e:
        raise RuntimeError(f"Failed to move {src} to {dst}: {e}")
