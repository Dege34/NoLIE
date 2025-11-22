"""
Seed management utilities for reproducibility.

Provides functions to set random seeds across different libraries and frameworks
to ensure reproducible results in deepfake forensics experiments.
"""

import random
import os
from typing import Optional, Union, Dict, Any
import numpy as np
import torch
from pathlib import Path


def set_seed(
    seed: int = 42,
    deterministic: bool = True,
    benchmark: bool = False,
    warn_only: bool = False,
) -> Dict[str, Any]:
    """
    Set random seeds for reproducibility across different libraries.
    
    Args:
        seed: Random seed value
        deterministic: Whether to use deterministic algorithms
        benchmark: Whether to use benchmark mode for cuDNN
        warn_only: Whether to only warn on non-deterministic operations
        
    Returns:
        Dictionary with seed information and status
    """
    seed_info = {
        "seed": seed,
        "deterministic": deterministic,
        "benchmark": benchmark,
        "warn_only": warn_only,
        "status": {},
    }
    
    # Python random
    random.seed(seed)
    seed_info["status"]["python"] = "set"
    
    # NumPy
    np.random.seed(seed)
    seed_info["status"]["numpy"] = "set"
    
    # PyTorch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        seed_info["status"]["torch_cuda"] = "set"
    seed_info["status"]["torch"] = "set"
    
    # Set deterministic algorithms
    if deterministic:
        # PyTorch deterministic operations
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
        # Set environment variables for additional determinism
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        os.environ["PYTHONHASHSEED"] = str(seed)
        
        # Enable deterministic algorithms in PyTorch
        torch.use_deterministic_algorithms(True, warn_only=warn_only)
        
        seed_info["status"]["deterministic"] = "enabled"
    else:
        # Allow non-deterministic algorithms for better performance
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = benchmark
        seed_info["status"]["deterministic"] = "disabled"
    
    # Set benchmark mode
    if benchmark and not deterministic:
        torch.backends.cudnn.benchmark = True
        seed_info["status"]["benchmark"] = "enabled"
    else:
        seed_info["status"]["benchmark"] = "disabled"
    
    return seed_info


def get_random_state() -> Dict[str, Any]:
    """
    Get current random state from all libraries.
    
    Returns:
        Dictionary containing random states
    """
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state()
    
    return state


def set_random_state(state: Dict[str, Any]) -> None:
    """
    Set random state from a previously saved state.
    
    Args:
        state: Dictionary containing random states
    """
    if "python" in state:
        random.setstate(state["python"])
    
    if "numpy" in state:
        np.random.set_state(state["numpy"])
    
    if "torch" in state:
        torch.set_rng_state(state["torch"])
    
    if "torch_cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state(state["torch_cuda"])


def save_random_state(filepath: Union[str, Path]) -> None:
    """
    Save current random state to file.
    
    Args:
        filepath: Path to save the random state
    """
    state = get_random_state()
    torch.save(state, filepath)


def load_random_state(filepath: Union[str, Path]) -> None:
    """
    Load random state from file.
    
    Args:
        filepath: Path to load the random state from
    """
    state = torch.load(filepath, map_location="cpu")
    set_random_state(state)


def create_deterministic_worker_init_fn(seed: int) -> callable:
    """
    Create a worker initialization function for DataLoader workers.
    
    Args:
        seed: Base seed value
        
    Returns:
        Worker initialization function
    """
    def worker_init_fn(worker_id: int) -> None:
        """Initialize worker with deterministic seed."""
        worker_seed = seed + worker_id
        set_seed(worker_seed, deterministic=True)
    
    return worker_init_fn


def ensure_reproducibility(
    seed: int = 42,
    deterministic: bool = True,
    benchmark: bool = False,
    log_seed_info: bool = True,
) -> Dict[str, Any]:
    """
    Ensure reproducibility by setting all random seeds and configurations.
    
    Args:
        seed: Random seed value
        deterministic: Whether to use deterministic algorithms
        benchmark: Whether to use benchmark mode for cuDNN
        log_seed_info: Whether to log seed information
        
    Returns:
        Dictionary with seed information and status
    """
    # Set seeds
    seed_info = set_seed(seed, deterministic, benchmark)
    
    # Log seed information if requested
    if log_seed_info:
        import logging
        logger = logging.getLogger(__name__)
        logger.info(f"Reproducibility enabled with seed: {seed}")
        logger.info(f"Deterministic: {deterministic}")
        logger.info(f"Benchmark: {benchmark}")
        
        for component, status in seed_info["status"].items():
            logger.info(f"  {component}: {status}")
    
    return seed_info


def check_reproducibility(
    seed: int = 42,
    num_tests: int = 5,
    tolerance: float = 1e-6,
) -> Dict[str, Any]:
    """
    Check if the current setup produces reproducible results.
    
    Args:
        seed: Random seed value
        num_tests: Number of tests to run
        tolerance: Tolerance for comparing results
        
    Returns:
        Dictionary with reproducibility test results
    """
    results = {
        "seed": seed,
        "num_tests": num_tests,
        "tolerance": tolerance,
        "tests": [],
        "reproducible": True,
    }
    
    # Test reproducibility
    for i in range(num_tests):
        # Set seed
        set_seed(seed, deterministic=True)
        
        # Generate random numbers
        python_rand = random.random()
        numpy_rand = np.random.random()
        torch_rand = torch.rand(1).item()
        
        # Store results
        test_result = {
            "test": i,
            "python": python_rand,
            "numpy": numpy_rand,
            "torch": torch_rand,
        }
        results["tests"].append(test_result)
    
    # Check if all tests produced the same results
    if len(results["tests"]) > 1:
        first_test = results["tests"][0]
        for test in results["tests"][1:]:
            for key in ["python", "numpy", "torch"]:
                if abs(test[key] - first_test[key]) > tolerance:
                    results["reproducible"] = False
                    break
            if not results["reproducible"]:
                break
    
    return results


# Global seed management
_global_seed: Optional[int] = None
_global_deterministic: bool = True


def set_global_seed(seed: int, deterministic: bool = True) -> None:
    """
    Set global seed configuration.
    
    Args:
        seed: Global seed value
        deterministic: Whether to use deterministic algorithms
    """
    global _global_seed, _global_deterministic
    _global_seed = seed
    _global_deterministic = deterministic
    set_seed(seed, deterministic)


def get_global_seed() -> Optional[int]:
    """
    Get the current global seed.
    
    Returns:
        Current global seed or None if not set
    """
    return _global_seed


def is_global_deterministic() -> bool:
    """
    Check if global deterministic mode is enabled.
    
    Returns:
        True if deterministic mode is enabled
    """
    return _global_deterministic
