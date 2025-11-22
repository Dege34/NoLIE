"""
Logging utilities for deepfake forensics.

Provides structured logging configuration with support for different log levels,
formatters, and handlers including file and console output.
"""

import logging
import logging.config
import sys
from pathlib import Path
from typing import Optional, Dict, Any
from rich.logging import RichHandler
from rich.console import Console


def setup_logging(
    level: str = "INFO",
    log_file: Optional[Path] = None,
    log_format: Optional[str] = None,
    use_rich: bool = True,
    project_name: str = "deepfake-forensics",
) -> logging.Logger:
    """
    Set up logging configuration for the application.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional path to log file
        log_format: Custom log format string
        use_rich: Whether to use rich console handler
        project_name: Name of the project for log formatting
        
    Returns:
        Configured logger instance
    """
    # Default log format
    if log_format is None:
        log_format = (
            "%(asctime)s - %(name)s - %(levelname)s - "
            "%(filename)s:%(lineno)d - %(message)s"
        )
    
    # Create log directory if log_file is specified
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Configure logging
    config: Dict[str, Any] = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "standard": {
                "format": log_format,
                "datefmt": "%Y-%m-%d %H:%M:%S",
            },
            "detailed": {
                "format": (
                    "%(asctime)s - %(name)s - %(levelname)s - "
                    "%(filename)s:%(lineno)d - %(funcName)s - %(message)s"
                ),
                "datefmt": "%Y-%m-%d %H:%M:%S",
            },
        },
        "handlers": {
            "console": {
                "class": "rich.logging.RichHandler" if use_rich else "logging.StreamHandler",
                "level": level,
                "formatter": "standard",
                "stream": sys.stdout,
                "rich_tracebacks": True,
                "show_time": True,
                "show_path": True,
            },
        },
        "loggers": {
            project_name: {
                "level": level,
                "handlers": ["console"],
                "propagate": False,
            },
            "deepfake_forensics": {
                "level": level,
                "handlers": ["console"],
                "propagate": False,
            },
        },
        "root": {
            "level": level,
            "handlers": ["console"],
        },
    }
    
    # Add file handler if log_file is specified
    if log_file:
        config["handlers"]["file"] = {
            "class": "logging.FileHandler",
            "level": level,
            "formatter": "detailed",
            "filename": str(log_file),
            "mode": "a",
        }
        
        # Add file handler to all loggers
        for logger_name in config["loggers"]:
            config["loggers"][logger_name]["handlers"].append("file")
        config["root"]["handlers"].append("file")
    
    # Apply configuration
    logging.config.dictConfig(config)
    
    # Get logger
    logger = get_logger(project_name)
    
    # Log configuration
    logger.info(f"Logging initialized with level: {level}")
    if log_file:
        logger.info(f"Log file: {log_file}")
    
    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance.
    
    Args:
        name: Logger name
        
    Returns:
        Logger instance
    """
    return logging.getLogger(name)


def setup_rich_logging(
    level: str = "INFO",
    console: Optional[Console] = None,
    show_time: bool = True,
    show_path: bool = True,
    rich_tracebacks: bool = True,
) -> RichHandler:
    """
    Set up rich logging handler.
    
    Args:
        level: Logging level
        console: Rich console instance
        show_time: Whether to show timestamps
        show_path: Whether to show file paths
        rich_tracebacks: Whether to use rich tracebacks
        
    Returns:
        Rich handler instance
    """
    if console is None:
        console = Console()
    
    handler = RichHandler(
        console=console,
        level=level,
        show_time=show_time,
        show_path=show_path,
        rich_tracebacks=rich_tracebacks,
    )
    
    return handler


def log_system_info(logger: logging.Logger) -> None:
    """
    Log system information for debugging.
    
    Args:
        logger: Logger instance
    """
    import platform
    import torch
    import sys
    
    logger.info("System Information:")
    logger.info(f"  Python version: {sys.version}")
    logger.info(f"  Platform: {platform.platform()}")
    logger.info(f"  PyTorch version: {torch.__version__}")
    logger.info(f"  CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"  CUDA version: {torch.version.cuda}")
        logger.info(f"  GPU count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            logger.info(f"  GPU {i}: {torch.cuda.get_device_name(i)}")


def log_config(logger: logging.Logger, config: Dict[str, Any], prefix: str = "Config") -> None:
    """
    Log configuration dictionary in a structured way.
    
    Args:
        logger: Logger instance
        config: Configuration dictionary
        prefix: Prefix for log messages
    """
    logger.info(f"{prefix}:")
    _log_dict(logger, config, indent=2)


def _log_dict(logger: logging.Logger, d: Dict[str, Any], indent: int = 0) -> None:
    """
    Recursively log dictionary contents.
    
    Args:
        logger: Logger instance
        d: Dictionary to log
        indent: Indentation level
    """
    for key, value in d.items():
        if isinstance(value, dict):
            logger.info(" " * indent + f"{key}:")
            _log_dict(logger, value, indent + 2)
        else:
            logger.info(" " * indent + f"{key}: {value}")


# Global logger instance
_global_logger: Optional[logging.Logger] = None


def get_global_logger() -> logging.Logger:
    """
    Get the global logger instance.
    
    Returns:
        Global logger instance
    """
    global _global_logger
    if _global_logger is None:
        _global_logger = setup_logging()
    return _global_logger


def set_global_logger(logger: logging.Logger) -> None:
    """
    Set the global logger instance.
    
    Args:
        logger: Logger instance to set as global
    """
    global _global_logger
    _global_logger = logger
