"""
Reproducibility utilities for consistent experiments.
"""

import random
import numpy as np
import torch
import os
from typing import Dict, Any


def set_seed(seed: int = 42) -> None:
    """
    Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # Ensure deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Set environment variables for additional reproducibility
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


def get_device_info() -> Dict[str, Any]:
    """
    Get information about available devices.
    
    Returns:
        Dictionary containing device information
    """
    device_info = {
        "cuda_available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "current_device": torch.cuda.current_device() if torch.cuda.is_available() else None,
    }
    
    if torch.cuda.is_available():
        device_info.update({
            "device_name": torch.cuda.get_device_name(0),
            "memory_allocated": torch.cuda.memory_allocated(0),
            "memory_reserved": torch.cuda.memory_reserved(0),
            "max_memory_allocated": torch.cuda.max_memory_allocated(0),
        })
    
    return device_info


def get_torch_version() -> str:
    """Get PyTorch version information."""
    return torch.__version__


def get_cuda_version() -> str:
    """Get CUDA version if available."""
    if torch.cuda.is_available():
        return torch.version.cuda
    return "CUDA not available"
