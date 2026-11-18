"""
Tests for reproducibility utilities.
"""

import pytest
import torch
import numpy as np
import random
import os

from llm_finetuning.utils.reproducibility import (
    set_seed,
    get_device_info,
    get_torch_version,
    get_cuda_version
)


class TestReproducibility:
    """Test cases for reproducibility utilities."""
    
    def test_set_seed(self):
        """Test seed setting functionality."""
        # Set seed
        set_seed(42)
        
        # Generate some random numbers
        python_random = random.random()
        numpy_random = np.random.random()
        torch_random = torch.rand(1).item()
        
        # Set seed again
        set_seed(42)
        
        # Generate same random numbers
        python_random_2 = random.random()
        numpy_random_2 = np.random.random()
        torch_random_2 = torch.rand(1).item()
        
        # Should be the same
        assert python_random == python_random_2
        assert numpy_random == numpy_random_2
        assert torch_random == torch_random_2
    
    def test_set_seed_environment_variables(self):
        """Test that environment variables are set correctly."""
        original_hashseed = os.environ.get("PYTHONHASHSEED")
        original_cublas = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
        
        try:
            set_seed(123)
            
            # Check environment variables
            assert os.environ["PYTHONHASHSEED"] == "123"
            assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"
            
        finally:
            # Restore original values
            if original_hashseed is not None:
                os.environ["PYTHONHASHSEED"] = original_hashseed
            elif "PYTHONHASHSEED" in os.environ:
                del os.environ["PYTHONHASHSEED"]
                
            if original_cublas is not None:
                os.environ["CUBLAS_WORKSPACE_CONFIG"] = original_cublas
            elif "CUBLAS_WORKSPACE_CONFIG" in os.environ:
                del os.environ["CUBLAS_WORKSPACE_CONFIG"]
    
    def test_get_device_info(self):
        """Test device information retrieval."""
        device_info = get_device_info()
        
        # Check required keys
        required_keys = ["cuda_available", "device_count", "current_device"]
        for key in required_keys:
            assert key in device_info
        
        # Check CUDA availability
        assert device_info["cuda_available"] == torch.cuda.is_available()
        
        if torch.cuda.is_available():
            # Check CUDA-specific info
            assert device_info["device_count"] == torch.cuda.device_count()
            assert device_info["current_device"] == torch.cuda.current_device()
            assert "device_name" in device_info
            assert "memory_allocated" in device_info
            assert "memory_reserved" in device_info
            assert "max_memory_allocated" in device_info
        else:
            # Check non-CUDA info
            assert device_info["device_count"] == 0
            assert device_info["current_device"] is None
    
    def test_get_torch_version(self):
        """Test PyTorch version retrieval."""
        version = get_torch_version()
        
        # Should return a string
        assert isinstance(version, str)
        
        # Should match torch.__version__
        assert version == torch.__version__
    
    def test_get_cuda_version(self):
        """Test CUDA version retrieval."""
        cuda_version = get_cuda_version()
        
        # Should return a string
        assert isinstance(cuda_version, str)
        
        if torch.cuda.is_available():
            # Should return actual CUDA version
            assert cuda_version == torch.version.cuda
        else:
            # Should return "CUDA not available"
            assert cuda_version == "CUDA not available"
    
    def test_deterministic_behavior(self):
        """Test that deterministic behavior is enabled."""
        set_seed(42)
        
        # Check that deterministic behavior is enabled
        assert torch.backends.cudnn.deterministic is True
        assert torch.backends.cudnn.benchmark is False
