"""
Tests for model utilities.
"""

import pytest
import torch
from unittest.mock import Mock, patch, MagicMock

from llm_finetuning.utils.model_utils import (
    setup_model_and_tokenizer,
    setup_lora_model,
    get_model_info,
    save_model_checkpoint,
    load_model_checkpoint
)


class TestModelUtils:
    """Test cases for model utilities."""
    
    @pytest.fixture
    def mock_model(self):
        """Create a mock model for testing."""
        model = Mock()
        model.num_parameters.return_value = 1000000
        model.config = Mock()
        model.config.use_cache = True
        model.config.pad_token_id = None
        return model
    
    @pytest.fixture
    def mock_tokenizer(self):
        """Create a mock tokenizer for testing."""
        tokenizer = Mock()
        tokenizer.pad_token = None
        tokenizer.eos_token = "<eos>"
        tokenizer.eos_token_id = 1
        return tokenizer
    
    @patch('llm_finetuning.utils.model_utils.AutoModelForCausalLM')
    @patch('llm_finetuning.utils.model_utils.AutoTokenizer')
    def test_setup_model_and_tokenizer(self, mock_tokenizer_class, mock_model_class, mock_tokenizer, mock_model):
        """Test model and tokenizer setup."""
        # Setup mocks
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        mock_model_class.from_pretrained.return_value = mock_model
        
        # Test setup
        model, tokenizer = setup_model_and_tokenizer("test-model")
        
        # Verify calls
        mock_tokenizer_class.from_pretrained.assert_called_once_with("test-model")
        mock_model_class.from_pretrained.assert_called_once()
        
        # Verify tokenizer setup
        assert tokenizer.pad_token == "<eos>"
        assert tokenizer.pad_token_id == 1
        
        # Verify model setup
        assert model.config.use_cache is False
        assert model.config.pad_token_id == 1
    
    @patch('llm_finetuning.utils.model_utils.AutoModelForCausalLM')
    @patch('llm_finetuning.utils.model_utils.AutoTokenizer')
    def test_setup_model_and_tokenizer_with_quantization(self, mock_tokenizer_class, mock_model_class):
        """Test model setup with quantization."""
        # Setup mocks
        mock_tokenizer = Mock()
        mock_tokenizer.pad_token = None
        mock_tokenizer.eos_token = "<eos>"
        mock_tokenizer.eos_token_id = 1
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        
        mock_model = Mock()
        mock_model.num_parameters.return_value = 1000000
        mock_model.config = Mock()
        mock_model_class.from_pretrained.return_value = mock_model
        
        # Test with 8-bit quantization
        model, tokenizer = setup_model_and_tokenizer(
            "test-model",
            load_in_8bit=True
        )
        
        # Verify quantization config was passed
        call_args = mock_model_class.from_pretrained.call_args
        assert "quantization_config" in call_args.kwargs
        quantization_config = call_args.kwargs["quantization_config"]
        assert quantization_config.load_in_8bit is True
    
    @patch('llm_finetuning.utils.model_utils.get_peft_model')
    @patch('llm_finetuning.utils.model_utils.LoraConfig')
    def test_setup_lora_model(self, mock_lora_config, mock_get_peft_model, mock_model):
        """Test LoRA model setup."""
        # Setup mocks
        mock_peft_config = Mock()
        mock_lora_config.return_value = mock_peft_config
        
        mock_lora_model = Mock()
        mock_lora_model.print_trainable_parameters = Mock()
        mock_get_peft_model.return_value = mock_lora_model
        
        # Test LoRA config
        lora_config = {
            "r": 16,
            "lora_alpha": 32,
            "lora_dropout": 0.1,
            "target_modules": ["q_proj", "v_proj"],
            "bias": "none"
        }
        
        result_model = setup_lora_model(mock_model, lora_config)
        
        # Verify LoRA config was created
        mock_lora_config.assert_called_once()
        call_args = mock_lora_config.call_args
        
        assert call_args.kwargs["r"] == 16
        assert call_args.kwargs["lora_alpha"] == 32
        assert call_args.kwargs["lora_dropout"] == 0.1
        assert call_args.kwargs["target_modules"] == ["q_proj", "v_proj"]
        assert call_args.kwargs["bias"] == "none"
        
        # Verify PEFT model was created
        mock_get_peft_model.assert_called_once_with(mock_model, mock_peft_config)
        
        # Verify trainable parameters were printed
        mock_lora_model.print_trainable_parameters.assert_called_once()
        
        assert result_model == mock_lora_model
    
    def test_get_model_info(self, mock_model):
        """Test model information retrieval."""
        # Mock parameters
        mock_param1 = Mock()
        mock_param1.numel.return_value = 1000
        mock_param1.element_size.return_value = 4
        mock_param1.requires_grad = True
        
        mock_param2 = Mock()
        mock_param2.numel.return_value = 2000
        mock_param2.element_size.return_value = 4
        mock_param2.requires_grad = False
        
        mock_model.parameters.return_value = [mock_param1, mock_param2]
        
        # Ensure no peft_config attribute exists
        if hasattr(mock_model, 'peft_config'):
            delattr(mock_model, 'peft_config')
        
        # Test without LoRA
        info = get_model_info(mock_model)
        
        assert info["num_parameters"] == 1000000
        assert info["num_trainable_parameters"] == 1000
        # Calculate expected size: (1000+2000)*4/1024/1024 = 0.01171875 MB
        expected_size = (1000 + 2000) * 4 / (1024 * 1024)
        assert info["model_size_mb"] == pytest.approx(expected_size, rel=1e-2)
        assert info["lora_enabled"] is False
    
    def test_get_model_info_with_lora(self, mock_model):
        """Test model information with LoRA."""
        # Mock LoRA config
        mock_model.peft_config = {"test": "config"}
        
        # Mock parameters
        mock_param = Mock()
        mock_param.numel.return_value = 1000
        mock_param.element_size.return_value = 4
        mock_param.requires_grad = True
        
        mock_model.parameters.return_value = [mock_param]
        
        info = get_model_info(mock_model)
        
        assert info["lora_enabled"] is True
        assert info["lora_config"] == {"test": "config"}
    
    @patch('os.makedirs')
    def test_save_model_checkpoint(self, mock_makedirs, mock_model, mock_tokenizer):
        """Test model checkpoint saving."""
        # Mock model save method
        mock_model.save_pretrained = Mock()
        mock_tokenizer.save_pretrained = Mock()
        
        save_model_checkpoint(mock_model, mock_tokenizer, "/test/output", "test_checkpoint")
        
        # Verify directory was created
        mock_makedirs.assert_called_once_with("/test/output/test_checkpoint", exist_ok=True)
        
        # Verify model and tokenizer were saved
        mock_model.save_pretrained.assert_called_once_with("/test/output/test_checkpoint")
        mock_tokenizer.save_pretrained.assert_called_once_with("/test/output/test_checkpoint")
    
    @patch('llm_finetuning.utils.model_utils.AutoTokenizer')
    @patch('llm_finetuning.utils.model_utils.AutoModelForCausalLM')
    @patch('peft.PeftModel')
    def test_load_model_checkpoint_with_lora(self, mock_peft_model, mock_model_class, mock_tokenizer_class):
        """Test loading model checkpoint with LoRA."""
        # Setup mocks
        mock_tokenizer = Mock()
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        
        mock_base_model = Mock()
        mock_model_class.from_pretrained.return_value = mock_base_model
        
        mock_lora_model = Mock()
        mock_peft_model.from_pretrained.return_value = mock_lora_model
        
        # Test loading
        model, tokenizer = load_model_checkpoint("/test/checkpoint")
        
        # Verify calls
        mock_tokenizer_class.from_pretrained.assert_called_once_with("/test/checkpoint")
        mock_model_class.from_pretrained.assert_called_once()
        mock_peft_model.from_pretrained.assert_called_once_with(mock_base_model, "/test/checkpoint")
        
        assert model == mock_lora_model
        assert tokenizer == mock_tokenizer
    
    @patch('llm_finetuning.utils.model_utils.AutoTokenizer')
    @patch('llm_finetuning.utils.model_utils.AutoModelForCausalLM')
    @patch('peft.PeftModel')
    def test_load_model_checkpoint_without_lora(self, mock_peft_model, mock_model_class, mock_tokenizer_class):
        """Test loading model checkpoint without LoRA."""
        # Setup mocks
        mock_tokenizer = Mock()
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        
        mock_base_model = Mock()
        mock_model_class.from_pretrained.return_value = mock_base_model
        
        # Make PeftModel.from_pretrained raise an exception
        mock_peft_model.from_pretrained.side_effect = Exception("No LoRA weights")
        
        # Test loading
        model, tokenizer = load_model_checkpoint("/test/checkpoint")
        
        # Verify calls
        mock_tokenizer_class.from_pretrained.assert_called_once_with("/test/checkpoint")
        mock_model_class.from_pretrained.assert_called_once()
        mock_peft_model.from_pretrained.assert_called_once()
        
        # Should return base model when LoRA loading fails
        assert model == mock_base_model
        assert tokenizer == mock_tokenizer
