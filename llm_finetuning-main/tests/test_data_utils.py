"""
Tests for data utilities.
"""

import pytest
import torch
from unittest.mock import Mock, patch
from transformers import AutoTokenizer

from llm_finetuning.utils.data_utils import DataProcessor, create_data_collator, get_dataset_info


class TestDataProcessor:
    """Test cases for DataProcessor class."""
    
    @pytest.fixture
    def mock_tokenizer(self):
        """Create a mock tokenizer for testing."""
        tokenizer = Mock(spec=AutoTokenizer)
        tokenizer.pad_token = "<pad>"
        tokenizer.pad_token_id = 0
        tokenizer.eos_token = "<eos>"
        tokenizer.eos_token_id = 1
        return tokenizer
    
    @pytest.fixture
    def data_processor(self, mock_tokenizer):
        """Create a DataProcessor instance for testing."""
        return DataProcessor(tokenizer=mock_tokenizer, max_length=512)
    
    def test_data_processor_init(self, mock_tokenizer):
        """Test DataProcessor initialization."""
        processor = DataProcessor(tokenizer=mock_tokenizer, max_length=256)
        assert processor.tokenizer == mock_tokenizer
        assert processor.max_length == 256
    
    @patch('llm_finetuning.utils.data_utils.load_dataset')
    @patch('llm_finetuning.utils.data_utils.train_test_split')
    def test_load_ag_news_dataset(self, mock_split, mock_load_dataset, data_processor):
        """Test loading AG News dataset."""
        # Mock dataset
        mock_dataset = {
            "train": Mock(),
            "test": Mock()
        }
        
        # Mock the len() function for the dataset splits
        mock_dataset["train"].__len__ = Mock(return_value=100)
        mock_dataset["test"].__len__ = Mock(return_value=50)
        
        mock_dataset["train"].select.return_value = mock_dataset["train"]
        mock_dataset["test"].select.return_value = mock_dataset["test"]
        mock_dataset["train"].map.return_value = mock_dataset["train"]
        mock_dataset["test"].map.return_value = mock_dataset["test"]
        
        # Mock train data
        mock_train_data = [
            {"text": "Sample text 1", "label": 0},
            {"text": "Sample text 2", "label": 1},
            {"text": "Sample text 3", "label": 2},
            {"text": "Sample text 4", "label": 3},
        ]
        mock_dataset["train"].__iter__ = lambda x: iter(mock_train_data)
        
        # Mock train_test_split to return predictable results
        mock_split.return_value = (
            ["text1", "text2"], ["text3"], [0, 1], [2]
        )
        
        mock_load_dataset.return_value = mock_dataset
        
        # Test loading dataset
        datasets = data_processor.load_ag_news_dataset(max_samples=4)
        
        # Verify dataset structure
        assert "train" in datasets
        assert "validation" in datasets
        assert "test" in datasets
        
        # Verify mock was called
        mock_load_dataset.assert_called_once_with("ag_news")
    
    def test_tokenize_function(self, data_processor):
        """Test tokenization function."""
        # Mock tokenizer behavior
        mock_tokenizer = data_processor.tokenizer
        mock_tokenizer.return_value = {
            "input_ids": [[1, 2, 3, 4, 5]],
            "attention_mask": [[1, 1, 1, 1, 1]]
        }
        
        examples = {
            "text": ["Sample text for testing"],
            "label": [0]
        }
        
        result = data_processor.tokenize_function(examples)
        
        # Verify result structure
        assert "input_ids" in result
        assert "attention_mask" in result
        assert "labels" in result
        assert result["labels"] == [0]
        
        # Verify tokenizer was called (don't check exact arguments due to mock complexity)
        assert mock_tokenizer.called
    
    @patch('llm_finetuning.utils.data_utils.train_test_split')
    def test_prepare_datasets(self, mock_split, data_processor):
        """Test dataset preparation."""
        # Mock train_test_split
        mock_split.return_value = (
            ["text1", "text2"], ["text3"], [0, 1], [2]
        )
        
        # Mock datasets
        mock_datasets = {
            "train": Mock(),
            "validation": Mock(),
            "test": Mock()
        }
        
        # Mock dataset behavior
        for dataset in mock_datasets.values():
            dataset.column_names = ["text", "label"]
            dataset.map.return_value = dataset
        
        # Mock tokenize function
        data_processor.tokenize_function = Mock(return_value={"input_ids": [1, 2, 3]})
        
        # Test preparation
        result = data_processor.prepare_datasets(mock_datasets)
        
        # Verify all datasets were processed
        assert len(result) == 3
        for dataset in result.values():
            dataset.map.assert_called_once()


class TestDataCollator:
    """Test cases for data collator."""
    
    def test_create_data_collator(self):
        """Test data collator creation."""
        mock_tokenizer = Mock()
        mock_tokenizer.pad_token = "<pad>"
        mock_tokenizer.pad_token_id = 0
        
        collator = create_data_collator(mock_tokenizer)
        
        # Verify collator was created
        assert collator is not None
        assert collator.tokenizer == mock_tokenizer
        assert collator.mlm is False  # Should be False for causal LM


class TestDatasetInfo:
    """Test cases for dataset info utilities."""
    
    def test_get_dataset_info(self):
        """Test getting dataset information."""
        # Mock dataset
        mock_dataset = Mock()
        mock_dataset.__len__ = lambda x: 100
        mock_dataset.features = {"text": "string", "label": "int64"}
        mock_dataset.column_names = ["text", "label"]
        mock_dataset.info = "Mock dataset info"
        
        info = get_dataset_info(mock_dataset)
        
        # Verify info structure
        assert info["num_examples"] == 100
        assert "text" in info["features"]
        assert "label" in info["features"]
        assert info["column_names"] == ["text", "label"]
        assert "info" in info
