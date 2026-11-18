"""
Data processing utilities for LLM fine-tuning.
"""

import torch
from typing import Dict, List, Any, Optional
from datasets import Dataset, load_dataset
from transformers import AutoTokenizer, DataCollatorForLanguageModeling
from sklearn.model_selection import train_test_split
import pandas as pd


class DataProcessor:
    """Handle dataset loading and preprocessing for text classification."""
    
    def __init__(self, tokenizer, max_length: int = 512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def load_ag_news_dataset(self, max_samples: Optional[int] = None) -> Dict[str, Dataset]:
        """
        Load and preprocess AG News dataset.
        
        Args:
            max_samples: Maximum number of samples to use (for demo purposes)
            
        Returns:
            Dictionary containing train, validation, and test datasets
        """
        # Load dataset
        dataset = load_dataset("ag_news")
        
        # Limit samples if specified
        if max_samples:
            for split in dataset:
                dataset[split] = dataset[split].select(range(min(max_samples, len(dataset[split]))))
        
        # Convert to text classification format
        def format_text(example):
            # Create a simple prompt for text classification
            text = f"Classify the following news article:\n\n{example['text']}\n\nCategory:"
            return {
                "text": text,
                "label": example["label"],
                "label_text": ["World", "Sports", "Business", "Science/Tech"][example["label"]]
            }
        
        # Apply formatting
        for split in dataset:
            dataset[split] = dataset[split].map(format_text)
        
        # Split train into train/validation
        train_data = dataset["train"]
        train_texts = [item["text"] for item in train_data]
        train_labels = [item["label"] for item in train_data]
        
        train_texts, val_texts, train_labels, val_labels = train_test_split(
            train_texts, train_labels, test_size=0.1, random_state=42, stratify=train_labels
        )
        
        # Create validation dataset
        val_data = [{"text": text, "label": label} for text, label in zip(val_texts, val_labels)]
        val_dataset = Dataset.from_list(val_data)
        
        # Update train dataset
        train_data = [{"text": text, "label": label} for text, label in zip(train_texts, train_labels)]
        train_dataset = Dataset.from_list(train_data)
        
        return {
            "train": train_dataset,
            "validation": val_dataset,
            "test": dataset["test"]
        }
    
    def tokenize_function(self, examples: Dict[str, List]) -> Dict[str, List]:
        """
        Tokenize text examples.
        
        Args:
            examples: Dictionary containing text and labels
            
        Returns:
            Tokenized examples
        """
        # Tokenize texts
        tokenized = self.tokenizer(
            examples["text"],
            truncation=True,
            padding=False,
            max_length=self.max_length,
            return_tensors=None
        )
        
        # Add labels
        tokenized["labels"] = examples["label"]
        
        return tokenized
    
    def prepare_datasets(self, datasets: Dict[str, Dataset]) -> Dict[str, Dataset]:
        """
        Prepare datasets by tokenizing them.
        
        Args:
            datasets: Dictionary containing train, validation, and test datasets
            
        Returns:
            Tokenized datasets
        """
        tokenized_datasets = {}
        
        for split, dataset in datasets.items():
            tokenized_datasets[split] = dataset.map(
                self.tokenize_function,
                batched=True,
                remove_columns=dataset.column_names
            )
        
        return tokenized_datasets


def create_data_collator(tokenizer) -> DataCollatorForLanguageModeling:
    """
    Create data collator for language modeling.
    
    Args:
        tokenizer: Hugging Face tokenizer
        
    Returns:
        Data collator instance
    """
    return DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # We're doing causal LM, not masked LM
        pad_to_multiple_of=8
    )


def get_dataset_info(dataset: Dataset) -> Dict[str, Any]:
    """
    Get information about a dataset.
    
    Args:
        dataset: Hugging Face dataset
        
    Returns:
        Dictionary containing dataset information
    """
    return {
        "num_examples": len(dataset),
        "features": list(dataset.features.keys()),
        "column_names": dataset.column_names,
        "info": str(dataset.info) if hasattr(dataset, 'info') else "No info available"
    }
