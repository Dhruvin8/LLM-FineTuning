"""
Training utilities for LLM fine-tuning.
"""

import os
import logging
from typing import Dict, Any, Optional
from transformers import (
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
    get_cosine_schedule_with_warmup
)
import torch
from torch.optim import AdamW

logger = logging.getLogger(__name__)


def setup_training_args(
    config: Dict[str, Any],
    output_dir: str
) -> TrainingArguments:
    """
    Setup training arguments from configuration.
    
    Args:
        config: Training configuration dictionary
        output_dir: Output directory for checkpoints
        
    Returns:
        TrainingArguments instance
    """
    training_config = config["training"]
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=training_config["num_train_epochs"],
        per_device_train_batch_size=training_config["per_device_train_batch_size"],
        per_device_eval_batch_size=training_config["per_device_eval_batch_size"],
        gradient_accumulation_steps=training_config["gradient_accumulation_steps"],
        learning_rate=training_config["learning_rate"],
        weight_decay=training_config["weight_decay"],
        warmup_ratio=training_config["warmup_ratio"],
        logging_steps=training_config["logging_steps"],
        eval_steps=training_config["eval_steps"],
        save_steps=training_config["save_steps"],
        save_total_limit=training_config["save_total_limit"],
        evaluation_strategy=training_config["evaluation_strategy"],
        save_strategy=training_config["save_strategy"],
        load_best_model_at_end=training_config["load_best_model_at_end"],
        metric_for_best_model=training_config["metric_for_best_model"],
        greater_is_better=training_config["greater_is_better"],
        fp16=config.get("fp16", True),
        dataloader_num_workers=config.get("dataloader_num_workers", 4),
        remove_unused_columns=config.get("remove_unused_columns", False),
        report_to=None,  # We'll use MLflow instead
        logging_dir=os.path.join(output_dir, "logs"),
        run_name=f"llm-finetuning-{config.get('seed', 42)}",
    )
    
    return training_args


def setup_trainer(
    model,
    training_args: TrainingArguments,
    train_dataset,
    eval_dataset,
    data_collator,
    tokenizer,
    config: Dict[str, Any]
) -> Trainer:
    """
    Setup trainer with callbacks and custom metrics.
    
    Args:
        model: Model to train
        training_args: Training arguments
        train_dataset: Training dataset
        eval_dataset: Evaluation dataset
        data_collator: Data collator
        tokenizer: Tokenizer
        config: Configuration dictionary
        
    Returns:
        Trainer instance
    """
    # Setup callbacks
    callbacks = []
    
    # Add early stopping if configured
    if config["training"].get("early_stopping_patience"):
        early_stopping = EarlyStoppingCallback(
            early_stopping_patience=config["training"]["early_stopping_patience"],
            early_stopping_threshold=config["training"].get("early_stopping_threshold", 0.001)
        )
        callbacks.append(early_stopping)
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        callbacks=callbacks,
    )
    
    return trainer


def compute_metrics(eval_preds):
    """
    Compute custom metrics for evaluation.
    
    Args:
        eval_preds: Evaluation predictions
        
    Returns:
        Dictionary of metrics
    """
    import numpy as np
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support
    
    predictions, labels = eval_preds
    
    # For causal LM, we need to extract the last token predictions
    # This is a simplified version - in practice, you'd need more sophisticated logic
    if len(predictions.shape) == 3:  # (batch_size, seq_len, vocab_size)
        predictions = np.argmax(predictions, axis=-1)
    
    # Flatten predictions and labels
    predictions = predictions.flatten()
    labels = labels.flatten()
    
    # Remove padding tokens (-100)
    mask = labels != -100
    predictions = predictions[mask]
    labels = labels[mask]
    
    # Compute metrics
    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average='weighted', zero_division=0
    )
    
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }


def get_learning_rate_scheduler(
    optimizer,
    num_training_steps: int,
    num_warmup_steps: int,
    scheduler_type: str = "cosine"
):
    """
    Get learning rate scheduler.
    
    Args:
        optimizer: Optimizer
        num_training_steps: Total number of training steps
        num_warmup_steps: Number of warmup steps
        scheduler_type: Type of scheduler
        
    Returns:
        Learning rate scheduler
    """
    if scheduler_type == "cosine":
        return get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps
        )
    else:
        raise ValueError(f"Unsupported scheduler type: {scheduler_type}")


def get_optimizer(model, learning_rate: float, weight_decay: float):
    """
    Get optimizer for training.
    
    Args:
        model: Model to optimize
        learning_rate: Learning rate
        weight_decay: Weight decay
        
    Returns:
        Optimizer instance
    """
    # Get trainable parameters
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    
    # Create optimizer
    optimizer = AdamW(
        trainable_params,
        lr=learning_rate,
        weight_decay=weight_decay,
        betas=(0.9, 0.999),
        eps=1e-8
    )
    
    return optimizer
