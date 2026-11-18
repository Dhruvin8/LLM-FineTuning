#!/usr/bin/env python3
"""
Training script for LLM fine-tuning demo.

This script implements fine-tuning of a small language model using LoRA
for parameter-efficient training on text classification tasks.
"""

import argparse
import logging
import os
import sys
import yaml
from pathlib import Path
from typing import Dict, Any

import torch
import mlflow
from transformers import set_seed as transformers_set_seed

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from llm_finetuning.utils import (
    set_seed,
    get_device_info,
    setup_mlflow,
    log_config_parameters,
    log_system_info,
    log_training_metrics,
    log_model_artifacts,
    DataProcessor,
    create_data_collator,
    setup_model_and_tokenizer,
    setup_lora_model,
    setup_training_args,
    setup_trainer,
    compute_metrics
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train LLM with LoRA fine-tuning"
    )
    
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to configuration file"
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for checkpoints (overrides config)"
    )
    
    parser.add_argument(
        "--model_name",
        type=str,
        default=None,
        help="Model name (overrides config)"
    )
    
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum number of samples to use (overrides config)"
    )
    
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=None,
        help="Learning rate (overrides config)"
    )
    
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Batch size (overrides config)"
    )
    
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Number of epochs (overrides config)"
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed (overrides config)"
    )
    
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint to resume from"
    )
    
    return parser.parse_args()


def override_config(config: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    """
    Override configuration with command line arguments.
    
    Args:
        config: Original configuration
        args: Command line arguments
        
    Returns:
        Updated configuration
    """
    if args.output_dir:
        config["training"]["output_dir"] = args.output_dir
    
    if args.model_name:
        config["model"]["name"] = args.model_name
    
    if args.max_samples:
        config["dataset"]["max_samples"] = args.max_samples
    
    if args.learning_rate:
        config["training"]["learning_rate"] = args.learning_rate
    
    if args.batch_size:
        config["training"]["per_device_train_batch_size"] = args.batch_size
        config["training"]["per_device_eval_batch_size"] = args.batch_size
    
    if args.epochs:
        config["training"]["num_train_epochs"] = args.epochs
    
    if args.seed:
        config["seed"] = args.seed
    
    return config


def main():
    """Main training function."""
    # Parse arguments
    args = parse_arguments()
    
    # Load configuration
    config = load_config(args.config)
    config = override_config(config, args)
    
    # Set random seed for reproducibility
    seed = config.get("seed", 42)
    set_seed(seed)
    transformers_set_seed(seed)
    
    logger.info(f"Starting training with seed: {seed}")
    logger.info(f"Configuration: {config}")
    
    # Setup MLflow
    setup_mlflow(config)
    
    # Start MLflow run
    with mlflow.start_run():
        # Log system information
        log_system_info()
        
        # Log configuration parameters
        log_config_parameters(config)
        
        # Log device information
        device_info = get_device_info()
        logger.info(f"Device info: {device_info}")
        for key, value in device_info.items():
            mlflow.log_param(f"device_{key}", value)
        
        # Setup model and tokenizer
        logger.info("Setting up model and tokenizer...")
        model, tokenizer = setup_model_and_tokenizer(
            model_name=config["model"]["name"],
            max_length=config["model"]["max_length"],
            use_cache=config["model"]["use_cache"]
        )
        
        # Setup LoRA
        logger.info("Setting up LoRA...")
        model = setup_lora_model(model, config["lora"])
        
        # Setup data processor
        logger.info("Setting up data processor...")
        data_processor = DataProcessor(
            tokenizer=tokenizer,
            max_length=config["model"]["max_length"]
        )
        
        # Load and prepare datasets
        logger.info("Loading datasets...")
        datasets = data_processor.load_ag_news_dataset(
            max_samples=config["dataset"]["max_samples"]
        )
        
        # Tokenize datasets
        logger.info("Tokenizing datasets...")
        tokenized_datasets = data_processor.prepare_datasets(datasets)
        
        # Create data collator
        data_collator = create_data_collator(tokenizer)
        
        # Setup training arguments
        logger.info("Setting up training arguments...")
        training_args = setup_training_args(config, config["training"]["output_dir"])
        
        # Setup trainer
        logger.info("Setting up trainer...")
        trainer = setup_trainer(
            model=model,
            training_args=training_args,
            train_dataset=tokenized_datasets["train"],
            eval_dataset=tokenized_datasets["validation"],
            data_collator=data_collator,
            tokenizer=tokenizer,
            config=config
        )
        
        # Add compute metrics to trainer
        trainer.compute_metrics = compute_metrics
        
        # Log dataset information
        for split, dataset in tokenized_datasets.items():
            mlflow.log_param(f"dataset_{split}_size", len(dataset))
        
        # Start training
        logger.info("Starting training...")
        
        # Resume from checkpoint if specified
        resume_from_checkpoint = args.resume_from_checkpoint
        
        # Train the model
        train_result = trainer.train(resume_from_checkpoint=resume_from_checkpoint)
        
        # Log training metrics
        log_training_metrics(train_result.metrics)
        
        # Evaluate the model
        logger.info("Evaluating model...")
        eval_result = trainer.evaluate()
        
        # Log evaluation metrics
        log_training_metrics(eval_result)
        
        # Save the final model
        logger.info("Saving final model...")
        trainer.save_model()
        tokenizer.save_pretrained(training_args.output_dir)
        
        # Log model artifacts to MLflow
        log_model_artifacts(model, tokenizer, config, training_args.output_dir)
        
        # Log final metrics
        mlflow.log_metric("final_train_loss", train_result.training_loss)
        mlflow.log_metric("final_eval_loss", eval_result["eval_loss"])
        
        logger.info("Training completed successfully!")
        logger.info(f"Final training loss: {train_result.training_loss:.4f}")
        logger.info(f"Final evaluation loss: {eval_result['eval_loss']:.4f}")
        
        if "eval_accuracy" in eval_result:
            logger.info(f"Final evaluation accuracy: {eval_result['eval_accuracy']:.4f}")


if __name__ == "__main__":
    main()
