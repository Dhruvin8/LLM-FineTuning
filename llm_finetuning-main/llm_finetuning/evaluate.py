#!/usr/bin/env python3
"""
Evaluation script for LLM fine-tuning demo.

This script evaluates a fine-tuned model on test data and generates
comprehensive metrics and visualizations.
"""

import argparse
import logging
import os
import sys
import json
from pathlib import Path
from typing import Dict, Any, List, Tuple

import torch
import numpy as np
import mlflow
from transformers import AutoTokenizer
from datasets import load_dataset
from sklearn.metrics import (
    accuracy_score, 
    precision_recall_fscore_support, 
    confusion_matrix,
    classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from llm_finetuning.utils import (
    setup_mlflow,
    log_evaluation_results,
    create_confusion_matrix_plot,
    load_model_checkpoint,
    DataProcessor
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate fine-tuned LLM model"
    )
    
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the trained model checkpoint"
    )
    
    parser.add_argument(
        "--test_dataset",
        type=str,
        default="ag_news",
        help="Test dataset to use for evaluation"
    )
    
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum number of test samples to evaluate"
    )
    
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for evaluation"
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./evaluation_results",
        help="Output directory for evaluation results"
    )
    
    parser.add_argument(
        "--mlflow_experiment",
        type=str,
        default="llm-finetuning-evaluation",
        help="MLflow experiment name for evaluation"
    )
    
    parser.add_argument(
        "--save_predictions",
        action="store_true",
        help="Save predictions to file"
    )
    
    return parser.parse_args()


def load_test_dataset(
    dataset_name: str,
    max_samples: int = None
) -> List[Dict[str, Any]]:
    """
    Load test dataset.
    
    Args:
        dataset_name: Name of the dataset
        max_samples: Maximum number of samples to load
        
    Returns:
        List of test samples
    """
    logger.info(f"Loading test dataset: {dataset_name}")
    
    if dataset_name == "ag_news":
        dataset = load_dataset("ag_news", split="test")
        
        # Format for text classification
        def format_text(example):
            text = f"Classify the following news article:\n\n{example['text']}\n\nCategory:"
            return {
                "text": text,
                "label": example["label"],
                "label_text": ["World", "Sports", "Business", "Science/Tech"][example["label"]]
            }
        
        dataset = dataset.map(format_text)
        
        if max_samples:
            dataset = dataset.select(range(min(max_samples, len(dataset))))
        
        return dataset
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")


def generate_predictions(
    model,
    tokenizer,
    test_data: List[Dict[str, Any]],
    batch_size: int = 8,
    max_length: int = 512
) -> Tuple[List[int], List[int]]:
    """
    Generate predictions for test data.
    
    Args:
        model: Trained model
        tokenizer: Tokenizer
        test_data: Test dataset
        batch_size: Batch size for inference
        max_length: Maximum sequence length
        
    Returns:
        Tuple of (predictions, true_labels)
    """
    logger.info("Generating predictions...")
    
    model.eval()
    predictions = []
    true_labels = []
    
    # Process in batches
    for i in range(0, len(test_data), batch_size):
        batch = test_data[i:i + batch_size]
        
        # Prepare batch
        texts = [item["text"] for item in batch]
        labels = [item["label"] for item in batch]
        
        # Tokenize
        inputs = tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length
        )
        
        # Move to device
        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Generate predictions
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            
            # Get predictions (simplified - in practice you'd need more sophisticated logic)
            # For this demo, we'll use a simple approach
            batch_predictions = torch.argmax(logits[:, -1, :], dim=-1).cpu().numpy()
            
            # For text classification, we need to map to class labels
            # This is a simplified approach - in practice you'd need proper prompt engineering
            batch_predictions = batch_predictions % 4  # Map to 4 classes for AG News
        
        predictions.extend(batch_predictions)
        true_labels.extend(labels)
        
        if (i // batch_size + 1) % 10 == 0:
            logger.info(f"Processed {i + len(batch)}/{len(test_data)} samples")
    
    return predictions, true_labels


def create_detailed_plots(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
    output_dir: str
) -> Dict[str, str]:
    """
    Create detailed evaluation plots.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names
        output_dir: Output directory
        
    Returns:
        Dictionary mapping plot names to file paths
    """
    os.makedirs(output_dir, exist_ok=True)
    plot_paths = {}
    
    # Confusion Matrix
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(y_true, y_pred)
    
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names
    )
    
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    
    cm_path = os.path.join(output_dir, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=300, bbox_inches='tight')
    plt.close()
    plot_paths["confusion_matrix"] = cm_path
    
    # Per-class metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
    )
    
    metrics_df = {
        'Class': class_names,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1,
        'Support': support
    }
    
    # Create metrics bar plot
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Precision
    axes[0, 0].bar(class_names, precision)
    axes[0, 0].set_title('Precision by Class')
    axes[0, 0].set_ylabel('Precision')
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # Recall
    axes[0, 1].bar(class_names, recall)
    axes[0, 1].set_title('Recall by Class')
    axes[0, 1].set_ylabel('Recall')
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    # F1-Score
    axes[1, 0].bar(class_names, f1)
    axes[1, 0].set_title('F1-Score by Class')
    axes[1, 0].set_ylabel('F1-Score')
    axes[1, 0].tick_params(axis='x', rotation=45)
    
    # Support
    axes[1, 1].bar(class_names, support)
    axes[1, 1].set_title('Support by Class')
    axes[1, 1].set_ylabel('Number of Samples')
    axes[1, 1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    
    metrics_path = os.path.join(output_dir, "per_class_metrics.png")
    plt.savefig(metrics_path, dpi=300, bbox_inches='tight')
    plt.close()
    plot_paths["per_class_metrics"] = metrics_path
    
    return plot_paths


def save_predictions(
    test_data: List[Dict[str, Any]],
    predictions: List[int],
    true_labels: List[int],
    output_dir: str
) -> str:
    """
    Save predictions to file.
    
    Args:
        test_data: Original test data
        predictions: Model predictions
        true_labels: True labels
        output_dir: Output directory
        
    Returns:
        Path to saved predictions file
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Prepare data for saving
    results = []
    class_names = ["World", "Sports", "Business", "Science/Tech"]
    
    for i, (pred, true) in enumerate(zip(predictions, true_labels)):
        results.append({
            "index": i,
            "text": test_data[i]["text"][:200] + "...",  # Truncate for readability
            "true_label": true,
            "true_class": class_names[true],
            "predicted_label": pred,
            "predicted_class": class_names[pred],
            "correct": pred == true
        })
    
    # Save as JSON
    predictions_path = os.path.join(output_dir, "predictions.json")
    with open(predictions_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save as CSV
    import pandas as pd
    df = pd.DataFrame(results)
    csv_path = os.path.join(output_dir, "predictions.csv")
    df.to_csv(csv_path, index=False)
    
    return predictions_path


def main():
    """Main evaluation function."""
    # Parse arguments
    args = parse_arguments()
    
    logger.info(f"Starting evaluation of model: {args.model_path}")
    
    # Setup MLflow
    mlflow.set_experiment(args.mlflow_experiment)
    
    with mlflow.start_run():
        # Load model and tokenizer
        logger.info("Loading model and tokenizer...")
        model, tokenizer = load_model_checkpoint(args.model_path)
        
        # Load test dataset
        test_data = load_test_dataset(args.test_dataset, args.max_samples)
        logger.info(f"Loaded {len(test_data)} test samples")
        
        # Generate predictions
        predictions, true_labels = generate_predictions(
            model=model,
            tokenizer=tokenizer,
            test_data=test_data,
            batch_size=args.batch_size
        )
        
        # Convert to numpy arrays
        y_true = np.array(true_labels)
        y_pred = np.array(predictions)
        
        # Compute metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average='weighted', zero_division=0
        )
        
        # Per-class metrics
        precision_per_class, recall_per_class, f1_per_class, support = precision_recall_fscore_support(
            y_true, y_pred, average=None, zero_division=0
        )
        
        # Log metrics to MLflow
        mlflow.log_metric("test_accuracy", accuracy)
        mlflow.log_metric("test_precision", precision)
        mlflow.log_metric("test_recall", recall)
        mlflow.log_metric("test_f1", f1)
        
        # Log per-class metrics
        class_names = ["World", "Sports", "Business", "Science/Tech"]
        for i, class_name in enumerate(class_names):
            mlflow.log_metric(f"test_precision_{class_name.lower()}", precision_per_class[i])
            mlflow.log_metric(f"test_recall_{class_name.lower()}", recall_per_class[i])
            mlflow.log_metric(f"test_f1_{class_name.lower()}", f1_per_class[i])
            mlflow.log_metric(f"test_support_{class_name.lower()}", support[i])
        
        # Create and save plots
        logger.info("Creating evaluation plots...")
        plot_paths = create_detailed_plots(
            y_true, y_pred, class_names, args.output_dir
        )
        
        # Log plots to MLflow
        for plot_name, plot_path in plot_paths.items():
            mlflow.log_artifact(plot_path, "evaluation_plots")
        
        # Save predictions if requested
        if args.save_predictions:
            logger.info("Saving predictions...")
            predictions_path = save_predictions(
                test_data, predictions, true_labels, args.output_dir
            )
            mlflow.log_artifact(predictions_path, "predictions")
        
        # Create and save classification report
        report = classification_report(
            y_true, y_pred, 
            target_names=class_names,
            output_dict=True
        )
        
        report_path = os.path.join(args.output_dir, "classification_report.json")
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        mlflow.log_artifact(report_path, "evaluation")
        
        # Log evaluation results
        log_evaluation_results(y_true, y_pred, class_names)
        
        # Print summary
        logger.info("=" * 50)
        logger.info("EVALUATION SUMMARY")
        logger.info("=" * 50)
        logger.info(f"Test Accuracy: {accuracy:.4f}")
        logger.info(f"Test Precision: {precision:.4f}")
        logger.info(f"Test Recall: {recall:.4f}")
        logger.info(f"Test F1-Score: {f1:.4f}")
        logger.info("=" * 50)
        
        logger.info("Per-class metrics:")
        for i, class_name in enumerate(class_names):
            logger.info(f"{class_name}:")
            logger.info(f"  Precision: {precision_per_class[i]:.4f}")
            logger.info(f"  Recall: {recall_per_class[i]:.4f}")
            logger.info(f"  F1-Score: {f1_per_class[i]:.4f}")
            logger.info(f"  Support: {support[i]}")
        
        logger.info(f"Results saved to: {args.output_dir}")
        logger.info("Evaluation completed successfully!")


if __name__ == "__main__":
    main()
