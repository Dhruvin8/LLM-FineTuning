"""
MLflow utilities for experiment tracking.
"""

import os
import mlflow
import mlflow.pytorch
import logging
from typing import Dict, Any, Optional
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np

logger = logging.getLogger(__name__)


def setup_mlflow(config: Dict[str, Any]) -> None:
    """
    Setup MLflow tracking.
    
    Args:
        config: Configuration dictionary containing MLflow settings
    """
    mlflow_config = config.get("mlflow", {})
    
    # Set tracking URI
    tracking_uri = mlflow_config.get("tracking_uri", "http://localhost:5000")
    mlflow.set_tracking_uri(tracking_uri)
    
    # Set experiment
    experiment_name = mlflow_config.get("experiment_name", "llm-finetuning-demo")
    mlflow.set_experiment(experiment_name)
    
    logger.info(f"MLflow tracking URI: {tracking_uri}")
    logger.info(f"MLflow experiment: {experiment_name}")


def log_training_metrics(
    metrics: Dict[str, float],
    step: Optional[int] = None
) -> None:
    """
    Log training metrics to MLflow.
    
    Args:
        metrics: Dictionary of metrics to log
        step: Step number (optional)
    """
    for metric_name, metric_value in metrics.items():
        if step is not None:
            mlflow.log_metric(metric_name, metric_value, step=step)
        else:
            mlflow.log_metric(metric_name, metric_value)


def log_model_artifacts(
    model,
    tokenizer,
    config: Dict[str, Any],
    output_dir: str
) -> None:
    """
    Log model artifacts to MLflow.
    
    Args:
        model: Trained model
        tokenizer: Tokenizer
        config: Configuration dictionary
        output_dir: Output directory containing model files
    """
    # Log model
    if config.get("mlflow", {}).get("log_model", False):
        mlflow.pytorch.log_model(
            pytorch_model=model,
            artifact_path="model",
            registered_model_name="llm-finetuning-model"
        )
    
    # Log artifacts
    if config.get("mlflow", {}).get("log_artifacts", False):
        mlflow.log_artifacts(output_dir, artifact_path="checkpoints")


def log_config_parameters(config: Dict[str, Any]) -> None:
    """
    Log configuration parameters to MLflow.
    
    Args:
        config: Configuration dictionary
    """
    # Log model parameters
    if "model" in config:
        for key, value in config["model"].items():
            mlflow.log_param(f"model_{key}", value)
    
    # Log training parameters
    if "training" in config:
        for key, value in config["training"].items():
            mlflow.log_param(f"training_{key}", value)
    
    # Log LoRA parameters
    if "lora" in config:
        for key, value in config["lora"].items():
            mlflow.log_param(f"lora_{key}", value)
    
    # Log other parameters
    for key, value in config.items():
        if key not in ["model", "training", "lora", "mlflow"]:
            mlflow.log_param(key, value)


def create_confusion_matrix_plot(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Optional[list] = None
) -> str:
    """
    Create confusion matrix plot and save it.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names
        
    Returns:
        Path to saved plot
    """
    plt.figure(figsize=(10, 8))
    
    cm = confusion_matrix(y_true, y_pred)
    
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=class_names or [f"Class {i}" for i in range(len(cm))],
        yticklabels=class_names or [f"Class {i}" for i in range(len(cm))]
    )
    
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    
    # Save plot
    plot_path = "confusion_matrix.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return plot_path


def create_training_plots(
    train_losses: list,
    eval_losses: list,
    eval_metrics: Dict[str, list]
) -> Dict[str, str]:
    """
    Create training plots and save them.
    
    Args:
        train_losses: List of training losses
        eval_losses: List of evaluation losses
        eval_metrics: Dictionary of evaluation metrics
        
    Returns:
        Dictionary mapping plot names to file paths
    """
    plot_paths = {}
    
    # Loss plot
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.plot(eval_losses, label='Validation Loss')
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    
    # Metrics plot
    plt.subplot(1, 2, 2)
    for metric_name, metric_values in eval_metrics.items():
        plt.plot(metric_values, label=metric_name)
    plt.xlabel('Steps')
    plt.ylabel('Metric Value')
    plt.title('Evaluation Metrics')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    
    # Save loss plot
    loss_plot_path = "training_plots.png"
    plt.savefig(loss_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    plot_paths["training_plots"] = loss_plot_path
    
    return plot_paths


def log_evaluation_results(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Optional[list] = None,
    additional_metrics: Optional[Dict[str, float]] = None
) -> None:
    """
    Log evaluation results to MLflow.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names
        additional_metrics: Additional metrics to log
    """
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support
    
    # Compute metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='weighted', zero_division=0
    )
    
    # Log metrics
    mlflow.log_metric("eval_accuracy", accuracy)
    mlflow.log_metric("eval_precision", precision)
    mlflow.log_metric("eval_recall", recall)
    mlflow.log_metric("eval_f1", f1)
    
    # Log additional metrics if provided
    if additional_metrics:
        for metric_name, metric_value in additional_metrics.items():
            mlflow.log_metric(metric_name, metric_value)
    
    # Create and log confusion matrix
    cm_plot_path = create_confusion_matrix_plot(y_true, y_pred, class_names)
    mlflow.log_artifact(cm_plot_path, "evaluation")
    
    # Create classification report
    report = classification_report(
        y_true, y_pred, 
        target_names=class_names,
        output_dict=True
    )
    
    # Log classification report as artifact
    import json
    report_path = "classification_report.json"
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    mlflow.log_artifact(report_path, "evaluation")


def log_system_info() -> None:
    """Log system information to MLflow."""
    import platform
    import psutil
    
    # System info
    mlflow.log_param("python_version", platform.python_version())
    mlflow.log_param("platform", platform.platform())
    mlflow.log_param("cpu_count", psutil.cpu_count())
    mlflow.log_param("memory_gb", round(psutil.virtual_memory().total / (1024**3), 2))
    
    # PyTorch info
    mlflow.log_param("pytorch_version", torch.__version__)
    mlflow.log_param("cuda_available", torch.cuda.is_available())
    
    if torch.cuda.is_available():
        mlflow.log_param("cuda_version", torch.version.cuda)
        mlflow.log_param("gpu_count", torch.cuda.device_count())
        mlflow.log_param("gpu_name", torch.cuda.get_device_name(0))
