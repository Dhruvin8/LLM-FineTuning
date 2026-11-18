#!/bin/bash

# LLM Fine-Tuning Demo Quick Start Script
# This script provides quick commands for common tasks

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check if MLflow server is running
check_mlflow() {
    if curl -s http://localhost:5000 > /dev/null 2>&1; then
        return 0
    else
        return 1
    fi
}

# Function to start MLflow server
start_mlflow() {
    print_status "Starting MLflow server..."
    if check_mlflow; then
        print_warning "MLflow server is already running at http://localhost:5000"
    else
        mlflow server --backend-store-uri sqlite:///mlruns/mlflow.db --default-artifact-root ./mlruns --host 0.0.0.0 --port 5000 &
        sleep 3
        if check_mlflow; then
            print_success "MLflow server started successfully at http://localhost:5000"
        else
            print_error "Failed to start MLflow server"
            exit 1
        fi
    fi
}

# Function to train model
train_model() {
    print_status "Starting model training..."
    python llm_finetuning/train.py --config llm_finetuning/configs/train.yaml
    print_success "Training completed!"
}

# Function to evaluate model
evaluate_model() {
    if [ ! -d "checkpoints" ] || [ -z "$(ls -A checkpoints)" ]; then
        print_error "No model checkpoints found. Please train a model first."
        exit 1
    fi
    
    # Find the latest checkpoint
    LATEST_CHECKPOINT=$(find checkpoints -name "checkpoint-*" -type d | sort -V | tail -1)
    
    if [ -z "$LATEST_CHECKPOINT" ]; then
        print_error "No checkpoint directories found in checkpoints/"
        exit 1
    fi
    
    print_status "Evaluating model from: $LATEST_CHECKPOINT"
    python llm_finetuning/evaluate.py --model_path "$LATEST_CHECKPOINT"
    print_success "Evaluation completed!"
}

# Function to show help
show_help() {
    echo "LLM Fine-Tuning Demo Quick Start"
    echo ""
    echo "Usage: $0 [COMMAND]"
    echo ""
    echo "Commands:"
    echo "  mlflow     Start MLflow tracking server"
    echo "  train      Train a model"
    echo "  evaluate   Evaluate the trained model"
    echo "  full       Run complete pipeline (mlflow + train + evaluate)"
    echo "  status     Check system status"
    echo "  help       Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 mlflow"
    echo "  $0 train"
    echo "  $0 full"
}

# Function to check system status
check_status() {
    print_status "Checking system status..."
    
    # Check Python
    if command -v python3 &> /dev/null; then
        PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
        print_success "Python: $PYTHON_VERSION"
    else
        print_error "Python 3 not found"
    fi
    
    # Check CUDA
    if python3 -c "import torch; print(torch.cuda.is_available())" 2>/dev/null | grep -q "True"; then
        print_success "CUDA: Available"
    else
        print_warning "CUDA: Not available (CPU only)"
    fi
    
    # Check MLflow
    if check_mlflow; then
        print_success "MLflow: Running at http://localhost:5000"
    else
        print_warning "MLflow: Not running"
    fi
    
    # Check checkpoints
    if [ -d "checkpoints" ] && [ "$(ls -A checkpoints)" ]; then
        CHECKPOINT_COUNT=$(find checkpoints -name "checkpoint-*" -type d | wc -l)
        print_success "Checkpoints: $CHECKPOINT_COUNT found"
    else
        print_warning "Checkpoints: None found"
    fi
}

# Function to run full pipeline
run_full_pipeline() {
    print_status "Running full pipeline..."
    
    # Start MLflow
    start_mlflow
    
    # Train model
    train_model
    
    # Evaluate model
    evaluate_model
    
    print_success "Full pipeline completed!"
    print_status "View results at: http://localhost:5000"
}

# Main script logic
case "${1:-help}" in
    "mlflow")
        start_mlflow
        ;;
    "train")
        train_model
        ;;
    "evaluate")
        evaluate_model
        ;;
    "full")
        run_full_pipeline
        ;;
    "status")
        check_status
        ;;
    "help"|*)
        show_help
        ;;
esac
