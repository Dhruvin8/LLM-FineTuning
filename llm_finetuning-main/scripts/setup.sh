#!/bin/bash

# LLM Fine-Tuning Demo Setup Script
# This script sets up the development environment

set -e

echo "🚀 Setting up LLM Fine-Tuning Demo..."

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.8+ first."
    exit 1
fi

# Check Python version
PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
REQUIRED_VERSION="3.8"

if [ "$(printf '%s\n' "$REQUIRED_VERSION" "$PYTHON_VERSION" | sort -V | head -n1)" != "$REQUIRED_VERSION" ]; then
    echo "❌ Python $REQUIRED_VERSION or higher is required. Found: $PYTHON_VERSION"
    exit 1
fi

echo "✅ Python version: $PYTHON_VERSION"

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "⬆️ Upgrading pip..."
pip install --upgrade pip

# Install the package in development mode
echo "📥 Installing package in development mode..."
pip install -e .

# Install development dependencies
echo "🛠️ Installing development dependencies..."
pip install -e ".[dev]"

# Create necessary directories
echo "📁 Creating necessary directories..."
mkdir -p checkpoints logs evaluation_results mlruns

# Check if CUDA is available
echo "🔍 Checking CUDA availability..."
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# Run tests to verify installation
echo "🧪 Running tests to verify installation..."
python3 -m pytest tests/ -v

echo "✅ Setup completed successfully!"
echo ""
echo "🎯 Next steps:"
echo "1. Start MLflow server: mlflow server --backend-store-uri sqlite:///mlruns/mlflow.db --default-artifact-root ./mlruns --host 0.0.0.0 --port 5000"
echo "2. Train a model: python llm_finetuning/train.py --config llm_finetuning/configs/train.yaml"
echo "3. Evaluate the model: python llm_finetuning/evaluate.py --model_path checkpoints/best_model"
echo "4. View results: http://localhost:5000"
echo ""
echo "📚 For more information, see README.md"
