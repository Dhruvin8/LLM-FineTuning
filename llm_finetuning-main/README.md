# LLM Fine-Tuning Demo

A professional demonstration of Large Language Model (LLM) fine-tuning using Hugging Face Transformers, PEFT/LoRA, and MLflow for experiment tracking. This project showcases modern MLOps practices.

## 🚀 Features

- **Parameter-Efficient Fine-Tuning**: Uses LoRA (Low-Rank Adaptation) for efficient training
- **Modern MLOps**: MLflow integration for experiment tracking and model versioning
- **Reproducible**: Docker support and comprehensive configuration management
- **Professional Structure**: Clean code organization with proper testing
- **Comprehensive Evaluation**: Detailed metrics, visualizations, and confusion matrices
- **CLI Interface**: Easy-to-use command-line tools for training and evaluation

## 📋 Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Configuration](#configuration)
- [Training](#training)
- [Evaluation](#evaluation)
- [MLflow Integration](#mlflow-integration)
- [Docker Support](#docker-support)
- [Testing](#testing)
- [API Reference](#api-reference)
- [Contributing](#contributing)
- [License](#license)

## 🛠 Installation

### Prerequisites

- Python 3.8+
- CUDA-compatible GPU (recommended)
- Docker (optional)

### Local Installation

1. **Clone the repository:**
   ```bash
   git clone <your-repo-url>
   cd llm_finetuning
   ```

2. **Install dependencies:**
   ```bash
   pip install -e .
   ```

3. **Install development dependencies (optional):**
   ```bash
   pip install -e ".[dev]"
   ```

### Docker Installation

1. **Build the Docker image:**
   ```bash
   docker build -t llm-finetuning-demo .
   ```

2. **Run with Docker Compose:**
   ```bash
   docker-compose up mlflow
   ```

## 🚀 Quick Start

### 1. Start MLflow Tracking Server

```bash
# Local
mlflow server --backend-store-uri sqlite:///mlruns/mlflow.db --default-artifact-root ./mlruns --host 0.0.0.0 --port 5000

# Or with Docker
docker-compose up mlflow
```

### 2. Train a Model

```bash
python llm_finetuning/train.py --config llm_finetuning/configs/train.yaml
```

### 3. Evaluate the Model

```bash
python llm_finetuning/evaluate.py --model_path checkpoints/best_model
```

### 4. View Results

Open your browser and navigate to `http://localhost:5000` to view the MLflow UI.

## 📁 Project Structure

```
llm_finetuning/
├── llm_finetuning/           # Main package
│   ├── __init__.py
│   ├── train.py              # Training script
│   ├── evaluate.py           # Evaluation script
│   ├── configs/              # Configuration files
│   │   └── train.yaml        # Training configuration
│   └── utils/                # Utility modules
│       ├── __init__.py
│       ├── data_utils.py     # Data processing utilities
│       ├── model_utils.py    # Model utilities
│       ├── training_utils.py # Training utilities
│       ├── mlflow_utils.py   # MLflow integration
│       └── reproducibility.py # Reproducibility utilities
├── tests/                    # Test suite
│   ├── __init__.py
│   ├── test_data_utils.py
│   ├── test_model_utils.py
│   └── test_reproducibility.py
├── checkpoints/              # Model checkpoints
├── logs/                     # Training logs
├── evaluation_results/       # Evaluation results
├── mlruns/                   # MLflow runs
├── pyproject.toml           # Project configuration
├── Dockerfile               # Docker configuration
├── docker-compose.yml       # Docker Compose configuration
├── pytest.ini              # Test configuration
└── README.md               # This file
```

## ⚙️ Configuration

The project uses YAML configuration files for easy parameter management. The main configuration file is `llm_finetuning/configs/train.yaml`.

### Key Configuration Sections

#### Model Configuration
```yaml
model:
  name: "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
  max_length: 512
  use_cache: false
```

#### LoRA Configuration
```yaml
lora:
  r: 16
  lora_alpha: 32
  lora_dropout: 0.1
  target_modules: ["q_proj", "v_proj", "k_proj", "o_proj"]
  bias: "none"
  task_type: "CAUSAL_LM"
```

#### Training Configuration
```yaml
training:
  num_train_epochs: 3
  per_device_train_batch_size: 4
  learning_rate: 2e-4
  weight_decay: 0.01
  warmup_ratio: 0.1
  early_stopping_patience: 3
```

#### MLflow Configuration
```yaml
mlflow:
  experiment_name: "llm-finetuning-demo"
  tracking_uri: "http://localhost:5000"
  log_model: true
  log_artifacts: true
```

## 🏋️ Training

### Basic Training

```bash
python llm_finetuning/train.py --config llm_finetuning/configs/train.yaml
```

### Training with Custom Parameters

```bash
python llm_finetuning/train.py \
    --config llm_finetuning/configs/train.yaml \
    --learning_rate 1e-4 \
    --batch_size 8 \
    --epochs 5 \
    --max_samples 5000
```

### Resume from Checkpoint

```bash
python llm_finetuning/train.py \
    --config llm_finetuning/configs/train.yaml \
    --resume_from_checkpoint checkpoints/checkpoint-1000
```

### Training Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--config` | Path to configuration file | Required |
| `--output_dir` | Output directory for checkpoints | From config |
| `--model_name` | Model name to use | From config |
| `--max_samples` | Maximum training samples | From config |
| `--learning_rate` | Learning rate | From config |
| `--batch_size` | Batch size | From config |
| `--epochs` | Number of epochs | From config |
| `--seed` | Random seed | From config |
| `--resume_from_checkpoint` | Resume from checkpoint | None |

## 📊 Evaluation

### Basic Evaluation

```bash
python llm_finetuning/evaluate.py --model_path checkpoints/best_model
```

### Evaluation with Custom Parameters

```bash
python llm_finetuning/evaluate.py \
    --model_path checkpoints/best_model \
    --test_dataset ag_news \
    --max_samples 1000 \
    --batch_size 16 \
    --save_predictions
```

### Evaluation Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--model_path` | Path to trained model | Required |
| `--test_dataset` | Test dataset name | `ag_news` |
| `--max_samples` | Maximum test samples | None (all) |
| `--batch_size` | Evaluation batch size | 8 |
| `--output_dir` | Output directory | `./evaluation_results` |
| `--mlflow_experiment` | MLflow experiment name | `llm-finetuning-evaluation` |
| `--save_predictions` | Save predictions to file | False |

### Evaluation Metrics

The evaluation script computes and logs the following metrics:

- **Accuracy**: Overall classification accuracy
- **Precision**: Weighted average precision
- **Recall**: Weighted average recall
- **F1-Score**: Weighted average F1-score
- **Per-class Metrics**: Precision, recall, F1, and support for each class
- **Confusion Matrix**: Visual representation of classification results

## 📈 MLflow Integration

### Starting MLflow Server

```bash
# Local
mlflow server --backend-store-uri sqlite:///mlruns/mlflow.db --default-artifact-root ./mlruns --host 0.0.0.0 --port 5000

# Docker
docker-compose up mlflow
```

### MLflow UI

Access the MLflow UI at `http://localhost:5000` to:

- View experiment runs and metrics
- Compare different model configurations
- Download model artifacts
- View training curves and visualizations
- Track hyperparameters and system information

### Logged Information

MLflow automatically logs:

- **Hyperparameters**: All configuration parameters
- **Metrics**: Training and validation losses, accuracy, F1-score
- **Artifacts**: Model checkpoints, evaluation plots, predictions
- **System Info**: Hardware specifications, software versions
- **Code Version**: Git commit hash (if available)

## 🐳 Docker Support

### Build and Run

```bash
# Build image
docker build -t llm-finetuning-demo .

# Run MLflow server
docker run -p 5000:5000 -v $(pwd)/mlruns:/app/mlruns llm-finetuning-demo

# Run training
docker run -v $(pwd)/mlruns:/app/mlruns -v $(pwd)/checkpoints:/app/checkpoints llm-finetuning-demo python llm_finetuning/train.py --config llm_finetuning/configs/train.yaml
```

### Docker Compose

```bash
# Start MLflow server
docker-compose up mlflow

# Run training (in another terminal)
docker-compose run --rm training python llm_finetuning/train.py --config llm_finetuning/configs/train.yaml

# Run evaluation (in another terminal)
docker-compose run --rm evaluation python llm_finetuning/evaluate.py --model_path checkpoints/best_model
```

## 🧪 Testing

### Run All Tests

```bash
pytest
```

### Run Tests with Coverage

```bash
pytest --cov=llm_finetuning --cov-report=html
```

### Run Specific Test Categories

```bash
# Unit tests only
pytest -m unit

# Integration tests only
pytest -m integration

# Skip slow tests
pytest -m "not slow"
```

### Test Structure

- **Unit Tests**: Test individual functions and classes
- **Integration Tests**: Test component interactions
- **Mocking**: Uses `unittest.mock` for external dependencies

## 📚 API Reference

### Core Classes

#### `DataProcessor`
Handles dataset loading and preprocessing for text classification.

```python
from llm_finetuning.utils import DataProcessor

processor = DataProcessor(tokenizer, max_length=512)
datasets = processor.load_ag_news_dataset(max_samples=1000)
tokenized_datasets = processor.prepare_datasets(datasets)
```

#### `setup_model_and_tokenizer`
Sets up model and tokenizer for fine-tuning.

```python
from llm_finetuning.utils import setup_model_and_tokenizer

model, tokenizer = setup_model_and_tokenizer(
    model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    max_length=512
)
```

#### `setup_lora_model`
Applies LoRA configuration to a model.

```python
from llm_finetuning.utils import setup_lora_model

lora_config = {
    "r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.1,
    "target_modules": ["q_proj", "v_proj"]
}
model = setup_lora_model(model, lora_config)
```

### Utility Functions

#### Reproducibility
```python
from llm_finetuning.utils import set_seed, get_device_info

set_seed(42)  # Set random seed
device_info = get_device_info()  # Get hardware information
```

#### MLflow Integration
```python
from llm_finetuning.utils import setup_mlflow, log_training_metrics

setup_mlflow(config)  # Setup MLflow tracking
log_training_metrics({"loss": 0.5, "accuracy": 0.8})  # Log metrics
```

## 🎯 Use Cases

This project is perfect for:

- **Portfolio Demonstration**: Showcase ML engineering skills
- **Learning**: Understand modern LLM fine-tuning practices
- **Research**: Experiment with different configurations
- **Production**: Use as a template for real-world projects

## 🔧 Customization

### Adding New Datasets

1. Extend the `DataProcessor` class
2. Add dataset loading logic
3. Update configuration files
4. Add tests

### Adding New Models

1. Update the model configuration
2. Modify `setup_model_and_tokenizer` if needed
3. Test with your model

### Adding New Metrics

1. Extend the `compute_metrics` function
2. Update evaluation scripts
3. Add visualization code

## 🐛 Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce batch size
   - Use gradient accumulation
   - Enable mixed precision training

2. **MLflow Connection Issues**
   - Check if MLflow server is running
   - Verify tracking URI in configuration

3. **Model Loading Issues**
   - Check model name and availability
   - Verify Hugging Face authentication

### Getting Help

- Check the logs in the `logs/` directory
- Review MLflow UI for detailed metrics
- Run tests to verify installation

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

### Development Setup

```bash
# Install development dependencies
pip install -e ".[dev]"

# Run pre-commit hooks
pre-commit install

# Run tests
pytest

# Format code
black llm_finetuning/
isort llm_finetuning/
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Hugging Face](https://huggingface.co/) for the Transformers library
- [MLflow](https://mlflow.org/) for experiment tracking
- [PEFT](https://github.com/huggingface/peft) for parameter-efficient fine-tuning
- [TinyLlama](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0) for the base model

## 📞 Contact

For questions or suggestions, please open an issue or contact [your-email@example.com].

---

**Happy Fine-Tuning! 🚀**
