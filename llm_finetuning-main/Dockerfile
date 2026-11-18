# LLM Fine-Tuning Demo Dockerfile
# This Dockerfile creates a reproducible environment for LLM fine-tuning

FROM python:3.9-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV DEBIAN_FRONTEND=noninteractive

# Set work directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY pyproject.toml ./
RUN pip install --no-cache-dir --upgrade pip setuptools wheel
RUN pip install --no-cache-dir -e .

# Copy project files
COPY . .

# Create necessary directories
RUN mkdir -p checkpoints logs evaluation_results

# Expose MLflow port
EXPOSE 5000

# Set default command
CMD ["python", "-m", "mlflow", "ui", "--host", "0.0.0.0", "--port", "5000"]
