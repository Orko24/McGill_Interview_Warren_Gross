# =============================================================================
# Llama 3.2-1B Quantization - Docker Image
# =============================================================================
# Build: docker build -t llama-quant .
# Run:   docker run --gpus all -v $(pwd)/results:/app/results llama-quant
# =============================================================================

# -----------------------------------------------------------------------------
# Layer 1: Base CUDA image with Python
# -----------------------------------------------------------------------------
FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04 AS base

# Prevent interactive prompts during install
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-dev \
    python3.11-venv \
    python3-pip \
    git \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.11 /usr/bin/python \
    && ln -sf /usr/bin/python3.11 /usr/bin/python3

# Upgrade pip
RUN python -m pip install --upgrade pip setuptools wheel

# -----------------------------------------------------------------------------
# Layer 2: PyTorch and CUDA dependencies (large, cache this)
# -----------------------------------------------------------------------------
FROM base AS pytorch

# Install PyTorch with CUDA support
RUN pip install --no-cache-dir \
    torch==2.2.0 \
    torchvision==0.17.0 \
    torchaudio==2.2.0 \
    --index-url https://download.pytorch.org/whl/cu121

# -----------------------------------------------------------------------------
# Layer 3: ML dependencies (transformers, accelerate, etc.)
# -----------------------------------------------------------------------------
FROM pytorch AS ml-deps

# Install transformers ecosystem
RUN pip install --no-cache-dir \
    transformers>=4.36.0 \
    accelerate>=0.25.0 \
    datasets>=2.16.0 \
    huggingface_hub>=0.20.0 \
    safetensors>=0.4.0 \
    sentencepiece>=0.1.99

# -----------------------------------------------------------------------------
# Layer 4: Quantization libraries
# -----------------------------------------------------------------------------
FROM ml-deps AS quant-deps

# BitsAndBytes (main quantization library)
RUN pip install --no-cache-dir bitsandbytes>=0.41.0

# Optional: GPTQ and AWQ (uncomment if needed, adds ~500MB)
# RUN pip install --no-cache-dir optimum>=1.15.0 auto-gptq>=0.6.0
# RUN pip install --no-cache-dir autoawq>=0.1.8

# -----------------------------------------------------------------------------
# Layer 5: Evaluation framework
# -----------------------------------------------------------------------------
FROM quant-deps AS eval-deps

# lm-evaluation-harness for CoQA benchmarking
RUN pip install --no-cache-dir lm-eval>=0.4.0

# -----------------------------------------------------------------------------
# Layer 6: Utilities and visualization
# -----------------------------------------------------------------------------
FROM eval-deps AS utils

RUN pip install --no-cache-dir \
    tqdm>=4.66.0 \
    numpy>=1.24.0 \
    pandas>=2.0.0 \
    matplotlib>=3.7.0 \
    seaborn>=0.12.0

# -----------------------------------------------------------------------------
# Layer 7: Application code
# -----------------------------------------------------------------------------
FROM utils AS app

# Set working directory
WORKDIR /app

# Copy application code
COPY config.py .
COPY quantize.py .
COPY evaluate.py .
COPY benchmark.py .
COPY main.py .
COPY sweep.py .
COPY visualize.py .

# Create results directory
RUN mkdir -p /app/results

# Create cache directory for HuggingFace models
ENV HF_HOME=/cache/huggingface
ENV TRANSFORMERS_CACHE=/cache/huggingface
RUN mkdir -p /cache/huggingface

# Default command: show help
CMD ["python", "main.py", "--help"]

# -----------------------------------------------------------------------------
# Labels for documentation
# -----------------------------------------------------------------------------
LABEL maintainer="Hemanto Bairagi"
LABEL description="Llama 3.2-1B Quantization Experiments"
LABEL version="1.0"


