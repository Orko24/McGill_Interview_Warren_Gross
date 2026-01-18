"""
Llama Quantization Package

A modular framework for quantizing Llama 3.2-1B and evaluating on CoQA.
Supports BitsAndBytes, GPTQ, and AWQ quantization methods.
"""

from llama_quant.core.config import (
    ExperimentConfig,
    QuantizationConfig,
    ModelConfig,
    EvalConfig,
    BenchmarkConfig,
    QuantMethod,
    ComputeDtype,
)
from llama_quant.models import load_model, get_model_size_mb
from llama_quant.evaluation import evaluate_model, extract_coqa_metrics
from llama_quant.benchmark import run_benchmarks

__version__ = "0.1.0"
__all__ = [
    # Config
    "ExperimentConfig",
    "QuantizationConfig", 
    "ModelConfig",
    "EvalConfig",
    "BenchmarkConfig",
    "QuantMethod",
    "ComputeDtype",
    # Model loading
    "load_model",
    "get_model_size_mb",
    # Evaluation
    "evaluate_model",
    "extract_coqa_metrics",
    # Benchmarking
    "run_benchmarks",
]



