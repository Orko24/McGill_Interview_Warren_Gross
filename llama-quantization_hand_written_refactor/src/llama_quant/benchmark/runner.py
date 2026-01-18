"""
Benchmark runner - orchestrates all hardware benchmarks.
"""

import logging
from dataclasses import dataclass
from typing import Dict, Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from llama_quant.core.config import ExperimentConfig
from llama_quant.benchmark.memory import (
    measure_model_size,
    get_gpu_memory_info,
    clear_gpu_memory,
)
from llama_quant.benchmark.latency import (
    measure_prefill_latency,
    measure_decode_latency,
)
from llama_quant.benchmark.throughput import measure_throughput

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    """Container for benchmark results."""
    model_size_mb: float
    peak_memory_mb: float
    allocated_memory_mb: float
    prefill_latency_ms: Dict[int, float]
    decode_latency_ms: float
    tokens_per_second: Dict[int, float]
    device_name: str
    dtype: str


def apply_torch_optimizations() -> None:
    """
    Apply PyTorch backend optimizations for faster inference.
    
    Enables:
    - TF32 for Ampere+ GPUs (A10G, A100, H100)
    - cuDNN benchmarking for optimized kernels
    - Medium precision matmul
    """
    # TF32 for faster matmuls on Ampere+ GPUs
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    # cuDNN auto-tuning
    torch.backends.cudnn.benchmark = True
    
    # Float32 matmul precision
    if hasattr(torch, 'set_float32_matmul_precision'):
        torch.set_float32_matmul_precision('medium')
    
    logger.info("PyTorch optimizations enabled: TF32, cuDNN benchmark")


def run_benchmarks(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    config: ExperimentConfig,
) -> Dict[str, Any]:
    """
    Run all hardware benchmarks.
    
    Measures:
    1. Model size (parameter memory)
    2. GPU memory usage
    3. Prefill latency (prompt processing)
    4. Decode latency (per-token generation)
    5. Throughput (tokens per second)
    
    Args:
        model: Model to benchmark
        tokenizer: Model tokenizer
        config: Experiment configuration
        
    Returns:
        Dictionary containing all benchmark results
    """
    bench_config = config.benchmark
    device = config.eval.device
    
    logger.info("Starting hardware benchmarks...")
    
    # Apply optimizations
    apply_torch_optimizations()
    
    # Clear memory
    clear_gpu_memory()
    
    # 1. Model size
    model_size_mb = measure_model_size(model)
    logger.info(f"Model size: {model_size_mb:.2f} MB")
    
    # 2. Memory usage
    clear_gpu_memory()
    dummy_input = tokenizer("Hello world", return_tensors="pt").to(device)
    
    with torch.inference_mode():
        _ = model(**dummy_input)
    
    memory_info = get_gpu_memory_info()
    logger.info(f"Memory usage: {memory_info}")
    
    # 3. Prefill latency
    prefill_latency = measure_prefill_latency(
        model=model,
        tokenizer=tokenizer,
        input_lengths=bench_config.input_lengths,
        warmup_runs=bench_config.warmup_runs,
        benchmark_runs=bench_config.benchmark_runs,
        device=device,
    )
    
    # 4. Decode latency
    decode_latency = measure_decode_latency(
        model=model,
        tokenizer=tokenizer,
        input_length=bench_config.input_lengths[0],
        output_length=bench_config.output_length,
        warmup_runs=bench_config.warmup_runs,
        benchmark_runs=bench_config.benchmark_runs,
        device=device,
    )
    
    # 5. Throughput
    throughput = measure_throughput(
        model=model,
        tokenizer=tokenizer,
        batch_sizes=bench_config.batch_sizes,
        input_length=bench_config.input_lengths[0],
        output_length=bench_config.output_length,
        warmup_runs=bench_config.warmup_runs,
        benchmark_runs=bench_config.benchmark_runs // 2,
        device=device,
    )
    
    # Compile results
    results = {
        "model_size_mb": model_size_mb,
        "memory_allocated_mb": memory_info["allocated_mb"],
        "memory_peak_mb": memory_info["peak_mb"],
        "prefill_latency_ms": prefill_latency,
        "decode_latency_ms_per_token": decode_latency,
        "throughput_tokens_per_sec": throughput,
        "device": torch.cuda.get_device_name() if torch.cuda.is_available() else "cpu",
    }
    
    return results


def print_benchmark_summary(results: Dict[str, Any]) -> None:
    """Print formatted benchmark summary."""
    print("\n" + "=" * 60)
    print("BENCHMARK SUMMARY")
    print("=" * 60)
    
    print(f"\nModel Size: {results['model_size_mb']:.2f} MB")
    print(f"Peak Memory: {results['memory_peak_mb']:.2f} MB")
    print(f"Device: {results['device']}")
    
    print("\nPrefill Latency:")
    for length, latency in results["prefill_latency_ms"].items():
        print(f"  {length} tokens: {latency:.2f} ms")
    
    print(f"\nDecode Latency: {results['decode_latency_ms_per_token']:.2f} ms/token")
    
    print("\nThroughput:")
    for batch_size, tps in results["throughput_tokens_per_sec"].items():
        print(f"  Batch {batch_size}: {tps:.2f} tokens/sec")
    
    print("=" * 60 + "\n")



