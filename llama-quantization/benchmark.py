"""
Hardware performance benchmarking
Measures: Memory usage, latency, throughput, model size
"""

import gc
import time
import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import ExperimentConfig, BenchmarkConfig

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    """Container for benchmark results"""
    model_size_mb: float
    peak_memory_mb: float
    allocated_memory_mb: float
    
    # Latency (seconds)
    prefill_latency_ms: Dict[int, float]  # input_length -> latency
    decode_latency_ms: float  # per token
    
    # Throughput
    tokens_per_second: Dict[int, float]  # batch_size -> tps
    
    # Additional info
    device_name: str
    dtype: str


def get_gpu_memory_info() -> Dict[str, float]:
    """Get current GPU memory usage"""
    if not torch.cuda.is_available():
        return {"allocated_mb": 0, "reserved_mb": 0, "peak_mb": 0}
    
    return {
        "allocated_mb": torch.cuda.memory_allocated() / 1024 / 1024,
        "reserved_mb": torch.cuda.memory_reserved() / 1024 / 1024,
        "peak_mb": torch.cuda.max_memory_allocated() / 1024 / 1024,
    }


def clear_gpu_memory():
    """Clear GPU memory cache"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def measure_model_size(model: AutoModelForCausalLM) -> float:
    """Calculate model size in MB"""
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    return (param_size + buffer_size) / 1024 / 1024


def measure_prefill_latency(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    input_lengths: List[int],
    warmup_runs: int = 5,
    benchmark_runs: int = 20,
    device: str = "cuda",
) -> Dict[int, float]:
    """
    Measure prefill (prompt processing) latency for different input lengths
    
    Returns:
        Dict mapping input_length -> latency in milliseconds
    """
    results = {}
    
    for length in input_lengths:
        # Create dummy input of specified length
        dummy_text = "Hello " * (length // 2)  # Approximate token count
        inputs = tokenizer(
            dummy_text,
            return_tensors="pt",
            max_length=length,
            truncation=True,
            padding="max_length",
        ).to(device)
        
        # Warmup
        for _ in range(warmup_runs):
            with torch.no_grad():
                _ = model(**inputs)
        
        # Synchronize GPU
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        # Benchmark
        latencies = []
        for _ in range(benchmark_runs):
            start = time.perf_counter()
            
            with torch.no_grad():
                _ = model(**inputs)
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            latencies.append((time.perf_counter() - start) * 1000)  # Convert to ms
        
        avg_latency = sum(latencies) / len(latencies)
        results[length] = avg_latency
        logger.info(f"Prefill latency (len={length}): {avg_latency:.2f}ms")
    
    return results


def measure_decode_latency(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    input_length: int = 128,
    output_length: int = 128,
    warmup_runs: int = 3,
    benchmark_runs: int = 10,
    device: str = "cuda",
) -> float:
    """
    Measure per-token decode latency during generation
    
    Returns:
        Average latency per generated token in milliseconds
    """
    # Create input
    dummy_text = "Hello " * (input_length // 2)
    inputs = tokenizer(
        dummy_text,
        return_tensors="pt",
        max_length=input_length,
        truncation=True,
    ).to(device)
    
    # Warmup
    for _ in range(warmup_runs):
        with torch.no_grad():
            _ = model.generate(
                **inputs,
                max_new_tokens=10,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    # Benchmark
    latencies = []
    for _ in range(benchmark_runs):
        start = time.perf_counter()
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=output_length,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        total_time = time.perf_counter() - start
        num_generated = outputs.shape[1] - inputs["input_ids"].shape[1]
        per_token_latency = (total_time * 1000) / num_generated  # ms per token
        latencies.append(per_token_latency)
    
    avg_latency = sum(latencies) / len(latencies)
    logger.info(f"Decode latency: {avg_latency:.2f}ms per token")
    
    return avg_latency


def measure_throughput(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    batch_sizes: List[int],
    input_length: int = 128,
    output_length: int = 128,
    warmup_runs: int = 2,
    benchmark_runs: int = 5,
    device: str = "cuda",
) -> Dict[int, float]:
    """
    Measure throughput (tokens per second) for different batch sizes
    
    Returns:
        Dict mapping batch_size -> tokens per second
    """
    results = {}
    
    for batch_size in batch_sizes:
        try:
            # Create batched input
            dummy_text = "Hello " * (input_length // 2)
            inputs = tokenizer(
                [dummy_text] * batch_size,
                return_tensors="pt",
                max_length=input_length,
                truncation=True,
                padding="max_length",
            ).to(device)
            
            # Warmup
            for _ in range(warmup_runs):
                with torch.no_grad():
                    _ = model.generate(
                        **inputs,
                        max_new_tokens=10,
                        do_sample=False,
                        pad_token_id=tokenizer.pad_token_id,
                    )
            
            clear_gpu_memory()
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            # Benchmark
            throughputs = []
            for _ in range(benchmark_runs):
                start = time.perf_counter()
                
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=output_length,
                        do_sample=False,
                        pad_token_id=tokenizer.pad_token_id,
                    )
                
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                
                elapsed = time.perf_counter() - start
                num_generated = (outputs.shape[1] - inputs["input_ids"].shape[1]) * batch_size
                tps = num_generated / elapsed
                throughputs.append(tps)
            
            avg_tps = sum(throughputs) / len(throughputs)
            results[batch_size] = avg_tps
            logger.info(f"Throughput (batch={batch_size}): {avg_tps:.2f} tokens/sec")
            
        except torch.cuda.OutOfMemoryError:
            logger.warning(f"OOM at batch_size={batch_size}, skipping...")
            results[batch_size] = 0.0
            clear_gpu_memory()
    
    return results


def run_benchmarks(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    config: ExperimentConfig,
) -> Dict[str, Any]:
    """
    Run all hardware benchmarks
    
    Returns:
        Dictionary containing all benchmark results
    """
    bench_config = config.benchmark
    device = config.eval.device
    
    logger.info("Starting hardware benchmarks...")
    
    # Clear memory before benchmarks
    clear_gpu_memory()
    
    # Model size
    model_size_mb = measure_model_size(model)
    logger.info(f"Model size: {model_size_mb:.2f} MB")
    
    # Memory usage during inference
    clear_gpu_memory()
    dummy_input = tokenizer("Hello world", return_tensors="pt").to(device)
    
    with torch.no_grad():
        _ = model(**dummy_input)
    
    memory_info = get_gpu_memory_info()
    logger.info(f"Memory usage: {memory_info}")
    
    # Prefill latency
    prefill_latency = measure_prefill_latency(
        model=model,
        tokenizer=tokenizer,
        input_lengths=bench_config.input_lengths,
        warmup_runs=bench_config.warmup_runs,
        benchmark_runs=bench_config.benchmark_runs,
        device=device,
    )
    
    # Decode latency
    decode_latency = measure_decode_latency(
        model=model,
        tokenizer=tokenizer,
        input_length=bench_config.input_lengths[0],
        output_length=bench_config.output_length,
        warmup_runs=bench_config.warmup_runs,
        benchmark_runs=bench_config.benchmark_runs,
        device=device,
    )
    
    # Throughput
    throughput = measure_throughput(
        model=model,
        tokenizer=tokenizer,
        batch_sizes=bench_config.batch_sizes,
        input_length=bench_config.input_lengths[0],
        output_length=bench_config.output_length,
        warmup_runs=bench_config.warmup_runs,
        benchmark_runs=bench_config.benchmark_runs // 2,  # Fewer runs for throughput
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


def print_benchmark_summary(results: Dict[str, Any]):
    """Print a formatted summary of benchmark results"""
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


def calculate_efficiency_score(
    accuracy: float,
    model_size_mb: float,
    baseline_accuracy: float = 0.8,
    baseline_size_mb: float = 2000.0,
) -> float:
    """
    Calculate an efficiency score balancing accuracy and size
    
    Higher is better. Penalizes accuracy drops, rewards size reduction.
    """
    accuracy_ratio = accuracy / baseline_accuracy
    size_ratio = baseline_size_mb / model_size_mb
    
    # Weighted geometric mean (more weight on accuracy)
    efficiency = (accuracy_ratio ** 0.7) * (size_ratio ** 0.3)
    
    return efficiency
