"""
Benchmark suite - orchestrates all hardware benchmarks.
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Type

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from llama_quant.core.config import ExperimentConfig
from llama_quant.benchmark.base import Benchmark, BenchmarkConfig, BenchmarkResult
from llama_quant.benchmark.memory import MemoryBenchmark
from llama_quant.benchmark.latency import LatencyBenchmark
from llama_quant.benchmark.throughput import ThroughputBenchmark

logger = logging.getLogger(__name__)


class BenchmarkSuite:
    """
    Orchestrates multiple benchmarks.
    
    Runs memory, latency, and throughput benchmarks with proper
    cleanup between runs.
    
    Example:
        >>> suite = BenchmarkSuite(model, tokenizer)
        >>> results = suite.run_all()
        >>> suite.print_summary(results)
    
    Or run individual benchmarks:
        >>> memory_result = suite.run_memory()
        >>> latency_result = suite.run_latency()
    """
    
    # Registry of available benchmarks
    BENCHMARKS: Dict[str, Type[Benchmark]] = {
        "memory": MemoryBenchmark,
        "latency": LatencyBenchmark,
        "throughput": ThroughputBenchmark,
    }
    
    def __init__(
        self,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        config: Optional[BenchmarkConfig] = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config or BenchmarkConfig()
        
        # Apply PyTorch optimizations
        self._apply_torch_optimizations()
    
    @classmethod
    def from_experiment_config(
        cls,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        experiment_config: ExperimentConfig,
    ) -> "BenchmarkSuite":
        """Create suite from experiment config."""
        bench_config = BenchmarkConfig(
            warmup_runs=experiment_config.benchmark.warmup_runs,
            benchmark_runs=experiment_config.benchmark.benchmark_runs,
            device=experiment_config.eval.device,
            input_lengths=experiment_config.benchmark.input_lengths,
            output_length=experiment_config.benchmark.output_length,
            batch_sizes=experiment_config.benchmark.batch_sizes,
        )
        return cls(model, tokenizer, bench_config)
    
    def _apply_torch_optimizations(self) -> None:
        """Apply PyTorch backend optimizations for faster inference."""
        # TF32 for faster matmuls on Ampere+ GPUs
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        
        # cuDNN auto-tuning
        torch.backends.cudnn.benchmark = True
        
        # Float32 matmul precision
        if hasattr(torch, 'set_float32_matmul_precision'):
            torch.set_float32_matmul_precision('medium')
        
        logger.info("PyTorch optimizations enabled: TF32, cuDNN benchmark")
    
    def _create_benchmark(self, benchmark_type: str) -> Benchmark:
        """Factory method to create benchmark instances."""
        if benchmark_type not in self.BENCHMARKS:
            raise ValueError(f"Unknown benchmark: {benchmark_type}. Available: {list(self.BENCHMARKS.keys())}")
        
        benchmark_class = self.BENCHMARKS[benchmark_type]
        return benchmark_class(self.model, self.tokenizer, self.config)
    
    def run_memory(self) -> BenchmarkResult:
        """Run memory benchmark."""
        benchmark = self._create_benchmark("memory")
        return benchmark.run()
    
    def run_latency(self) -> BenchmarkResult:
        """Run latency benchmark."""
        benchmark = self._create_benchmark("latency")
        return benchmark.run()
    
    def run_throughput(self) -> BenchmarkResult:
        """Run throughput benchmark."""
        benchmark = self._create_benchmark("throughput")
        return benchmark.run()
    
    def run_all(self) -> Dict[str, Any]:
        """
        Run all benchmarks.
        
        Returns:
            Consolidated dictionary with all results
        """
        logger.info("Starting benchmark suite...")
        
        # Memory benchmark
        MemoryBenchmark.clear_gpu_memory()
        memory_result = self.run_memory()
        
        # Latency benchmark  
        MemoryBenchmark.clear_gpu_memory()
        latency_result = self.run_latency()
        
        # Throughput benchmark
        MemoryBenchmark.clear_gpu_memory()
        throughput_result = self.run_throughput()
        
        # Consolidate results
        results = {
            "model_size_mb": memory_result.metrics["model_size_mb"],
            "memory_allocated_mb": memory_result.metrics["allocated_mb"],
            "memory_peak_mb": memory_result.metrics["peak_mb"],
            "prefill_latency_ms": latency_result.metrics["prefill_latency_ms"],
            "decode_latency_ms_per_token": latency_result.metrics["decode_ms_per_token"],
            "throughput_tokens_per_sec": throughput_result.metrics["tokens_per_sec"],
            "device": memory_result.device,
        }
        
        return results
    
    def print_summary(self, results: Dict[str, Any]) -> None:
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


# Backwards-compatible function wrapper
def run_benchmarks(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    config: ExperimentConfig,
) -> Dict[str, Any]:
    """
    Run all benchmarks (backwards-compatible function).
    
    For new code, prefer using BenchmarkSuite directly.
    """
    suite = BenchmarkSuite.from_experiment_config(model, tokenizer, config)
    return suite.run_all()
