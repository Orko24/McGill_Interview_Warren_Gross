"""
Benchmark module - hardware performance measurement.

Classes:
    BenchmarkSuite: Orchestrates all benchmarks
    MemoryBenchmark: Model size and GPU memory usage
    LatencyBenchmark: Prefill and decode latency
    ThroughputBenchmark: Tokens per second at different batch sizes

Example:
    >>> from llama_quant.benchmark import BenchmarkSuite
    >>> 
    >>> suite = BenchmarkSuite(model, tokenizer)
    >>> results = suite.run_all()
    >>> suite.print_summary(results)
"""

from llama_quant.benchmark.base import Benchmark, BenchmarkConfig, BenchmarkResult
from llama_quant.benchmark.memory import MemoryBenchmark, clear_gpu_memory
from llama_quant.benchmark.latency import LatencyBenchmark
from llama_quant.benchmark.throughput import ThroughputBenchmark
from llama_quant.benchmark.runner import BenchmarkSuite, run_benchmarks

__all__ = [
    # Base classes
    "Benchmark",
    "BenchmarkConfig", 
    "BenchmarkResult",
    
    # Concrete benchmarks
    "MemoryBenchmark",
    "LatencyBenchmark", 
    "ThroughputBenchmark",
    
    # Suite
    "BenchmarkSuite",
    
    # Backwards-compatible function
    "run_benchmarks",
    "clear_gpu_memory",
]
