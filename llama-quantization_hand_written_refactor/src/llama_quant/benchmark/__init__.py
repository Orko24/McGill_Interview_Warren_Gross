"""
Hardware benchmarking module.

Measures:
- Memory usage (allocated, peak)
- Prefill latency (prompt processing)
- Decode latency (per-token generation)
- Throughput (tokens per second)

Usage:
    from llama_quant.benchmark import run_benchmarks
    
    results = run_benchmarks(model, tokenizer, config)
"""

from llama_quant.benchmark.runner import (
    run_benchmarks,
    BenchmarkResult,
    apply_torch_optimizations,
)
from llama_quant.benchmark.memory import (
    measure_model_size,
    get_gpu_memory_info,
    clear_gpu_memory,
)
from llama_quant.benchmark.latency import (
    measure_prefill_latency,
    measure_decode_latency,
)
from llama_quant.benchmark.throughput import (
    measure_throughput,
)

__all__ = [
    "run_benchmarks",
    "BenchmarkResult",
    "apply_torch_optimizations",
    "measure_model_size",
    "get_gpu_memory_info",
    "clear_gpu_memory",
    "measure_prefill_latency",
    "measure_decode_latency",
    "measure_throughput",
]



