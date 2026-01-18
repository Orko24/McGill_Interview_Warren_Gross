"""
Memory benchmark - measures model size and GPU memory usage.
"""

import gc
import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from llama_quant.benchmark.base import Benchmark, BenchmarkConfig, BenchmarkResult

logger = logging.getLogger(__name__)


@dataclass
class MemoryStats:
    """GPU memory statistics."""
    allocated_mb: float
    reserved_mb: float
    peak_mb: float
    
    def to_dict(self) -> Dict[str, float]:
        return {
            "allocated_mb": self.allocated_mb,
            "reserved_mb": self.reserved_mb,
            "peak_mb": self.peak_mb,
        }


class MemoryBenchmark(Benchmark):
    """
    Benchmarks model memory footprint.
    
    Measures:
        - Model parameter size (weights + buffers)
        - GPU memory allocated
        - Peak GPU memory usage
    
    Example:
        >>> benchmark = MemoryBenchmark(model, tokenizer)
        >>> result = benchmark.run()
        >>> print(f"Model size: {result.metrics['model_size_mb']:.2f} MB")
    """
    
    @property
    def name(self) -> str:
        return "Memory"
    
    def run(self) -> BenchmarkResult:
        """Run memory benchmark."""
        logger.info("Running memory benchmark...")
        
        # Clear previous stats
        self.clear_gpu_memory()
        
        # Measure model size
        model_size = self.get_model_size()
        logger.info(f"Model size: {model_size:.2f} MB")
        
        # Trigger forward pass to measure runtime memory
        dummy_input = self._create_dummy_input(length=128)
        with torch.inference_mode():
            _ = self.model(**dummy_input)
        
        # Get memory stats
        memory_stats = self.get_gpu_memory_stats()
        logger.info(f"Peak memory: {memory_stats.peak_mb:.2f} MB")
        
        return BenchmarkResult(
            name=self.name,
            metrics={
                "model_size_mb": model_size,
                **memory_stats.to_dict(),
            },
            device=self._get_device_name(),
            dtype=self._get_model_dtype(),
        )
    
    def get_model_size(self) -> float:
        """
        Calculate model size in megabytes.
        
        Includes:
            - Parameters (weights)
            - Buffers (batch norm stats, etc.)
        """
        param_size = sum(p.nelement() * p.element_size() for p in self.model.parameters())
        buffer_size = sum(b.nelement() * b.element_size() for b in self.model.buffers())
        return (param_size + buffer_size) / 1024 / 1024
    
    def get_gpu_memory_stats(self) -> MemoryStats:
        """Get current GPU memory statistics."""
        if not torch.cuda.is_available():
            return MemoryStats(0, 0, 0)
        
        return MemoryStats(
            allocated_mb=torch.cuda.memory_allocated() / 1024 / 1024,
            reserved_mb=torch.cuda.memory_reserved() / 1024 / 1024,
            peak_mb=torch.cuda.max_memory_allocated() / 1024 / 1024,
        )
    
    @staticmethod
    def clear_gpu_memory() -> None:
        """Clear GPU memory cache and reset peak stats."""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()


# Convenience function for backwards compatibility
def clear_gpu_memory() -> None:
    """Clear GPU memory (module-level function)."""
    MemoryBenchmark.clear_gpu_memory()
