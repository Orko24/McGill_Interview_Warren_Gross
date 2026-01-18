"""
Memory measurement utilities.

Measures:
- Model parameter size
- GPU memory allocation
- Peak memory usage
"""

import gc
import logging
from typing import Dict

import torch
from transformers import AutoModelForCausalLM

logger = logging.getLogger(__name__)


def measure_model_size(model: AutoModelForCausalLM) -> float:
    """
    Calculate model size in megabytes.
    
    Sums:
    - Parameter memory (weights)
    - Buffer memory (batch norm stats, etc.)
    
    Args:
        model: Model to measure
        
    Returns:
        Size in megabytes
    """
    param_size = sum(p.nelement() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.nelement() * b.element_size() for b in model.buffers())
    return (param_size + buffer_size) / 1024 / 1024


def get_gpu_memory_info() -> Dict[str, float]:
    """
    Get current GPU memory usage statistics.
    
    Returns:
        Dictionary with:
        - allocated_mb: Currently allocated memory
        - reserved_mb: Reserved memory (includes fragmentation)
        - peak_mb: Peak allocated memory since last reset
    """
    if not torch.cuda.is_available():
        return {"allocated_mb": 0, "reserved_mb": 0, "peak_mb": 0}
    
    return {
        "allocated_mb": torch.cuda.memory_allocated() / 1024 / 1024,
        "reserved_mb": torch.cuda.memory_reserved() / 1024 / 1024,
        "peak_mb": torch.cuda.max_memory_allocated() / 1024 / 1024,
    }


def clear_gpu_memory() -> None:
    """
    Clear GPU memory cache and reset peak stats.
    
    Should be called between benchmarks for accurate measurements.
    """
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def log_memory_stats() -> None:
    """Log current memory statistics."""
    stats = get_gpu_memory_info()
    logger.info(f"GPU Memory - Allocated: {stats['allocated_mb']:.2f} MB, "
                f"Reserved: {stats['reserved_mb']:.2f} MB, "
                f"Peak: {stats['peak_mb']:.2f} MB")

