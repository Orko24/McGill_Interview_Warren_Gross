"""
Base benchmark class - abstract interface for all benchmarks.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class BenchmarkConfig:
    """Configuration for benchmarks."""
    warmup_runs: int = 3
    benchmark_runs: int = 10
    device: str = "cuda"
    
    # Latency config
    input_lengths: list = field(default_factory=lambda: [128, 256, 512, 1024])
    output_length: int = 128
    
    # Throughput config
    batch_sizes: list = field(default_factory=lambda: [1, 4, 8])


@dataclass
class BenchmarkResult:
    """Container for benchmark results."""
    name: str
    metrics: Dict[str, Any]
    device: str
    dtype: str
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "metrics": self.metrics,
            "device": self.device,
            "dtype": self.dtype,
        }


class Benchmark(ABC):
    """
    Abstract base class for all benchmarks.
    
    Subclasses must implement:
        - name: Human-readable benchmark name
        - run(): Execute the benchmark and return results
    """
    
    def __init__(
        self,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        config: Optional[BenchmarkConfig] = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config or BenchmarkConfig()
        self._device = self.config.device
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable benchmark name."""
        pass
    
    @abstractmethod
    def run(self) -> BenchmarkResult:
        """Execute the benchmark and return results."""
        pass
    
    def _get_device_name(self) -> str:
        """Get GPU device name or 'cpu'."""
        if torch.cuda.is_available():
            return torch.cuda.get_device_name()
        return "cpu"
    
    def _get_model_dtype(self) -> str:
        """Get model's primary dtype."""
        for param in self.model.parameters():
            return str(param.dtype)
        return "unknown"
    
    def _create_dummy_input(self, length: int, batch_size: int = 1) -> Dict[str, torch.Tensor]:
        """Create dummy input for benchmarking."""
        dummy_text = "Hello " * (length // 2)
        texts = [dummy_text] * batch_size
        
        inputs = self.tokenizer(
            texts if batch_size > 1 else dummy_text,
            return_tensors="pt",
            max_length=length,
            truncation=True,
            padding="max_length" if batch_size > 1 else False,
        )
        return {k: v.to(self._device) for k, v in inputs.items()}
    
    def _synchronize(self) -> None:
        """Synchronize CUDA if available."""
        if torch.cuda.is_available():
            torch.cuda.synchronize()

