"""
Latency benchmark - measures prefill and decode latency.
"""

import time
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from llama_quant.benchmark.base import Benchmark, BenchmarkConfig, BenchmarkResult
from llama_quant.benchmark.memory import MemoryBenchmark

logger = logging.getLogger(__name__)


@dataclass
class LatencyStats:
    """Latency measurement statistics."""
    mean_ms: float
    min_ms: float
    max_ms: float
    std_ms: float
    
    def to_dict(self) -> Dict[str, float]:
        return {
            "mean_ms": self.mean_ms,
            "min_ms": self.min_ms,
            "max_ms": self.max_ms,
            "std_ms": self.std_ms,
        }


class LatencyBenchmark(Benchmark):
    """
    Benchmarks inference latency.
    
    Measures:
        - Prefill latency: Time to process input prompt (compute-bound)
        - Decode latency: Time per generated token (memory-bandwidth bound)
    
    Example:
        >>> benchmark = LatencyBenchmark(model, tokenizer)
        >>> result = benchmark.run()
        >>> print(f"Decode: {result.metrics['decode_ms_per_token']:.2f} ms/token")
    """
    
    @property
    def name(self) -> str:
        return "Latency"
    
    def run(self) -> BenchmarkResult:
        """Run full latency benchmark (prefill + decode)."""
        logger.info("Running latency benchmark...")
        
        prefill_results = self.measure_prefill()
        decode_ms = self.measure_decode()
        
        return BenchmarkResult(
            name=self.name,
            metrics={
                "prefill_latency_ms": prefill_results,
                "decode_ms_per_token": decode_ms,
            },
            device=self._get_device_name(),
            dtype=self._get_model_dtype(),
        )
    
    def measure_prefill(self) -> Dict[int, float]:
        """
        Measure prefill (prompt processing) latency.
        
        Prefill is the forward pass over the input prompt before generation.
        This is compute-bound and benefits from larger batch sizes.
        
        Returns:
            Dict mapping input_length -> latency in milliseconds
        """
        results = {}
        input_lengths = self.config.input_lengths
        
        for length in tqdm(input_lengths, desc="📊 Prefill latency", unit="len"):
            latencies = self._benchmark_prefill_single(length)
            results[length] = sum(latencies) / len(latencies)
            logger.debug(f"Prefill (len={length}): {results[length]:.2f}ms")
        
        return results
    
    def _benchmark_prefill_single(self, length: int) -> List[float]:
        """Benchmark prefill for a single input length."""
        inputs = self._create_dummy_input(length)
        
        # Warmup
        for _ in range(self.config.warmup_runs):
            with torch.inference_mode():
                _ = self.model(**inputs)
        
        self._synchronize()
        
        # Benchmark
        latencies = []
        for _ in range(self.config.benchmark_runs):
            start = time.perf_counter()
            
            with torch.inference_mode():
                _ = self.model(**inputs)
            
            self._synchronize()
            latencies.append((time.perf_counter() - start) * 1000)
        
        return latencies
    
    def measure_decode(self) -> float:
        """
        Measure per-token decode latency during generation.
        
        Decode latency is the time to generate each new token.
        This is memory-bandwidth bound due to KV-cache access.
        
        Returns:
            Average latency per generated token in milliseconds
        """
        input_length = self.config.input_lengths[0]
        output_length = self.config.output_length
        
        inputs = self._create_dummy_input(input_length)
        
        # Warmup
        for _ in range(self.config.warmup_runs):
            with torch.inference_mode():
                _ = self.model.generate(
                    **inputs,
                    max_new_tokens=10,
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id,
                )
        
        self._synchronize()
        
        # Benchmark
        latencies = []
        for _ in tqdm(range(self.config.benchmark_runs), desc="📊 Decode latency", unit="run"):
            start = time.perf_counter()
            
            with torch.inference_mode():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=output_length,
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id,
                )
            
            self._synchronize()
            
            total_time = time.perf_counter() - start
            num_generated = outputs.shape[1] - inputs["input_ids"].shape[1]
            per_token_ms = (total_time * 1000) / num_generated
            latencies.append(per_token_ms)
        
        avg_latency = sum(latencies) / len(latencies)
        logger.info(f"Decode latency: {avg_latency:.2f}ms per token")
        
        return avg_latency
