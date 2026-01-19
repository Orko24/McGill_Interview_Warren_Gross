"""
Throughput benchmark - measures tokens generated per second.
"""

import time
import logging
from typing import Any, Dict, List, Optional

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from llama_quant.benchmark.base import Benchmark, BenchmarkConfig, BenchmarkResult
from llama_quant.benchmark.memory import MemoryBenchmark

logger = logging.getLogger(__name__)


class ThroughputBenchmark(Benchmark):
    """
    Benchmarks generation throughput (tokens per second).
    
    Throughput scales with batch size until memory-bound.
    Quantized models allow larger batches = higher throughput.
    
    Example:
        >>> benchmark = ThroughputBenchmark(model, tokenizer)
        >>> result = benchmark.run()
        >>> print(f"Batch 8: {result.metrics['tokens_per_sec'][8]:.2f} tok/s")
    """
    
    @property
    def name(self) -> str:
        return "Throughput"
    
    def run(self) -> BenchmarkResult:
        """Run throughput benchmark across batch sizes."""
        logger.info("Running throughput benchmark...")
        
        throughput_results = self.measure_throughput()
        
        return BenchmarkResult(
            name=self.name,
            metrics={
                "tokens_per_sec": throughput_results,
            },
            device=self._get_device_name(),
            dtype=self._get_model_dtype(),
        )
    
    def measure_throughput(self) -> Dict[int, float]:
        """
        Measure throughput for different batch sizes.
        
        Returns:
            Dict mapping batch_size -> tokens per second
        """
        results = {}
        batch_sizes = self.config.batch_sizes
        input_length = self.config.input_lengths[0]
        output_length = self.config.output_length
        
        for batch_size in tqdm(batch_sizes, desc="📊 Throughput", unit="batch"):
            try:
                tps = self._benchmark_batch_size(batch_size, input_length, output_length)
                results[batch_size] = tps
                logger.debug(f"Throughput (batch={batch_size}): {tps:.2f} tok/s")
                
            except torch.cuda.OutOfMemoryError:
                logger.warning(f"OOM at batch_size={batch_size}, skipping...")
                results[batch_size] = 0.0
                MemoryBenchmark.clear_gpu_memory()
        
        return results
    
    def _benchmark_batch_size(
        self, 
        batch_size: int, 
        input_length: int, 
        output_length: int
    ) -> float:
        """Benchmark throughput for a single batch size."""
        inputs = self._create_dummy_input(input_length, batch_size)
        
        # Warmup
        for _ in range(self.config.warmup_runs):
            with torch.inference_mode():
                _ = self.model.generate(
                    **inputs,
                    max_new_tokens=10,
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id,
                )
        
        MemoryBenchmark.clear_gpu_memory()
        self._synchronize()
        
        # Benchmark
        throughputs = []
        for _ in range(self.config.benchmark_runs):
            start = time.perf_counter()
            
            with torch.inference_mode():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=output_length,
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id,
                )
            
            self._synchronize()
            
            elapsed = time.perf_counter() - start
            num_generated = (outputs.shape[1] - inputs["input_ids"].shape[1]) * batch_size
            tps = num_generated / elapsed
            throughputs.append(tps)
        
        return sum(throughputs) / len(throughputs)
