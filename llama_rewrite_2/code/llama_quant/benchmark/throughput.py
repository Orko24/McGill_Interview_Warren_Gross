"""
Throughput measurement utilities.

Measures tokens generated per second at different batch sizes.
"""

import time
import logging
from typing import Dict, List

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from llama_quant.benchmark.memory import clear_gpu_memory

logger = logging.getLogger(__name__)


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
    Measure throughput (tokens per second) for different batch sizes.
    
    Throughput scales with batch size until memory-bound.
    Quantized models allow larger batches = higher throughput.
    
    Args:
        model: Model to benchmark
        tokenizer: Model tokenizer
        batch_sizes: List of batch sizes to test
        input_length: Length of input prompts
        output_length: Number of tokens to generate
        warmup_runs: Number of warmup iterations
        benchmark_runs: Number of measured iterations
        device: Device to run on
        
    Returns:
        Dict mapping batch_size -> tokens per second
    """
    results = {}
    
    for batch_size in tqdm(batch_sizes, desc="📊 Throughput", unit="batch"):
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
                with torch.inference_mode():
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
                
                with torch.inference_mode():
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
            logger.debug(f"Throughput (batch={batch_size}): {avg_tps:.2f} tokens/sec")
            
        except torch.cuda.OutOfMemoryError:
            logger.warning(f"OOM at batch_size={batch_size}, skipping...")
            results[batch_size] = 0.0
            clear_gpu_memory()
    
    return results

