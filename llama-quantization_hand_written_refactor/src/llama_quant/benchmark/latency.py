"""
Latency measurement utilities.

Measures:
- Prefill latency: Time to process input prompt
- Decode latency: Time per generated token
"""

import time
import logging
from typing import Dict, List

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)


def measure_prefill_latency(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    input_lengths: List[int],
    warmup_runs: int = 5,
    benchmark_runs: int = 20,
    device: str = "cuda",
) -> Dict[int, float]:
    """
    Measure prefill (prompt processing) latency for different input lengths.
    
    Prefill is the forward pass over the input prompt before generation.
    This is compute-bound and benefits from batch processing.
    
    Args:
        model: Model to benchmark
        tokenizer: Model tokenizer
        input_lengths: List of input lengths to test
        warmup_runs: Number of warmup iterations
        benchmark_runs: Number of measured iterations
        device: Device to run on
        
    Returns:
        Dict mapping input_length -> latency in milliseconds
    """
    results = {}
    
    for length in tqdm(input_lengths, desc="📊 Prefill latency", unit="len"):
        # Create dummy input of specified length
        dummy_text = "Hello " * (length // 2)
        inputs = tokenizer(
            dummy_text,
            return_tensors="pt",
            max_length=length,
            truncation=True,
            padding="max_length",
        ).to(device)
        
        # Warmup
        for _ in range(warmup_runs):
            with torch.inference_mode():
                _ = model(**inputs)
        
        # Synchronize
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        # Benchmark
        latencies = []
        for _ in range(benchmark_runs):
            start = time.perf_counter()
            
            with torch.inference_mode():
                _ = model(**inputs)
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            latencies.append((time.perf_counter() - start) * 1000)
        
        avg_latency = sum(latencies) / len(latencies)
        results[length] = avg_latency
        logger.debug(f"Prefill latency (len={length}): {avg_latency:.2f}ms")
    
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
    Measure per-token decode latency during generation.
    
    Decode latency is the time to generate each new token.
    This is memory-bandwidth bound due to KV-cache access.
    
    Args:
        model: Model to benchmark
        tokenizer: Model tokenizer
        input_length: Length of input prompt
        output_length: Number of tokens to generate
        warmup_runs: Number of warmup iterations
        benchmark_runs: Number of measured iterations
        device: Device to run on
        
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
        with torch.inference_mode():
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
    for _ in tqdm(range(benchmark_runs), desc="📊 Decode latency", unit="run"):
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
        
        total_time = time.perf_counter() - start
        num_generated = outputs.shape[1] - inputs["input_ids"].shape[1]
        per_token_latency = (total_time * 1000) / num_generated
        latencies.append(per_token_latency)
    
    avg_latency = sum(latencies) / len(latencies)
    logger.info(f"Decode latency: {avg_latency:.2f}ms per token")
    
    return avg_latency



