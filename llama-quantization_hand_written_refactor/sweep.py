"""
Systematic hyperparameter sweeps for quantization experiments

This script runs a grid search over quantization configurations
to find the optimal accuracy/compression tradeoff.
"""

import itertools
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any
from dataclasses import replace

import torch

from config import (
    ExperimentConfig,
    QuantizationConfig,
    ModelConfig,
    EvalConfig,
    BenchmarkConfig,
    QuantMethod,
    ComputeDtype,
)
from main import run_experiment

logger = logging.getLogger(__name__)


# ============================================================================
# HYPERPARAMETER SEARCH SPACE
# ============================================================================

# BitsAndBytes 4-bit configurations
BNB_4BIT_CONFIGS = {
    "quant_type": ["nf4", "fp4"],
    "double_quant": [True, False],
    "compute_dtype": [ComputeDtype.FP16, ComputeDtype.BF16],
}

# GPTQ configurations
GPTQ_CONFIGS = {
    "bits": [2, 3, 4, 8],
    "group_size": [32, 64, 128, -1],  # -1 = per-column
    "desc_act": [True, False],
    "sym": [True, False],
}

# AWQ configurations
AWQ_CONFIGS = {
    "bits": [4],
    "group_size": [32, 64, 128],
    "zero_point": [True, False],
}


def generate_bnb_configs() -> List[ExperimentConfig]:
    """Generate all BitsAndBytes 4-bit configurations"""
    configs = []
    
    for quant_type, double_quant, compute_dtype in itertools.product(
        BNB_4BIT_CONFIGS["quant_type"],
        BNB_4BIT_CONFIGS["double_quant"],
        BNB_4BIT_CONFIGS["compute_dtype"],
    ):
        name = f"bnb_4bit_{quant_type}_dq{int(double_quant)}_{compute_dtype.value}"
        
        config = ExperimentConfig(
            name=name,
            quantization=QuantizationConfig(
                method=QuantMethod.BITSANDBYTES_4BIT,
                bnb_4bit_quant_type=quant_type,
                bnb_4bit_use_double_quant=double_quant,
                bnb_4bit_compute_dtype=compute_dtype,
            ),
        )
        configs.append(config)
    
    # Also add 8-bit baseline
    configs.append(ExperimentConfig(
        name="bnb_8bit",
        quantization=QuantizationConfig(method=QuantMethod.BITSANDBYTES_8BIT),
    ))
    
    return configs


def generate_gptq_configs() -> List[ExperimentConfig]:
    """Generate GPTQ configurations"""
    configs = []
    
    # Don't enumerate all combinations (too many), pick strategic ones
    strategic_configs = [
        (4, 128, False, True),   # Standard 4-bit
        (4, 64, False, True),    # Smaller group
        (4, 32, False, True),    # Even smaller group
        (4, 128, True, True),    # With desc_act
        (3, 128, False, True),   # 3-bit aggressive
        (3, 64, False, True),    # 3-bit smaller group
        (8, 128, False, True),   # 8-bit high quality
        (2, 128, False, True),   # 2-bit extreme (likely poor quality)
    ]
    
    for bits, group_size, desc_act, sym in strategic_configs:
        name = f"gptq_{bits}bit_g{group_size}_da{int(desc_act)}_sym{int(sym)}"
        
        config = ExperimentConfig(
            name=name,
            quantization=QuantizationConfig(
                method=QuantMethod.GPTQ,
                gptq_bits=bits,
                gptq_group_size=group_size,
                gptq_desc_act=desc_act,
                gptq_sym=sym,
            ),
        )
        configs.append(config)
    
    return configs


def generate_awq_configs() -> List[ExperimentConfig]:
    """Generate AWQ configurations"""
    configs = []
    
    for bits, group_size, zero_point in itertools.product(
        AWQ_CONFIGS["bits"],
        AWQ_CONFIGS["group_size"],
        AWQ_CONFIGS["zero_point"],
    ):
        name = f"awq_{bits}bit_g{group_size}_zp{int(zero_point)}"
        
        config = ExperimentConfig(
            name=name,
            quantization=QuantizationConfig(
                method=QuantMethod.AWQ,
                awq_bits=bits,
                awq_group_size=group_size,
                awq_zero_point=zero_point,
            ),
        )
        configs.append(config)
    
    return configs


def generate_all_configs() -> List[ExperimentConfig]:
    """Generate all experiment configurations"""
    configs = []
    
    # FP16 baseline
    configs.append(ExperimentConfig(
        name="fp16_baseline",
        quantization=QuantizationConfig(method=QuantMethod.NONE),
    ))
    
    # BitsAndBytes configs
    configs.extend(generate_bnb_configs())
    
    # GPTQ configs
    configs.extend(generate_gptq_configs())
    
    # AWQ configs
    configs.extend(generate_awq_configs())
    
    return configs


def run_sweep(
    configs: List[ExperimentConfig] = None,
    eval_limit: int = None,
    output_dir: str = "./sweep_results",
) -> Dict[str, Any]:
    """
    Run a sweep over multiple configurations
    
    Args:
        configs: List of configs to run. If None, generates all.
        eval_limit: Limit eval samples (for faster iteration)
        output_dir: Where to save results
        
    Returns:
        Summary of all results
    """
    if configs is None:
        configs = generate_all_configs()
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Running sweep over {len(configs)} configurations")
    
    all_results = []
    
    for i, config in enumerate(configs):
        logger.info(f"\n{'='*60}")
        logger.info(f"Configuration {i+1}/{len(configs)}: {config.name}")
        logger.info(f"{'='*60}")
        
        # Apply eval limit
        if eval_limit is not None:
            config.eval.limit = eval_limit
        
        config.output_dir = str(output_path)
        
        try:
            result = run_experiment(config)
            all_results.append(result)
            
            # Save intermediate results
            intermediate_file = output_path / f"intermediate_{datetime.now():%Y%m%d_%H%M%S}.json"
            with open(intermediate_file, "w") as f:
                json.dump(all_results, f, indent=2, default=str)
                
        except Exception as e:
            logger.error(f"Config {config.name} failed: {e}")
            all_results.append({
                "experiment_name": config.name,
                "error": str(e),
            })
        
        # Clear memory between experiments
        torch.cuda.empty_cache()
    
    # Create summary
    summary = create_sweep_summary(all_results)
    
    # Save final results
    final_file = output_path / f"sweep_final_{datetime.now():%Y%m%d_%H%M%S}.json"
    with open(final_file, "w") as f:
        json.dump({
            "summary": summary,
            "all_results": all_results,
        }, f, indent=2, default=str)
    
    logger.info(f"\nSweep complete! Results saved to {final_file}")
    
    return summary


def create_sweep_summary(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Create a summary of sweep results"""
    summary = {
        "total_configs": len(results),
        "successful": 0,
        "failed": 0,
        "results": [],
    }
    
    for r in results:
        if "error" in r:
            summary["failed"] += 1
            continue
        
        summary["successful"] += 1
        
        entry = {
            "name": r.get("experiment_name"),
            "method": r.get("quantization_method"),
        }
        
        # Extract accuracy
        if "evaluation" in r and "coqa_metrics" in r["evaluation"]:
            entry.update(r["evaluation"]["coqa_metrics"])
        
        # Extract benchmarks
        if "benchmarks" in r:
            bench = r["benchmarks"]
            entry["model_size_mb"] = bench.get("model_size_mb")
            entry["memory_peak_mb"] = bench.get("memory_peak_mb")
            
            if "throughput_tokens_per_sec" in bench:
                # Get best throughput
                entry["max_throughput"] = max(bench["throughput_tokens_per_sec"].values())
        
        summary["results"].append(entry)
    
    # Sort by accuracy (descending)
    summary["results"].sort(
        key=lambda x: x.get("coqa_f1", 0),
        reverse=True,
    )
    
    return summary


def print_sweep_summary(summary: Dict[str, Any]):
    """Print a formatted sweep summary"""
    print("\n" + "=" * 100)
    print("SWEEP SUMMARY")
    print("=" * 100)
    print(f"Total: {summary['total_configs']} | Successful: {summary['successful']} | Failed: {summary['failed']}")
    print("-" * 100)
    print(f"{'Name':<35} {'Method':<15} {'CoQA F1':<10} {'Size MB':<10} {'Peak MB':<10} {'TPS':<10}")
    print("-" * 100)
    
    for r in summary["results"]:
        name = r.get("name", "?")[:34]
        method = r.get("method", "?")[:14]
        f1 = r.get("coqa_f1", "N/A")
        size = r.get("model_size_mb", "N/A")
        peak = r.get("memory_peak_mb", "N/A")
        tps = r.get("max_throughput", "N/A")
        
        f1_str = f"{f1:.4f}" if isinstance(f1, float) else str(f1)
        size_str = f"{size:.1f}" if isinstance(size, float) else str(size)
        peak_str = f"{peak:.1f}" if isinstance(peak, float) else str(peak)
        tps_str = f"{tps:.1f}" if isinstance(tps, float) else str(tps)
        
        print(f"{name:<35} {method:<15} {f1_str:<10} {size_str:<10} {peak_str:<10} {tps_str:<10}")
    
    print("=" * 100 + "\n")


# ============================================================================
# QUICK EXPERIMENTS FOR ITERATION
# ============================================================================

def run_quick_comparison(eval_limit: int = 50):
    """
    Run a quick comparison of main quantization methods
    Good for rapid iteration
    """
    configs = [
        ExperimentConfig(
            name="fp16_baseline",
            quantization=QuantizationConfig(method=QuantMethod.NONE),
        ),
        ExperimentConfig(
            name="bnb_8bit",
            quantization=QuantizationConfig(method=QuantMethod.BITSANDBYTES_8BIT),
        ),
        ExperimentConfig(
            name="bnb_4bit_nf4",
            quantization=QuantizationConfig(
                method=QuantMethod.BITSANDBYTES_4BIT,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            ),
        ),
    ]
    
    return run_sweep(configs, eval_limit=eval_limit, output_dir="./quick_results")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run quantization sweep")
    parser.add_argument("--quick", action="store_true", help="Run quick comparison only")
    parser.add_argument("--limit", type=int, default=None, help="Eval sample limit")
    parser.add_argument("--method", choices=["bnb", "gptq", "awq", "all"], default="all")
    parser.add_argument("--output", type=str, default="./sweep_results")
    
    args = parser.parse_args()
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    
    if args.quick:
        summary = run_quick_comparison(eval_limit=args.limit or 50)
    else:
        # Generate configs based on method
        if args.method == "bnb":
            configs = generate_bnb_configs()
        elif args.method == "gptq":
            configs = generate_gptq_configs()
        elif args.method == "awq":
            configs = generate_awq_configs()
        else:
            configs = generate_all_configs()
        
        summary = run_sweep(configs, eval_limit=args.limit, output_dir=args.output)
    
    print_sweep_summary(summary)
