"""
Modal serverless GPU runner for quantization experiments.

Usage:
    # Run quick comparison
    modal run infra/modal_app.py
    
    # Run specific experiment
    modal run infra/modal_app.py --experiment bnb_4bit_nf4
    
    # Run all experiments
    modal run infra/modal_app.py --all

Requires:
    - Modal account and CLI authenticated
    - HuggingFace token as Modal secret (HF_TOKEN)
"""

import os
import sys
from pathlib import Path

import modal

# =============================================================================
# Modal Configuration
# =============================================================================

app = modal.App("llama-quantization")

# GPU image with all dependencies
image = (
    modal.Image.from_registry("nvidia/cuda:12.1.0-devel-ubuntu22.04", add_python="3.11")
    .apt_install("git")
    .pip_install(
        "torch>=2.1.0",
        "transformers>=4.36.0",
        "accelerate>=0.25.0",
        "datasets>=2.16.0",
        "bitsandbytes>=0.43.0",
        "optimum>=1.15.0",
        "lm-eval>=0.4.0",
        "sentencepiece>=0.1.99",
        "safetensors>=0.4.0",
        "huggingface-hub>=0.20.0",
        "tqdm>=4.66.0",
        "numpy<2.0.0",  # Pin for compatibility
        "pyyaml>=6.0",
    )
)

# Add local source code
image = image.add_local_python_source("src")

# Persistent volume for results and model cache
volume = modal.Volume.from_name("llama-quant-cache", create_if_missing=True)

CACHE_DIR = "/cache"
RESULTS_DIR = "/cache/results"
HF_CACHE_DIR = "/cache/huggingface"


# =============================================================================
# GPU Functions
# =============================================================================

@app.function(
    image=image,
    gpu="A10G",
    timeout=3600,
    secrets=[modal.Secret.from_name("huggingface-secret")],
    volumes={CACHE_DIR: volume},
)
def run_single_experiment(experiment_name: str, limit: int = None) -> dict:
    """
    Run a single quantization experiment on GPU.
    
    Args:
        experiment_name: Name of experiment (e.g., "bnb_4bit_nf4")
        limit: Optional limit on eval samples
        
    Returns:
        Dictionary with experiment results
    """
    import json
    import torch
    
    # Setup HuggingFace cache
    os.environ["HF_HOME"] = HF_CACHE_DIR
    os.environ["TRANSFORMERS_CACHE"] = HF_CACHE_DIR
    
    # Import after setting up path
    sys.path.insert(0, "/root/src")
    
    from llama_quant.core.config import get_experiment
    from llama_quant.models import load_model, get_model_size_mb
    from llama_quant.evaluation import evaluate_model
    from llama_quant.benchmark import run_benchmarks
    from llama_quant.utils.serialization import save_results
    
    print(f"Running experiment: {experiment_name}")
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"CUDA Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # Get config
    config = get_experiment(experiment_name)
    config.output_dir = RESULTS_DIR
    
    if limit is not None:
        config.eval.limit = limit
    
    results = {
        "experiment_name": experiment_name,
        "quantization_method": config.quantization.method.value,
    }
    
    try:
        # Load model
        print("Loading model...")
        model, tokenizer = load_model(config)
        results["model_size_mb"] = get_model_size_mb(model)
        print(f"Model size: {results['model_size_mb']:.2f} MB")
        
        # Run evaluation
        print("Running evaluation...")
        eval_results = evaluate_model(model, tokenizer, config)
        results["coqa_metrics"] = eval_results.get("coqa_metrics", {})
        print(f"CoQA F1: {results['coqa_metrics'].get('coqa_f1', 'N/A')}")
        
        # Run benchmarks
        print("Running benchmarks...")
        benchmark_results = run_benchmarks(model, tokenizer, config)
        results["benchmarks"] = benchmark_results
        
        results["status"] = "success"
        
    except Exception as e:
        print(f"Error: {e}")
        results["status"] = "error"
        results["error"] = str(e)
    
    # Save results
    os.makedirs(RESULTS_DIR, exist_ok=True)
    output_path = f"{RESULTS_DIR}/{experiment_name}_results.json"
    save_results(results, output_path)
    
    # Commit volume
    volume.commit()
    
    return results


@app.function(
    image=image,
    gpu="A10G",
    timeout=7200,
    secrets=[modal.Secret.from_name("huggingface-secret")],
    volumes={CACHE_DIR: volume},
)
def run_comparison(
    experiments: list = None,
    limit: int = 100,
) -> dict:
    """
    Run multiple experiments and compare results.
    
    Args:
        experiments: List of experiment names (default: core experiments)
        limit: Eval sample limit per experiment
        
    Returns:
        Comparison results dictionary
    """
    import json
    import torch
    
    os.environ["HF_HOME"] = HF_CACHE_DIR
    os.environ["TRANSFORMERS_CACHE"] = HF_CACHE_DIR
    
    sys.path.insert(0, "/root/src")
    
    from llama_quant.core.config import get_experiment, get_all_experiments
    from llama_quant.models import load_model, get_model_size_mb
    from llama_quant.evaluation import evaluate_model
    from llama_quant.benchmark import run_benchmarks
    from llama_quant.utils.serialization import save_results
    
    if experiments is None:
        # Default comparison: FP16 vs main quantization methods
        experiments = ["fp16_baseline", "bnb_4bit_nf4", "bnb_4bit_fp4"]
    
    print(f"Running comparison: {experiments}")
    print(f"GPU: {torch.cuda.get_device_name()}")
    
    all_results = []
    
    for exp_name in experiments:
        print(f"\n{'='*60}")
        print(f"Experiment: {exp_name}")
        print(f"{'='*60}\n")
        
        try:
            config = get_experiment(exp_name)
            config.eval.limit = limit
            
            model, tokenizer = load_model(config)
            
            results = {
                "experiment_name": exp_name,
                "model_size_mb": get_model_size_mb(model),
            }
            
            # Evaluation
            eval_results = evaluate_model(model, tokenizer, config, skip_sanity_check=True)
            results["coqa_metrics"] = eval_results.get("coqa_metrics", {})
            
            # Quick benchmark (reduced iterations)
            config.benchmark.benchmark_runs = 5
            config.benchmark.warmup_runs = 2
            benchmark_results = run_benchmarks(model, tokenizer, config)
            results["benchmarks"] = benchmark_results
            
            results["status"] = "success"
            
            # Cleanup
            del model
            del tokenizer
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"Error in {exp_name}: {e}")
            results = {
                "experiment_name": exp_name,
                "status": "error",
                "error": str(e),
            }
        
        all_results.append(results)
        
        # Save incrementally
        os.makedirs(RESULTS_DIR, exist_ok=True)
        save_results(all_results, f"{RESULTS_DIR}/comparison_results.json")
        volume.commit()
    
    # Print summary
    print("\n" + "="*60)
    print("COMPARISON SUMMARY")
    print("="*60)
    
    for r in all_results:
        name = r.get("experiment_name", "unknown")
        status = r.get("status", "unknown")
        
        if status == "success":
            f1 = r.get("coqa_metrics", {}).get("coqa_f1", "N/A")
            size = r.get("model_size_mb", "N/A")
            print(f"{name}: F1={f1:.4f if isinstance(f1, float) else f1}, Size={size:.2f if isinstance(size, float) else size} MB")
        else:
            print(f"{name}: {status} - {r.get('error', 'Unknown error')}")
    
    return {"experiments": all_results}


# =============================================================================
# CLI Entry Point
# =============================================================================

@app.local_entrypoint()
def main(
    experiment: str = None,
    all_experiments: bool = False,
    limit: int = 100,
):
    """
    Run quantization experiments on Modal.
    
    Args:
        experiment: Single experiment name to run
        all_experiments: Run all experiments
        limit: Eval sample limit
    """
    if all_experiments:
        experiments = [
            "fp16_baseline",
            "bnb_4bit_nf4",
            "bnb_4bit_fp4",
            "bnb_4bit_nf4_no_double",
            "bnb_4bit_nf4_bf16",
        ]
        results = run_comparison.remote(experiments=experiments, limit=limit)
        print(f"\nResults saved to Modal volume")
        
    elif experiment:
        results = run_single_experiment.remote(experiment, limit=limit)
        print(f"\nResults: {results}")
        
    else:
        # Default: quick comparison
        results = run_comparison.remote(limit=limit)
        print(f"\nResults saved to Modal volume")

