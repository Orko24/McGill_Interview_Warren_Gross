"""
Modal serverless GPU runner for quantization experiments.

Usage:
    # Run quick comparison
    modal run code/infra/modal_app.py
    
    # Run specific experiment
    modal run code/infra/modal_app.py --experiment bnb_4bit_nf4
    
    # Run all experiments
    modal run code/infra/modal_app.py --all

Requires:
    - Modal account and CLI authenticated
    - Modal secret: huggingface-secret with HF_TOKEN
"""

import json
import os
import sys
from datetime import datetime
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

# Add local source code (llama_quant package is inside code/)
image = image.add_local_python_source("code")

# Persistent volume for results and model cache
volume = modal.Volume.from_name("llama-quant-cache", create_if_missing=True)

# HuggingFace token - reads from Modal secret store
# Create with: modal secret create huggingface-secret HF_TOKEN=hf_your_token
hf_secret = modal.Secret.from_name("huggingface-secret")

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
    secrets=[hf_secret],
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
    sys.path.insert(0, "/root/code")
    
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
    secrets=[hf_secret],
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
    
    sys.path.insert(0, "/root/code")
    
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
            f1_str = f"{f1:.4f}" if isinstance(f1, (int, float)) else str(f1)
            size_str = f"{size:.2f}" if isinstance(size, (int, float)) else str(size)
            print(f"{name}: F1={f1_str}, Size={size_str} MB")
        else:
            print(f"{name}: {status} - {r.get('error', 'Unknown error')}")
    
    return {"experiments": all_results}


# =============================================================================
# Local Results Management
# =============================================================================

def get_next_results_filename(results_dir: Path) -> Path:
    """Get next available results filename (results1.json, results2.json, etc.)"""
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Find existing results files
    existing = list(results_dir.glob("results*.json"))
    existing_nums = []
    for f in existing:
        name = f.stem  # e.g., "results1", "results2"
        try:
            num = int(name.replace("results", ""))
            existing_nums.append(num)
        except ValueError:
            pass
    
    # Get next number
    next_num = max(existing_nums, default=0) + 1
    return results_dir / f"results{next_num}.json"


def save_results_locally(results: dict, results_dir: Path) -> Path:
    """Save results to local JSON file."""
    output_path = get_next_results_filename(results_dir)
    
    # Add metadata
    results["timestamp"] = datetime.now().isoformat()
    results["saved_to"] = str(output_path)
    
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    
    return output_path


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
    # Local results directory (relative to where modal run is called)
    local_results_dir = Path(__file__).parent.parent.parent / "results"
    
    if all_experiments:
        experiments = [
            "fp16_baseline",
            "bnb_4bit_nf4",
            "bnb_4bit_fp4",
            "bnb_4bit_nf4_no_double",
            "bnb_4bit_nf4_bf16",
        ]
        results = run_comparison.remote(experiments=experiments, limit=limit)
        
    elif experiment:
        results = run_single_experiment.remote(experiment, limit=limit)
        # Wrap single experiment result for consistency
        results = {"experiments": [results], "run_type": "single"}
        
    else:
        # Default: quick comparison
        results = run_comparison.remote(limit=limit)
    
    # Save results locally
    results["limit"] = limit
    results["run_type"] = "all" if all_experiments else ("single" if experiment else "comparison")
    output_path = save_results_locally(results, local_results_dir)
    
    print(f"\n{'='*60}")
    print(f"Results saved to: {output_path}")
    print(f"{'='*60}")

