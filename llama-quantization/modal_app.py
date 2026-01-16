"""
Modal App for Llama 3.2-1B Quantization Experiments

Run experiments on cloud GPUs without any VM setup.

Usage:
    # Setup (one time)
    pip install modal
    modal setup  # Authenticate via browser

    # Run single experiment
    modal run modal_app.py --experiment fp16_baseline --limit 100
    
    # Run quick comparison (FP16, 8bit, 4bit)
    modal run modal_app.py --quick --limit 50
    
    # Run full sweep
    modal run modal_app.py --sweep --method bnb --limit 200
"""

import modal
import json
from pathlib import Path

# =============================================================================
# MODAL APP SETUP
# =============================================================================

# Define the Docker image with all dependencies
image = modal.Image.debian_slim(python_version="3.11").pip_install(
    # Core ML
    "torch>=2.1.0",
    "transformers>=4.36.0",
    "accelerate>=0.25.0",
    "datasets>=2.16.0",
    # Quantization
    "bitsandbytes>=0.41.0",
    # Evaluation
    "lm-eval>=0.4.0",
    # Utilities
    "tqdm>=4.66.0",
    "numpy>=1.24.0",
    "pandas>=2.0.0",
    "sentencepiece>=0.1.99",
).env({
    "HF_HOME": "/cache/huggingface",
    "TRANSFORMERS_CACHE": "/cache/huggingface",
})

# Create the Modal app
app = modal.App("llama-quantization", image=image)

# Persistent volume for caching models and results
volume = modal.Volume.from_name("llama-quant-cache", create_if_missing=True)

# HuggingFace token secret (set this in Modal dashboard)
# modal secret create huggingface HF_TOKEN=<your_token>
hf_secret = modal.Secret.from_name("huggingface", required_keys=["HF_TOKEN"])


# =============================================================================
# EXPERIMENT CODE (runs on GPU)
# =============================================================================

@app.function(
    gpu="A10G",  # 24GB VRAM, good for 1B model
    timeout=3600,  # 1 hour max
    volumes={"/cache": volume},
    secrets=[hf_secret],
)
def run_single_experiment(
    experiment_name: str,
    limit: int = None,
) -> dict:
    """
    Run a single quantization experiment on Modal GPU.
    
    Args:
        experiment_name: One of the predefined experiments from config.py
        limit: Number of eval samples (None = full dataset)
        
    Returns:
        Dictionary with all results
    """
    import os
    import torch
    from huggingface_hub import login
    
    # Login to HuggingFace
    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        login(token=hf_token)
    
    # Import after login
    from config import EXPERIMENTS
    from quantize import load_quantized_model, get_model_memory_footprint
    from evaluate import evaluate_model
    from benchmark import run_benchmarks
    
    print(f"=" * 60)
    print(f"Running experiment: {experiment_name}")
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    print(f"=" * 60)
    
    # Get config
    if experiment_name not in EXPERIMENTS:
        raise ValueError(f"Unknown experiment: {experiment_name}. "
                        f"Available: {list(EXPERIMENTS.keys())}")
    
    config = EXPERIMENTS[experiment_name]
    
    # Apply limit
    if limit is not None:
        config.eval.limit = limit
    
    results = {
        "experiment_name": experiment_name,
        "quantization_method": config.quantization.method.value,
    }
    
    # Step 1: Load model
    print("\n[1/3] Loading model...")
    try:
        model, tokenizer = load_quantized_model(config)
        memory = get_model_memory_footprint(model)
        results["memory_footprint_mb"] = memory.get("total_mb", 0)
        print(f"Model loaded. Memory: {results['memory_footprint_mb']:.1f} MB")
    except Exception as e:
        print(f"ERROR loading model: {e}")
        results["error"] = str(e)
        return results
    
    # Step 2: Run evaluation
    print("\n[2/3] Running CoQA evaluation...")
    try:
        eval_results = evaluate_model(model, tokenizer, config)
        results["evaluation"] = eval_results
        
        if "coqa_metrics" in eval_results:
            metrics = eval_results["coqa_metrics"]
            print(f"CoQA F1: {metrics.get('coqa_f1', 'N/A')}")
            print(f"CoQA EM: {metrics.get('coqa_em', 'N/A')}")
    except Exception as e:
        print(f"ERROR in evaluation: {e}")
        results["eval_error"] = str(e)
    
    # Step 3: Run benchmarks
    print("\n[3/3] Running benchmarks...")
    try:
        bench_results = run_benchmarks(model, tokenizer, config)
        results["benchmarks"] = bench_results
        print(f"Peak memory: {bench_results.get('memory_peak_mb', 'N/A')} MB")
    except Exception as e:
        print(f"ERROR in benchmarks: {e}")
        results["benchmark_error"] = str(e)
    
    # Cleanup
    del model
    del tokenizer
    torch.cuda.empty_cache()
    
    print(f"\n{'=' * 60}")
    print(f"Experiment {experiment_name} complete!")
    print(f"{'=' * 60}")
    
    return results


@app.function(
    gpu="A10G",
    timeout=7200,  # 2 hours for sweep
    volumes={"/cache": volume},
    secrets=[hf_secret],
)
def run_quick_comparison(limit: int = 50) -> dict:
    """
    Run FP16, 8-bit, and 4-bit comparison.
    This is the minimum viable experiment set.
    """
    import torch
    import os
    from huggingface_hub import login
    
    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        login(token=hf_token)
    
    from config import EXPERIMENTS, ExperimentConfig, QuantizationConfig, QuantMethod
    from quantize import load_quantized_model, get_model_memory_footprint
    from evaluate import evaluate_model
    from benchmark import run_benchmarks
    
    experiments = ["fp16_baseline", "bnb_8bit", "bnb_4bit_nf4"]
    all_results = []
    
    for exp_name in experiments:
        print(f"\n{'=' * 60}")
        print(f"Running: {exp_name}")
        print(f"{'=' * 60}")
        
        config = EXPERIMENTS[exp_name]
        config.eval.limit = limit
        
        result = {"experiment_name": exp_name}
        
        try:
            model, tokenizer = load_quantized_model(config)
            result["memory_mb"] = get_model_memory_footprint(model).get("total_mb", 0)
            
            eval_results = evaluate_model(model, tokenizer, config)
            result["evaluation"] = eval_results
            
            bench_results = run_benchmarks(model, tokenizer, config)
            result["benchmarks"] = bench_results
            
            del model, tokenizer
            torch.cuda.empty_cache()
            
        except Exception as e:
            result["error"] = str(e)
        
        all_results.append(result)
        
        # Print summary
        if "evaluation" in result and "coqa_metrics" in result["evaluation"]:
            f1 = result["evaluation"]["coqa_metrics"].get("coqa_f1", "N/A")
            print(f"  CoQA F1: {f1}")
        if "memory_mb" in result:
            print(f"  Memory: {result['memory_mb']:.1f} MB")
    
    return {
        "experiments": experiments,
        "results": all_results,
        "summary": create_summary(all_results),
    }


def create_summary(results: list) -> dict:
    """Create a summary table from results."""
    summary = []
    for r in results:
        entry = {"name": r.get("experiment_name", "unknown")}
        
        if "evaluation" in r and "coqa_metrics" in r["evaluation"]:
            entry["coqa_f1"] = r["evaluation"]["coqa_metrics"].get("coqa_f1")
            entry["coqa_em"] = r["evaluation"]["coqa_metrics"].get("coqa_em")
        
        if "memory_mb" in r:
            entry["memory_mb"] = r["memory_mb"]
        
        if "benchmarks" in r:
            entry["peak_memory_mb"] = r["benchmarks"].get("memory_peak_mb")
        
        summary.append(entry)
    
    return summary


# =============================================================================
# LOCAL ENTRYPOINT (runs on your machine)
# =============================================================================

@app.local_entrypoint()
def main(
    experiment: str = None,
    quick: bool = False,
    sweep: bool = False,
    method: str = "bnb",
    limit: int = None,
    output: str = "./results",
):
    """
    Local entrypoint - dispatches to Modal GPU functions.
    
    Args:
        experiment: Run single experiment by name
        quick: Run quick comparison (FP16, 8bit, 4bit)
        sweep: Run full sweep
        method: For sweep, which method (bnb, gptq, awq, all)
        limit: Eval sample limit
        output: Output directory for results
    """
    import os
    
    output_dir = Path(output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if quick:
        print("Running quick comparison (FP16, 8-bit, 4-bit NF4)...")
        results = run_quick_comparison.remote(limit=limit or 50)
        
        # Save results
        output_file = output_dir / "quick_comparison.json"
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\nResults saved to {output_file}")
        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        for entry in results.get("summary", []):
            print(f"{entry['name']:<20} F1: {entry.get('coqa_f1', 'N/A'):<10} Memory: {entry.get('memory_mb', 'N/A')}")
        
    elif experiment:
        print(f"Running single experiment: {experiment}")
        results = run_single_experiment.remote(
            experiment_name=experiment,
            limit=limit,
        )
        
        # Save results
        output_file = output_dir / f"{experiment}_results.json"
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\nResults saved to {output_file}")
        
        # Print summary
        if "evaluation" in results and "coqa_metrics" in results["evaluation"]:
            metrics = results["evaluation"]["coqa_metrics"]
            print(f"\nCoQA F1: {metrics.get('coqa_f1', 'N/A')}")
            print(f"CoQA EM: {metrics.get('coqa_em', 'N/A')}")
        
    elif sweep:
        print(f"Running {method} sweep...")
        # For sweep, we'll run experiments sequentially
        from config import EXPERIMENTS
        
        if method == "bnb":
            exp_names = ["fp16_baseline", "bnb_8bit", "bnb_4bit_nf4", "bnb_4bit_fp4"]
        elif method == "gptq":
            exp_names = ["fp16_baseline", "gptq_4bit_g128", "gptq_4bit_g32", "gptq_3bit_g128"]
        elif method == "awq":
            exp_names = ["fp16_baseline", "awq_4bit_g128"]
        else:
            exp_names = list(EXPERIMENTS.keys())
        
        all_results = []
        for exp_name in exp_names:
            print(f"\n--- Running {exp_name} ---")
            try:
                result = run_single_experiment.remote(
                    experiment_name=exp_name,
                    limit=limit,
                )
                all_results.append(result)
            except Exception as e:
                print(f"Failed: {e}")
                all_results.append({"experiment_name": exp_name, "error": str(e)})
        
        # Save results
        output_file = output_dir / f"sweep_{method}.json"
        with open(output_file, "w") as f:
            json.dump(all_results, f, indent=2, default=str)
        
        print(f"\nSweep results saved to {output_file}")
        
    else:
        print("Usage:")
        print("  modal run modal_app.py --quick --limit 50")
        print("  modal run modal_app.py --experiment fp16_baseline --limit 100")
        print("  modal run modal_app.py --sweep --method bnb --limit 200")
        print("\nAvailable experiments:")
        from config import EXPERIMENTS
        for name in EXPERIMENTS.keys():
            print(f"  - {name}")


# =============================================================================
# STANDALONE FUNCTIONS FOR TESTING
# =============================================================================

@app.function(gpu="A10G", secrets=[hf_secret], volumes={"/cache": volume})
def test_gpu():
    """Quick test that GPU is working."""
    import torch
    import os
    from huggingface_hub import login
    
    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        login(token=hf_token)
        print("HuggingFace login successful")
    
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    return {"status": "ok", "gpu": torch.cuda.get_device_name()}


if __name__ == "__main__":
    # For local testing without Modal
    print("Run with: modal run modal_app.py --help")

