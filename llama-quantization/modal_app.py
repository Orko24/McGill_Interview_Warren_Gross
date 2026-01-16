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
# Use standard debian_slim - simpler and more compatible
image = modal.Image.debian_slim(python_version="3.11").pip_install(
    # Pin NumPy to 1.x to avoid compatibility issues
    "numpy<2.0.0",
    # Core ML
    "torch>=2.1.0",
    "transformers>=4.36.0",
    "accelerate>=0.25.0",
    "datasets>=2.16.0",
    # Quantization
    "bitsandbytes>=0.43.0",
    # Evaluation
    "lm-eval>=0.4.0",
    # Utilities
    "tqdm>=4.66.0",
    "pandas>=2.0.0",
    "sentencepiece>=0.1.99",
).env({
    "HF_HOME": "/cache/huggingface",
    "TRANSFORMERS_CACHE": "/cache/huggingface",
    "BITSANDBYTES_NOWELCOME": "1",
})

# Add local Python files to the image
image = image.add_local_python_source("config", "quantize", "evaluate", "benchmark", "main", "sweep")

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
    import sys
    import torch
    from huggingface_hub import login
    
    # Add mounted code to path
    sys.path.insert(0, "/root")
    
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
    import sys
    from huggingface_hub import login
    
    # Add mounted code to path
    sys.path.insert(0, "/root")
    
    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        login(token=hf_token)
    
    from config import EXPERIMENTS, ExperimentConfig, QuantizationConfig, QuantMethod
    from quantize import load_quantized_model, get_model_memory_footprint
    from evaluate import evaluate_model
    from benchmark import run_benchmarks
    
    # Skip 8-bit - has persistent CUDA kernel bug on A10G
    # The LLM.int8() implementation has issues with this GPU architecture
    experiments = ["fp16_baseline", "bnb_4bit_nf4"]
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
            print(f"\n❌ EXPERIMENT FAILED: {exp_name}")
            print(f"   Error: {e}")
            print(f"\n🛑 FAIL-FAST: Stopping to save compute credits.")
            all_results.append(result)
            # Return early with partial results
            return {
                "experiments": experiments,
                "results": all_results,
                "summary": create_summary(all_results),
                "failed": True,
                "failed_experiment": exp_name,
            }
        
        all_results.append(result)
        
        # Print summary
        if "evaluation" in result and "coqa_metrics" in result["evaluation"]:
            f1 = result["evaluation"]["coqa_metrics"].get("coqa_f1", "N/A")
            print(f"  CoQA F1: {f1}")
        if "memory_mb" in result:
            print(f"  Memory: {result['memory_mb']:.1f} MB")
    
    print(f"\n✅ All {len(experiments)} experiments completed successfully!")
    
    # Save results to volume for persistence
    import json
    from datetime import datetime
    results_data = {
        "experiments": experiments,
        "results": all_results,
        "summary": create_summary(all_results),
        "failed": False,
        "timestamp": datetime.now().isoformat(),
    }
    
    # Convert any non-serializable objects to strings
    def make_serializable(obj):
        if isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [make_serializable(v) for v in obj]
        elif isinstance(obj, (int, float, str, bool, type(None))):
            return obj
        else:
            return str(obj)
    
    results_data = make_serializable(results_data)
    
    results_path = "/cache/results/quick_comparison.json"
    import os
    os.makedirs("/cache/results", exist_ok=True)
    with open(results_path, "w") as f:
        json.dump(results_data, f, indent=2)
    print(f"\n📁 Results saved to {results_path}")
    
    # Print summary table
    print("\n" + "=" * 60)
    print("FINAL RESULTS SUMMARY")
    print("=" * 60)
    for r in all_results:
        name = r.get("experiment_name", "unknown")
        mem = r.get("memory_mb", 0)
        f1 = r.get("evaluation", {}).get("coqa_metrics", {}).get("coqa_f1", "N/A")
        f1_str = f"{f1:.4f}" if isinstance(f1, (int, float)) else str(f1)
        print(f"{name:20s} | F1: {f1_str} | Memory: {mem:.1f} MB")
    print("=" * 60)
    
    return results_data


@app.function(
    gpu="A10G",
    timeout=7200,
    volumes={"/cache": volume},
    secrets=[hf_secret],
)
def run_extended_comparison(limit: int = 50) -> dict:
    """
    Extended comparison: FP16, BnB NF4, BnB FP4, QLoRA config
    Saves to results2.json
    """
    import torch
    import os
    import sys
    import json
    from datetime import datetime
    from huggingface_hub import login
    
    sys.path.insert(0, "/root")
    
    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        login(token=hf_token)
    
    from config import EXPERIMENTS, ExperimentConfig, QuantizationConfig, QuantMethod, ModelConfig
    from quantize import load_quantized_model, get_model_memory_footprint, load_tokenizer
    from evaluate import evaluate_model
    from benchmark import run_benchmarks
    
    # Extended experiment list
    experiments = [
        "fp16_baseline",
        "bnb_4bit_nf4", 
        "bnb_4bit_fp4",  # FP4 variant
    ]
    
    all_results = []
    
    for exp_name in experiments:
        print(f"\n{'=' * 60}")
        print(f"Running: {exp_name}")
        print(f"{'=' * 60}")
        
        config = EXPERIMENTS[exp_name]
        config.eval.limit = limit
        
        result = {
            "experiment_name": exp_name,
            "quantization_method": config.quantization.method.value,
        }
        
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
            import traceback
            result["error"] = str(e)
            result["traceback"] = traceback.format_exc()
            print(f"\n❌ EXPERIMENT FAILED: {exp_name}")
            print(f"   Error: {e}")
            print(f"\n🛑 FAIL-FAST: Stopping to save compute credits.")
            all_results.append(result)
            break
        
        all_results.append(result)
        
        # Print summary
        if "evaluation" in result and "coqa_metrics" in result["evaluation"]:
            f1 = result["evaluation"]["coqa_metrics"].get("coqa_f1", "N/A")
            f1_str = f"{f1:.4f}" if isinstance(f1, (int, float)) else str(f1)
            print(f"  CoQA F1: {f1_str}")
        if "memory_mb" in result:
            print(f"  Memory: {result['memory_mb']:.1f} MB")
    
    # Now add QLoRA memory estimate (load 4-bit + prepare for LoRA)
    print(f"\n{'=' * 60}")
    print("Running: QLoRA Memory Estimation")
    print("{'=' * 60}")
    
    try:
        from transformers import AutoModelForCausalLM, BitsAndBytesConfig
        
        # QLoRA config: 4-bit base model
        qlora_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        
        model = AutoModelForCausalLM.from_pretrained(
            "meta-llama/Llama-3.2-1B",
            quantization_config=qlora_config,
            device_map="auto",
            trust_remote_code=True,
        )
        
        # Estimate LoRA adapter overhead (typical r=16, alpha=32)
        # LoRA adds ~0.1-1% of base model params
        base_params = sum(p.numel() for p in model.parameters())
        lora_overhead_mb = (base_params * 0.01 * 4) / (1024 * 1024)  # ~1% params in FP32
        
        qlora_result = {
            "experiment_name": "qlora_ready",
            "quantization_method": "bnb_4bit + LoRA",
            "memory_mb": get_model_memory_footprint(model).get("total_mb", 0),
            "lora_overhead_mb_estimate": lora_overhead_mb,
            "note": "4-bit base ready for LoRA fine-tuning. LoRA adapters add ~1% params.",
        }
        
        del model
        torch.cuda.empty_cache()
        
        all_results.append(qlora_result)
        print(f"  Base Memory: {qlora_result['memory_mb']:.1f} MB")
        print(f"  LoRA Overhead (est): {lora_overhead_mb:.1f} MB")
        
    except Exception as e:
        qlora_result = {
            "experiment_name": "qlora_ready",
            "error": str(e),
        }
        all_results.append(qlora_result)
        print(f"  QLoRA estimation failed: {e}")
    
    print(f"\n✅ Extended comparison completed!")
    
    # Save results
    def make_serializable(obj):
        if isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [make_serializable(v) for v in obj]
        elif isinstance(obj, (int, float, str, bool, type(None))):
            return obj
        else:
            return str(obj)
    
    results_data = {
        "experiments": [r.get("experiment_name") for r in all_results],
        "results": make_serializable(all_results),
        "summary": create_summary(all_results),
        "timestamp": datetime.now().isoformat(),
        "hardware": {"gpu": "A10G", "platform": "Modal"},
    }
    
    results_path = "/cache/results/extended_comparison.json"
    os.makedirs("/cache/results", exist_ok=True)
    with open(results_path, "w") as f:
        json.dump(results_data, f, indent=2)
    print(f"\n📁 Results saved to {results_path}")
    
    # Print summary table
    print("\n" + "=" * 60)
    print("FINAL RESULTS SUMMARY")
    print("=" * 60)
    for r in all_results:
        name = r.get("experiment_name", "unknown")
        mem = r.get("memory_mb", 0)
        f1 = r.get("evaluation", {}).get("coqa_metrics", {}).get("coqa_f1", "N/A") if "evaluation" in r else "N/A"
        f1_str = f"{f1:.4f}" if isinstance(f1, (int, float)) else str(f1)
        print(f"{name:20s} | F1: {f1_str:10s} | Memory: {mem:.1f} MB")
    print("=" * 60)
    
    return results_data


@app.function(
    gpu="A10G",
    timeout=7200,
    volumes={"/cache": volume},
    secrets=[hf_secret],
)
def run_hyperparam_sweep(limit: int = 50) -> dict:
    """
    Hyperparameter ablation study:
    - Double quant ON vs OFF
    - NF4 vs FP4
    - FP16 vs BF16 compute dtype
    
    Saves to results3.json
    """
    import torch
    import os
    import sys
    import json
    from datetime import datetime
    from huggingface_hub import login
    
    sys.path.insert(0, "/root")
    
    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        login(token=hf_token)
    
    from config import EXPERIMENTS
    from quantize import load_quantized_model, get_model_memory_footprint
    from evaluate import evaluate_model
    from benchmark import run_benchmarks
    
    # Hyperparameter ablation experiments
    experiments = [
        "fp16_baseline",           # Reference
        "bnb_4bit_nf4",            # NF4 + double quant (best so far)
        "bnb_4bit_nf4_no_double",  # NF4 without double quant
        "bnb_4bit_nf4_bf16",       # NF4 + BF16 compute dtype
        "bnb_4bit_fp4",            # FP4 + double quant
        "bnb_4bit_fp4_no_double",  # FP4 without double quant
    ]
    
    all_results = []
    
    for exp_name in experiments:
        print(f"\n{'=' * 60}")
        print(f"Running: {exp_name}")
        print(f"{'=' * 60}")
        
        if exp_name not in EXPERIMENTS:
            print(f"  ⚠️ Experiment {exp_name} not found in config, skipping")
            continue
        
        config = EXPERIMENTS[exp_name]
        config.eval.limit = limit
        
        # Extract hyperparams for logging
        qc = config.quantization
        result = {
            "experiment_name": exp_name,
            "hyperparams": {
                "method": qc.method.value,
                "quant_type": qc.bnb_4bit_quant_type if "4bit" in exp_name else "N/A",
                "double_quant": qc.bnb_4bit_use_double_quant if "4bit" in exp_name else "N/A",
                "compute_dtype": qc.bnb_4bit_compute_dtype.value if "4bit" in exp_name else "N/A",
            },
        }
        
        try:
            model, tokenizer = load_quantized_model(config)
            result["memory_mb"] = get_model_memory_footprint(model).get("total_mb", 0)
            
            eval_results = evaluate_model(model, tokenizer, config)
            result["evaluation"] = eval_results
            
            # Skip benchmarks for speed (focus on accuracy comparison)
            # bench_results = run_benchmarks(model, tokenizer, config)
            # result["benchmarks"] = bench_results
            
            del model, tokenizer
            torch.cuda.empty_cache()
            
        except Exception as e:
            import traceback
            result["error"] = str(e)
            result["traceback"] = traceback.format_exc()
            print(f"\n❌ EXPERIMENT FAILED: {exp_name}")
            print(f"   Error: {e}")
            print(f"\n🛑 FAIL-FAST: Stopping to save compute credits.")
            all_results.append(result)
            break
        
        all_results.append(result)
        
        # Print summary
        if "evaluation" in result and "coqa_metrics" in result["evaluation"]:
            f1 = result["evaluation"]["coqa_metrics"].get("coqa_f1", "N/A")
            f1_str = f"{f1:.4f}" if isinstance(f1, (int, float)) else str(f1)
            print(f"  CoQA F1: {f1_str}")
        if "memory_mb" in result:
            print(f"  Memory: {result['memory_mb']:.1f} MB")
        print(f"  Hyperparams: {result['hyperparams']}")
        
        # === INCREMENTAL SAVE after each experiment ===
        def make_serializable_inner(obj):
            if isinstance(obj, dict):
                return {k: make_serializable_inner(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [make_serializable_inner(v) for v in obj]
            elif isinstance(obj, (int, float, str, bool, type(None))):
                return obj
            else:
                return str(obj)
        
        partial_results = {
            "title": "Hyperparameter Ablation Study (In Progress)",
            "completed_experiments": [r.get("experiment_name") for r in all_results],
            "total_planned": len(experiments),
            "results": make_serializable_inner(all_results),
            "summary": create_summary(all_results),
            "timestamp": datetime.now().isoformat(),
            "hardware": {"gpu": "A10G", "platform": "Modal"},
        }
        
        results_path = "/cache/results/hyperparam_sweep.json"
        os.makedirs("/cache/results", exist_ok=True)
        with open(results_path, "w") as f:
            json.dump(partial_results, f, indent=2)
        print(f"  💾 Incremental save: {len(all_results)}/{len(experiments)} experiments saved")
    
    print(f"\n✅ Hyperparameter sweep completed!")
    
    # Save results
    def make_serializable(obj):
        if isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [make_serializable(v) for v in obj]
        elif isinstance(obj, (int, float, str, bool, type(None))):
            return obj
        else:
            return str(obj)
    
    results_data = {
        "title": "Hyperparameter Ablation Study",
        "experiments": [r.get("experiment_name") for r in all_results],
        "results": make_serializable(all_results),
        "summary": create_summary(all_results),
        "timestamp": datetime.now().isoformat(),
        "hardware": {"gpu": "A10G", "platform": "Modal"},
        "ablation_factors": [
            "quant_type: nf4 vs fp4",
            "double_quant: True vs False",
            "compute_dtype: fp16 vs bf16",
        ],
    }
    
    results_path = "/cache/results/hyperparam_sweep.json"
    os.makedirs("/cache/results", exist_ok=True)
    with open(results_path, "w") as f:
        json.dump(results_data, f, indent=2)
    print(f"\n📁 Results saved to {results_path}")
    
    # Print summary table
    print("\n" + "=" * 70)
    print("HYPERPARAMETER ABLATION SUMMARY")
    print("=" * 70)
    print(f"{'Experiment':<25} {'Quant':<6} {'Double':<7} {'Dtype':<7} {'F1':<8} {'Memory':<10}")
    print("-" * 70)
    for r in all_results:
        name = r.get("experiment_name", "unknown")[:24]
        hp = r.get("hyperparams", {})
        qt = str(hp.get("quant_type", "N/A"))[:5]
        dq = "Yes" if hp.get("double_quant") == True else "No" if hp.get("double_quant") == False else "N/A"
        dt = str(hp.get("compute_dtype", "N/A"))[:6]
        f1 = r.get("evaluation", {}).get("coqa_metrics", {}).get("coqa_f1", "N/A") if "evaluation" in r else "ERR"
        f1_str = f"{f1:.4f}" if isinstance(f1, (int, float)) else str(f1)[:7]
        mem = r.get("memory_mb", 0)
        mem_str = f"{mem:.1f} MB" if isinstance(mem, (int, float)) else "N/A"
        print(f"{name:<25} {qt:<6} {dq:<7} {dt:<7} {f1_str:<8} {mem_str:<10}")
    print("=" * 70)
    
    return results_data


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
    extended: bool = False,
    hyperparam: bool = False,  # NEW: Hyperparameter ablation study
    sweep: bool = False,
    method: str = "bnb",
    limit: int = None,
    output: str = "./results",
):
    """
    Local entrypoint - dispatches to Modal GPU functions.
    
    Args:
        experiment: Run single experiment by name
        quick: Run quick comparison (FP16, 4bit NF4)
        extended: Run extended comparison (FP16, NF4, FP4, QLoRA)
        hyperparam: Run hyperparameter ablation (double_quant, compute_dtype, etc.)
        sweep: Run full sweep
        method: For sweep, which method (bnb, gptq, awq, all)
        limit: Eval sample limit
        output: Output directory for results
    """
    import os
    
    output_dir = Path(output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if extended:
        print("Running EXTENDED comparison (FP16, NF4, FP4, QLoRA)...")
        results = run_extended_comparison.remote(limit=limit or 50)
        
        # Save results locally as results2.json
        output_file = output_dir / "results2.json"
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\n📁 Results saved to {output_file}")
        print("\n" + "=" * 60)
        print("EXTENDED COMPARISON SUMMARY")
        print("=" * 60)
        for entry in results.get("summary", []):
            name = entry.get('name', 'unknown')
            f1 = entry.get('coqa_f1', 'N/A')
            mem = entry.get('memory_mb', 'N/A')
            f1_str = f"{f1:.4f}" if isinstance(f1, (int, float)) else str(f1)
            mem_str = f"{mem:.1f}" if isinstance(mem, (int, float)) else str(mem)
            print(f"{name:<20} F1: {f1_str:<10} Memory: {mem_str} MB")
        print("=" * 60)
    
    elif hyperparam:
        print("Running HYPERPARAMETER ablation (double_quant, compute_dtype, quant_type)...")
        results = run_hyperparam_sweep.remote(limit=limit or 50)
        
        # Save results locally as results3.json
        output_file = output_dir / "results3.json"
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\n📁 Results saved to {output_file}")
        print("\n" + "=" * 70)
        print("HYPERPARAMETER ABLATION SUMMARY")
        print("=" * 70)
        for entry in results.get("summary", []):
            name = entry.get('name', 'unknown')
            f1 = entry.get('coqa_f1', 'N/A')
            mem = entry.get('memory_mb', 'N/A')
            f1_str = f"{f1:.4f}" if isinstance(f1, (int, float)) else str(f1)
            mem_str = f"{mem:.1f}" if isinstance(mem, (int, float)) else str(mem)
            print(f"{name:<25} F1: {f1_str:<10} Memory: {mem_str} MB")
        print("=" * 70)
        
    elif quick:
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

