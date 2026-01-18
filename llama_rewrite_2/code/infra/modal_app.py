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
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import modal


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class ModalConfig:
    """Configuration for Modal infrastructure."""
    app_name: str = "llama-quantization"
    gpu_type: str = "A10G"
    timeout: int = 3600
    cache_dir: str = "/cache"
    results_dir: str = "/cache/results"
    hf_cache_dir: str = "/cache/huggingface"


@dataclass
class ExperimentRequest:
    """Request to run an experiment."""
    name: str
    limit: Optional[int] = None
    
    
@dataclass
class ExperimentResult:
    """Result from running an experiment."""
    experiment_name: str
    status: str  # "success" or "error"
    model_size_mb: Optional[float] = None
    coqa_metrics: Dict[str, Any] = field(default_factory=dict)
    benchmarks: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        result = {
            "experiment_name": self.experiment_name,
            "status": self.status,
        }
        if self.model_size_mb is not None:
            result["model_size_mb"] = self.model_size_mb
        if self.coqa_metrics:
            result["coqa_metrics"] = self.coqa_metrics
        if self.benchmarks:
            result["benchmarks"] = self.benchmarks
        if self.error:
            result["error"] = self.error
        return result


# =============================================================================
# Modal Infrastructure Setup
# =============================================================================

config = ModalConfig()
app = modal.App(config.app_name)

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
        "numpy<2.0.0",
        "pyyaml>=6.0",
    )
)

image = image.add_local_python_source("code")
volume = modal.Volume.from_name("llama-quant-cache", create_if_missing=True)
hf_secret = modal.Secret.from_name("huggingface-secret")


# =============================================================================
# Experiment Runner (runs on GPU)
# =============================================================================

class ExperimentRunner:
    """
    Runs quantization experiments on GPU.
    
    Handles:
        - Model loading with specified quantization
        - CoQA evaluation via lm-eval-harness
        - Hardware benchmarks (memory, latency, throughput)
    """
    
    def __init__(self, hf_cache_dir: str = "/cache/huggingface"):
        self.hf_cache_dir = hf_cache_dir
        self._setup_environment()
    
    def _setup_environment(self) -> None:
        """Configure environment for HuggingFace."""
        os.environ["HF_HOME"] = self.hf_cache_dir
        os.environ["TRANSFORMERS_CACHE"] = self.hf_cache_dir
        sys.path.insert(0, "/root/code")
    
    def run(self, request: ExperimentRequest) -> ExperimentResult:
        """
        Run a single experiment.
        
        Args:
            request: Experiment configuration
            
        Returns:
            ExperimentResult with metrics or error
        """
        import torch
        from llama_quant.core.config import get_experiment
        from llama_quant.models import load_model, get_model_size_mb
        from llama_quant.evaluation import evaluate_model
        from llama_quant.benchmark import BenchmarkSuite
        
        print(f"\n{'='*60}")
        print(f"Running: {request.name}")
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"{'='*60}\n")
        
        try:
            # Load config
            exp_config = get_experiment(request.name)
            if request.limit is not None:
                exp_config.eval.limit = request.limit
            
            # Load model
            print("Loading model...")
            model, tokenizer = load_model(exp_config)
            model_size = get_model_size_mb(model)
            print(f"Model size: {model_size:.2f} MB")
            
            # Evaluate
            print("Running evaluation...")
            eval_results = evaluate_model(model, tokenizer, exp_config, skip_sanity_check=True)
            coqa_metrics = eval_results.get("coqa_metrics", {})
            print(f"CoQA F1: {coqa_metrics.get('coqa_f1', 'N/A')}")
            
            # Benchmark
            print("Running benchmarks...")
            exp_config.benchmark.benchmark_runs = 5
            exp_config.benchmark.warmup_runs = 2
            suite = BenchmarkSuite.from_experiment_config(model, tokenizer, exp_config)
            benchmark_results = suite.run_all()
            
            # Cleanup
            del model, tokenizer
            torch.cuda.empty_cache()
            
            return ExperimentResult(
                experiment_name=request.name,
                status="success",
                model_size_mb=model_size,
                coqa_metrics=coqa_metrics,
                benchmarks=benchmark_results,
            )
            
        except Exception as e:
            print(f"Error: {e}")
            return ExperimentResult(
                experiment_name=request.name,
                status="error",
                error=str(e),
            )


class ComparisonRunner:
    """
    Runs multiple experiments and compares results.
    
    Iterates through experiment configurations, collects results,
    and provides summary statistics.
    """
    
    DEFAULT_EXPERIMENTS = ["fp16_baseline", "bnb_4bit_nf4", "bnb_4bit_fp4"]
    
    def __init__(self, hf_cache_dir: str = "/cache/huggingface"):
        self.runner = ExperimentRunner(hf_cache_dir)
    
    def run(
        self, 
        experiments: Optional[List[str]] = None, 
        limit: int = 100
    ) -> List[ExperimentResult]:
        """
        Run comparison across multiple experiments.
        
        Args:
            experiments: List of experiment names (default: FP16, NF4, FP4)
            limit: Evaluation sample limit
            
        Returns:
            List of ExperimentResult objects
        """
        if experiments is None:
            experiments = self.DEFAULT_EXPERIMENTS
        
        print(f"Running comparison: {experiments}")
        
        results = []
        for exp_name in experiments:
            request = ExperimentRequest(name=exp_name, limit=limit)
            result = self.runner.run(request)
            results.append(result)
        
        self._print_summary(results)
        return results
    
    def _print_summary(self, results: List[ExperimentResult]) -> None:
        """Print formatted comparison summary."""
        print("\n" + "="*60)
        print("COMPARISON SUMMARY")
        print("="*60)
        
        for r in results:
            if r.status == "success":
                f1 = r.coqa_metrics.get("coqa_f1", "N/A")
                f1_str = f"{f1:.4f}" if isinstance(f1, (int, float)) else str(f1)
                size_str = f"{r.model_size_mb:.2f}" if r.model_size_mb else "N/A"
                print(f"{r.experiment_name}: F1={f1_str}, Size={size_str} MB")
            else:
                print(f"{r.experiment_name}: ERROR - {r.error}")


# =============================================================================
# Results Manager (runs locally)
# =============================================================================

class ResultsManager:
    """
    Manages local storage of experiment results.
    
    Handles:
        - Auto-incrementing filenames (results1.json, results2.json, ...)
        - JSON serialization with metadata
    """
    
    def __init__(self, results_dir: Path):
        self.results_dir = results_dir
        self.results_dir.mkdir(parents=True, exist_ok=True)
    
    def save(self, results: Dict[str, Any]) -> Path:
        """
        Save results to next available JSON file.
        
        Args:
            results: Dictionary of results to save
            
        Returns:
            Path to saved file
        """
        output_path = self._get_next_filename()
        
        # Add metadata
        results["timestamp"] = datetime.now().isoformat()
        results["saved_to"] = str(output_path)
        
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        
        return output_path
    
    def _get_next_filename(self) -> Path:
        """Get next available results filename."""
        existing = list(self.results_dir.glob("results*.json"))
        existing_nums = []
        
        for f in existing:
            try:
                num = int(f.stem.replace("results", ""))
                existing_nums.append(num)
            except ValueError:
                pass
        
        next_num = max(existing_nums, default=0) + 1
        return self.results_dir / f"results{next_num}.json"


# =============================================================================
# Modal GPU Functions
# =============================================================================

@app.function(
    image=image,
    gpu=config.gpu_type,
    timeout=config.timeout,
    secrets=[hf_secret],
    volumes={config.cache_dir: volume},
)
def run_single_experiment(experiment_name: str, limit: int = None) -> dict:
    """Modal function: Run single experiment on GPU."""
    runner = ExperimentRunner(config.hf_cache_dir)
    request = ExperimentRequest(name=experiment_name, limit=limit)
    result = runner.run(request)
    
    volume.commit()
    return result.to_dict()


@app.function(
    image=image,
    gpu=config.gpu_type,
    timeout=config.timeout * 2,
    secrets=[hf_secret],
    volumes={config.cache_dir: volume},
)
def run_comparison(experiments: list = None, limit: int = 100) -> dict:
    """Modal function: Run comparison on GPU."""
    runner = ComparisonRunner(config.hf_cache_dir)
    results = runner.run(experiments, limit)
    
    volume.commit()
    return {"experiments": [r.to_dict() for r in results]}


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
    # Setup local results manager
    local_results_dir = Path(__file__).parent.parent.parent / "results"
    results_manager = ResultsManager(local_results_dir)
    
    # Determine what to run
    if all_experiments:
        experiments = [
            "fp16_baseline",
            "bnb_4bit_nf4",
            "bnb_4bit_fp4",
            "bnb_4bit_nf4_no_double",
            "bnb_4bit_nf4_bf16",
        ]
        results = run_comparison.remote(experiments=experiments, limit=limit)
        run_type = "all"
        
    elif experiment:
        results = run_single_experiment.remote(experiment, limit=limit)
        results = {"experiments": [results], "run_type": "single"}
        run_type = "single"
        
    else:
        results = run_comparison.remote(limit=limit)
        run_type = "comparison"
    
    # Save locally
    results["limit"] = limit
    results["run_type"] = run_type
    output_path = results_manager.save(results)
    
    print(f"\n{'='*60}")
    print(f"Results saved to: {output_path}")
    print(f"{'='*60}")
