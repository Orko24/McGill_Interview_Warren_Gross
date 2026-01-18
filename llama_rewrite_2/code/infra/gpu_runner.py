"""
GPU-side experiment runner.

This module runs ONLY in the Modal cloud environment where torch and 
llama_quant are installed. All imports are at the top as they should be.

Do not import this module locally - it will fail without GPU dependencies.
"""

import os
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# GPU-only imports (available in Modal cloud environment)
import torch
from llama_quant.core.config import get_experiment
from llama_quant.models import load_model, get_model_size_mb
from llama_quant.evaluation import evaluate_model
from llama_quant.benchmark import BenchmarkSuite


@dataclass
class ExperimentRequest:
    """Request to run an experiment."""
    name: str
    limit: Optional[int] = None


@dataclass 
class ExperimentResult:
    """Result from running an experiment."""
    experiment_name: str
    status: str
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
        
        # Ensure llama_quant is importable
        if "/root/code" not in sys.path:
            sys.path.insert(0, "/root/code")
    
    def run(self, request: ExperimentRequest) -> ExperimentResult:
        """
        Run a single experiment.
        
        Args:
            request: Experiment configuration
            
        Returns:
            ExperimentResult with metrics or error
        """
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
    
    # Note: bnb_8bit excluded due to CUDA kernel bug on A10G (bitsandbytes issue)
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

