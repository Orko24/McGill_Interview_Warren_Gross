"""
Evaluation module using lm-evaluation-harness
Primary benchmark: CoQA (Conversational Question Answering)
"""

import logging
import json
from typing import Dict, Any, Optional
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import ExperimentConfig

logger = logging.getLogger(__name__)


def run_lm_eval(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    config: ExperimentConfig,
) -> Dict[str, Any]:
    """
    Run lm-evaluation-harness on the model
    
    Args:
        model: The (quantized) model to evaluate
        tokenizer: Model tokenizer
        config: Experiment configuration
        
    Returns:
        Dictionary containing evaluation results
    """
    try:
        import lm_eval
        from lm_eval.models.huggingface import HFLM
    except ImportError:
        raise ImportError(
            "lm-eval not installed. Install with: "
            "pip install lm-eval"
        )
    
    logger.info(f"Running evaluation on tasks: {config.eval.tasks}")
    
    # Wrap model for lm-eval
    lm = HFLM(
        pretrained=model,
        tokenizer=tokenizer,
        batch_size=config.eval.batch_size,
        max_batch_size=config.eval.max_batch_size,
    )
    
    # Run evaluation
    results = lm_eval.simple_evaluate(
        model=lm,
        tasks=config.eval.tasks,
        num_fewshot=config.eval.num_fewshot,
        limit=config.eval.limit,
        batch_size=config.eval.batch_size,
    )
    
    return results


def extract_coqa_metrics(results: Dict[str, Any]) -> Dict[str, float]:
    """
    Extract key metrics from CoQA evaluation results
    
    CoQA uses F1 score and Exact Match (EM) as primary metrics
    """
    metrics = {}
    
    if "results" in results:
        for task_name, task_results in results["results"].items():
            if "coqa" in task_name.lower():
                # Extract available metrics
                if "f1" in task_results:
                    metrics["coqa_f1"] = task_results["f1"]
                if "f1,none" in task_results:
                    metrics["coqa_f1"] = task_results["f1,none"]
                if "em" in task_results:
                    metrics["coqa_em"] = task_results["em"]
                if "em,none" in task_results:
                    metrics["coqa_em"] = task_results["em,none"]
                    
                # Get stderr if available
                if "f1_stderr" in task_results:
                    metrics["coqa_f1_stderr"] = task_results["f1_stderr"]
                if "f1_stderr,none" in task_results:
                    metrics["coqa_f1_stderr"] = task_results["f1_stderr,none"]
    
    return metrics


def run_quick_sanity_check(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    device: str = "cuda",
) -> Dict[str, Any]:
    """
    Run a quick sanity check to ensure model generates coherent text
    before running full evaluation
    """
    logger.info("Running sanity check...")
    
    test_prompts = [
        "The capital of France is",
        "def fibonacci(n):\n    '''Calculate the nth Fibonacci number'''\n",
        "Question: What is 2 + 2?\nAnswer:",
    ]
    
    results = []
    
    for prompt in test_prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=50,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        
        generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
        results.append({
            "prompt": prompt,
            "generated": generated,
        })
        
        logger.info(f"Prompt: {prompt[:50]}...")
        logger.info(f"Generated: {generated[:100]}...")
    
    return {"sanity_check": results}


def evaluate_model(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    config: ExperimentConfig,
    skip_sanity_check: bool = False,
) -> Dict[str, Any]:
    """
    Main evaluation entry point
    
    Args:
        model: Model to evaluate
        tokenizer: Model tokenizer
        config: Experiment config
        skip_sanity_check: Whether to skip initial sanity check
        
    Returns:
        Dictionary with all evaluation results
    """
    all_results = {}
    
    # Sanity check
    if not skip_sanity_check:
        sanity_results = run_quick_sanity_check(model, tokenizer)
        all_results.update(sanity_results)
    
    # Run lm-eval
    logger.info("Starting lm-evaluation-harness...")
    eval_results = run_lm_eval(model, tokenizer, config)
    all_results["lm_eval_results"] = eval_results
    
    # Extract key metrics
    coqa_metrics = extract_coqa_metrics(eval_results)
    all_results["coqa_metrics"] = coqa_metrics
    
    logger.info(f"CoQA Metrics: {coqa_metrics}")
    
    return all_results


def save_eval_results(
    results: Dict[str, Any],
    output_dir: str,
    experiment_name: str,
):
    """Save evaluation results to JSON"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    results_file = output_path / f"{experiment_name}_eval_results.json"
    
    # Make results JSON serializable
    def make_serializable(obj):
        if isinstance(obj, torch.Tensor):
            return obj.tolist()
        elif isinstance(obj, (set, frozenset)):
            return list(obj)
        elif hasattr(obj, "__dict__"):
            return str(obj)
        return obj
    
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2, default=make_serializable)
    
    logger.info(f"Results saved to {results_file}")


def compare_results(
    results_list: list[Dict[str, Any]],
    experiment_names: list[str],
) -> Dict[str, Any]:
    """
    Compare results across multiple experiments
    
    Returns a summary table of key metrics
    """
    comparison = {
        "experiments": [],
    }
    
    for name, results in zip(experiment_names, results_list):
        entry = {"name": name}
        
        if "coqa_metrics" in results:
            entry.update(results["coqa_metrics"])
        
        if "benchmark_results" in results:
            bench = results["benchmark_results"]
            if "memory_mb" in bench:
                entry["memory_mb"] = bench["memory_mb"]
            if "tokens_per_second" in bench:
                entry["tokens_per_second"] = bench["tokens_per_second"]
        
        comparison["experiments"].append(entry)
    
    return comparison


def print_comparison_table(comparison: Dict[str, Any]):
    """Print a formatted comparison table"""
    experiments = comparison["experiments"]
    
    if not experiments:
        print("No experiments to compare")
        return
    
    # Get all keys
    all_keys = set()
    for exp in experiments:
        all_keys.update(exp.keys())
    all_keys.discard("name")
    all_keys = sorted(all_keys)
    
    # Print header
    header = ["Experiment"] + list(all_keys)
    print("\n" + "=" * 80)
    print(f"{'Experiment':<25}", end="")
    for key in all_keys:
        print(f"{key:<15}", end="")
    print("\n" + "-" * 80)
    
    # Print rows
    for exp in experiments:
        print(f"{exp['name']:<25}", end="")
        for key in all_keys:
            val = exp.get(key, "N/A")
            if isinstance(val, float):
                print(f"{val:<15.4f}", end="")
            else:
                print(f"{str(val):<15}", end="")
        print()
    
    print("=" * 80 + "\n")
