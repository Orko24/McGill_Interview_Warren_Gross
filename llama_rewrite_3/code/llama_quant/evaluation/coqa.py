"""
CoQA-specific evaluation utilities.

CoQA (Conversational Question Answering) benchmark:
- Multi-turn QA on diverse domains
- Uses F1 score and Exact Match (EM) as metrics
- Evaluates reading comprehension and reasoning
"""

import logging
from typing import Dict, Any, List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)


def extract_coqa_metrics(results: Dict[str, Any]) -> Dict[str, float]:
    """
    Extract key metrics from CoQA evaluation results.
    
    CoQA reports:
    - F1: Token-level overlap between prediction and answer
    - EM (Exact Match): Percentage of exact string matches
    
    Args:
        results: Raw lm-eval results dictionary
        
    Returns:
        Dictionary with extracted metrics:
        - coqa_f1: F1 score (0-1)
        - coqa_em: Exact match score (0-1)
        - coqa_f1_stderr: Standard error of F1 (if available)
    """
    metrics = {}
    
    if "results" not in results:
        logger.warning("No 'results' key in evaluation output")
        return metrics
    
    for task_name, task_results in results["results"].items():
        if "coqa" not in task_name.lower():
            continue
            
        # Extract F1 score (try different key formats)
        for key in ["f1", "f1,none"]:
            if key in task_results:
                metrics["coqa_f1"] = task_results[key]
                break
        
        # Extract Exact Match
        for key in ["em", "em,none"]:
            if key in task_results:
                metrics["coqa_em"] = task_results[key]
                break
        
        # Extract standard errors if available
        for key in ["f1_stderr", "f1_stderr,none"]:
            if key in task_results:
                metrics["coqa_f1_stderr"] = task_results[key]
                break
        
        for key in ["em_stderr", "em_stderr,none"]:
            if key in task_results:
                metrics["coqa_em_stderr"] = task_results[key]
                break
    
    return metrics


def run_sanity_check(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    device: str = "cuda",
) -> List[Dict[str, str]]:
    """
    Run quick sanity check to verify model generates coherent text.
    
    Tests:
    1. Factual knowledge completion
    2. Code generation
    3. Simple QA
    
    Args:
        model: Model to test
        tokenizer: Model tokenizer
        device: Device to run on
        
    Returns:
        List of prompt/response pairs
    """
    test_prompts = [
        "The capital of France is",
        "def fibonacci(n):\n    '''Calculate the nth Fibonacci number'''\n",
        "Question: What is 2 + 2?\nAnswer:",
    ]
    
    results = []
    
    for prompt in test_prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        
        with torch.inference_mode():
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
        
        logger.debug(f"Sanity check - Prompt: {prompt[:30]}...")
        logger.debug(f"Sanity check - Generated: {generated[:50]}...")
    
    return results


def format_coqa_results(metrics: Dict[str, float]) -> str:
    """Format CoQA metrics for display."""
    lines = ["CoQA Results:"]
    
    if "coqa_f1" in metrics:
        f1 = metrics["coqa_f1"]
        stderr = metrics.get("coqa_f1_stderr", 0)
        lines.append(f"  F1 Score: {f1:.4f} (±{stderr:.4f})")
    
    if "coqa_em" in metrics:
        em = metrics["coqa_em"]
        stderr = metrics.get("coqa_em_stderr", 0)
        lines.append(f"  Exact Match: {em:.4f} (±{stderr:.4f})")
    
    return "\n".join(lines)

