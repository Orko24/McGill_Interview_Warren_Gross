"""
lm-evaluation-harness integration.

Provides wrappers for running evaluations using the lm-eval library.
"""

import logging
from typing import Dict, Any

from transformers import AutoModelForCausalLM, AutoTokenizer

from llama_quant.core.config import ExperimentConfig
from llama_quant.evaluation.coqa import extract_coqa_metrics, run_sanity_check

logger = logging.getLogger(__name__)


def run_lm_eval(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    config: ExperimentConfig,
) -> Dict[str, Any]:
    """
    Run lm-evaluation-harness on the model.
    
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
            "lm-eval not installed. Install with: pip install lm-eval"
        )
    
    logger.info(f"Running evaluation on tasks: {config.eval.tasks}")
    logger.info(f"  Num fewshot: {config.eval.num_fewshot}")
    logger.info(f"  Batch size: {config.eval.batch_size}")
    logger.info(f"  Limit: {config.eval.limit or 'full'}")
    
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


def evaluate_model(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    config: ExperimentConfig,
    skip_sanity_check: bool = False,
) -> Dict[str, Any]:
    """
    Main evaluation entry point.
    
    Runs:
    1. Quick sanity check (optional)
    2. Full lm-evaluation-harness benchmark
    3. Extracts key metrics
    
    Args:
        model: Model to evaluate
        tokenizer: Model tokenizer
        config: Experiment config
        skip_sanity_check: Whether to skip initial sanity check
        
    Returns:
        Dictionary with all evaluation results including:
        - sanity_check: Quick generation tests
        - lm_eval_results: Full evaluation results
        - coqa_metrics: Extracted CoQA F1/EM scores
    """
    all_results = {}
    
    # Run sanity check
    if not skip_sanity_check:
        logger.info("Running sanity check...")
        sanity_results = run_sanity_check(model, tokenizer, config.eval.device)
        all_results["sanity_check"] = sanity_results
        logger.info("Sanity check passed")
    
    # Run lm-eval
    logger.info("Starting lm-evaluation-harness...")
    eval_results = run_lm_eval(model, tokenizer, config)
    all_results["lm_eval_results"] = eval_results
    
    # Extract key metrics
    coqa_metrics = extract_coqa_metrics(eval_results)
    all_results["coqa_metrics"] = coqa_metrics
    
    logger.info(f"CoQA Metrics: {coqa_metrics}")
    
    return all_results

