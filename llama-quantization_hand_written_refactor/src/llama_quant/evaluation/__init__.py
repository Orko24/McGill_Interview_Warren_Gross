"""
Model evaluation module using lm-evaluation-harness.

Primary benchmark: CoQA (Conversational Question Answering)

Usage:
    from llama_quant.evaluation import evaluate_model, extract_coqa_metrics
    
    results = evaluate_model(model, tokenizer, config)
    metrics = extract_coqa_metrics(results)
"""

from llama_quant.evaluation.harness import (
    evaluate_model,
    run_lm_eval,
)
from llama_quant.evaluation.coqa import (
    extract_coqa_metrics,
    run_sanity_check,
)

__all__ = [
    "evaluate_model",
    "run_lm_eval",
    "extract_coqa_metrics",
    "run_sanity_check",
]



