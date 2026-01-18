"""
Model loading with quantization support.

Provides a unified interface for loading models with different quantization methods:
- BitsAndBytes (4-bit, 8-bit)
- GPTQ
- AWQ
- FP16 baseline

Usage:
    from llama_quant.models import load_model
    
    model, tokenizer = load_model(config)
"""

from llama_quant.models.base import (
    ModelLoader,
    load_model,
    load_tokenizer,
    get_model_size_mb,
    get_torch_dtype,
)

__all__ = [
    "ModelLoader",
    "load_model",
    "load_tokenizer",
    "get_model_size_mb",
    "get_torch_dtype",
]

