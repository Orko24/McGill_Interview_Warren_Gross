"""
Base model loading utilities and abstract interface.

This module provides:
- Abstract ModelLoader class for implementing quantization-specific loaders
- Factory function to get appropriate loader based on quantization method
- Common utilities for tokenizer loading and model size calculation
"""

import logging
from abc import ABC, abstractmethod
from typing import Tuple, Optional, Dict, Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from llama_quant.core.config import (
    ExperimentConfig,
    QuantMethod,
    ComputeDtype,
)

logger = logging.getLogger(__name__)

# Default attention implementation (SDPA includes Flash Attention 2 when available)
DEFAULT_ATTN_IMPLEMENTATION = "sdpa"


def get_torch_dtype(dtype: ComputeDtype) -> torch.dtype:
    """Convert ComputeDtype enum to torch.dtype."""
    mapping = {
        ComputeDtype.FP16: torch.float16,
        ComputeDtype.BF16: torch.bfloat16,
        ComputeDtype.FP32: torch.float32,
    }
    return mapping[dtype]


def load_tokenizer(config: ExperimentConfig) -> AutoTokenizer:
    """
    Load tokenizer for the model.
    
    Sets padding_side="left" for decoder-only models during generation.
    """
    tokenizer = AutoTokenizer.from_pretrained(
        config.model.model_id,
        revision=config.model.revision,
        trust_remote_code=config.model.trust_remote_code,
        padding_side="left",
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    return tokenizer


def get_model_size_mb(model: AutoModelForCausalLM) -> float:
    """Calculate model size in megabytes."""
    param_size = sum(p.nelement() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.nelement() * b.element_size() for b in model.buffers())
    return (param_size + buffer_size) / 1024 / 1024


def get_memory_footprint(model: AutoModelForCausalLM) -> Dict[str, float]:
    """Get detailed memory footprint of the model."""
    try:
        footprint = model.get_memory_footprint()
        return {"total_mb": footprint / 1024 / 1024}
    except AttributeError:
        return {"total_mb": get_model_size_mb(model)}


class ModelLoader(ABC):
    """
    Abstract base class for model loaders.
    
    Implement this class to add support for new quantization methods.
    """
    
    @abstractmethod
    def load(self, config: ExperimentConfig) -> AutoModelForCausalLM:
        """Load and return the quantized model."""
        pass
    
    @property
    @abstractmethod
    def method(self) -> QuantMethod:
        """Return the quantization method this loader handles."""
        pass


class FP16Loader(ModelLoader):
    """Loader for FP16 baseline (no quantization)."""
    
    @property
    def method(self) -> QuantMethod:
        return QuantMethod.NONE
    
    def load(self, config: ExperimentConfig) -> AutoModelForCausalLM:
        logger.info(f"Loading {config.model.model_id} in FP16 (baseline)...")
        
        model = AutoModelForCausalLM.from_pretrained(
            config.model.model_id,
            revision=config.model.revision,
            dtype=get_torch_dtype(config.model.torch_dtype),  # Use dtype instead of torch_dtype (deprecated)
            device_map=config.model.device_map,
            trust_remote_code=config.model.trust_remote_code,
            attn_implementation=config.model.attn_implementation or DEFAULT_ATTN_IMPLEMENTATION,
        )
        
        return model


def _get_loader(method: QuantMethod) -> ModelLoader:
    """Factory function to get appropriate loader for quantization method."""
    # Import here to avoid circular imports
    from llama_quant.models.bnb import BitsAndBytes4BitLoader, BitsAndBytes8BitLoader
    from llama_quant.models.gptq import GPTQLoader
    from llama_quant.models.awq import AWQLoader
    
    loaders = {
        QuantMethod.NONE: FP16Loader(),
        QuantMethod.BITSANDBYTES_4BIT: BitsAndBytes4BitLoader(),
        QuantMethod.BITSANDBYTES_8BIT: BitsAndBytes8BitLoader(),
        QuantMethod.GPTQ: GPTQLoader(),
        QuantMethod.AWQ: AWQLoader(),
    }
    
    if method not in loaders:
        raise ValueError(f"Unsupported quantization method: {method}")
    
    return loaders[method]


def load_model(config: ExperimentConfig) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """
    Main entry point: Load model with specified quantization method.
    
    Args:
        config: Experiment configuration specifying model and quantization settings
        
    Returns:
        Tuple of (model, tokenizer)
        
    Example:
        >>> from llama_quant.core.config import get_experiment
        >>> config = get_experiment("bnb_4bit_nf4")
        >>> model, tokenizer = load_model(config)
    """
    method = config.quantization.method
    
    # Load tokenizer (same for all methods)
    tokenizer = load_tokenizer(config)
    
    # Get appropriate loader and load model
    loader = _get_loader(method)
    model = loader.load(config)
    
    # Log model info
    size_mb = get_model_size_mb(model)
    logger.info(f"Model loaded: {config.model.model_id}")
    logger.info(f"Quantization: {method.value}")
    logger.info(f"Model size: {size_mb:.2f} MB")
    
    return model, tokenizer

