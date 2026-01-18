"""
BitsAndBytes quantization loaders (4-bit and 8-bit).

BitsAndBytes provides:
- NF4: Normal Float 4-bit, optimal for normally-distributed weights
- FP4: Floating Point 4-bit, uniform quantization levels
- Double quantization: Quantize the quantization constants for extra compression
- 8-bit: LLM.int8() with outlier handling
"""

import logging
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

from llama_quant.core.config import ExperimentConfig, QuantMethod
from llama_quant.models.base import ModelLoader, get_torch_dtype, DEFAULT_ATTN_IMPLEMENTATION

logger = logging.getLogger(__name__)


class BitsAndBytes4BitLoader(ModelLoader):
    """
    Loader for BitsAndBytes 4-bit quantization.
    
    Supports:
    - NF4 (Normal Float 4-bit): Optimal for normally-distributed weights
    - FP4 (Floating Point 4-bit): Uniform quantization levels
    - Double quantization: Quantize the scaling factors for extra compression
    """
    
    @property
    def method(self) -> QuantMethod:
        return QuantMethod.BITSANDBYTES_4BIT
    
    def load(self, config: ExperimentConfig) -> AutoModelForCausalLM:
        quant_config = config.quantization
        
        logger.info(f"Loading {config.model.model_id} with BitsAndBytes 4-bit...")
        logger.info(f"  Quant type: {quant_config.bnb_4bit_quant_type}")
        logger.info(f"  Double quant: {quant_config.bnb_4bit_use_double_quant}")
        logger.info(f"  Compute dtype: {quant_config.bnb_4bit_compute_dtype.value}")
        
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=get_torch_dtype(quant_config.bnb_4bit_compute_dtype),
            bnb_4bit_quant_type=quant_config.bnb_4bit_quant_type,
            bnb_4bit_use_double_quant=quant_config.bnb_4bit_use_double_quant,
        )
        
        model = AutoModelForCausalLM.from_pretrained(
            config.model.model_id,
            revision=config.model.revision,
            quantization_config=bnb_config,
            device_map=config.model.device_map,
            trust_remote_code=config.model.trust_remote_code,
            attn_implementation=config.model.attn_implementation or DEFAULT_ATTN_IMPLEMENTATION,
        )
        
        return model


class BitsAndBytes8BitLoader(ModelLoader):
    """
    Loader for BitsAndBytes 8-bit quantization (LLM.int8()).
    
    Uses mixed-precision decomposition:
    - Most weights in int8
    - Outlier features kept in FP16
    - Threshold parameter controls outlier detection
    
    Note: May have CUDA kernel issues on some GPUs (A10G).
    """
    
    @property
    def method(self) -> QuantMethod:
        return QuantMethod.BITSANDBYTES_8BIT
    
    def load(self, config: ExperimentConfig) -> AutoModelForCausalLM:
        quant_config = config.quantization
        
        logger.info(f"Loading {config.model.model_id} with BitsAndBytes 8-bit...")
        logger.info(f"  Threshold: {quant_config.bnb_8bit_threshold}")
        
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=quant_config.bnb_8bit_threshold,
            llm_int8_has_fp16_weight=quant_config.bnb_8bit_has_fp16_weight,
        )
        
        model = AutoModelForCausalLM.from_pretrained(
            config.model.model_id,
            revision=config.model.revision,
            quantization_config=bnb_config,
            device_map=config.model.device_map,
            trust_remote_code=config.model.trust_remote_code,
            attn_implementation=config.model.attn_implementation or DEFAULT_ATTN_IMPLEMENTATION,
        )
        
        return model

