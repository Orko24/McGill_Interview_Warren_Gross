"""
AWQ (Activation-Aware Weight Quantization) loader.

AWQ quantization:
- Identifies "salient" weights based on activation patterns
- Protects important weights during quantization
- Achieves better accuracy than naive quantization at same bit-width
- Supports GEMM and GEMV kernels for inference
"""

import logging
from transformers import AutoModelForCausalLM, AwqConfig

from llama_quant.core.config import ExperimentConfig, QuantMethod
from llama_quant.models.base import (
    ModelLoader,
    load_tokenizer,
    DEFAULT_ATTN_IMPLEMENTATION,
)

logger = logging.getLogger(__name__)


class AWQLoader(ModelLoader):
    """
    Loader for AWQ quantization.
    
    AWQ identifies and protects important weights:
    1. Analyze activation magnitudes on calibration data
    2. Identify "salient" weights that significantly affect output
    3. Apply per-channel scaling to protect important weights
    4. Quantize with the scaling applied
    
    Parameters:
    - bits: Quantization bits (typically 4)
    - group_size: Number of weights per quantization group
    - zero_point: Use asymmetric quantization with zero point
    - version: Kernel version ("GEMM" or "GEMV")
    """
    
    @property
    def method(self) -> QuantMethod:
        return QuantMethod.AWQ
    
    def load(self, config: ExperimentConfig) -> AutoModelForCausalLM:
        quant_config = config.quantization
        
        logger.info(f"Loading {config.model.model_id} with AWQ...")
        logger.info(f"  Bits: {quant_config.awq_bits}")
        logger.info(f"  Group size: {quant_config.awq_group_size}")
        logger.info(f"  Zero point: {quant_config.awq_zero_point}")
        logger.info(f"  Version: {quant_config.awq_version}")
        
        # Try using autoawq for quantization
        try:
            return self._quantize_with_autoawq(config)
        except ImportError:
            logger.warning("autoawq not installed, trying transformers AWQ...")
            return self._load_with_transformers(config)
    
    def _quantize_with_autoawq(self, config: ExperimentConfig) -> AutoModelForCausalLM:
        """Quantize model using autoawq library."""
        from awq import AutoAWQForCausalLM
        from datasets import load_dataset
        
        quant_config = config.quantization
        
        # Load model for AWQ
        logger.info("Loading model for AWQ quantization...")
        model = AutoAWQForCausalLM.from_pretrained(
            config.model.model_id,
            trust_remote_code=config.model.trust_remote_code,
        )
        
        tokenizer = load_tokenizer(config)
        
        # Load calibration data
        logger.info(f"Loading calibration data from {config.model.calibration_dataset}...")
        calib_dataset = load_dataset(
            config.model.calibration_dataset,
            split="train",
            streaming=True,
        )
        
        calib_data = []
        for i, example in enumerate(calib_dataset):
            if i >= config.model.calibration_samples:
                break
            calib_data.append(example["text"])
        
        # Quantize
        logger.info("Quantizing model with AWQ (this may take a while)...")
        model.quantize(
            tokenizer,
            quant_config={
                "zero_point": quant_config.awq_zero_point,
                "q_group_size": quant_config.awq_group_size,
                "w_bit": quant_config.awq_bits,
                "version": quant_config.awq_version,
            },
            calib_data=calib_data,
        )
        
        return model
    
    def _load_with_transformers(self, config: ExperimentConfig) -> AutoModelForCausalLM:
        """Load AWQ model using transformers."""
        quant_config = config.quantization
        
        awq_config = AwqConfig(
            bits=quant_config.awq_bits,
            group_size=quant_config.awq_group_size,
            zero_point=quant_config.awq_zero_point,
            version=quant_config.awq_version,
        )
        
        model = AutoModelForCausalLM.from_pretrained(
            config.model.model_id,
            revision=config.model.revision,
            quantization_config=awq_config,
            device_map=config.model.device_map,
            trust_remote_code=config.model.trust_remote_code,
            attn_implementation=config.model.attn_implementation or DEFAULT_ATTN_IMPLEMENTATION,
        )
        
        return model

