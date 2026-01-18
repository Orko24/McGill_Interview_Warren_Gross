"""
GPTQ quantization loader.

GPTQ (Generative Pre-trained Transformer Quantization):
- Post-training quantization using calibration data
- Layer-by-layer quantization minimizing reconstruction error
- Supports 2-8 bit quantization
- Group-wise quantization for better accuracy
"""

import logging
import torch
from transformers import AutoModelForCausalLM, GPTQConfig

from llama_quant.core.config import ExperimentConfig, QuantMethod
from llama_quant.models.base import (
    ModelLoader,
    load_tokenizer,
    DEFAULT_ATTN_IMPLEMENTATION,
)

logger = logging.getLogger(__name__)


class GPTQLoader(ModelLoader):
    """
    Loader for GPTQ quantization.
    
    GPTQ performs calibration-based quantization:
    1. Load base model in FP16
    2. Run calibration data through model
    3. Quantize weights layer-by-layer to minimize output error
    
    Parameters:
    - bits: Quantization bits (2-8, typically 4)
    - group_size: Number of weights per quantization group (-1 for per-column)
    - desc_act: Use activation order descending (can improve accuracy)
    - sym: Symmetric vs asymmetric quantization
    - damp_percent: Dampening for Hessian inverse (stability)
    """
    
    @property
    def method(self) -> QuantMethod:
        return QuantMethod.GPTQ
    
    def load(self, config: ExperimentConfig) -> AutoModelForCausalLM:
        quant_config = config.quantization
        
        logger.info(f"Loading {config.model.model_id} with GPTQ...")
        logger.info(f"  Bits: {quant_config.gptq_bits}")
        logger.info(f"  Group size: {quant_config.gptq_group_size}")
        logger.info(f"  Symmetric: {quant_config.gptq_sym}")
        
        # Try using optimum for quantization
        try:
            return self._quantize_with_optimum(config)
        except ImportError:
            logger.warning("optimum not installed, trying pre-quantized model...")
            return self._load_prequantized(config)
    
    def _quantize_with_optimum(self, config: ExperimentConfig) -> AutoModelForCausalLM:
        """Quantize model using optimum library."""
        from optimum.gptq import GPTQQuantizer
        from datasets import load_dataset
        
        quant_config = config.quantization
        
        # Load base model
        logger.info("Loading base model for GPTQ quantization...")
        model = AutoModelForCausalLM.from_pretrained(
            config.model.model_id,
            revision=config.model.revision,
            torch_dtype=torch.float16,
            device_map=config.model.device_map,
            trust_remote_code=config.model.trust_remote_code,
            attn_implementation=config.model.attn_implementation or DEFAULT_ATTN_IMPLEMENTATION,
        )
        
        tokenizer = load_tokenizer(config)
        
        # Prepare calibration data
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
            tokenized = tokenizer(
                example["text"],
                truncation=True,
                max_length=config.model.calibration_seq_length,
                return_tensors="pt",
            )
            calib_data.append(tokenized["input_ids"])
        
        # Initialize quantizer
        quantizer = GPTQQuantizer(
            bits=quant_config.gptq_bits,
            group_size=quant_config.gptq_group_size,
            desc_act=quant_config.gptq_desc_act,
            sym=quant_config.gptq_sym,
            damp_percent=quant_config.gptq_damp_percent,
        )
        
        # Quantize
        logger.info("Quantizing model with GPTQ (this may take a while)...")
        quantized_model = quantizer.quantize_model(model, tokenizer)
        
        return quantized_model
    
    def _load_prequantized(self, config: ExperimentConfig) -> AutoModelForCausalLM:
        """Load a pre-quantized GPTQ model."""
        quant_config = config.quantization
        
        gptq_config = GPTQConfig(
            bits=quant_config.gptq_bits,
            group_size=quant_config.gptq_group_size,
            desc_act=quant_config.gptq_desc_act,
            sym=quant_config.gptq_sym,
            damp_percent=quant_config.gptq_damp_percent,
            dataset=config.model.calibration_dataset,
        )
        
        model = AutoModelForCausalLM.from_pretrained(
            config.model.model_id,
            revision=config.model.revision,
            quantization_config=gptq_config,
            device_map=config.model.device_map,
            trust_remote_code=config.model.trust_remote_code,
            attn_implementation=config.model.attn_implementation or DEFAULT_ATTN_IMPLEMENTATION,
        )
        
        return model

