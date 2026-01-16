"""
Quantization methods for Llama 3.2-1B
Supports: BitsAndBytes (4/8-bit), GPTQ, AWQ
"""

import torch
import logging
from typing import Tuple, Optional
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    GPTQConfig,
    AwqConfig,
)
from config import (
    ExperimentConfig,
    QuantMethod,
    ComputeDtype,
)

logger = logging.getLogger(__name__)


def get_torch_dtype(dtype: ComputeDtype) -> torch.dtype:
    """Convert config dtype to torch dtype"""
    mapping = {
        ComputeDtype.FP16: torch.float16,
        ComputeDtype.BF16: torch.bfloat16,
        ComputeDtype.FP32: torch.float32,
    }
    return mapping[dtype]


def load_tokenizer(config: ExperimentConfig) -> AutoTokenizer:
    """Load tokenizer for the model"""
    tokenizer = AutoTokenizer.from_pretrained(
        config.model.model_id,
        revision=config.model.revision,
        trust_remote_code=config.model.trust_remote_code,
    )
    
    # Ensure padding token is set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    return tokenizer


def create_bnb_config(config: ExperimentConfig) -> BitsAndBytesConfig:
    """Create BitsAndBytes quantization config"""
    quant_config = config.quantization
    
    if quant_config.method == QuantMethod.BITSANDBYTES_8BIT:
        return BitsAndBytesConfig(
            load_in_8bit=True,
        )
    
    elif quant_config.method == QuantMethod.BITSANDBYTES_4BIT:
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=get_torch_dtype(quant_config.bnb_4bit_compute_dtype),
            bnb_4bit_quant_type=quant_config.bnb_4bit_quant_type,
            bnb_4bit_use_double_quant=quant_config.bnb_4bit_use_double_quant,
        )
    
    raise ValueError(f"Invalid BnB method: {quant_config.method}")


def create_gptq_config(config: ExperimentConfig) -> GPTQConfig:
    """Create GPTQ quantization config"""
    quant_config = config.quantization
    
    return GPTQConfig(
        bits=quant_config.gptq_bits,
        group_size=quant_config.gptq_group_size,
        desc_act=quant_config.gptq_desc_act,
        sym=quant_config.gptq_sym,
        damp_percent=quant_config.gptq_damp_percent,
        dataset=config.model.calibration_dataset,
    )


def create_awq_config(config: ExperimentConfig) -> AwqConfig:
    """Create AWQ quantization config"""
    quant_config = config.quantization
    
    return AwqConfig(
        bits=quant_config.awq_bits,
        group_size=quant_config.awq_group_size,
        zero_point=quant_config.awq_zero_point,
        version=quant_config.awq_version,
    )


def load_model_fp16(config: ExperimentConfig) -> AutoModelForCausalLM:
    """Load model in FP16 (baseline, no quantization)"""
    logger.info(f"Loading {config.model.model_id} in FP16...")
    
    model = AutoModelForCausalLM.from_pretrained(
        config.model.model_id,
        revision=config.model.revision,
        torch_dtype=get_torch_dtype(config.model.torch_dtype),
        device_map=config.model.device_map,
        trust_remote_code=config.model.trust_remote_code,
    )
    
    return model


def load_model_bnb(config: ExperimentConfig) -> AutoModelForCausalLM:
    """Load model with BitsAndBytes quantization"""
    method = config.quantization.method
    logger.info(f"Loading {config.model.model_id} with {method.value}...")
    
    bnb_config = create_bnb_config(config)
    
    model = AutoModelForCausalLM.from_pretrained(
        config.model.model_id,
        revision=config.model.revision,
        quantization_config=bnb_config,
        device_map=config.model.device_map,
        trust_remote_code=config.model.trust_remote_code,
    )
    
    return model


def load_model_gptq(config: ExperimentConfig) -> AutoModelForCausalLM:
    """
    Load model with GPTQ quantization
    
    Note: GPTQ requires calibration data. For pre-quantized models,
    you can load directly. For quantizing yourself, use AutoGPTQ library.
    """
    logger.info(f"Loading {config.model.model_id} with GPTQ...")
    
    # Check if we're loading a pre-quantized model or quantizing ourselves
    # For this assignment, we'll quantize ourselves using optimum
    try:
        from optimum.gptq import GPTQQuantizer
        from datasets import load_dataset
        
        # Load base model in FP16 first
        model = AutoModelForCausalLM.from_pretrained(
            config.model.model_id,
            revision=config.model.revision,
            torch_dtype=torch.float16,
            device_map=config.model.device_map,
            trust_remote_code=config.model.trust_remote_code,
        )
        
        tokenizer = load_tokenizer(config)
        
        # Load calibration dataset
        logger.info("Loading calibration dataset...")
        calib_dataset = load_dataset(
            config.model.calibration_dataset,
            split="train",
            streaming=True,
        )
        
        # Prepare calibration data
        def prepare_calibration_data(examples):
            return tokenizer(
                examples["text"],
                truncation=True,
                max_length=config.model.calibration_seq_length,
                return_tensors="pt",
                padding=True,
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
        quant_config = config.quantization
        quantizer = GPTQQuantizer(
            bits=quant_config.gptq_bits,
            group_size=quant_config.gptq_group_size,
            desc_act=quant_config.gptq_desc_act,
            sym=quant_config.gptq_sym,
            damp_percent=quant_config.gptq_damp_percent,
        )
        
        # Quantize
        logger.info("Quantizing model with GPTQ...")
        quantized_model = quantizer.quantize_model(model, tokenizer)
        
        return quantized_model
        
    except ImportError:
        logger.warning("optimum not installed, trying to load pre-quantized model...")
        
        # Try loading with transformers GPTQ support
        gptq_config = create_gptq_config(config)
        
        model = AutoModelForCausalLM.from_pretrained(
            config.model.model_id,
            revision=config.model.revision,
            quantization_config=gptq_config,
            device_map=config.model.device_map,
            trust_remote_code=config.model.trust_remote_code,
        )
        
        return model


def load_model_awq(config: ExperimentConfig) -> AutoModelForCausalLM:
    """
    Load model with AWQ quantization
    
    Note: AWQ also requires calibration. We can use the autoawq library
    or load pre-quantized models.
    """
    logger.info(f"Loading {config.model.model_id} with AWQ...")
    
    try:
        from awq import AutoAWQForCausalLM
        from datasets import load_dataset
        
        # Load model for AWQ quantization
        model = AutoAWQForCausalLM.from_pretrained(
            config.model.model_id,
            trust_remote_code=config.model.trust_remote_code,
        )
        
        tokenizer = load_tokenizer(config)
        
        # Load calibration data
        logger.info("Loading calibration dataset for AWQ...")
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
        quant_config = config.quantization
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
        
    except ImportError:
        logger.warning("autoawq not installed, trying transformers AWQ...")
        
        awq_config = create_awq_config(config)
        
        model = AutoModelForCausalLM.from_pretrained(
            config.model.model_id,
            revision=config.model.revision,
            quantization_config=awq_config,
            device_map=config.model.device_map,
            trust_remote_code=config.model.trust_remote_code,
        )
        
        return model


def load_quantized_model(
    config: ExperimentConfig,
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """
    Main entry point: Load model with specified quantization method
    
    Returns:
        Tuple of (model, tokenizer)
    """
    method = config.quantization.method
    
    # Load tokenizer (same for all methods)
    tokenizer = load_tokenizer(config)
    
    # Load model based on quantization method
    if method == QuantMethod.NONE:
        model = load_model_fp16(config)
    
    elif method in [QuantMethod.BITSANDBYTES_4BIT, QuantMethod.BITSANDBYTES_8BIT]:
        model = load_model_bnb(config)
    
    elif method == QuantMethod.GPTQ:
        model = load_model_gptq(config)
    
    elif method == QuantMethod.AWQ:
        model = load_model_awq(config)
    
    else:
        raise ValueError(f"Unknown quantization method: {method}")
    
    # Log model info
    logger.info(f"Model loaded: {config.model.model_id}")
    logger.info(f"Quantization: {method.value}")
    
    return model, tokenizer


def get_model_size_mb(model: AutoModelForCausalLM) -> float:
    """Calculate model size in MB"""
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    size_mb = (param_size + buffer_size) / 1024 / 1024
    return size_mb


def get_model_memory_footprint(model: AutoModelForCausalLM) -> dict:
    """Get detailed memory footprint of the model"""
    try:
        # For models with get_memory_footprint method
        footprint = model.get_memory_footprint()
        return {"total_mb": footprint / 1024 / 1024}
    except AttributeError:
        # Manual calculation
        return {"total_mb": get_model_size_mb(model)}


def save_quantized_model(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    save_path: str,
    config: ExperimentConfig,
):
    """Save quantized model to disk"""
    logger.info(f"Saving quantized model to {save_path}...")
    
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    
    # Save config for reproducibility
    import json
    from dataclasses import asdict
    
    config_path = f"{save_path}/experiment_config.json"
    with open(config_path, "w") as f:
        # Convert enums to strings for JSON serialization
        config_dict = asdict(config)
        json.dump(config_dict, f, indent=2, default=str)
    
    logger.info(f"Model and config saved to {save_path}")
