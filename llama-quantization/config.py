"""
Configuration for Llama 3.2-1B Quantization Experiments
All hyperparameters in one place for easy iteration with Cursor
"""

from dataclasses import dataclass, field
from typing import Optional, List
from enum import Enum


class QuantMethod(Enum):
    NONE = "none"  # FP16 baseline
    BITSANDBYTES_8BIT = "bnb_8bit"
    BITSANDBYTES_4BIT = "bnb_4bit"
    GPTQ = "gptq"
    AWQ = "awq"


class ComputeDtype(Enum):
    FP16 = "float16"
    BF16 = "bfloat16"
    FP32 = "float32"


@dataclass
class QuantizationConfig:
    """Quantization hyperparameters"""
    
    method: QuantMethod = QuantMethod.BITSANDBYTES_4BIT
    
    # BitsAndBytes specific
    bnb_4bit_compute_dtype: ComputeDtype = ComputeDtype.FP16
    bnb_4bit_quant_type: str = "nf4"  # "nf4" or "fp4"
    bnb_4bit_use_double_quant: bool = True  # Nested quantization
    
    # GPTQ specific
    gptq_bits: int = 4
    gptq_group_size: int = 128  # -1 for per-column, 32/64/128 common
    gptq_desc_act: bool = False  # Activation order descending
    gptq_sym: bool = True  # Symmetric quantization
    gptq_damp_percent: float = 0.1
    
    # AWQ specific
    awq_bits: int = 4
    awq_group_size: int = 128
    awq_zero_point: bool = True
    awq_version: str = "GEMM"  # "GEMM" or "GEMV"


@dataclass
class ModelConfig:
    """Model configuration"""
    
    model_id: str = "meta-llama/Llama-3.2-1B"
    revision: str = "main"
    trust_remote_code: bool = True
    torch_dtype: ComputeDtype = ComputeDtype.FP16
    device_map: str = "auto"
    
    # For GPTQ calibration
    calibration_dataset: str = "c4"
    calibration_samples: int = 128
    calibration_seq_length: int = 2048


@dataclass
class EvalConfig:
    """Evaluation configuration"""
    
    tasks: List[str] = field(default_factory=lambda: ["coqa"])
    num_fewshot: int = 0  # CoQA typically uses 0-shot
    batch_size: str = "auto"  # or int
    max_batch_size: int = 16
    device: str = "cuda"
    limit: Optional[int] = None  # Set to small number for debugging


@dataclass
class BenchmarkConfig:
    """Hardware benchmark configuration"""
    
    warmup_runs: int = 5
    benchmark_runs: int = 20
    input_lengths: List[int] = field(default_factory=lambda: [128, 256, 512, 1024])
    output_length: int = 128
    batch_sizes: List[int] = field(default_factory=lambda: [1, 4, 8])


@dataclass
class ExperimentConfig:
    """Full experiment configuration"""
    
    name: str = "baseline"
    output_dir: str = "./results"
    seed: int = 42
    
    quantization: QuantizationConfig = field(default_factory=QuantizationConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    benchmark: BenchmarkConfig = field(default_factory=BenchmarkConfig)
    
    # Logging
    log_level: str = "INFO"
    save_quantized_model: bool = False
    quantized_model_path: Optional[str] = None


# Pre-defined experiment configurations for systematic comparison
EXPERIMENTS = {
    "fp16_baseline": ExperimentConfig(
        name="fp16_baseline",
        quantization=QuantizationConfig(method=QuantMethod.NONE),
    ),
    
    "bnb_8bit": ExperimentConfig(
        name="bnb_8bit",
        quantization=QuantizationConfig(method=QuantMethod.BITSANDBYTES_8BIT),
    ),
    
    "bnb_4bit_nf4": ExperimentConfig(
        name="bnb_4bit_nf4",
        quantization=QuantizationConfig(
            method=QuantMethod.BITSANDBYTES_4BIT,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        ),
    ),
    
    "bnb_4bit_fp4": ExperimentConfig(
        name="bnb_4bit_fp4",
        quantization=QuantizationConfig(
            method=QuantMethod.BITSANDBYTES_4BIT,
            bnb_4bit_quant_type="fp4",
            bnb_4bit_use_double_quant=True,
        ),
    ),
    
    "gptq_4bit_g128": ExperimentConfig(
        name="gptq_4bit_g128",
        quantization=QuantizationConfig(
            method=QuantMethod.GPTQ,
            gptq_bits=4,
            gptq_group_size=128,
        ),
    ),
    
    "gptq_4bit_g32": ExperimentConfig(
        name="gptq_4bit_g32",
        quantization=QuantizationConfig(
            method=QuantMethod.GPTQ,
            gptq_bits=4,
            gptq_group_size=32,
        ),
    ),
    
    "gptq_3bit_g128": ExperimentConfig(
        name="gptq_3bit_g128",
        quantization=QuantizationConfig(
            method=QuantMethod.GPTQ,
            gptq_bits=3,
            gptq_group_size=128,
        ),
    ),
    
    "awq_4bit_g128": ExperimentConfig(
        name="awq_4bit_g128",
        quantization=QuantizationConfig(
            method=QuantMethod.AWQ,
            awq_bits=4,
            awq_group_size=128,
        ),
    ),
}
