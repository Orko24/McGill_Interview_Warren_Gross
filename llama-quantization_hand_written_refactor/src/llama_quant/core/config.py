"""
Configuration dataclasses for quantization experiments.

All hyperparameters are defined here for:
- Quantization methods (BnB, GPTQ, AWQ)
- Model loading
- Evaluation settings
- Hardware benchmarking

This module has NO external dependencies beyond stdlib.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from enum import Enum
from pathlib import Path
import yaml
import json


class QuantMethod(Enum):
    """Supported quantization methods."""
    NONE = "none"  # FP16 baseline
    BITSANDBYTES_8BIT = "bnb_8bit"
    BITSANDBYTES_4BIT = "bnb_4bit"
    GPTQ = "gptq"
    AWQ = "awq"


class ComputeDtype(Enum):
    """Compute precision for quantized operations."""
    FP16 = "float16"
    BF16 = "bfloat16"
    FP32 = "float32"


@dataclass
class QuantizationConfig:
    """Quantization hyperparameters for all supported methods."""
    
    method: QuantMethod = QuantMethod.BITSANDBYTES_4BIT
    
    # === BitsAndBytes 4-bit ===
    bnb_4bit_compute_dtype: ComputeDtype = ComputeDtype.FP16
    bnb_4bit_quant_type: str = "nf4"  # "nf4" or "fp4"
    bnb_4bit_use_double_quant: bool = True
    
    # === BitsAndBytes 8-bit ===
    bnb_8bit_threshold: float = 6.0
    bnb_8bit_has_fp16_weight: bool = False
    
    # === GPTQ ===
    gptq_bits: int = 4
    gptq_group_size: int = 128  # -1 for per-column
    gptq_desc_act: bool = False
    gptq_sym: bool = True
    gptq_damp_percent: float = 0.1
    
    # === AWQ ===
    awq_bits: int = 4
    awq_group_size: int = 128
    awq_zero_point: bool = True
    awq_version: str = "GEMM"  # "GEMM" or "GEMV"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "method": self.method.value,
            "bnb_4bit_compute_dtype": self.bnb_4bit_compute_dtype.value,
            "bnb_4bit_quant_type": self.bnb_4bit_quant_type,
            "bnb_4bit_use_double_quant": self.bnb_4bit_use_double_quant,
            "gptq_bits": self.gptq_bits,
            "gptq_group_size": self.gptq_group_size,
            "awq_bits": self.awq_bits,
            "awq_group_size": self.awq_group_size,
        }


@dataclass
class ModelConfig:
    """Model and tokenizer configuration."""
    
    model_id: str = "meta-llama/Llama-3.2-1B"
    revision: str = "main"
    trust_remote_code: bool = True
    torch_dtype: ComputeDtype = ComputeDtype.FP16
    device_map: str = "auto"
    
    # Attention implementation
    attn_implementation: str = "sdpa"  # "sdpa", "flash_attention_2", "eager"
    
    # Calibration (for GPTQ/AWQ)
    calibration_dataset: str = "c4"
    calibration_samples: int = 128
    calibration_seq_length: int = 2048


@dataclass
class EvalConfig:
    """Evaluation configuration for lm-evaluation-harness."""
    
    tasks: List[str] = field(default_factory=lambda: ["coqa"])
    num_fewshot: int = 0
    batch_size: str = "auto"
    max_batch_size: int = 16
    device: str = "cuda"
    limit: Optional[int] = None  # None = full eval


@dataclass
class BenchmarkConfig:
    """Hardware benchmark configuration."""
    
    warmup_runs: int = 5
    benchmark_runs: int = 20
    input_lengths: List[int] = field(default_factory=lambda: [128, 256, 512, 1024])
    output_length: int = 128
    batch_sizes: List[int] = field(default_factory=lambda: [1, 4, 8])


@dataclass
class ExperimentConfig:
    """
    Complete experiment configuration.
    
    Combines all sub-configs and experiment metadata.
    Can be loaded from YAML files for easy iteration.
    """
    
    name: str = "baseline"
    description: str = ""
    output_dir: str = "./results"
    seed: int = 42
    
    # Sub-configurations
    quantization: QuantizationConfig = field(default_factory=QuantizationConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    benchmark: BenchmarkConfig = field(default_factory=BenchmarkConfig)
    
    # Execution flags
    skip_eval: bool = False
    skip_benchmark: bool = False
    save_quantized_model: bool = False
    quantized_model_path: Optional[str] = None
    
    # Logging
    log_level: str = "INFO"

    @classmethod
    def from_yaml(cls, path: str) -> "ExperimentConfig":
        """Load configuration from YAML file."""
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls._from_dict(data)
    
    @classmethod
    def from_json(cls, path: str) -> "ExperimentConfig":
        """Load configuration from JSON file."""
        with open(path) as f:
            data = json.load(f)
        return cls._from_dict(data)
    
    @classmethod
    def _from_dict(cls, data: Dict[str, Any]) -> "ExperimentConfig":
        """Create config from dictionary."""
        # Parse nested configs
        quant_data = data.pop("quantization", {})
        if "method" in quant_data:
            quant_data["method"] = QuantMethod(quant_data["method"])
        if "bnb_4bit_compute_dtype" in quant_data:
            quant_data["bnb_4bit_compute_dtype"] = ComputeDtype(quant_data["bnb_4bit_compute_dtype"])
        
        model_data = data.pop("model", {})
        if "torch_dtype" in model_data:
            model_data["torch_dtype"] = ComputeDtype(model_data["torch_dtype"])
        
        eval_data = data.pop("eval", {})
        benchmark_data = data.pop("benchmark", {})
        
        return cls(
            quantization=QuantizationConfig(**quant_data) if quant_data else QuantizationConfig(),
            model=ModelConfig(**model_data) if model_data else ModelConfig(),
            eval=EvalConfig(**eval_data) if eval_data else EvalConfig(),
            benchmark=BenchmarkConfig(**benchmark_data) if benchmark_data else BenchmarkConfig(),
            **data,
        )
    
    def to_yaml(self, path: str) -> None:
        """Save configuration to YAML file."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(self._to_dict(), f, default_flow_style=False)
    
    def _to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "name": self.name,
            "description": self.description,
            "output_dir": self.output_dir,
            "seed": self.seed,
            "quantization": {
                "method": self.quantization.method.value,
                "bnb_4bit_compute_dtype": self.quantization.bnb_4bit_compute_dtype.value,
                "bnb_4bit_quant_type": self.quantization.bnb_4bit_quant_type,
                "bnb_4bit_use_double_quant": self.quantization.bnb_4bit_use_double_quant,
                "gptq_bits": self.quantization.gptq_bits,
                "gptq_group_size": self.quantization.gptq_group_size,
                "awq_bits": self.quantization.awq_bits,
                "awq_group_size": self.quantization.awq_group_size,
            },
            "model": {
                "model_id": self.model.model_id,
                "torch_dtype": self.model.torch_dtype.value,
                "attn_implementation": self.model.attn_implementation,
            },
            "eval": {
                "tasks": self.eval.tasks,
                "num_fewshot": self.eval.num_fewshot,
                "limit": self.eval.limit,
            },
            "benchmark": {
                "warmup_runs": self.benchmark.warmup_runs,
                "benchmark_runs": self.benchmark.benchmark_runs,
                "input_lengths": self.benchmark.input_lengths,
                "batch_sizes": self.benchmark.batch_sizes,
            },
            "skip_eval": self.skip_eval,
            "skip_benchmark": self.skip_benchmark,
        }


# =============================================================================
# Pre-defined Experiment Configurations
# =============================================================================

def get_experiment(name: str) -> ExperimentConfig:
    """Get a pre-defined experiment configuration by name."""
    experiments = get_all_experiments()
    if name not in experiments:
        raise ValueError(f"Unknown experiment: {name}. Available: {list(experiments.keys())}")
    return experiments[name]


def get_all_experiments() -> Dict[str, ExperimentConfig]:
    """Get all pre-defined experiment configurations."""
    return {
        # Baseline
        "fp16_baseline": ExperimentConfig(
            name="fp16_baseline",
            description="FP16 baseline without quantization",
            quantization=QuantizationConfig(method=QuantMethod.NONE),
        ),
        
        # BitsAndBytes 8-bit
        "bnb_8bit": ExperimentConfig(
            name="bnb_8bit",
            description="BitsAndBytes 8-bit quantization",
            quantization=QuantizationConfig(method=QuantMethod.BITSANDBYTES_8BIT),
        ),
        
        # BitsAndBytes 4-bit NF4
        "bnb_4bit_nf4": ExperimentConfig(
            name="bnb_4bit_nf4",
            description="BitsAndBytes 4-bit with NF4 and double quantization",
            quantization=QuantizationConfig(
                method=QuantMethod.BITSANDBYTES_4BIT,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            ),
        ),
        
        # BitsAndBytes 4-bit FP4
        "bnb_4bit_fp4": ExperimentConfig(
            name="bnb_4bit_fp4",
            description="BitsAndBytes 4-bit with FP4 and double quantization",
            quantization=QuantizationConfig(
                method=QuantMethod.BITSANDBYTES_4BIT,
                bnb_4bit_quant_type="fp4",
                bnb_4bit_use_double_quant=True,
            ),
        ),
        
        # BitsAndBytes 4-bit NF4 without double quant
        "bnb_4bit_nf4_no_double": ExperimentConfig(
            name="bnb_4bit_nf4_no_double",
            description="BitsAndBytes 4-bit NF4 without double quantization",
            quantization=QuantizationConfig(
                method=QuantMethod.BITSANDBYTES_4BIT,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=False,
            ),
        ),
        
        # BitsAndBytes 4-bit with BF16 compute
        "bnb_4bit_nf4_bf16": ExperimentConfig(
            name="bnb_4bit_nf4_bf16",
            description="BitsAndBytes 4-bit NF4 with BF16 compute dtype",
            quantization=QuantizationConfig(
                method=QuantMethod.BITSANDBYTES_4BIT,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=ComputeDtype.BF16,
            ),
        ),
        
        # GPTQ configurations
        "gptq_4bit_g128": ExperimentConfig(
            name="gptq_4bit_g128",
            description="GPTQ 4-bit with group size 128",
            quantization=QuantizationConfig(
                method=QuantMethod.GPTQ,
                gptq_bits=4,
                gptq_group_size=128,
            ),
        ),
        
        "gptq_4bit_g32": ExperimentConfig(
            name="gptq_4bit_g32",
            description="GPTQ 4-bit with group size 32",
            quantization=QuantizationConfig(
                method=QuantMethod.GPTQ,
                gptq_bits=4,
                gptq_group_size=32,
            ),
        ),
        
        "gptq_3bit_g128": ExperimentConfig(
            name="gptq_3bit_g128",
            description="GPTQ 3-bit with group size 128",
            quantization=QuantizationConfig(
                method=QuantMethod.GPTQ,
                gptq_bits=3,
                gptq_group_size=128,
            ),
        ),
        
        # AWQ configurations
        "awq_4bit_g128": ExperimentConfig(
            name="awq_4bit_g128",
            description="AWQ 4-bit with group size 128",
            quantization=QuantizationConfig(
                method=QuantMethod.AWQ,
                awq_bits=4,
                awq_group_size=128,
            ),
        ),
    }



