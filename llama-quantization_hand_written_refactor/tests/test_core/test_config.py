"""Tests for configuration module."""

import pytest
import tempfile
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from llama_quant.core.config import (
    ExperimentConfig,
    QuantizationConfig,
    QuantMethod,
    ComputeDtype,
    get_experiment,
    get_all_experiments,
)


class TestQuantMethod:
    """Tests for QuantMethod enum."""
    
    def test_enum_values(self):
        assert QuantMethod.NONE.value == "none"
        assert QuantMethod.BITSANDBYTES_4BIT.value == "bnb_4bit"
        assert QuantMethod.GPTQ.value == "gptq"
        assert QuantMethod.AWQ.value == "awq"


class TestQuantizationConfig:
    """Tests for QuantizationConfig dataclass."""
    
    def test_default_values(self):
        config = QuantizationConfig()
        assert config.method == QuantMethod.BITSANDBYTES_4BIT
        assert config.bnb_4bit_quant_type == "nf4"
        assert config.bnb_4bit_use_double_quant is True
    
    def test_custom_values(self):
        config = QuantizationConfig(
            method=QuantMethod.GPTQ,
            gptq_bits=4,
            gptq_group_size=32,
        )
        assert config.method == QuantMethod.GPTQ
        assert config.gptq_bits == 4
        assert config.gptq_group_size == 32


class TestExperimentConfig:
    """Tests for ExperimentConfig dataclass."""
    
    def test_default_values(self):
        config = ExperimentConfig()
        assert config.name == "baseline"
        assert config.seed == 42
    
    def test_yaml_roundtrip(self):
        """Test saving and loading from YAML."""
        config = ExperimentConfig(
            name="test_experiment",
            description="Test description",
        )
        
        with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as f:
            config.to_yaml(f.name)
            loaded = ExperimentConfig.from_yaml(f.name)
        
        assert loaded.name == config.name
        assert loaded.description == config.description


class TestGetExperiment:
    """Tests for get_experiment function."""
    
    def test_get_valid_experiment(self):
        config = get_experiment("bnb_4bit_nf4")
        assert config.name == "bnb_4bit_nf4"
        assert config.quantization.method == QuantMethod.BITSANDBYTES_4BIT
    
    def test_get_invalid_experiment(self):
        with pytest.raises(ValueError):
            get_experiment("nonexistent_experiment")
    
    def test_get_all_experiments(self):
        experiments = get_all_experiments()
        assert "fp16_baseline" in experiments
        assert "bnb_4bit_nf4" in experiments
        assert len(experiments) >= 5
