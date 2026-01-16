# Llama 3.2-1B Quantization Study

Systematic evaluation of quantization methods for Llama 3.2-1B, minimizing bit-width while maintaining accuracy on CoQA.

---

## 🚀 Quick Start with Modal (Recommended)

Modal handles everything in Docker containers on cloud GPUs. No VM setup needed.

### One-Time Setup
```bash
# 1. Install modal locally
pip install modal

# 2. Authenticate (opens browser)
modal setup

# 3. Add your HuggingFace token to Modal secrets
#    Go to: https://modal.com/secrets
#    Create secret named "huggingface" with key HF_TOKEN=<your_token>
```

### Run Experiments
```bash
# Quick comparison: FP16 vs 8-bit vs 4-bit (minimum viable)
modal run modal_app.py --quick --limit 50

# Single experiment
modal run modal_app.py --experiment bnb_4bit_nf4 --limit 100

# Full BnB sweep
modal run modal_app.py --sweep --method bnb --limit 200

# Test GPU is working
modal run modal_app.py::test_gpu
```

Results are saved to `./results/` locally.

---

## 🖥️ Local/SSH Quick Start (Alternative)

```bash
# Install dependencies
pip install -r requirements.txt

# Login to Hugging Face (required for Llama)
huggingface-cli login

# Run quick comparison (FP16 vs 8-bit vs 4-bit)
python sweep.py --quick --limit 50

# Run single experiment
python main.py --experiment bnb_4bit_nf4

# Run full sweep
python sweep.py --method all --limit 100
```

## Project Structure

```
├── config.py          # All hyperparameters and experiment configs
├── quantize.py        # Quantization methods (BnB, GPTQ, AWQ)
├── evaluate.py        # lm-evaluation-harness wrapper for CoQA
├── benchmark.py       # Memory, latency, throughput measurements
├── main.py            # Main orchestration script
├── sweep.py           # Hyperparameter sweep utilities
├── requirements.txt   # Dependencies
└── results/           # Output directory
```

## Quantization Methods

### BitsAndBytes (Recommended for Ease of Use)
- **8-bit**: Minimal accuracy loss, ~50% memory reduction
- **4-bit NF4**: Best quality 4-bit, uses normal distribution quantiles
- **4-bit FP4**: Alternative 4-bit, uses floating point quantization
- **Double quantization**: Further reduces memory by quantizing quantization constants

### GPTQ (Best for Deployment)
- Post-training quantization with calibration data
- Supports 2/3/4/8-bit precision
- Group size controls accuracy/compression tradeoff
- Requires ~128 calibration samples

### AWQ (Activation-Aware)
- Preserves salient weights based on activation patterns
- Generally better quality than GPTQ at same bit-width
- 4-bit only

## Key Hyperparameters

### BitsAndBytes 4-bit
| Parameter | Options | Notes |
|-----------|---------|-------|
| `quant_type` | `nf4`, `fp4` | NF4 typically better for LLMs |
| `double_quant` | `True`, `False` | Extra compression, minimal impact |
| `compute_dtype` | `fp16`, `bf16` | BF16 more stable if supported |

### GPTQ
| Parameter | Options | Notes |
|-----------|---------|-------|
| `bits` | 2, 3, 4, 8 | 4-bit is sweet spot |
| `group_size` | 32, 64, 128, -1 | Smaller = better accuracy, larger model |
| `desc_act` | `True`, `False` | Activation order, sometimes helps |
| `sym` | `True`, `False` | Symmetric quantization |

### AWQ
| Parameter | Options | Notes |
|-----------|---------|-------|
| `bits` | 4 | Only 4-bit supported |
| `group_size` | 32, 64, 128 | Similar to GPTQ |
| `zero_point` | `True`, `False` | Asymmetric quantization |

## Usage Examples

### Run Specific Configuration

```python
from config import ExperimentConfig, QuantizationConfig, QuantMethod
from main import run_experiment

config = ExperimentConfig(
    name="my_experiment",
    quantization=QuantizationConfig(
        method=QuantMethod.BITSANDBYTES_4BIT,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    ),
)

results = run_experiment(config)
```

### Custom Sweep

```python
from sweep import run_sweep
from config import ExperimentConfig, QuantizationConfig, QuantMethod

configs = [
    ExperimentConfig(
        name="4bit_g128",
        quantization=QuantizationConfig(
            method=QuantMethod.GPTQ,
            gptq_bits=4,
            gptq_group_size=128,
        ),
    ),
    ExperimentConfig(
        name="4bit_g64",
        quantization=QuantizationConfig(
            method=QuantMethod.GPTQ,
            gptq_bits=4,
            gptq_group_size=64,
        ),
    ),
]

summary = run_sweep(configs, eval_limit=100)
```

### Debugging with Limited Samples

```bash
# Quick iteration with 50 eval samples
python main.py --experiment bnb_4bit_nf4 --limit 50
```

## Expected Results

Based on typical quantization performance:

| Method | Bits | Model Size | CoQA F1 (approx) |
|--------|------|------------|------------------|
| FP16 (baseline) | 16 | ~2GB | ~0.75-0.80 |
| BnB 8-bit | 8 | ~1GB | ~0.74-0.79 |
| BnB 4-bit NF4 | 4 | ~0.6GB | ~0.70-0.76 |
| GPTQ 4-bit | 4 | ~0.6GB | ~0.72-0.77 |
| GPTQ 3-bit | 3 | ~0.45GB | ~0.65-0.72 |
| AWQ 4-bit | 4 | ~0.6GB | ~0.73-0.77 |

*Note: Actual results will vary. Run experiments to get precise numbers.*

## Hardware Requirements

- **Minimum**: 8GB VRAM (T4, RTX 3070)
- **Recommended**: 16GB+ VRAM (A100, RTX 4090)
- Llama 3.2-1B is small enough to run on free Colab T4

## Cloud Options

### RunPod / Vast.ai (Recommended)
```bash
# SSH into instance, then:
git clone <your-repo>
cd llama-quantization
pip install -r requirements.txt
python sweep.py --quick
```

### Modal
```python
import modal

app = modal.App("llama-quant")

@app.function(gpu="T4")
def run_quant():
    # Your code here
    pass
```

## Report Structure (4 pages)

1. **Introduction** (0.5 pages)
   - Problem statement
   - Why quantization matters

2. **Methods** (1 page)
   - Quantization techniques overview
   - Experimental setup

3. **Results** (1.5 pages)
   - Accuracy vs compression curves
   - Latency/throughput analysis
   - Memory footprint comparison

4. **Discussion** (1 page)
   - Best configurations
   - Tradeoffs and recommendations
   - Limitations

## Citation

If you use this code, please cite:

```bibtex
@misc{llama-quant-2024,
  title={Quantization Analysis for Llama 3.2-1B},
  author={Your Name},
  year={2024},
}
```

## License

MIT
