# Code Walkthrough Log #2: gpu_runner.py

**Date:** January 18, 2026  
**File:** `infra/gpu_runner.py`  
**Runs:** Modal cloud ONLY (needs torch, transformers, etc.)

---

## What This File Does

This is where **ALL the actual work happens**:
1. Load a model (FP16 or quantized)
2. Run CoQA evaluation
3. Measure performance (memory, latency, throughput)

---

## Part-by-Part Breakdown

### Part 1: Imports (lines 10-20)

```python
import torch
from llama_quant.core.config import get_experiment    # Get experiment settings
from llama_quant.models import load_model             # Load model with quantization
from llama_quant.evaluation import evaluate_model     # Run CoQA
from llama_quant.benchmark import BenchmarkSuite      # Measure performance
```

These imports **ONLY work in Modal cloud** where the packages are installed.

---

### Part 2: Data Classes (lines 23-53)

#### ExperimentRequest
```python
@dataclass
class ExperimentRequest:
    name: str              # e.g., "bnb_4bit_nf4"
    limit: Optional[int]   # How many CoQA samples (None = all)
```

**What it is:** Just a container saying "run this experiment with this many samples"

**Example:**
```python
request = ExperimentRequest(name="bnb_4bit_nf4", limit=50)
```

#### ExperimentResult
```python
@dataclass 
class ExperimentResult:
    experiment_name: str        # "bnb_4bit_nf4"
    status: str                 # "success" or "error"
    model_size_mb: float        # 965.13
    coqa_metrics: dict          # {"coqa_f1": 0.676, "coqa_em": 0.58}
    benchmarks: dict            # {"throughput": 199, "latency": 24.2}
    error: Optional[str]        # Error message if failed
```

**What it is:** Container for all results from one experiment

---

### Part 3: ExperimentRunner (lines 56-137)

This is the **main class** that runs a single experiment.

#### Constructor
```python
def __init__(self, hf_cache_dir: str = "/cache/huggingface"):
    self.hf_cache_dir = hf_cache_dir
    self._setup_environment()  # Set HF_HOME, etc.
```

Just sets up environment variables for HuggingFace caching.

#### The run() method - THE CORE LOGIC

```python
def run(self, request: ExperimentRequest) -> ExperimentResult:
```

**Step-by-step what it does:**

```python
# Step 1: Get the experiment configuration
exp_config = get_experiment(request.name)
# This returns a config object with model settings, quant settings, etc.
```

```python
# Step 2: Load the model (with quantization if specified)
model, tokenizer = load_model(exp_config)
model_size = get_model_size_mb(model)
# model_size = 965.13 for 4-bit, 2357.13 for FP16
```

```python
# Step 3: Run CoQA evaluation
eval_results = evaluate_model(model, tokenizer, exp_config)
coqa_metrics = eval_results.get("coqa_metrics", {})
# coqa_metrics = {"coqa_f1": 0.676, "coqa_em": 0.58}
```

```python
# Step 4: Run benchmarks (memory, latency, throughput)
suite = BenchmarkSuite.from_experiment_config(model, tokenizer, exp_config)
benchmark_results = suite.run_all()
# benchmark_results = {"throughput": 199, "latency_ms": 24.2, ...}
```

```python
# Step 5: Cleanup and return
del model, tokenizer
torch.cuda.empty_cache()

return ExperimentResult(
    experiment_name=request.name,
    status="success",
    model_size_mb=model_size,
    coqa_metrics=coqa_metrics,
    benchmarks=benchmark_results,
)
```

**If anything fails**, it catches the exception and returns an error result:
```python
except Exception as e:
    return ExperimentResult(
        experiment_name=request.name,
        status="error",
        error=str(e),
    )
```

---

### Part 4: ComparisonRunner (lines 140-198)

This runs **multiple experiments** and prints a summary.

```python
class ComparisonRunner:
    DEFAULT_EXPERIMENTS = ["fp16_baseline", "bnb_4bit_nf4", "bnb_4bit_fp4"]
    
    def run(self, experiments=None, limit=100):
        if experiments is None:
            experiments = self.DEFAULT_EXPERIMENTS
        
        results = []
        for exp_name in experiments:
            result = self.runner.run(ExperimentRequest(exp_name, limit))
            results.append(result)
        
        self._print_summary(results)
        return results
```

**What it does:**
1. Loop through each experiment name
2. Run ExperimentRunner for each one
3. Collect all results
4. Print a summary table

---

## Flow Diagram

```
ComparisonRunner.run(["fp16_baseline", "bnb_4bit_nf4", "bnb_4bit_fp4"])
│
├── Loop 1: ExperimentRunner.run("fp16_baseline")
│   ├── get_experiment("fp16_baseline") → config
│   ├── load_model(config) → FP16 model, tokenizer
│   ├── evaluate_model() → {f1: 0.625}
│   ├── BenchmarkSuite.run_all() → {throughput: 383, ...}
│   └── return ExperimentResult
│
├── Loop 2: ExperimentRunner.run("bnb_4bit_nf4")
│   ├── get_experiment("bnb_4bit_nf4") → config
│   ├── load_model(config) → NF4 quantized model, tokenizer
│   ├── evaluate_model() → {f1: 0.676}
│   ├── BenchmarkSuite.run_all() → {throughput: 199, ...}
│   └── return ExperimentResult
│
├── Loop 3: ExperimentRunner.run("bnb_4bit_fp4")
│   ├── get_experiment("bnb_4bit_fp4") → config
│   ├── load_model(config) → FP4 quantized model, tokenizer
│   ├── evaluate_model() → {f1: 0.587}
│   ├── BenchmarkSuite.run_all() → {throughput: 200, ...}
│   └── return ExperimentResult
│
└── Return [result1, result2, result3]
```

---

## Key Takeaways

1. **ExperimentRunner** = runs ONE experiment (load → eval → benchmark)
2. **ComparisonRunner** = loops over ExperimentRunner for multiple experiments
3. **Everything is try/except wrapped** so one failure doesn't crash the whole run
4. **Results are returned as dataclasses** that convert to dict for JSON serialization

---

## Next: What does `load_model()` do?

See `code_walkthrough_3.md` for the model loading logic.

