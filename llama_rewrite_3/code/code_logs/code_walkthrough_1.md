# Code Walkthrough Log #1: modal_app.py (Entry Point)

**Date:** January 18, 2026  
**File:** `infra/modal_app.py`  
**Purpose:** Understand the entry point of the codebase

---

## Conversation Log

### User Request:
> "ok .... like even as a software engineering... the excessive abstraction on the code is messing with me. i think lets walk through this code piece by code piece. entry point so i understand exactly whats going on... start with the entry point. what we'll work through this like im 5"

### Response:

Let's walk through `modal_app.py` - the file you run with `modal run code/infra/modal_app.py`

---

## What is this file?

This file does **3 things**:
1. Sets up Modal (cloud GPU service)
2. Sends work to GPU in the cloud
3. Saves results locally

---

## Part-by-Part Breakdown

### Part 1: The Imports (lines 17-38)

```python
import json          # To save results as JSON files
import modal         # The cloud GPU service
from pathlib import Path  # To handle file paths
```

**Q: What's that weird try/except block at lines 34-38?**

```python
try:
    from infra.gpu_runner import ExperimentRunner, ExperimentRequest, ComparisonRunner
    GPU_IMPORTS_AVAILABLE = True
except ImportError:
    GPU_IMPORTS_AVAILABLE = False
```

**A:** This exists because:
- **On your laptop**: `gpu_runner.py` imports FAIL (no torch installed locally)
- **In Modal cloud**: `gpu_runner.py` imports WORK (torch is installed there)

We don't actually need those imports locally - they only run in the cloud. The try/except prevents your laptop from crashing when you run the command.

---

### Part 2: Configuration (lines 45-53)

```python
@dataclass
class ModalConfig:
    app_name: str = "llama-quantization"  # Name of the Modal app
    gpu_type: str = "A100"                 # Which GPU to use
    timeout: int = 3600                    # 1 hour max runtime
    cache_dir: str = "/cache"              # Where to cache stuff
    hf_cache_dir: str = "/cache/huggingface"  # Where to cache models
```

**Q: What is this?**

**A:** Just settings. Nothing runs here. It's a container for configuration values.

**Q: Why A100?**

**A:** We tried A10G first but hit an 8-bit CUDA bug. A100 has more memory (40GB vs 24GB) and is more compatible.

---

### Part 3: Modal Setup (lines 61-86)

```python
CONFIG = ModalConfig()  # Create config instance

app = modal.App(CONFIG.app_name)  # Create Modal app called "llama-quantization"

image = (
    modal.Image.debian_slim(python_version="3.11")  # Start with clean Linux
    .apt_install("git")                              # Install git
    .pip_install(                                    # Install Python packages
        "torch>=2.1.0",
        "transformers>=4.36.0",
        "accelerate>=0.25.0",
        "bitsandbytes>=0.43.0",
        # ... more packages
    )
    .add_local_python_source("code")  # Upload our code folder
)

volume = modal.Volume.from_name("llama-quant-cache", create_if_missing=True)
hf_secret = modal.Secret.from_name("huggingface-secret")
```

**Q: What does this do?**

**A:** This tells Modal:
- What Linux image to use (debian)
- What packages to install (torch, transformers, etc.)
- What code to upload (our `code/` folder)
- Where to store cached models (volume)
- How to authenticate with HuggingFace (secret)

**Q: When does this run?**

**A:** When you first run `modal run`, Modal builds this image in the cloud. After that, it's cached.

---

### Part 4: ResultsManager (lines 93-126)

```python
class ResultsManager:
    """Manages local storage of experiment results."""
    
    def __init__(self, results_dir: Path):
        self.results_dir = results_dir
        self.results_dir.mkdir(parents=True, exist_ok=True)  # Create folder if needed
    
    def save(self, results: Dict[str, Any]) -> Path:
        """Save results with timestamp metadata."""
        output_path = self._get_next_filename()  # results1.json, results2.json, ...
        
        results["timestamp"] = datetime.now().isoformat()
        results["saved_to"] = str(output_path)
        
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        
        return output_path
    
    def _get_next_filename(self) -> Path:
        """Get next available results filename."""
        # Finds existing files: results1.json, results2.json
        # Returns next number: results3.json
```

**Q: What does this do?**

**A:** Simple helper that:
1. Creates the results folder if it doesn't exist
2. Saves JSON files with auto-incrementing names (results1.json, results2.json, ...)
3. Adds timestamp to the results

**Q: Where does it save?**

**A:** To `llama_rewrite_2/results/` folder on your laptop.

---

### Part 5: GPU Functions (lines 133-163)

```python
@app.function(
    image=image,           # Use the image we defined above
    gpu=CONFIG.gpu_type,   # A100
    timeout=CONFIG.timeout, # 1 hour
    secrets=[hf_secret],   # HuggingFace token
    volumes={CONFIG.cache_dir: volume},  # Persistent storage
)
def run_single_experiment(experiment_name: str, limit: Optional[int] = None) -> dict:
    """Run single experiment on GPU."""
    runner = ExperimentRunner(CONFIG.hf_cache_dir)
    request = ExperimentRequest(name=experiment_name, limit=limit)
    result = runner.run(request)
    
    volume.commit()  # Save cached models
    return result.to_dict()
```

**Q: What does `@app.function` do?**

**A:** It's a decorator that says: "Run this function on Modal's cloud GPU, not my laptop"

When you call `run_single_experiment.remote(...)`, Modal:
1. Spins up a cloud GPU
2. Loads the image with all packages
3. Uploads your code
4. Runs the function
5. Returns the result

**Q: What's inside the function?**

**A:** 
1. Create an `ExperimentRunner` (from gpu_runner.py)
2. Create a request saying what experiment to run
3. Run it and get results
4. Save cached models to the volume
5. Return results as a dictionary

---

### Part 6: Entry Point - main() (lines 170-217)

```python
@app.local_entrypoint()
def main(
    experiment: str = None,       # --experiment nf4
    all_experiments: bool = False, # --all
    limit: int = 100,             # --limit 50
):
```

**Q: What does `@app.local_entrypoint()` mean?**

**A:** This is what runs on YOUR LAPTOP when you type `modal run code/infra/modal_app.py`

**The logic inside:**

```python
    # Create results manager (saves to llama_rewrite_2/results/)
    results_manager = ResultsManager(
        Path(__file__).parent.parent.parent / "results"
    )
    
    # Decide what to run based on command line args
    if all_experiments:
        # Run ALL experiments (FP16, NF4, FP4, etc.)
        experiments = ["fp16_baseline", "bnb_4bit_nf4", "bnb_4bit_fp4", ...]
        results = run_comparison.remote(experiments=experiments, limit=limit)
        
    elif experiment:
        # Run ONE specific experiment
        results = run_single_experiment.remote(experiment, limit=limit)
        
    else:
        # Default: run comparison of FP16 vs NF4 vs FP4
        results = run_comparison.remote(limit=limit)
    
    # Save results locally
    output_path = results_manager.save(results)
    print(f"Results saved to: {output_path}")
```

**Q: What does `.remote()` mean?**

**A:** "Send this to the cloud and wait for results"

- `run_comparison.remote()` = Run `run_comparison()` on Modal's cloud GPU
- The results come back as a Python dictionary
- Then we save them locally

---

## Complete Flow Diagram

```
YOU TYPE: modal run code/infra/modal_app.py --limit 50

YOUR LAPTOP                              MODAL CLOUD (A100 GPU)
───────────                              ──────────────────────

1. main() starts
   │
2. results_manager created
   │
3. Calls run_comparison.remote()
   │                                     
   └─────────────────────────────────────► 4. Modal receives request
                                              - Spins up A100 GPU
                                              - Installs packages (cached)
                                              - Uploads code folder
                                           
                                           5. run_comparison() executes
                                              - Loads FP16 model
                                              - Evaluates on CoQA
                                              - Runs benchmarks
                                              - Repeats for NF4, FP4
                                           
   ◄───────────────────────────────────── 6. Returns results dict

7. results_manager.save()
   │
   └── Writes results/results3.json

8. Prints "Results saved to: ..."
```

---

## Command Line Examples

```bash
# Default: Compare FP16 vs NF4 vs FP4 with 100 samples
modal run code/infra/modal_app.py

# Same but with only 50 samples (faster)
modal run code/infra/modal_app.py --limit 50

# Run only NF4 experiment
modal run code/infra/modal_app.py --experiment bnb_4bit_nf4

# Run ALL experiments (including ablations)
modal run code/infra/modal_app.py --all --limit 500
```

---

## Key Files Referenced

| File | What it does | Where it runs |
|------|--------------|---------------|
| `modal_app.py` | Entry point, orchestration | Your laptop |
| `gpu_runner.py` | Actual experiment logic | Modal cloud |
| `results/results1.json` | Output | Your laptop |

---

---

## Continuing the Walkthrough

### User Question:
> "cool so now run_comparison. where is this defined. again update everything on code walkthrough 1 by 1 as you would any log file"

### Answer:

`run_comparison` is defined in **the same file** (`modal_app.py`) at lines 150-163:

```python
@app.function(
    image=image,                    # Use our Docker image with torch, etc.
    gpu=CONFIG.gpu_type,            # A100
    timeout=CONFIG.timeout * 2,     # 2 hours (longer for multiple experiments)
    secrets=[hf_secret],            # HuggingFace token
    volumes={CONFIG.cache_dir: volume},  # Persistent model cache
)
def run_comparison(experiments: Optional[List[str]] = None, limit: int = 100) -> dict:
    """Run comparison across multiple experiments on GPU."""
    runner = ComparisonRunner(CONFIG.hf_cache_dir)
    results = runner.run(experiments, limit)
    
    volume.commit()
    return {"experiments": [r.to_dict() for r in results]}
```

### Breaking it down:

**Line 150-156: The decorator `@app.function(...)`**

This tells Modal: "When someone calls `run_comparison.remote()`, run this function on a cloud GPU with these settings"

| Setting | Value | Meaning |
|---------|-------|---------|
| `image` | our custom image | Use Debian + torch + transformers |
| `gpu` | "A100" | Use an NVIDIA A100 GPU |
| `timeout` | 7200 seconds | 2 hour max runtime |
| `secrets` | HF token | Authenticate with HuggingFace |
| `volumes` | cache volume | Persistent storage for models |

**Line 157: Function signature**

```python
def run_comparison(experiments: Optional[List[str]] = None, limit: int = 100) -> dict:
```

- `experiments`: List of experiment names like `["fp16_baseline", "bnb_4bit_nf4"]`, or None for defaults
- `limit`: How many CoQA samples to evaluate
- Returns: A dictionary with results

**Line 159: Create ComparisonRunner**

```python
runner = ComparisonRunner(CONFIG.hf_cache_dir)
```

`ComparisonRunner` is imported from `gpu_runner.py`. This is where we jump to the next file.

**Line 160: Run the experiments**

```python
results = runner.run(experiments, limit)
```

This is the actual work - runs FP16, NF4, FP4 and returns results.

**Line 162: Save cached models**

```python
volume.commit()
```

Persists the HuggingFace model cache so next run is faster.

**Line 163: Return results as dict**

```python
return {"experiments": [r.to_dict() for r in results]}
```

Converts `ExperimentResult` objects to dictionaries for JSON serialization.

---

### So where does the ACTUAL work happen?

```
modal_app.py                          gpu_runner.py
────────────                          ──────────────

run_comparison()
    │
    └── runner = ComparisonRunner()
            │
            └── runner.run(experiments, limit)
                    │
                    └── This is in gpu_runner.py!
```

**Next:** Let's look at `ComparisonRunner` in `gpu_runner.py`

---

### User Confirmation:
> "oh ok so ComparisonRunner is from the gpu runner right... just saw lines 150-153"

### Answer:

**Yes!** Look at line 35 of `modal_app.py`:

```python
from infra.gpu_runner import ExperimentRunner, ExperimentRequest, ComparisonRunner
```

So the import chain is:

```
modal_app.py
    │
    ├── line 35: from infra.gpu_runner import ComparisonRunner
    │
    └── line 159: runner = ComparisonRunner(CONFIG.hf_cache_dir)
                      │
                      └── This class is defined in gpu_runner.py!
```

**Now let's jump to `gpu_runner.py` to see what `ComparisonRunner` does.**

---

## Jumping to gpu_runner.py

See `code_walkthrough_2.md` for the full breakdown. Here's the key part:

### ComparisonRunner (gpu_runner.py, lines 140-198)

```python
class ComparisonRunner:
    """Runs multiple experiments and compares results."""
    
    DEFAULT_EXPERIMENTS = ["fp16_baseline", "bnb_4bit_nf4", "bnb_4bit_fp4"]
    
    def __init__(self, hf_cache_dir: str = "/cache/huggingface"):
        self.runner = ExperimentRunner(hf_cache_dir)
    
    def run(self, experiments=None, limit=100) -> List[ExperimentResult]:
        if experiments is None:
            experiments = self.DEFAULT_EXPERIMENTS  # ["fp16_baseline", "bnb_4bit_nf4", "bnb_4bit_fp4"]
        
        results = []
        for exp_name in experiments:
            request = ExperimentRequest(name=exp_name, limit=limit)
            result = self.runner.run(request)  # ← ExperimentRunner does the work
            results.append(result)
        
        return results
```

**What this does:**

1. If no experiments specified, use defaults: FP16, NF4, FP4
2. Loop through each experiment name
3. For each one, call `ExperimentRunner.run()`
4. Collect all results and return them

**The chain continues:**

```
ComparisonRunner.run()
    │
    └── for each experiment:
            │
            └── ExperimentRunner.run(request)
                    │
                    └── WHERE IS THIS? (Same file, lines 56-137)
```

---

---

### User Question:
> "aight explain the gpu runner cuz where did you get the coqa metrics from kinda confused here"

### Answer: The CoQA Metrics Chain

Let me trace exactly where CoQA metrics come from:

**Step 1: gpu_runner.py (lines 106-110)**
```python
# Evaluate
print("Running evaluation...")
eval_results = evaluate_model(model, tokenizer, exp_config, skip_sanity_check=True)
coqa_metrics = eval_results.get("coqa_metrics", {})
```

**Step 2: Where is `evaluate_model`?**

It's imported at line 19:
```python
from llama_quant.evaluation import evaluate_model
```

**Step 3: llama_quant/evaluation/harness.py (lines 67-113)**

```python
def evaluate_model(model, tokenizer, config, skip_sanity_check=False):
    """Main evaluation entry point."""
    all_results = {}
    
    # Run lm-eval (the actual CoQA evaluation)
    eval_results = run_lm_eval(model, tokenizer, config)
    all_results["lm_eval_results"] = eval_results
    
    # Extract CoQA metrics from raw results
    coqa_metrics = extract_coqa_metrics(eval_results)
    all_results["coqa_metrics"] = coqa_metrics
    
    return all_results
```

**Step 4: What does `run_lm_eval` do? (lines 18-64)**

```python
def run_lm_eval(model, tokenizer, config):
    import lm_eval
    from lm_eval.models.huggingface import HFLM
    
    # Wrap our model for lm-eval library
    lm = HFLM(pretrained=model, tokenizer=tokenizer, ...)
    
    # Run actual evaluation!
    results = lm_eval.simple_evaluate(
        model=lm,
        tasks=["coqa"],      # The CoQA benchmark
        limit=config.limit,  # e.g., 50 samples
    )
    
    return results
```

**Step 5: What does `extract_coqa_metrics` do? (coqa.py lines 19-69)**

```python
def extract_coqa_metrics(results):
    metrics = {}
    
    # Find CoQA results in the raw output
    for task_name, task_results in results["results"].items():
        if "coqa" in task_name:
            metrics["coqa_f1"] = task_results["f1"]      # e.g., 0.676
            metrics["coqa_em"] = task_results["em"]      # e.g., 0.58
    
    return metrics
```

---

### The Complete Flow Diagram:

```
gpu_runner.py
    │
    └── evaluate_model(model, tokenizer, config)
            │
            └── harness.py
                    │
                    ├── run_lm_eval()
                    │       │
                    │       └── lm_eval.simple_evaluate(tasks=["coqa"])
                    │               │
                    │               └── Returns raw results dict with F1, EM
                    │
                    └── extract_coqa_metrics(raw_results)
                            │
                            └── Returns {"coqa_f1": 0.676, "coqa_em": 0.58}
```

---

### In Plain English:

1. **gpu_runner.py** calls `evaluate_model()`
2. **harness.py** wraps our model for `lm-eval` library
3. **lm-eval** (external library) runs the actual CoQA benchmark
4. **coqa.py** extracts F1 and EM scores from raw results
5. Results flow back: `{"coqa_f1": 0.676, "coqa_em": 0.58}`

---

### Key Files:

| File | What it does |
|------|--------------|
| `gpu_runner.py` | Calls evaluate_model() |
| `evaluation/harness.py` | Wraps model for lm-eval |
| `evaluation/coqa.py` | Extracts F1/EM from results |
| `lm_eval` (external) | Actually runs CoQA benchmark |

---

---

### User Question:
> "@coqa.py (1-139) why arent they in classes and how did u get them... are they being pulled from hugging face?"

### Answer Part 1: Why No Classes?

These are **stateless utility functions** - they don't need classes.

```python
def extract_coqa_metrics(results):  # Input → Output, no state
    metrics = {}
    # ... process results ...
    return metrics
```

**When to use a class:**
- Need to maintain **state** (e.g., `self.model`, `self.cache`)
- Need **initialization** (e.g., loading something)
- Functions share data between calls

**When functions are fine:**
- Stateless transformations (input dict → output dict)
- Simple utilities
- No shared state needed

`extract_coqa_metrics()` takes a dict, extracts values, returns a dict. No state = no class needed.

**This is actually GOOD design** - not everything needs OOP. Python isn't Java.

---

### Answer Part 2: Where Does CoQA Data Come From?

**Yes, from HuggingFace!** But not directly - through the `lm-eval` library.

```python
# harness.py
results = lm_eval.simple_evaluate(
    model=lm,
    tasks=["coqa"],  # ← This tells lm-eval to use CoQA benchmark
)
```

**What happens inside `lm_eval`:**

```
lm_eval.simple_evaluate(tasks=["coqa"])
    │
    └── lm-eval library internally:
            │
            ├── 1. Downloads CoQA dataset from HuggingFace
            │      URL: https://huggingface.co/datasets/stanfordnlp/coqa
            │      (cached after first download)
            │
            ├── 2. For each question in dataset:
            │      - Feeds question + context to model
            │      - Gets model's answer
            │
            ├── 3. Compares model answers to ground truth
            │
            └── 4. Calculates metrics:
                   - F1: Word overlap between prediction and answer
                   - EM: Exact string match percentage
                         │
                         └── Returns: {"results": {"coqa": {"f1": 0.676, "em": 0.58}}}
```

**Our code's role:**

| Component | What it does |
|-----------|--------------|
| `lm-eval` (external) | Downloads data, runs benchmark, calculates scores |
| `harness.py` | Wraps our model so lm-eval can use it |
| `coqa.py` | Extracts F1/EM from lm-eval's messy output |

**We don't implement CoQA evaluation ourselves** - we use the standard `lm-eval` library that everyone uses. This ensures our results are comparable to published papers.

---

### The coqa.py Functions Explained:

| Function | Purpose |
|----------|---------|
| `extract_coqa_metrics()` | Pull F1/EM from lm-eval's nested dict |
| `run_sanity_check()` | Quick test: can model generate text? |
| `format_coqa_results()` | Pretty-print metrics for logging |

All simple utilities. No state. No classes needed.

---

---

### User Confirmation:
> "@harness.py (104-108) lines 104-108 you mean huh"

### Answer:

**Yes!** Lines 104-108 in `harness.py`:

```python
# Line 104: Run the benchmark
eval_results = run_lm_eval(model, tokenizer, config)

# Line 105: Store raw results  
all_results["lm_eval_results"] = eval_results

# Line 108: Extract F1 and EM scores
coqa_metrics = extract_coqa_metrics(eval_results)
```

**The chain:**

```
Line 104: run_lm_eval(model, tokenizer, config)
    │
    └── Calls lm_eval.simple_evaluate(tasks=["coqa"])
            │
            ├── Downloads CoQA dataset from HuggingFace
            ├── Runs model on questions
            └── Returns: {"results": {"coqa": {"f1": 0.676, "em": 0.58}}}

Line 108: extract_coqa_metrics(eval_results)
    │
    └── Returns: {"coqa_f1": 0.676, "coqa_em": 0.58}
```

---

## Summary So Far

We've traced the full path from entry point to CoQA metrics:

```
modal run code/infra/modal_app.py
    │
    └── modal_app.py: main() → run_comparison.remote()
            │
            └── gpu_runner.py: ComparisonRunner → ExperimentRunner.run()
                    │
                    ├── load_model() → Loads FP16/NF4/FP4 model
                    │
                    └── evaluate_model() → harness.py
                            │
                            ├── run_lm_eval() → lm_eval.simple_evaluate()
                            │       │
                            │       └── HuggingFace CoQA dataset
                            │
                            └── extract_coqa_metrics() → {"coqa_f1": 0.676}
```

---

---

### User Returns:
> "k im back so where were we... and its probably better we just finish up our understanding here"

### Final Section: Model Loading (bnb.py)

**How quantization actually works:**

```python
# bnb.py lines 42-56

# Step 1: Create quantization config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,                    # Use 4-bit quantization
    bnb_4bit_quant_type="nf4",            # NF4 (distribution-aware) or FP4 (uniform)
    bnb_4bit_use_double_quant=True,       # Quantize the scales too
)

# Step 2: Load model - quantization happens automatically!
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.2-1B",
    quantization_config=bnb_config,       # ← This triggers quantization
)
```

**That's it.** The `transformers` library + `bitsandbytes` handle everything.

---

## COMPLETE FLOW (Final Summary)

```
YOU RUN: modal run code/infra/modal_app.py --limit 50

┌─────────────────────────────────────────────────────────────┐
│ YOUR LAPTOP (modal_app.py)                                  │
├─────────────────────────────────────────────────────────────┤
│ main() → run_comparison.remote() → sends to cloud           │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│ MODAL CLOUD GPU (gpu_runner.py)                             │
├─────────────────────────────────────────────────────────────┤
│ ComparisonRunner loops through ["fp16", "nf4", "fp4"]:      │
│                                                             │
│   ExperimentRunner.run():                                   │
│     │                                                       │
│     ├── 1. load_model() ← bnb.py                           │
│     │       └── BitsAndBytesConfig + from_pretrained()     │
│     │                                                       │
│     ├── 2. evaluate_model() ← harness.py                   │
│     │       └── lm_eval.simple_evaluate(tasks=["coqa"])    │
│     │           └── Downloads from HuggingFace, runs test  │
│     │                                                       │
│     └── 3. BenchmarkSuite.run_all()                        │
│             └── Measures memory, latency, throughput        │
│                                                             │
│   Returns: {f1: 0.676, memory: 965MB, ...}                 │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│ YOUR LAPTOP (modal_app.py)                                  │
├─────────────────────────────────────────────────────────────┤
│ ResultsManager.save() → results/results2.json               │
└─────────────────────────────────────────────────────────────┘
```

---

## Key Files Summary

| File | Purpose | Key Function |
|------|---------|--------------|
| `modal_app.py` | Entry point, orchestration | `main()`, `run_comparison()` |
| `gpu_runner.py` | Runs experiments on GPU | `ExperimentRunner.run()` |
| `bnb.py` | Loads quantized models | `BitsAndBytes4BitLoader.load()` |
| `harness.py` | Runs CoQA evaluation | `run_lm_eval()` |
| `coqa.py` | Extracts F1/EM scores | `extract_coqa_metrics()` |

---

## What the Assignment Required (and what AI helped with)

The user noted: "this assignment requires AI tools right... there were a ton of stuff you cant just expect me to know"

**Things that required specialized knowledge:**
- Modal serverless GPU setup
- BitsAndBytes quantization config options
- lm-evaluation-harness library usage
- CUDA kernel bugs and workarounds (8-bit issue)
- HuggingFace transformers API

**The codebase demonstrates:**
1. Clean separation of concerns
2. Modular architecture
3. Reproducible experiments
4. Proper error handling

---

## HOW QUANTIZATION IS APPLIED TO HUGGINGFACE MODELS

**User Question:** How is quantization applied to the HuggingFace model?

### The Flow

```
ExperimentRunner.run()
    │
    └── load_model(config)                    ← base.py line 136
            │
            ├── method = config.quantization.method
            │
            └── loader = _get_loader(method)  ← Factory pattern (base.py line 115)
                    │
                    └── Returns the appropriate Loader class
```

### The Factory (_get_loader)

```python
# base.py lines 122-128
loaders = {
    QuantMethod.NONE: FP16Loader(),
    QuantMethod.BITSANDBYTES_4BIT: BitsAndBytes4BitLoader(),
    QuantMethod.BITSANDBYTES_8BIT: BitsAndBytes8BitLoader(),
    QuantMethod.GPTQ: GPTQLoader(),
    QuantMethod.AWQ: AWQLoader(),
}
```

### Key Insight: All Quantization Goes Through Transformers

All quantization methods use the **same pattern**:

1. Create a quantization config object
2. Pass it to `AutoModelForCausalLM.from_pretrained()`
3. Transformers handles the quantization internally

### BitsAndBytes (bnb.py)

```python
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",  # or "fp4"
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.2-1B",
    quantization_config=bnb_config,  # ← Quantization applied here
)
```

### AWQ (awq.py lines 106-126)

```python
from transformers import AutoModelForCausalLM, AwqConfig

awq_config = AwqConfig(
    bits=4,
    group_size=128,
    zero_point=True,
    version="GEMM",
)

model = AutoModelForCausalLM.from_pretrained(
    config.model.model_id,
    quantization_config=awq_config,  # ← Same pattern
)
```

### GPTQ (gptq.py lines 116-138)

```python
from transformers import AutoModelForCausalLM, GPTQConfig

gptq_config = GPTQConfig(
    bits=4,
    group_size=128,
    desc_act=True,
    sym=True,
    damp_percent=0.1,
    dataset="c4",
)

model = AutoModelForCausalLM.from_pretrained(
    config.model.model_id,
    quantization_config=gptq_config,  # ← Same pattern
)
```

### Summary

**All 4 quantization methods (FP16, BnB, AWQ, GPTQ) follow the same pattern:**

| Method | Config Class | Key Parameter |
|--------|--------------|---------------|
| FP16 (baseline) | None | `dtype=torch.float16` |
| BitsAndBytes 4-bit | `BitsAndBytesConfig` | `load_in_4bit=True` |
| BitsAndBytes 8-bit | `BitsAndBytesConfig` | `load_in_8bit=True` |
| AWQ | `AwqConfig` | `bits=4` |
| GPTQ | `GPTQConfig` | `bits=4` |

The `transformers` library does all the heavy lifting. Our code just:
1. Picks the right config class based on the experiment
2. Sets the hyperparameters from our YAML config
3. Calls `from_pretrained()` with `quantization_config=...`

---

---

## DISCOVERY: AWQ AND GPTQ WERE NOT BEING RUN

**User Question:** Shouldn't we run gptq_4bit_g128, gptq_4bit_g32, gptq_3bit_g128, awq_4bit_g128?

### Finding

The configs were **DEFINED** in `config.py` (lines 308-347) but **NOT INCLUDED** in the default experiments:

```python
# gpu_runner.py line 150 (BEFORE)
DEFAULT_EXPERIMENTS = ["fp16_baseline", "bnb_4bit_nf4", "bnb_4bit_fp4"]
```

AWQ and GPTQ loaders existed but were never used!

### Fix Applied

Updated `gpu_runner.py` to include GPTQ and AWQ:

```python
# gpu_runner.py (AFTER)
DEFAULT_EXPERIMENTS = [
    "fp16_baseline", 
    "bnb_4bit_nf4", 
    "bnb_4bit_fp4",
    "gptq_4bit_g128",
    "awq_4bit_g128",
]
```

Also added required packages to `modal_app.py`:
- `auto-gptq>=0.7.0`
- `autoawq>=0.2.0`

---

## END OF WALKTHROUGH

This log documents the complete code understanding session.
Date: January 18, 2026
