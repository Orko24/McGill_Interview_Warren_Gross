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

## Next Steps

See `code_walkthrough_2.md` for breakdown of `gpu_runner.py` (where the actual GPU work happens).
