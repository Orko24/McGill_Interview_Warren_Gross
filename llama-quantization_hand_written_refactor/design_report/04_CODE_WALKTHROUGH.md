# Complete Code Walkthrough - Line by Line

## Overview

This document walks through every file in the codebase, explaining what each section does in plain English.

---

## File 1: `config.py` — The Settings Hub

### Purpose
Store ALL configuration in one place so you never have to hunt through multiple files to change something.

### Key Sections

#### 1. Enum Definitions (Lines 11-22)
```python
class QuantMethod(Enum):
    NONE = "none"              # No quantization (FP16 baseline)
    BITSANDBYTES_8BIT = "bnb_8bit"   # 8-bit quantization
    BITSANDBYTES_4BIT = "bnb_4bit"   # 4-bit quantization
    GPTQ = "gptq"              # GPTQ method
    AWQ = "awq"                # AWQ method

class ComputeDtype(Enum):
    FP16 = "float16"           # 16-bit floats
    BF16 = "bfloat16"          # Brain Float 16
    FP32 = "float32"           # 32-bit floats (rarely used)
```

**Why Enums?** Prevents typos. Instead of typing "nf4" and maybe mistyping "nf5", you use `QuantMethod.NF4` which your IDE autocompletes.

#### 2. Quantization Config (Lines 26-47)
```python
@dataclass
class QuantizationConfig:
    method: QuantMethod = QuantMethod.BITSANDBYTES_4BIT
    
    # THE KEY KNOBS:
    bnb_4bit_quant_type: str = "nf4"        # "nf4" or "fp4"
    bnb_4bit_use_double_quant: bool = True  # Extra compression
    bnb_4bit_compute_dtype: ComputeDtype = ComputeDtype.FP16
```

**What this controls:**
- `quant_type`: Which 4-bit format (NF4 wins!)
- `double_quant`: Compress the scales too (free!)
- `compute_dtype`: What precision for math operations

#### 3. Experiment Definitions (Lines 109-173)
```python
EXPERIMENTS = {
    "fp16_baseline": ExperimentConfig(
        name="fp16_baseline",
        quantization=QuantizationConfig(method=QuantMethod.NONE),
    ),
    
    "bnb_4bit_nf4": ExperimentConfig(
        name="bnb_4bit_nf4",
        quantization=QuantizationConfig(
            method=QuantMethod.BITSANDBYTES_4BIT,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        ),
    ),
    # ... more experiments
}
```

**How to add a new experiment:**
```python
"my_new_experiment": ExperimentConfig(
    name="my_new_experiment",
    quantization=QuantizationConfig(
        method=QuantMethod.BITSANDBYTES_4BIT,
        bnb_4bit_quant_type="fp4",  # Try FP4 this time
    ),
),
```

---

## File 2: `quantize.py` — Model Loading

### Purpose
Load a model with the specified quantization settings.

### Key Function: `load_quantized_model()`

```python
def load_quantized_model(config: ExperimentConfig):
    """
    This is the main entry point.
    Give it a config, get back a model + tokenizer.
    """
    
    # Step 1: Load tokenizer (same for all configs)
    tokenizer = load_tokenizer(config)
    
    # Step 2: Route to correct loader based on method
    method = config.quantization.method
    
    if method == QuantMethod.NONE:
        model = load_model_fp16(config)        # No quantization
    elif method in [QuantMethod.BITSANDBYTES_8BIT, QuantMethod.BITSANDBYTES_4BIT]:
        model = load_model_bnb(config)         # BitsAndBytes
    elif method == QuantMethod.GPTQ:
        model = load_model_gptq(config)        # GPTQ
    elif method == QuantMethod.AWQ:
        model = load_model_awq(config)         # AWQ
    
    return model, tokenizer
```

### BitsAndBytes Loading (The Important One)

```python
def load_model_bnb(config: ExperimentConfig):
    # Create the BnB config from our settings
    bnb_config = create_bnb_config(config)
    
    # Load model with quantization applied automatically
    model = AutoModelForCausalLM.from_pretrained(
        config.model.model_id,           # "meta-llama/Llama-3.2-1B"
        quantization_config=bnb_config,  # Apply quantization!
        device_map="auto",               # Let it figure out GPU
        attn_implementation="sdpa",      # Use fast attention
    )
    
    return model

def create_bnb_config(config):
    """Turn our config into what BitsAndBytes expects"""
    
    if config.quantization.method == QuantMethod.BITSANDBYTES_4BIT:
        return BitsAndBytesConfig(
            load_in_4bit=True,                      # Enable 4-bit
            bnb_4bit_quant_type="nf4",             # NF4 format
            bnb_4bit_use_double_quant=True,        # Extra compression
            bnb_4bit_compute_dtype=torch.float16,  # Math in FP16
        )
```

**What happens when this runs:**
1. HuggingFace downloads model weights (cached after first time)
2. BitsAndBytes quantizes each layer as it's loaded
3. Model ends up on GPU, already quantized
4. No extra steps needed!

---

## File 3: `evaluate.py` — Measuring Accuracy

### Purpose
Run the CoQA benchmark and return accuracy metrics.

### Main Function

```python
def evaluate_model(model, tokenizer, config):
    """
    Returns: {
        "sanity_check": [...],      # Quick test outputs
        "coqa_metrics": {
            "coqa_f1": 0.6758,      # Main accuracy metric
            "coqa_em": 0.52,        # Exact match
        }
    }
    """
    
    results = {}
    
    # Step 1: Sanity check (does the model work at all?)
    results["sanity_check"] = run_sanity_check(model, tokenizer)
    
    # Step 2: Run actual CoQA evaluation
    results["lm_eval_results"] = run_lm_eval(model, tokenizer, config)
    
    # Step 3: Extract the metrics we care about
    results["coqa_metrics"] = extract_metrics(results["lm_eval_results"])
    
    return results
```

### Sanity Check (Quick Test)

```python
def run_sanity_check(model, tokenizer):
    """
    Generate a few outputs to make sure model isn't broken.
    """
    test_prompts = [
        "The capital of France is",           # Should say "Paris"
        "def fibonacci(n):\n    '''...",     # Should write code
        "Question: What is 2 + 2?\nAnswer:",  # Should say "4"
    ]
    
    results = []
    for prompt in test_prompts:
        output = model.generate(tokenizer(prompt))
        results.append({
            "prompt": prompt,
            "generated": tokenizer.decode(output)
        })
    
    return results
```

**Why do this?** 
- Sometimes quantization completely breaks a model
- This catches obvious failures before running the slow evaluation
- Quick visual check: "Does the output make sense?"

### CoQA Evaluation (The Real Test)

```python
def run_lm_eval(model, tokenizer, config):
    """
    Use lm-evaluation-harness to run CoQA benchmark.
    """
    from lm_eval import simple_evaluate
    from lm_eval.models.huggingface import HFLM
    
    # Wrap our model in lm-eval's format
    lm = HFLM(pretrained=model, tokenizer=tokenizer)
    
    # Run evaluation
    results = simple_evaluate(
        model=lm,
        tasks=["coqa"],           # The benchmark to run
        num_fewshot=0,            # Zero-shot (no examples)
        limit=config.eval.limit,  # How many samples (50 for quick test)
    )
    
    return results
```

**What happens:**
1. lm-eval loads CoQA dataset
2. For each question, asks the model to generate an answer
3. Compares generated answer to correct answer
4. Calculates F1 and Exact Match scores

---

## File 4: `benchmark.py` — Measuring Speed

### Purpose
Measure how fast the model runs (latency, throughput, memory).

### Memory Measurement

```python
def get_model_memory_footprint(model):
    """
    How much VRAM does this model use?
    """
    total_bytes = 0
    
    # Count all parameter memory
    for param in model.parameters():
        total_bytes += param.numel() * param.element_size()
    
    # Count all buffer memory (like batch norm running stats)
    for buffer in model.buffers():
        total_bytes += buffer.numel() * buffer.element_size()
    
    return {
        "total_mb": total_bytes / (1024 * 1024),
        "total_gb": total_bytes / (1024 * 1024 * 1024),
    }
```

**Why this matters:**
- Smaller memory = can run on cheaper GPUs
- Smaller memory = can batch more requests
- This is the main benefit of quantization!

### Latency Measurement

```python
def measure_decode_latency(model, tokenizer, config):
    """
    How long to generate each token?
    """
    prompt = "The quick brown fox"
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    
    latencies = []
    
    # Warm up GPU (first runs are slower)
    for _ in range(config.benchmark.warmup_runs):
        model.generate(inputs, max_new_tokens=10)
    
    # Actual measurement
    for _ in tqdm(range(config.benchmark.benchmark_runs), desc="📊 Decode latency"):
        torch.cuda.synchronize()  # Make sure GPU is done
        start = time.perf_counter()
        
        model.generate(inputs, max_new_tokens=128)
        
        torch.cuda.synchronize()  # Wait for GPU to finish
        end = time.perf_counter()
        
        latencies.append((end - start) / 128)  # Per-token time
    
    return {
        "mean_ms": np.mean(latencies) * 1000,
        "std_ms": np.std(latencies) * 1000,
    }
```

**Key points:**
- `torch.cuda.synchronize()` — GPU operations are async, this waits for completion
- Warmup runs — First runs are slower due to CUDA compilation
- Multiple runs — Get average to reduce variance

---

## File 5: `modal_app.py` — The Orchestrator

### Purpose
- Define what runs on cloud GPU (Modal functions)
- Provide CLI interface (`modal run modal_app.py --hyperparam`)
- Coordinate loading → evaluating → benchmarking → saving

### Modal Setup

```python
import modal

# Create the app
app = modal.App("llama-quantization")

# Create a persistent storage volume for caching
volume = modal.Volume.from_name("llama-cache", create_if_missing=True)

# Get HuggingFace token from Modal secrets
hf_secret = modal.Secret.from_name("huggingface")

# Define the Docker image with all dependencies
image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "torch>=2.1.0",
    "transformers>=4.36.0",
    "bitsandbytes>=0.43.0",
    "lm-eval>=0.4.0",
    # ... more packages
)
```

**What this does:**
1. Creates a Modal "app" container
2. Sets up persistent storage (models only download once)
3. Loads secrets (HuggingFace token for gated models)
4. Defines the environment (like a Dockerfile)

### GPU Function Definition

```python
@app.function(
    gpu="A10G",           # Request NVIDIA A10G GPU
    timeout=7200,         # Max 2 hours
    volumes={"/cache": volume},  # Mount our storage
    secrets=[hf_secret],  # Pass HF token
)
def run_hyperparam_sweep(limit: int = 50):
    """
    This function runs ON THE CLOUD GPU.
    """
    import torch
    from huggingface_hub import login
    
    # Login to HuggingFace (needed for Llama)
    login(token=os.environ.get("HF_TOKEN"))
    
    # Run experiments...
    experiments = ["fp16_baseline", "bnb_4bit_nf4", ...]
    
    for exp_name in experiments:
        config = EXPERIMENTS[exp_name]
        model, tokenizer = load_quantized_model(config)
        results = evaluate_model(model, tokenizer, config)
        # ... save results
```

**The `@app.function` decorator:**
- `gpu="A10G"` — Requests GPU hardware
- `timeout=7200` — 2 hour limit
- `volumes` — Attach persistent storage
- `secrets` — Pass environment variables securely

### CLI Entry Point

```python
@app.local_entrypoint()
def main(
    quick: bool = False,
    extended: bool = False,
    hyperparam: bool = False,
    limit: int = None,
):
    """
    This runs on YOUR MACHINE, dispatches to cloud.
    """
    if hyperparam:
        print("Running hyperparam sweep...")
        # .remote() sends this to the cloud GPU
        results = run_hyperparam_sweep.remote(limit=limit or 50)
        
        # Save results locally
        with open("results/results3.json", "w") as f:
            json.dump(results, f)
```

**The `.remote()` call:**
- Serializes your function arguments
- Sends them to Modal's cloud
- Modal provisions a GPU, runs your function
- Returns the results to your machine

---

## Data Flow Summary

```
┌─────────────────────────────────────────────────────────────────┐
│                      YOUR TERMINAL                              │
│                                                                 │
│  $ modal run modal_app.py --hyperparam --limit 50              │
│                         │                                       │
└─────────────────────────│───────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                      MODAL CLOUD (A10G GPU)                     │
│                                                                 │
│  1. Load config.py EXPERIMENTS dictionary                       │
│                          │                                       │
│  2. For each experiment: │                                       │
│     ┌────────────────────▼────────────────────┐                │
│     │ quantize.py: load_quantized_model()     │                │
│     │ - Download model from HuggingFace       │                │
│     │ - Apply BitsAndBytes quantization       │                │
│     │ - Move to GPU                           │                │
│     └────────────────────┬────────────────────┘                │
│                          │                                       │
│     ┌────────────────────▼────────────────────┐                │
│     │ evaluate.py: evaluate_model()           │                │
│     │ - Run sanity check                      │                │
│     │ - Run CoQA evaluation                   │                │
│     │ - Extract F1 and EM scores             │                │
│     └────────────────────┬────────────────────┘                │
│                          │                                       │
│     ┌────────────────────▼────────────────────┐                │
│     │ Save incremental results to volume      │                │
│     └────────────────────┬────────────────────┘                │
│                          │                                       │
│  3. Return all results   │                                       │
│                          │                                       │
└──────────────────────────│──────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                      YOUR MACHINE                               │
│                                                                 │
│  4. Save to results/results3.json                              │
│  5. Print summary table                                         │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Key Code Patterns

### Pattern 1: Config-Driven Design
```python
# BAD: Hardcoded values scattered everywhere
model = load_model(bits=4, quant_type="nf4", double_quant=True)

# GOOD: Single config object
config = EXPERIMENTS["bnb_4bit_nf4"]
model = load_quantized_model(config)
```

### Pattern 2: Fail-Fast
```python
# Stop on first error to save time/money
try:
    result = run_experiment(exp)
except Exception as e:
    print(f"❌ Failed: {e}")
    return partial_results  # Don't continue
```

### Pattern 3: Incremental Saves
```python
# Save after each step, not just at the end
for exp in experiments:
    result = run(exp)
    all_results.append(result)
    save_json(all_results)  # Save NOW
```

### Pattern 4: Progress Feedback
```python
# Users hate staring at blank screens
for i in tqdm(range(20), desc="📊 Measuring latency"):
    measure_one_run()
```


