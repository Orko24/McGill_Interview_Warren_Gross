# Llama 3.2-1B Quantization — Code Breakdown & Execution Plan

## Assignment Summary
**Goal**: Minimize bit-width of Llama 3.2-1B while maximizing CoQA accuracy  
**Deliverables**: 4-page report + modular, reproducible code  
**Interviewer**: Warren Gross (McGill) — hardware efficiency expert

---

## Codebase Structure

```
llama-quantization/
├── Dockerfile           # Multi-stage Docker build (7 layers)
├── docker-compose.yaml  # Container orchestration
├── run.sh               # Convenience script for Docker commands
├── env.example          # Environment variables template
│
├── config.py            # All hyperparameters & experiment definitions
├── quantize.py          # Model loading with BnB/GPTQ/AWQ quantization
├── evaluate.py          # lm-evaluation-harness wrapper for CoQA
├── benchmark.py         # Memory/latency/throughput profiling
├── main.py              # Single experiment runner
├── sweep.py             # Grid search over configurations
├── visualize.py         # Publication-ready figures
├── requirements.txt     # Dependencies
│
├── modal_app.py         # Modal deployment (alternative to Docker)
├── NOTES_AND_PLAN.md    # This file
└── README.md            # Quick start guide
```

---

## File-by-File Breakdown

### 1. `config.py` — The Control Center
**What it does**: Defines all experiment configurations using Python dataclasses.

**Key classes**:
- `QuantMethod` — Enum: NONE (FP16), BNB_8BIT, BNB_4BIT, GPTQ, AWQ
- `QuantizationConfig` — All quantization hyperparameters
- `ModelConfig` — Model ID, calibration settings
- `EvalConfig` — CoQA task settings, batch size, sample limits
- `ExperimentConfig` — Combines all configs into one experiment

**Pre-defined experiments** (in `EXPERIMENTS` dict):
| Name | Method | Notes |
|------|--------|-------|
| `fp16_baseline` | None | Full precision baseline |
| `bnb_8bit` | BitsAndBytes 8-bit | Good accuracy, 2x compression |
| `bnb_4bit_nf4` | BitsAndBytes 4-bit NF4 | Best 4-bit method |
| `bnb_4bit_fp4` | BitsAndBytes 4-bit FP4 | Alternative 4-bit |
| `gptq_4bit_g128` | GPTQ 4-bit | Calibration-based |
| `gptq_3bit_g128` | GPTQ 3-bit | Aggressive compression |
| `awq_4bit_g128` | AWQ 4-bit | Activation-aware |

---

### 2. `quantize.py` — Model Loading
**What it does**: Loads Llama 3.2-1B with specified quantization.

**Key functions**:
- `load_quantized_model(config)` → Returns `(model, tokenizer)`
- `load_model_bnb(config)` → BitsAndBytes quantization (instant, no calibration)
- `load_model_gptq(config)` → GPTQ (requires calibration data, slower)
- `load_model_awq(config)` → AWQ (requires calibration, slower)
- `get_model_memory_footprint(model)` → Memory usage in MB

**Why BitsAndBytes is your friend**: No calibration needed. Just load and go. GPTQ/AWQ require running calibration data through the model first.

---

### 3. `evaluate.py` — CoQA Benchmarking
**What it does**: Wraps lm-evaluation-harness to evaluate on CoQA.

**Key functions**:
- `run_lm_eval(model, tokenizer, config)` → Runs evaluation
- `extract_coqa_metrics(results)` → Pulls out F1 and EM scores
- `run_quick_sanity_check(model, tokenizer)` → Quick generation test

**CoQA Metrics**:
- **F1 Score**: Primary metric (word overlap with reference)
- **Exact Match (EM)**: Stricter metric (exact string match)

---

### 4. `benchmark.py` — Hardware Profiling
**What it does**: Measures memory, latency, throughput.

**Key functions**:
- `measure_memory_usage(model)` → Peak GPU memory
- `measure_inference_latency(model, tokenizer, ...)` → Time per token
- `measure_throughput(model, tokenizer, ...)` → Tokens/second
- `run_benchmarks(model, tokenizer, config)` → All benchmarks

**Warren Gross cares about these numbers** — include them in your report!

---

### 5. `main.py` — Experiment Runner
**What it does**: Orchestrates a single experiment.

**Usage**:
```bash
# Run specific experiment
python main.py --experiment fp16_baseline --limit 100

# Run multiple
python main.py --experiments fp16_baseline bnb_8bit bnb_4bit_nf4 --limit 100
```

**Flow**:
1. Load & quantize model
2. Run CoQA evaluation
3. Run hardware benchmarks
4. Save results to JSON

---

### 6. `sweep.py` — Grid Search
**What it does**: Systematically tests many configurations.

**Usage**:
```bash
# Quick comparison (3 configs, 50 samples each)
python sweep.py --quick --limit 50

# Full BnB sweep
python sweep.py --method bnb --limit 200

# Everything (takes hours)
python sweep.py --method all --limit 500
```

---

### 7. `visualize.py` — Report Figures
**What it does**: Creates publication-ready plots.

**Generates**:
- Accuracy vs Compression scatter plot
- Memory usage bar chart
- Latency comparison
- Pareto frontier (accuracy vs bits)

---

## Execution Plan with Docker

### Docker Image Layers (Dockerfile)

```
Layer 1: nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04  (base)
    ↓
Layer 2: PyTorch 2.2.0 + CUDA 12.1                    (~5GB, cached)
    ↓
Layer 3: transformers, accelerate, datasets           (~500MB)
    ↓
Layer 4: bitsandbytes (quantization)                  (~100MB)
    ↓
Layer 5: lm-eval (evaluation harness)                 (~200MB)
    ↓
Layer 6: numpy, pandas, matplotlib, seaborn           (~300MB)
    ↓
Layer 7: Application code (your .py files)            (~50KB)
```

### Quick Start with Docker

```bash
# 1. Setup environment
cp env.example .env
# Edit .env and add your HF_TOKEN

# 2. Build image (one time, ~10 min)
./run.sh build

# 3. Test GPU works
./run.sh test

# 4. Run experiments
./run.sh quick          # FP16 vs 8-bit vs 4-bit (minimum viable)
./run.sh bnb            # Full BnB sweep
./run.sh exp bnb_4bit_nf4 100   # Specific experiment with limit
```

### Alternative: Modal (Serverless)

If you don't want to manage Docker locally:
```bash
pip install modal
modal setup
modal run modal_app.py --quick --limit 50
```

---

## Minimum Viable Submission (2-3 hours)

If time is tight, just do these 3 experiments:

| Experiment | Why |
|------------|-----|
| `fp16_baseline` | Reference point |
| `bnb_8bit` | 2x compression, minimal accuracy loss |
| `bnb_4bit_nf4` | 4x compression, the interesting tradeoff |

This gives you:
- Clear accuracy vs compression curve
- Memory reduction numbers
- Enough for a solid 4-page report

---

## Report Outline (for later)

1. **Introduction** (0.5 page)
   - Problem: quantization for efficient deployment
   - Approach: compare BnB 4/8-bit quantization

2. **Method** (1 page)
   - Quantization techniques (NF4, group size)
   - Evaluation setup (CoQA, lm-eval-harness)

3. **Results** (1.5 pages)
   - Accuracy table (F1 scores)
   - Memory/latency figures
   - Pareto analysis

4. **Discussion** (1 page)
   - Best accuracy/compression tradeoff
   - Hardware implications
   - Recommendations

---

## Next Steps

1. ✅ Code breakdown (this document)
2. 🔄 Create `modal_app.py` for Docker-based GPU execution
3. ⏳ Run experiments
4. ⏳ Generate figures
5. ⏳ Write report

