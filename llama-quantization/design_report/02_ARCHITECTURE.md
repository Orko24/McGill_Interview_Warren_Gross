# System Architecture - Complete Overview

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           YOUR LOCAL MACHINE                                     │
│                                                                                 │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐            │
│  │   .env file     │    │  modal_app.py   │    │   results/      │            │
│  │  (HF_TOKEN)     │───▶│  (CLI entry)    │───▶│  JSON outputs   │            │
│  └─────────────────┘    └────────┬────────┘    └─────────────────┘            │
│                                  │                                              │
│                                  │ modal run --hyperparam                       │
│                                  ▼                                              │
└─────────────────────────────────────────────────────────────────────────────────┘
                                   │
                                   │ HTTPS (Modal CLI)
                                   ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           MODAL CLOUD (GPU)                                      │
│                                                                                 │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │                        NVIDIA A10G GPU (24GB VRAM)                       │   │
│  │                                                                          │   │
│  │  ┌──────────────┐   ┌──────────────┐   ┌──────────────┐                │   │
│  │  │  quantize.py │   │ evaluate.py  │   │ benchmark.py │                │   │
│  │  │              │   │              │   │              │                │   │
│  │  │ Load model   │──▶│ Run CoQA     │──▶│ Measure      │                │   │
│  │  │ Apply quant  │   │ evaluation   │   │ latency/mem  │                │   │
│  │  └──────────────┘   └──────────────┘   └──────────────┘                │   │
│  │         │                   │                  │                        │   │
│  │         └───────────────────┴──────────────────┘                        │   │
│  │                             │                                            │   │
│  │                             ▼                                            │   │
│  │                    ┌──────────────┐                                     │   │
│  │                    │   config.py  │ ◀── All hyperparameters             │   │
│  │                    └──────────────┘                                     │   │
│  │                                                                          │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                 │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │                      Modal Volume (/cache)                               │   │
│  │  - HuggingFace model cache (downloaded once, reused)                    │   │
│  │  - Results JSON files (incremental saves)                               │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## File-by-File Breakdown

### 1. `config.py` - The Brain (All Settings in One Place)

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              config.py                                           │
│                                                                                 │
│  PURPOSE: Single source of truth for ALL hyperparameters                        │
│                                                                                 │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │  class QuantizationConfig:          # WHAT TO TUNE                      │   │
│  │  ├── method: QuantMethod            # BnB, GPTQ, or AWQ                 │   │
│  │  ├── bnb_4bit_quant_type: str      # "nf4" or "fp4"                    │   │
│  │  ├── bnb_4bit_use_double_quant     # True/False                        │   │
│  │  └── bnb_4bit_compute_dtype        # FP16 or BF16                      │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                 │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │  EXPERIMENTS = {                    # PRE-DEFINED CONFIGS               │   │
│  │    "fp16_baseline":        → No quantization                           │   │
│  │    "bnb_4bit_nf4":         → NF4 + double quant ✓ BEST                │   │
│  │    "bnb_4bit_nf4_no_double": → NF4, no double quant                   │   │
│  │    "bnb_4bit_fp4":         → FP4 + double quant                       │   │
│  │    ...                                                                  │   │
│  │  }                                                                      │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                 │
│  WHY THIS DESIGN:                                                               │
│  - Change settings in ONE place                                                │
│  - Easy to add new experiments                                                 │
│  - Reproducible: config is saved with results                                  │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

### 2. `quantize.py` - The Loader (Gets Models Ready)

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              quantize.py                                         │
│                                                                                 │
│  PURPOSE: Load models with different quantization settings                      │
│                                                                                 │
│  INPUT: ExperimentConfig                                                        │
│  OUTPUT: (model, tokenizer) tuple                                               │
│                                                                                 │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │                                                                          │   │
│  │  load_quantized_model(config)                                           │   │
│  │           │                                                              │   │
│  │           ├── method == NONE ────────────▶ load_model_fp16()            │   │
│  │           │                                    │                         │   │
│  │           │                                    ▼                         │   │
│  │           │                              AutoModelForCausalLM.           │   │
│  │           │                              from_pretrained(                │   │
│  │           │                                dtype=float16,                │   │
│  │           │                                attn_implementation="sdpa"    │   │
│  │           │                              )                               │   │
│  │           │                                                              │   │
│  │           ├── method == BNB_4BIT ────────▶ load_model_bnb()             │   │
│  │           │                                    │                         │   │
│  │           │                                    ▼                         │   │
│  │           │                              create_bnb_config() ────────┐  │   │
│  │           │                                    │                     │  │   │
│  │           │                                    ▼                     │  │   │
│  │           │                              BitsAndBytesConfig(         │  │   │
│  │           │                                load_in_4bit=True,        │  │   │
│  │           │                                bnb_4bit_quant_type=...,  │  │   │
│  │           │                                bnb_4bit_use_double_quant │  │   │
│  │           │                              )                           │  │   │
│  │           │                                                          │  │   │
│  │           └── method == GPTQ/AWQ ───────▶ (Not implemented fully)   │  │   │
│  │                                                                      │  │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                                 │
│  KEY OPTIMIZATIONS:                                                            │
│  - attn_implementation="sdpa" → Uses Flash Attention (faster)                  │
│  - padding_side="left" → Required for decoder-only models                      │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

### 3. `evaluate.py` - The Tester (Measures Accuracy)

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              evaluate.py                                         │
│                                                                                 │
│  PURPOSE: Run CoQA evaluation using lm-evaluation-harness                       │
│                                                                                 │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │                                                                          │   │
│  │  evaluate_model(model, tokenizer, config)                               │   │
│  │           │                                                              │   │
│  │           ├── 1. Sanity Check ──────────────────────────────────────┐   │   │
│  │           │       │                                                  │   │   │
│  │           │       ▼                                                  │   │   │
│  │           │   Generate 3 test prompts:                               │   │   │
│  │           │   - "The capital of France is" → Should say "Paris"     │   │   │
│  │           │   - def fibonacci(n): → Should write code               │   │   │
│  │           │   - "What is 2+2?" → Should say "4"                     │   │   │
│  │           │                                                          │   │   │
│  │           ├── 2. CoQA Evaluation ───────────────────────────────────┘   │   │
│  │           │       │                                                      │   │
│  │           │       ▼                                                      │   │
│  │           │   lm_eval.simple_evaluate(                                  │   │
│  │           │     model=model,                                            │   │
│  │           │     tasks=["coqa"],                                         │   │
│  │           │     limit=50,  # or full dataset                           │   │
│  │           │   )                                                         │   │
│  │           │                                                              │   │
│  │           └── 3. Extract Metrics ────────────────────────────────────┐   │   │
│  │                   │                                                  │   │   │
│  │                   ▼                                                  │   │   │
│  │               return {                                               │   │   │
│  │                 "coqa_f1": 0.6758,   # Main metric                  │   │   │
│  │                 "coqa_em": 0.52,     # Exact match                  │   │   │
│  │               }                                                      │   │   │
│  │                                                                      │   │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

### 4. `benchmark.py` - The Speedometer (Measures Performance)

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              benchmark.py                                        │
│                                                                                 │
│  PURPOSE: Measure hardware performance metrics                                  │
│                                                                                 │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │                                                                          │   │
│  │  run_benchmarks(model, tokenizer, config)                               │   │
│  │           │                                                              │   │
│  │           ├── measure_prefill_latency() ──────────────────────────────┐│   │
│  │           │       │                                                    ││   │
│  │           │       ▼                                                    ││   │
│  │           │   Time how long to process input tokens                   ││   │
│  │           │   (the "thinking" phase before generating)                ││   │
│  │           │                                                            ││   │
│  │           ├── measure_decode_latency() ───────────────────────────────┘│   │
│  │           │       │                                                     │   │
│  │           │       ▼                                                     │   │
│  │           │   Time per-token generation                                │   │
│  │           │   (how fast it spits out words)                           │   │
│  │           │                                                             │   │
│  │           ├── measure_throughput() ────────────────────────────────────┤   │
│  │           │       │                                                     │   │
│  │           │       ▼                                                     │   │
│  │           │   Tokens per second at different batch sizes              │   │
│  │           │                                                             │   │
│  │           └── get_memory_usage() ──────────────────────────────────────┤   │
│  │                   │                                                     │   │
│  │                   ▼                                                     │   │
│  │               Peak GPU memory during inference                         │   │
│  │                                                                         │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                 │
│  PYTORCH OPTIMIZATIONS ENABLED:                                                │
│  - torch.backends.cuda.matmul.allow_tf32 = True  (faster matrix math)         │
│  - torch.backends.cudnn.benchmark = True         (auto-tune convolutions)     │
│  - torch.inference_mode()                        (no gradient tracking)       │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

### 5. `modal_app.py` - The Orchestrator (Runs Everything)

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              modal_app.py                                        │
│                                                                                 │
│  PURPOSE: CLI interface + Modal cloud deployment                                │
│                                                                                 │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │  CLI OPTIONS (run from terminal)                                        │   │
│  │  ════════════════════════════════════════════════════════════════════  │   │
│  │                                                                          │   │
│  │  modal run modal_app.py --quick --limit 50                              │   │
│  │       │                    │        │                                    │   │
│  │       │                    │        └── Eval on 50 samples              │   │
│  │       │                    └── Run quick comparison (FP16 vs NF4)       │   │
│  │       └── Use Modal to run on cloud GPU                                 │   │
│  │                                                                          │   │
│  │  --quick      → FP16 + NF4 only           → quick_comparison.json      │   │
│  │  --extended   → FP16 + NF4 + FP4 + QLoRA  → results2.json              │   │
│  │  --hyperparam → Full ablation study       → results3.json              │   │
│  │                                                                          │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                 │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │  MODAL FUNCTIONS (run on cloud GPU)                                     │   │
│  │  ════════════════════════════════════════════════════════════════════  │   │
│  │                                                                          │   │
│  │  @app.function(gpu="A10G", ...)                                         │   │
│  │  def run_quick_comparison():                                            │   │
│  │      for exp in ["fp16_baseline", "bnb_4bit_nf4"]:                      │   │
│  │          model = load_quantized_model(config)                           │   │
│  │          results = evaluate_model(model)                                │   │
│  │          benchmarks = run_benchmarks(model)                             │   │
│  │          save_results()  # Incremental save!                           │   │
│  │                                                                          │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                 │
│  KEY FEATURES:                                                                 │
│  - Fail-fast: Stops on first error to save compute credits                    │
│  - Incremental saves: Results saved after each experiment                     │
│  - Progress bars: tqdm for visual feedback                                    │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## Data Flow Diagram

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   config    │     │  quantize   │     │  evaluate   │     │  benchmark  │
│             │     │             │     │             │     │             │
│ EXPERIMENTS │────▶│ Load model  │────▶│ Run CoQA    │────▶│ Measure     │
│ dictionary  │     │ with quant  │     │ evaluation  │     │ performance │
│             │     │             │     │             │     │             │
└─────────────┘     └──────┬──────┘     └──────┬──────┘     └──────┬──────┘
                          │                   │                   │
                          │    model          │    f1_score       │    latency
                          │    object         │    em_score       │    memory
                          │                   │                   │
                          └───────────────────┴───────────────────┘
                                              │
                                              ▼
                                    ┌─────────────────┐
                                    │   modal_app.py  │
                                    │                 │
                                    │ Aggregate all   │
                                    │ results + save  │
                                    │ to JSON         │
                                    │                 │
                                    └────────┬────────┘
                                             │
                                             ▼
                                    ┌─────────────────┐
                                    │ results/*.json  │
                                    │                 │
                                    │ - quick_...     │
                                    │ - results2.json │
                                    │ - results3.json │
                                    └─────────────────┘
```


