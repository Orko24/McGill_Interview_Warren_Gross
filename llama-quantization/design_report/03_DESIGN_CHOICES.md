# Design Choices - Why We Did What We Did

## Overview

This document explains every major design decision in plain English.

---

## 1. Why Modal Instead of Local/Docker?

### The Problem
- User has no local GPU (only 13GB RAM, AMD Ryzen CPU)
- Docker alone can't provide GPU access
- Need cloud GPU for experiments

### The Solution
**Modal** - A serverless GPU platform that:
- Provisions GPUs on-demand
- Charges by the second (~$0.50-1/hour for A10G)
- Handles all the Docker/CUDA complexity automatically

### Trade-off
| Option | Pros | Cons |
|--------|------|------|
| Local CPU | Free | Too slow, not enough RAM |
| Colab | Free tier | Session timeouts, limited |
| **Modal** ✓ | Fast, easy, reliable | Costs ~$3-5 for full experiments |
| Lambda/AWS | More control | Complex setup |

---

## 2. Why BitsAndBytes Over GPTQ/AWQ?

### The Contenders

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  QUANTIZATION METHODS COMPARISON                                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  BitsAndBytes (BnB)                                                        │
│  ├── Setup: pip install bitsandbytes ✓ Easy                               │
│  ├── Usage: Just add quantization_config to from_pretrained()             │
│  ├── Calibration: NOT REQUIRED ✓                                          │
│  └── Quality: Good (NF4 especially)                                        │
│                                                                             │
│  GPTQ                                                                       │
│  ├── Setup: pip install optimum auto-gptq                                  │
│  ├── Usage: Need calibration data + quantization step                     │
│  ├── Calibration: REQUIRED (128+ samples recommended)                     │
│  └── Quality: Often slightly better than BnB                              │
│                                                                             │
│  AWQ                                                                        │
│  ├── Setup: pip install autoawq                                            │
│  ├── Usage: Need calibration data + quantization step                     │
│  ├── Calibration: REQUIRED                                                │
│  └── Quality: Often best, but newer                                       │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Why We Chose BitsAndBytes
1. **Simplest setup** - No calibration data needed
2. **Well-tested** - Widely used in the community
3. **Good enough** - NF4 achieves excellent results
4. **Assignment scope** - Focus on systematic comparison, not every method

### What We'd Add Given More Time
- GPTQ with different group sizes (32, 64, 128)
- AWQ for comparison
- 8-bit (if CUDA bug is fixed)

---

## 3. Why NF4 Over FP4?

### The Experiment
We tested both at identical memory usage (965 MB):

```
┌───────────────────────────────────────────────────────────────┐
│                    NF4 vs FP4 Results                         │
├───────────────────────────────────────────────────────────────┤
│                                                               │
│  NF4 (Normal Float 4):     ████████████████████  67.6% F1    │
│  FP4 (Floating Point 4):   ████████████          58.1% F1    │
│                                                               │
│  Difference: +9.5% F1 (16.4% relative improvement!)          │
│                                                               │
└───────────────────────────────────────────────────────────────┘
```

### Why NF4 Wins

**The Math (simplified):**

Neural network weights follow a bell curve (normal distribution):
```
                          ▲
                         ╱│╲
                        ╱ │ ╲
                       ╱  │  ╲
                      ╱   │   ╲
                     ╱    │    ╲
          ──────────╱─────│─────╲──────────
          -3σ     -1σ    0    +1σ     +3σ
          (rare)  (common)    (common) (rare)
```

- **FP4**: Distributes its 16 values evenly across the range
- **NF4**: Packs more values near zero (where weights cluster)

**Result**: NF4 preserves more information where it matters most.

---

## 4. Why Double Quantization = Free Compression

### What Double Quantization Does

```
WITHOUT DOUBLE QUANT:
┌─────────────────────────────────────────────────────────────┐
│  Weights (4-bit): ████████████████  (compressed)           │
│  Scales (FP32):   ████████████████████████████████████████ │
│                   ▲                                         │
│                   └── These are BIG! 32 bits each          │
└─────────────────────────────────────────────────────────────┘

WITH DOUBLE QUANT:
┌─────────────────────────────────────────────────────────────┐
│  Weights (4-bit):    ████████████████  (compressed)        │
│  Scales (8-bit):     ████████          (also compressed!)  │
│  Scale-scales (FP32): ████████████████████████████████████ │
│                       ▲                                     │
│                       └── Only a few of these now          │
└─────────────────────────────────────────────────────────────┘
```

### Our Finding
- With double quant: 67.58% F1
- Without double quant: 67.58% F1
- **Same accuracy, smaller footprint!**

### Recommendation
Always enable double quantization - it's free compression.

---

## 5. Why We Skip 8-bit Quantization

### The Bug

```
Error invalid configuration argument at line 380 in file /src/csrc/ops.cu
```

### What Happened
The BitsAndBytes 8-bit CUDA kernel has a bug on A10G GPUs. This is a known issue with certain GPU architectures.

### What We Tried
1. Added `llm_int8_threshold=6.0`
2. Set `llm_int8_has_fp16_weight=False`
3. Used NVIDIA CUDA 12.1 base image

### Result
Bug persisted. Not worth debugging further for this assignment since 4-bit is more interesting anyway (more aggressive compression).

---

## 6. Why SDPA (Flash Attention)?

### What It Is
SDPA = Scaled Dot-Product Attention, PyTorch's optimized attention implementation.

```python
# We enable this with:
attn_implementation="sdpa"
```

### Benefits
```
┌────────────────────────────────────────────────────────────────┐
│  Standard Attention           │  SDPA / Flash Attention       │
├────────────────────────────────────────────────────────────────┤
│                               │                                │
│  Memory: O(n²)                │  Memory: O(n)                 │
│  Speed:  Baseline             │  Speed:  2-4x faster          │
│                               │                                │
│  For 1024 tokens:             │  For 1024 tokens:             │
│  ~4GB memory                  │  ~100MB memory                │
│                               │                                │
└────────────────────────────────────────────────────────────────┘
```

### Why We Use It
- **Free speedup** - Just a config flag
- **Lower memory** - Can fit longer sequences
- **PyTorch native** - No extra dependencies

---

## 7. Why Incremental Saves?

### The Problem
Cloud GPU time costs money. If an experiment crashes, we lose all data.

### The Solution
Save results after EACH experiment, not just at the end:

```python
for exp_name in experiments:
    # Run experiment
    result = run_experiment(exp_name)
    all_results.append(result)
    
    # SAVE IMMEDIATELY
    with open("results.json", "w") as f:
        json.dump(all_results, f)
    print(f"💾 Saved {len(all_results)}/{len(experiments)}")
```

### Benefits
- If experiment 4/6 crashes, we still have results 1-3
- Can monitor progress in real-time
- No wasted compute credits

---

## 8. Why Fail-Fast?

### The Problem
If one experiment fails, running more will likely fail too (same bug).

### The Solution
Stop immediately on first error:

```python
try:
    result = run_experiment(exp_name)
except Exception as e:
    print(f"❌ FAILED: {e}")
    print("🛑 STOPPING to save compute credits")
    return partial_results  # Return what we have
```

### Benefits
- Saves money (cloud GPU = $$$/hour)
- Faster debugging (see error immediately)
- Still get partial results

---

## 9. Why These Specific Hyperparameters?

### Ablation Study Design

We tested **one variable at a time** (scientific method):

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  ABLATION STUDY: Testing Each Variable Independently                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Baseline: NF4 + double_quant=True + FP16                                  │
│                                                                             │
│  Test 1: Change QUANT TYPE only                                            │
│  ├── NF4 → 67.58% F1                                                       │
│  └── FP4 → 58.07% F1  ← Quant type matters A LOT                          │
│                                                                             │
│  Test 2: Change DOUBLE QUANT only                                          │
│  ├── double_quant=True  → 67.58% F1                                        │
│  └── double_quant=False → 67.58% F1  ← No difference                      │
│                                                                             │
│  Test 3: Change COMPUTE DTYPE only                                         │
│  ├── FP16 → 67.58% F1                                                      │
│  └── BF16 → 67.58% F1  ← No difference                                    │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Why This Matters
- **Quant type**: Most important choice (NF4 >> FP4)
- **Double quant**: Always enable (free)
- **Compute dtype**: Doesn't matter for accuracy (use hardware preference)

---

## 10. Project Structure Choice

### Why This Layout?

```
llama-quantization/
├── config.py        # ALL settings in one place (easy to find/change)
├── quantize.py      # Model loading only (single responsibility)
├── evaluate.py      # Evaluation only (single responsibility)
├── benchmark.py     # Performance measurement only (single responsibility)
├── modal_app.py     # Orchestration + CLI (ties everything together)
├── results/         # All outputs (easy to find)
└── design_report/   # Documentation (separate from code)
```

### Design Principles
1. **Single Responsibility**: Each file does ONE thing
2. **Config Centralization**: All settings in `config.py`
3. **Separation of Concerns**: Load → Evaluate → Benchmark → Save
4. **Easy Navigation**: Logical folder structure

### Benefits
- Easy to understand (read one file at a time)
- Easy to modify (change one thing in one place)
- Easy to test (each module is independent)
- Easy to reproduce (config saved with results)


