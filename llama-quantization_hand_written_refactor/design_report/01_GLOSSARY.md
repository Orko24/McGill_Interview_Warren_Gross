# Quantization Glossary - Plain English Definitions

## What is Quantization?

**Quantization** = Making a model smaller by using less precise numbers.

Think of it like this: Instead of measuring your height as "5 feet 11.372841 inches" (very precise), you just say "6 feet" (less precise, but good enough). The model uses less memory and runs faster, but might be slightly less accurate.

---

## Bit-Widths Explained

| Term | Meaning | Size | Use Case |
|------|---------|------|----------|
| **FP32** | Full Precision (32-bit floating point) | 4 bytes per number | Training (most accurate) |
| **FP16** | Half Precision (16-bit floating point) | 2 bytes per number | Inference baseline |
| **BF16** | Brain Float 16 (Google's format) | 2 bytes per number | Same range as FP32, less precision |
| **INT8** | 8-bit integer | 1 byte per number | 2x smaller than FP16 |
| **INT4** | 4-bit integer | 0.5 bytes per number | 4x smaller than FP16 |

### Visual Size Comparison
```
FP32:  ████████████████████████████████  (32 bits)
FP16:  ████████████████                  (16 bits) - Our baseline
INT8:  ████████                          (8 bits)  - 2x compression
INT4:  ████                              (4 bits)  - 4x compression
```

---

## Quantization Data Types

### NF4 (Normal Float 4-bit)
**What it is**: A special 4-bit format designed specifically for neural network weights.

**Why it's good**: Neural network weights follow a "bell curve" (normal distribution). NF4 is optimized for this shape, giving more precision where weights are common (near zero) and less where they're rare (extremes).

```
Regular INT4:    | Equal spacing between all values |
NF4:             |  Dense near 0  | Sparse at edges |
                 ▼▼▼▼▼▼▼▼▼▼▼      ▼         ▼
                 More precision   Less precision
                 (common values)  (rare values)
```

**Result**: NF4 preserves model quality better than regular 4-bit.

---

### FP4 (Float Point 4-bit)
**What it is**: Standard 4-bit floating point representation.

**Why it's worse**: Treats all value ranges equally, doesn't account for neural network weight distributions.

**Our finding**: FP4 was 9.5% worse than NF4 at the same memory cost!

---

## BitsAndBytes-Specific Terms

### Double Quantization (Nested Quantization)
**What it is**: Quantizing the quantization parameters themselves.

**Plain English**: When you quantize weights, you need to store some small numbers (scales) to "undo" the quantization. Double quantization compresses these small numbers too.

```
Without Double Quant:
  Weights: 4-bit ████
  Scales:  FP32  ████████████████████████████████

With Double Quant:
  Weights: 4-bit ████
  Scales:  8-bit ████████  ← Also compressed!
```

**Benefit**: Saves ~0.4 bits per parameter (free compression!)
**Our finding**: No accuracy loss from double quantization.

---

### Compute Dtype
**What it is**: The precision used for actual math operations during inference.

| Setting | Meaning |
|---------|---------|
| **FP16** | Math done in 16-bit floats |
| **BF16** | Math done in Brain Float 16 |

**Our finding**: Both give identical accuracy. Use whatever your GPU prefers.

---

## Quantization Methods

### BitsAndBytes (BnB)
**What it is**: A library by Tim Dettmers for easy quantization.
**How it works**: Quantizes weights "on the fly" when loading the model.
**Pros**: No calibration data needed, just load and go.
**Cons**: Slightly slower than pre-quantized models.

### GPTQ (GPT Quantization)
**What it is**: Post-training quantization with calibration data.
**How it works**: Analyzes which weights matter most using example data, then quantizes smartly.
**Pros**: Often better quality than BnB.
**Cons**: Requires calibration data and extra setup.

### AWQ (Activation-Aware Weight Quantization)
**What it is**: Like GPTQ but focuses on preserving "salient" (important) weights.
**How it works**: Identifies which weights have the biggest impact on activations.
**Pros**: Often the best quality.
**Cons**: Requires setup, newer method.

---

## Hardware Terms

### GPU Memory (VRAM)
The memory on your graphics card. More VRAM = bigger models can fit.

### Memory Bandwidth
How fast data moves between GPU memory and compute units. Quantization helps because smaller models = less data to move.

### A10G
The GPU we used (NVIDIA A10G with 24GB VRAM). A good mid-range GPU for inference.

---

## Evaluation Terms

### CoQA (Conversational Question Answering)
A benchmark dataset where models answer questions about passages in a conversational style.

### F1 Score
A measure of accuracy that balances precision and recall. 
- 0% = completely wrong
- 100% = perfect
- Our models: ~62-68% (good for a 1B parameter model!)

### EM (Exact Match)
Stricter than F1 - only counts if the answer matches exactly.

---

## Our Key Numbers

| Metric | FP16 Baseline | Best 4-bit (NF4) |
|--------|---------------|------------------|
| F1 Score | 64.2% | 67.6% |
| Memory | 2,357 MB | 965 MB |
| Compression | 1x | 2.44x |

**Bottom Line**: We made the model 2.44x smaller with NO accuracy loss!


