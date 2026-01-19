# Llama 3.2-1B Quantization Study

A systematic evaluation of quantization methods for Llama 3.2-1B on the CoQA conversational QA benchmark.

## Project Structure

```
llama_rewrite_2/
├── code/                      # Source code
│   ├── infra/                 # Infrastructure (Modal deployment)
│   │   └── modal_app.py       # GPU runner for experiments
│   └── llama_quant/           # Core Python package
│       ├── core/              # Configuration & data classes
│       ├── models/            # Model loaders (FP16, BnB, GPTQ, AWQ)
│       ├── evaluation/        # CoQA evaluation via lm-eval-harness
│       ├── benchmark/         # Memory, latency, throughput benchmarks
│       └── utils/             # Logging & serialization
│
├── results/                   # Experiment results (JSON)
│   └── results1.json          # Latest run data
│
├── visualizations/            # Plotting scripts
│   ├── visualize.py           # General visualization
│   └── generate_figures.py    # Publication figures
│
├── reports/                   # LaTeX academic report
│   ├── main.tex               # Main document
│   ├── main.pdf               # Compiled PDF
│   ├── sections/              # Report sections
│   ├── figures/               # Generated figures (PDF/PNG)
│   ├── references.bib         # Bibliography
│   └── Makefile               # Build: `make` or `make clean`
│
└── design_report/             # Technical documentation
    ├── 01_GLOSSARY.md         # Key terms (NF4, FP4, etc.)
    ├── 02_ARCHITECTURE.md     # System design
    ├── 03_DESIGN_CHOICES.md   # Why we made certain decisions
    ├── 04_CODE_WALKTHROUGH.md # Line-by-line explanation
    └── 05_RESULTS_SUMMARY.md  # Consolidated findings
```

## Quick Start

### Prerequisites
- Modal account (`pip install modal && modal setup`)
- HuggingFace token (create Modal secret: `modal secret create huggingface-secret HF_TOKEN=hf_xxx`)

### Run Experiments

```bash
cd llama_rewrite_2

# Quick comparison (FP16 vs NF4 vs FP4)
modal run code/infra/modal_app.py --limit 50

# Full evaluation
modal run code/infra/modal_app.py --limit 500

# All quantization methods
modal run code/infra/modal_app.py --all --limit 100
```

Results auto-save to `results/results1.json`, `results2.json`, etc.

### Generate Figures

```bash
cd visualizations
python generate_figures.py ../results/results1.json
```

### Build Report

```bash
cd reports
make        # Compiles main.pdf
make clean  # Remove build artifacts
```

## Key Results (50 samples)

| Method | F1 Score | Model Size | Compression |
|--------|----------|------------|-------------|
| FP16 (baseline) | 0.6418 | 2357 MB | 1.0x |
| BnB 4-bit NF4 | **0.6758** | 965 MB | **2.4x** |
| BnB 4-bit FP4 | 0.5807 | 965 MB | 2.4x |

**Key Finding**: NF4 quantization achieves comparable (or better) accuracy to FP16 with 2.4x memory reduction.

## Learning Resources

Start with the `design_report/` folder:
1. **01_GLOSSARY.md** - Understand the terminology
2. **02_ARCHITECTURE.md** - See how components connect
3. **04_CODE_WALKTHROUGH.md** - Detailed code explanation

## Author

Hemanto Bairagi

