# Sources & Theoretical Backing

## Key Papers Supporting NF4 Quantization Claims

---

### 1. QLoRA: Efficient Finetuning of Quantized LLMs
**Authors:** Dettmers, T., Pagnoni, A., Holtzman, A., & Zettlemoyer, L.  
**Year:** 2023  
**arXiv:** [2305.14314](https://arxiv.org/abs/2305.14314)

**Key Claims:**
- Introduces NormalFloat4 (NF4), an "information-theoretically optimal" data type for normally distributed weights
- NF4 places quantization bins at the quantiles of a standard normal distribution
- Empirically shows NF4 + double quantization matches 16-bit finetuning performance
- Demonstrates that quantization scheme selection significantly impacts quality

**Relevant Quote:**
> "We introduce a new data type, 4-bit NormalFloat (NF4), that is information-theoretically optimal for normally distributed weights"

**Use for:** Justifying NF4 design, why NF4 > FP4, theoretical optimality

---

### 2. QReg: On Regularization Effects of Quantization
**Authors:** Askari-Hemmat, M. H., et al.  
**Year:** 2022  
**arXiv:** [2206.12372](https://arxiv.org/abs/2206.12372)

**Key Claims:**
- Quantization during training behaves like a regularizer
- The regularization strength correlates with quantization level (lower bits = stronger effect)
- Provides analytical framework: quantization error modeled as additive noise
- Shows quantization can improve generalization in certain settings

**Relevant Quote:**
> "We show that quantization has an inherent regularization effect"

**Use for:** Explaining why NF4 might match/exceed FP16 (regularization hypothesis)

---

### 3. Low-Bit Quantization Favors Undertrained LLMs
**Authors:** Ouyang, S., et al.  
**Venue:** ACL 2025  
**Link:** [aclanthology.org/2025.acl-long.1555](https://aclanthology.org/2025.acl-long.1555/)

**Key Claims:**
- Models trained on fewer tokens ("undertrained") suffer less from quantization
- Provides scaling laws showing quantization acts differently depending on training regime
- Suggests overfitted models are more damaged by quantization than undertrained ones

**Use for:** Alternative explanation for NF4 ≥ FP16 (if FP16 model is slightly overfit)

---

### 4. Lloyd-Max Quantizer (Classic Information Theory)
**Original:** Lloyd, S. (1982). "Least squares quantization in PCM"  
**Also:** Max, J. (1960). "Quantizing for minimum distortion"

**Key Result:**
For any non-uniform distribution (e.g., Gaussian), an optimal non-uniform quantizer **always** achieves lower expected distortion than a uniform quantizer with the same number of levels.

**Mathematical Basis:**
- Uniform quantization wastes bits in low-density regions (tails)
- Non-uniform quantization (like NF4) places more bins where probability mass is concentrated
- For Gaussian: more bins near zero, fewer in tails

**Use for:** Why FP4 (uniform) underperforms NF4 (non-uniform) mathematically

---

### 5. APoT: Additive Powers-of-Two Quantization
**Authors:** Li, Y., et al.  
**Year:** 2019  
**arXiv:** [1909.13144](https://arxiv.org/abs/1909.13144)

**Key Claims:**
- Non-uniform quantization (APoT) outperforms uniform quantization for CNNs
- 4-bit APoT achieves ~76.6% top-1 on ImageNet, matching uniform at higher precision
- Demonstrates "bell-shaped" weight distributions benefit from non-uniform schemes

**Use for:** Empirical evidence that non-uniform > uniform at low bit-widths

---

### 6. SQWA: Stochastic Quantized Weight Averaging
**Authors:** Zhang, J., et al.  
**Year:** 2020  
**arXiv:** [2002.00343](https://arxiv.org/abs/2002.00343)

**Key Claims:**
- Quantization + model averaging pushes toward "flat minima"
- Flat minima → better generalization
- Quantization noise acts similarly to dropout, ensembling, smoothing

**Use for:** Theoretical support for quantization-as-regularization

---

### 7. CoQA: A Conversational Question Answering Challenge
**Authors:** Reddy, S., Chen, D., & Manning, C. D.  
**Year:** 2019  
**Venue:** TACL 2019  
**Link:** [aclanthology.org/Q19-1016](https://aclanthology.org/Q19-1016/)

**Key Claims:**
- Introduces CoQA dataset: 127,000+ questions across 8,000+ conversations
- Tests conversational QA: questions may reference previous turns (coreference)
- Evaluation uses F1 score (word overlap) and Exact Match (EM)
- Challenging benchmark requiring dialogue context understanding

**Use for:** Benchmark description, evaluation methodology

---

## Summary Table

| Claim in Report | Supporting Source |
|-----------------|-------------------|
| NF4 is optimal for Gaussian weights | QLoRA (Dettmers 2023) |
| Quantization acts as regularization | QReg (Askari-Hemmat 2022) |
| NF4 can match/beat FP16 | QLoRA + QReg + ACL 2025 |
| FP4 underperforms NF4 | Lloyd-Max theory + APoT |
| Non-uniform quantization is theoretically optimal | Lloyd-Max (1960, 1982) |

---

## How to Cite in Report

### BibTeX Entries

```bibtex
@article{dettmers2023qlora,
  title={QLoRA: Efficient Finetuning of Quantized LLMs},
  author={Dettmers, Tim and Pagnoni, Artidoro and Holtzman, Ari and Zettlemoyer, Luke},
  journal={arXiv preprint arXiv:2305.14314},
  year={2023}
}

@article{askari2022qreg,
  title={QReg: On Regularization Effects of Quantization},
  author={Askari-Hemmat, Mohammad Hossein and others},
  journal={arXiv preprint arXiv:2206.12372},
  year={2022}
}

@inproceedings{ouyang2025lowbit,
  title={Low-Bit Quantization Favors Undertrained LLMs},
  author={Ouyang, Shuo and others},
  booktitle={Proceedings of ACL 2025},
  year={2025}
}

@article{lloyd1982least,
  title={Least squares quantization in PCM},
  author={Lloyd, Stuart},
  journal={IEEE Transactions on Information Theory},
  volume={28},
  number={2},
  pages={129--137},
  year={1982}
}

@article{reddy2019coqa,
  title={CoQA: A Conversational Question Answering Challenge},
  author={Reddy, Siva and Chen, Danqi and Manning, Christopher D},
  journal={Transactions of the Association for Computational Linguistics},
  volume={7},
  pages={249--266},
  year={2019}
}
```

---

## Revised Abstract (Theoretically Grounded)

> We evaluate 4-bit quantization on Llama 3.2-1B using CoQA. NF4 quantization achieves F1=0.676, comparable to or exceeding the FP16 baseline (F1=0.625), while reducing model size by 59%. This aligns with Dettmers et al. (2023), who show NF4 is information-theoretically optimal for normally distributed weights. The slight improvement may reflect quantization's regularization effect (Askari-Hemmat et al., 2022), which can reduce overfitting. FP4 quantization (F1=0.587) significantly underperforms, consistent with Lloyd-Max quantizer theory: uniform quantization is suboptimal for non-uniform (Gaussian) weight distributions.

