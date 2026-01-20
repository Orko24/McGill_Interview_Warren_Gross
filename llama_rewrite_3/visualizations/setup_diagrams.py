"""
Generate 3 separate experimental setup diagrams for the report.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch

# Color scheme
colors = {
    'bg': '#1a1a2e',
    'model': '#4a90d9',
    'quant_nf4': '#4ecdc4',
    'quant_fp4': '#ff6b6b',
    'quant_fp16': '#636e72',
    'infra': '#00b894',
    'gpu': '#74b816',
    'framework': '#e17055',
    'eval': '#0f3460',
    'benchmark': '#6c5ce7',
    'text': '#ffffff',
}

def draw_box(ax, x, y, width, height, label, sublabel=None, color='#4a90d9'):
    """Draw a rounded rectangle with label."""
    box = FancyBboxPatch(
        (x - width/2, y - height/2), width, height,
        boxstyle="round,pad=0.02,rounding_size=0.2",
        facecolor=color, edgecolor='white', linewidth=2.5, alpha=0.95
    )
    ax.add_patch(box)
    if sublabel:
        ax.text(x, y + 0.12, label, ha='center', va='center', 
                fontsize=13, fontweight='bold', color='white')
        ax.text(x, y - 0.15, sublabel, ha='center', va='center', 
                fontsize=9, color='white', alpha=0.85)
    else:
        ax.text(x, y, label, ha='center', va='center', 
                fontsize=13, fontweight='bold', color='white')

def draw_arrow(ax, start, end, color='white'):
    """Draw an arrow between two points."""
    ax.annotate('', xy=end, xytext=start,
                arrowprops=dict(arrowstyle='->', color=color, lw=2.5))

# =============================================================================
# FIGURE 1: HuggingFace Model Loading & Quantization
# =============================================================================

fig1, ax1 = plt.subplots(figsize=(8, 6))
ax1.set_xlim(0, 8)
ax1.set_ylim(0, 6)
ax1.set_aspect('equal')
ax1.axis('off')
fig1.patch.set_facecolor(colors['bg'])
ax1.set_facecolor(colors['bg'])

# Title
ax1.text(4, 5.6, 'Model Loading & Quantization', 
         ha='center', va='center', fontsize=16, fontweight='bold', color='white')

# HuggingFace Hub
draw_box(ax1, 4, 4.8, 3.5, 0.7, 'HuggingFace Hub', 'meta-llama/Llama-3.2-1B', color='#2d3436')

# Arrow
draw_arrow(ax1, (4, 4.45), (4, 3.95))

# Llama model
draw_box(ax1, 4, 3.5, 3.5, 0.8, 'Llama 3.2-1B', '1B params | RoPE | SwiGLU', color=colors['model'])

# Arrow to quantization
draw_arrow(ax1, (4, 3.1), (4, 2.6))

# Quantization label
ax1.text(4, 2.35, 'BitsAndBytes Quantization', ha='center', va='center', 
         fontsize=11, fontweight='bold', color='#00d9ff')

# Three quantization options
draw_box(ax1, 1.8, 1.4, 2, 0.7, 'FP16', 'Baseline (16-bit)', color=colors['quant_fp16'])
draw_box(ax1, 4, 1.4, 2, 0.7, 'NF4', '4-bit (optimal)', color=colors['quant_nf4'])
draw_box(ax1, 6.2, 1.4, 2, 0.7, 'FP4', '4-bit (uniform)', color=colors['quant_fp4'])

# Arrows to quantization options
draw_arrow(ax1, (2.8, 3.1), (1.8, 1.75))
draw_arrow(ax1, (4, 3.1), (4, 1.75))
draw_arrow(ax1, (5.2, 3.1), (6.2, 1.75))

# Output sizes
ax1.text(1.8, 0.7, '2.3 GB', ha='center', va='center', fontsize=10, color='#aaa')
ax1.text(4, 0.7, '0.97 GB', ha='center', va='center', fontsize=10, color='#4ecdc4')
ax1.text(6.2, 0.7, '0.97 GB', ha='center', va='center', fontsize=10, color='#aaa')

plt.tight_layout()
fig1.savefig('figures/setup_1_model.png', dpi=300, bbox_inches='tight', 
             facecolor=colors['bg'], edgecolor='none')
fig1.savefig('figures/setup_1_model.pdf', bbox_inches='tight', 
             facecolor=colors['bg'], edgecolor='none')
print("Saved: setup_1_model.png")

# =============================================================================
# FIGURE 2: Modal Cloud Infrastructure
# =============================================================================

fig2, ax2 = plt.subplots(figsize=(6, 6))
ax2.set_xlim(0, 6)
ax2.set_ylim(0, 6)
ax2.set_aspect('equal')
ax2.axis('off')
fig2.patch.set_facecolor(colors['bg'])
ax2.set_facecolor(colors['bg'])

# Title
ax2.text(3, 5.6, 'Infrastructure Stack', 
         ha='center', va='center', fontsize=16, fontweight='bold', color='white')

# Modal Cloud
draw_box(ax2, 3, 4.7, 3.5, 0.7, 'Modal Cloud', 'Serverless GPUs', color=colors['infra'])

# Arrow
draw_arrow(ax2, (3, 4.35), (3, 3.85))

# GPU
draw_box(ax2, 3, 3.4, 3.5, 0.8, 'NVIDIA A100', '40GB VRAM | CUDA 12.x', color=colors['gpu'])

# Arrow
draw_arrow(ax2, (3, 3.0), (3, 2.5))

# PyTorch
draw_box(ax2, 3, 2.1, 3.5, 0.7, 'PyTorch 2.1+', 'Transformers 4.36+', color=colors['framework'])

# Arrow
draw_arrow(ax2, (3, 1.75), (3, 1.25))

# BitsAndBytes
draw_box(ax2, 3, 0.8, 3.5, 0.7, 'BitsAndBytes', '4-bit Quantization Library', color='#d63031')

plt.tight_layout()
fig2.savefig('figures/setup_2_infra.png', dpi=300, bbox_inches='tight', 
             facecolor=colors['bg'], edgecolor='none')
fig2.savefig('figures/setup_2_infra.pdf', bbox_inches='tight', 
             facecolor=colors['bg'], edgecolor='none')
print("Saved: setup_2_infra.png")

# =============================================================================
# FIGURE 3: Evaluation Pipeline
# =============================================================================

fig3, ax3 = plt.subplots(figsize=(8, 5))
ax3.set_xlim(0, 8)
ax3.set_ylim(0, 5)
ax3.set_aspect('equal')
ax3.axis('off')
fig3.patch.set_facecolor(colors['bg'])
ax3.set_facecolor(colors['bg'])

# Title
ax3.text(4, 4.6, 'Evaluation Pipeline', 
         ha='center', va='center', fontsize=16, fontweight='bold', color='white')

# lm-evaluation-harness
draw_box(ax3, 4, 3.7, 4, 0.8, 'lm-evaluation-harness', 'Gao et al., 2023', color=colors['eval'])

# Arrow
draw_arrow(ax3, (4, 3.3), (4, 2.8))

# CoQA
draw_box(ax3, 4, 2.4, 4, 0.7, 'CoQA Benchmark', 'Conversational QA | Zero-shot', color=colors['benchmark'])

# Arrow
draw_arrow(ax3, (4, 2.05), (4, 1.55))

# Metrics box
ax3.text(4, 1.2, 'Metrics', ha='center', va='center', fontsize=11, 
         fontweight='bold', color='#00d9ff')

# Metric boxes in a row
metrics = [('F1 Score', 1.3), ('EM Score', 3), ('Memory', 4.7), ('Latency', 6.4)]
for label, x in metrics:
    box = FancyBboxPatch(
        (x - 0.7, 0.4), 1.4, 0.5,
        boxstyle="round,pad=0.02,rounding_size=0.15",
        facecolor='#2d3436', edgecolor='white', linewidth=1.5, alpha=0.9
    )
    ax3.add_patch(box)
    ax3.text(x, 0.65, label, ha='center', va='center', fontsize=9, 
             fontweight='bold', color='white')

plt.tight_layout()
fig3.savefig('figures/setup_3_eval.png', dpi=300, bbox_inches='tight', 
             facecolor=colors['bg'], edgecolor='none')
fig3.savefig('figures/setup_3_eval.pdf', bbox_inches='tight', 
             facecolor=colors['bg'], edgecolor='none')
print("Saved: setup_3_eval.png")

print("\nAll 3 diagrams generated!")


