"""
Generate experimental setup architecture diagram for the report.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

# Set up the figure
fig, ax = plt.subplots(1, 1, figsize=(12, 8))
ax.set_xlim(0, 12)
ax.set_ylim(0, 10)
ax.set_aspect('equal')
ax.axis('off')

# Color scheme - dark theme with accent colors
colors = {
    'bg': '#1a1a2e',
    'model': '#4a90d9',
    'quant': '#e94560',
    'eval': '#0f3460',
    'infra': '#16213e',
    'arrow': '#eaeaea',
    'text': '#ffffff',
    'accent1': '#00d9ff',
    'accent2': '#ff6b6b',
    'accent3': '#4ecdc4',
}

fig.patch.set_facecolor(colors['bg'])
ax.set_facecolor(colors['bg'])

def draw_box(ax, x, y, width, height, label, sublabel=None, color='#4a90d9', text_color='white'):
    """Draw a rounded rectangle with label."""
    box = FancyBboxPatch(
        (x - width/2, y - height/2), width, height,
        boxstyle="round,pad=0.02,rounding_size=0.15",
        facecolor=color, edgecolor='white', linewidth=2, alpha=0.9
    )
    ax.add_patch(box)
    ax.text(x, y + (0.15 if sublabel else 0), label, 
            ha='center', va='center', fontsize=11, fontweight='bold', color=text_color)
    if sublabel:
        ax.text(x, y - 0.25, sublabel, ha='center', va='center', fontsize=8, color=text_color, alpha=0.8)

def draw_arrow(ax, start, end, color='white'):
    """Draw an arrow between two points."""
    ax.annotate('', xy=end, xytext=start,
                arrowprops=dict(arrowstyle='->', color=color, lw=2))

# Title
ax.text(6, 9.5, 'Experimental Setup: Llama 3.2-1B Quantization Study', 
        ha='center', va='center', fontsize=16, fontweight='bold', color='white')

# ============= Left Column: Model & Quantization =============

# Model source
draw_box(ax, 2, 7.5, 3, 0.8, 'HuggingFace Hub', 'meta-llama/Llama-3.2-1B', color='#2d3436')

# Arrow down
draw_arrow(ax, (2, 7.1), (2, 6.6))

# Main model box
draw_box(ax, 2, 6, 3, 0.9, 'Llama 3.2-1B', '1B params • RoPE • SwiGLU', color=colors['model'])

# Arrow to quantization
draw_arrow(ax, (2, 5.55), (2, 5.0))

# Quantization options (3 branches)
ax.text(2, 4.7, 'Quantization', ha='center', va='center', fontsize=10, 
        fontweight='bold', color=colors['accent1'])

# Three quantization paths
draw_box(ax, 0.8, 3.8, 1.8, 0.7, 'FP16', 'Baseline', color='#636e72')
draw_box(ax, 2, 3.8, 1.8, 0.7, 'NF4', '4-bit', color=colors['accent3'])
draw_box(ax, 3.2, 3.8, 1.8, 0.7, 'FP4', '4-bit', color=colors['accent2'])

# Arrows from model to quant options
draw_arrow(ax, (1.3, 5.55), (0.8, 4.15))
draw_arrow(ax, (2, 5.55), (2, 4.15))
draw_arrow(ax, (2.7, 5.55), (3.2, 4.15))

# ============= Middle Column: Evaluation =============

# Merge arrows to evaluation
draw_arrow(ax, (0.8, 3.45), (5.5, 2.8))
draw_arrow(ax, (2, 3.45), (5.5, 2.7))
draw_arrow(ax, (3.2, 3.45), (5.5, 2.6))

# Evaluation harness
draw_box(ax, 6, 2.5, 3.5, 1, 'lm-evaluation-harness', 'Gao et al., 2023', color=colors['eval'])

# Arrow to CoQA
draw_arrow(ax, (6, 2.0), (6, 1.5))

# CoQA benchmark
draw_box(ax, 6, 1, 3, 0.8, 'CoQA Benchmark', '50 samples • Zero-shot', color='#6c5ce7')

# ============= Right Column: Infrastructure =============

# Modal cloud
draw_box(ax, 10, 7, 2.5, 0.8, 'Modal Cloud', 'Serverless GPUs', color='#00b894')

# Arrow down
draw_arrow(ax, (10, 6.6), (10, 6.1))

# GPU
draw_box(ax, 10, 5.5, 2.5, 0.9, 'NVIDIA A100', '40GB • CUDA 12.x', color='#74b816')

# Arrow down
draw_arrow(ax, (10, 5.05), (10, 4.5))

# Frameworks
draw_box(ax, 10, 4, 2.5, 0.8, 'PyTorch 2.1+', 'Transformers 4.36+', color='#e17055')

# Arrow down
draw_arrow(ax, (10, 3.6), (10, 3.1))

# BitsAndBytes
draw_box(ax, 10, 2.6, 2.5, 0.8, 'BitsAndBytes', '4-bit Quantization', color='#d63031')

# Connect infrastructure to evaluation
draw_arrow(ax, (8.75, 2.5), (7.75, 2.5))

# ============= Output =============

# Results arrow
draw_arrow(ax, (6, 0.6), (6, 0.2))

# Results box at bottom
ax.text(6, -0.2, 'Results: F1 Score, Memory, Latency, Throughput', 
        ha='center', va='center', fontsize=10, color='white', 
        bbox=dict(boxstyle='round', facecolor='#2d3436', edgecolor='white', alpha=0.8))

# ============= Legend =============
legend_y = 8.5
ax.add_patch(FancyBboxPatch((8.5, legend_y), 0.3, 0.3, boxstyle="round", 
                             facecolor=colors['model'], edgecolor='white', lw=1))
ax.text(9, legend_y + 0.15, 'Model', fontsize=8, color='white', va='center')

ax.add_patch(FancyBboxPatch((8.5, legend_y - 0.5), 0.3, 0.3, boxstyle="round", 
                             facecolor=colors['accent3'], edgecolor='white', lw=1))
ax.text(9, legend_y - 0.35, 'NF4 (Best)', fontsize=8, color='white', va='center')

ax.add_patch(FancyBboxPatch((10.2, legend_y), 0.3, 0.3, boxstyle="round", 
                             facecolor='#74b816', edgecolor='white', lw=1))
ax.text(10.7, legend_y + 0.15, 'Infrastructure', fontsize=8, color='white', va='center')

ax.add_patch(FancyBboxPatch((10.2, legend_y - 0.5), 0.3, 0.3, boxstyle="round", 
                             facecolor=colors['eval'], edgecolor='white', lw=1))
ax.text(10.7, legend_y - 0.35, 'Evaluation', fontsize=8, color='white', va='center')

# Save
plt.tight_layout()
plt.savefig('figures/experimental_setup.png', dpi=300, bbox_inches='tight', 
            facecolor=colors['bg'], edgecolor='none')
plt.savefig('figures/experimental_setup.pdf', bbox_inches='tight', 
            facecolor=colors['bg'], edgecolor='none')
print("Saved: figures/experimental_setup.png and .pdf")

plt.show()

