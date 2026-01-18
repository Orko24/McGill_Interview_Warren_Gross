#!/usr/bin/env python3
"""
Generate publication-ready figures for Llama quantization report.
Reads results from JSON and creates matplotlib figures.
"""

import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path

# Style configuration for publication
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'legend.fontsize': 9,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})

# Colors - professional palette
COLORS = {
    'fp16': '#2E86AB',       # Blue
    'nf4': '#28A745',        # Green  
    'fp4': '#DC3545',        # Red
    'nf4_variant': '#20C997', # Teal
    'fp4_variant': '#FD7E14', # Orange
    'baseline': '#6C757D',   # Gray
}

def load_results():
    """Load results from JSON files"""
    results_dir = Path(__file__).parent.parent.parent / 'results'
    
    with open(results_dir / 'results3_summary.json') as f:
        return json.load(f)

def fig1_accuracy_vs_memory(data, output_dir):
    """
    Figure 1: Accuracy vs Memory tradeoff scatter plot
    Shows compression vs accuracy for all configurations
    """
    fig, ax = plt.subplots(figsize=(6, 4))
    
    results = data['results']
    
    for r in results:
        name = r['name']
        f1 = r['coqa_f1']
        mem = r['memory_mb']
        
        # Determine color and marker
        if 'fp16' in name:
            color, marker, label = COLORS['fp16'], 's', 'FP16 Baseline'
        elif 'nf4' in name:
            color, marker = COLORS['nf4'], 'o'
            label = 'NF4 variants' if 'nf4' in name else None
        elif 'fp4' in name:
            color, marker = COLORS['fp4'], '^'
            label = 'FP4 variants' if 'fp4' in name else None
        else:
            color, marker, label = COLORS['baseline'], 'x', name
        
        ax.scatter(mem, f1, c=color, marker=marker, s=120, 
                   edgecolors='black', linewidths=0.5, zorder=5)
        
        # Annotate key points
        offset = (10, 5) if 'fp16' in name else (-15, 10) if 'nf4' in name and 'no_double' not in name else (10, -15)
        ax.annotate(name.replace('bnb_4bit_', '').replace('_', '\n'), 
                    (mem, f1), textcoords="offset points", 
                    xytext=offset, fontsize=7, ha='center')
    
    # Draw arrow showing compression direction
    ax.annotate('', xy=(1000, 0.68), xytext=(2300, 0.68),
                arrowprops=dict(arrowstyle='->', color='gray', lw=1.5))
    ax.text(1650, 0.69, '2.44× compression', fontsize=8, ha='center', color='gray')
    
    # Reference lines
    ax.axhline(y=0.6418, color=COLORS['fp16'], linestyle='--', alpha=0.5, label='FP16 baseline F1')
    
    ax.set_xlabel('GPU Memory (MB)')
    ax.set_ylabel('CoQA F1 Score')
    ax.set_title('Accuracy vs Memory: Quantization Tradeoff')
    
    # Custom legend
    legend_elements = [
        mpatches.Patch(color=COLORS['fp16'], label='FP16 (16-bit)'),
        mpatches.Patch(color=COLORS['nf4'], label='NF4 (4-bit)'),
        mpatches.Patch(color=COLORS['fp4'], label='FP4 (4-bit)'),
    ]
    ax.legend(handles=legend_elements, loc='lower right')
    
    ax.set_xlim(800, 2600)
    ax.set_ylim(0.55, 0.72)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'fig1_accuracy_vs_memory.pdf')
    plt.savefig(output_dir / 'fig1_accuracy_vs_memory.png')
    plt.close()
    print("✓ Generated fig1_accuracy_vs_memory.pdf")

def fig2_all_configurations_bar(data, output_dir):
    """
    Figure 2: Bar chart comparing all configurations
    """
    fig, ax = plt.subplots(figsize=(8, 4))
    
    results = data['results']
    names = [r['name'].replace('bnb_4bit_', '').replace('fp16_baseline', 'FP16\nBaseline') 
             for r in results]
    f1_scores = [r['coqa_f1'] for r in results]
    
    colors = []
    for r in results:
        if 'fp16' in r['name']:
            colors.append(COLORS['fp16'])
        elif 'nf4' in r['name']:
            colors.append(COLORS['nf4'])
        else:
            colors.append(COLORS['fp4'])
    
    bars = ax.bar(names, f1_scores, color=colors, edgecolor='black', linewidth=0.5)
    
    # Add value labels on bars
    for bar, score in zip(bars, f1_scores):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005, 
                f'{score:.3f}', ha='center', va='bottom', fontsize=8)
    
    # Baseline reference line
    ax.axhline(y=0.6418, color=COLORS['fp16'], linestyle='--', alpha=0.7, 
               label='FP16 Baseline')
    
    ax.set_ylabel('CoQA F1 Score')
    ax.set_title('Quantization Configuration Comparison')
    ax.set_ylim(0.5, 0.75)
    
    # Rotate x labels for readability
    plt.xticks(rotation=45, ha='right')
    
    ax.legend(loc='upper right')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'fig2_all_configurations.pdf')
    plt.savefig(output_dir / 'fig2_all_configurations.png')
    plt.close()
    print("✓ Generated fig2_all_configurations.pdf")

def fig3_nf4_vs_fp4(data, output_dir):
    """
    Figure 3: Direct NF4 vs FP4 comparison (key finding)
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3.5))
    
    # Left plot: F1 scores comparison
    methods = ['NF4', 'FP4']
    f1_scores = [0.6758, 0.5807]  # Best of each type
    
    bars = ax1.bar(methods, f1_scores, color=[COLORS['nf4'], COLORS['fp4']], 
                   edgecolor='black', linewidth=1, width=0.6)
    
    # Add value labels
    for bar, score in zip(bars, f1_scores):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{score:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Add delta annotation
    ax1.annotate('', xy=(1, 0.6758), xytext=(1, 0.5807),
                arrowprops=dict(arrowstyle='<->', color='black', lw=2))
    ax1.text(1.15, 0.63, '+9.5%\nF1', fontsize=10, fontweight='bold', color='black')
    
    ax1.axhline(y=0.6418, color=COLORS['fp16'], linestyle='--', alpha=0.7)
    ax1.text(1.5, 0.645, 'FP16 baseline', fontsize=8, color=COLORS['fp16'])
    
    ax1.set_ylabel('CoQA F1 Score')
    ax1.set_title('Accuracy Comparison')
    ax1.set_ylim(0.5, 0.75)
    
    # Right plot: Memory (same for both)
    memory = [965.1, 965.1]
    bars2 = ax2.bar(methods, memory, color=[COLORS['nf4'], COLORS['fp4']], 
                    edgecolor='black', linewidth=1, width=0.6)
    
    ax2.axhline(y=2357.1, color=COLORS['fp16'], linestyle='--', alpha=0.7)
    ax2.text(1.5, 2400, 'FP16: 2357 MB', fontsize=8, color=COLORS['fp16'])
    
    # Add compression ratio
    ax2.text(0.5, 1200, '2.44×\ncompression', fontsize=10, ha='center', 
             fontweight='bold', color='gray')
    
    ax2.set_ylabel('GPU Memory (MB)')
    ax2.set_title('Memory Usage')
    ax2.set_ylim(0, 2700)
    
    plt.suptitle('NF4 vs FP4: Same Memory, Different Accuracy', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'fig3_nf4_vs_fp4.pdf')
    plt.savefig(output_dir / 'fig3_nf4_vs_fp4.png')
    plt.close()
    print("✓ Generated fig3_nf4_vs_fp4.pdf")

def fig4_ablation_heatmap(data, output_dir):
    """
    Figure 4: Ablation study as a visual table/heatmap
    """
    fig, ax = plt.subplots(figsize=(7, 4))
    
    # Create matrix for heatmap
    # Rows: NF4, FP4
    # Columns: With DQ, Without DQ
    matrix = np.array([
        [0.6758, 0.6758],  # NF4: with DQ, without DQ
        [0.5807, 0.5886],  # FP4: with DQ, without DQ
    ])
    
    im = ax.imshow(matrix, cmap='RdYlGn', aspect='auto', vmin=0.55, vmax=0.70)
    
    # Labels
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Double Quant\nEnabled', 'Double Quant\nDisabled'])
    ax.set_yticks([0, 1])
    ax.set_yticklabels(['NF4', 'FP4'])
    
    # Add text annotations
    for i in range(2):
        for j in range(2):
            text = ax.text(j, i, f'{matrix[i, j]:.4f}', 
                          ha='center', va='center', fontsize=12, fontweight='bold',
                          color='white' if matrix[i, j] < 0.62 else 'black')
    
    ax.set_title('Ablation Study: Quantization Type × Double Quantization\n(CoQA F1 Scores)')
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('F1 Score')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'fig4_ablation_heatmap.pdf')
    plt.savefig(output_dir / 'fig4_ablation_heatmap.png')
    plt.close()
    print("✓ Generated fig4_ablation_heatmap.pdf")

def fig5_compression_waterfall(data, output_dir):
    """
    Figure 5: Waterfall chart showing memory breakdown
    """
    fig, ax = plt.subplots(figsize=(6, 4))
    
    # Data for waterfall
    categories = ['FP16\nBaseline', 'Weight\nQuantization', '4-bit\nResult']
    values = [2357.1, -1392.0, 965.1]
    
    # Calculate positions
    running_total = 0
    bottoms = []
    for i, v in enumerate(values):
        if i == 0:
            bottoms.append(0)
            running_total = v
        elif i == len(values) - 1:
            bottoms.append(0)
        else:
            if v < 0:
                bottoms.append(running_total + v)
                running_total += v
            else:
                bottoms.append(running_total)
                running_total += v
    
    colors = [COLORS['fp16'], '#DC3545', COLORS['nf4']]
    
    bars = ax.bar(categories, [abs(v) for v in values], bottom=bottoms, 
                  color=colors, edgecolor='black', linewidth=1)
    
    # Annotations
    ax.text(0, 2357.1 + 50, '2357 MB', ha='center', fontsize=10, fontweight='bold')
    ax.text(1, 1650, '-1392 MB\n(59% reduction)', ha='center', fontsize=9, color='white', fontweight='bold')
    ax.text(2, 965.1 + 50, '965 MB', ha='center', fontsize=10, fontweight='bold')
    
    # Arrow showing savings
    ax.annotate('', xy=(2.4, 2357.1), xytext=(2.4, 965.1),
                arrowprops=dict(arrowstyle='<->', color='black', lw=2))
    ax.text(2.55, 1650, '2.44×\nsmaller', fontsize=9, fontweight='bold')
    
    ax.set_ylabel('GPU Memory (MB)')
    ax.set_title('Memory Reduction through 4-bit Quantization')
    ax.set_ylim(0, 2800)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'fig5_compression_waterfall.pdf')
    plt.savefig(output_dir / 'fig5_compression_waterfall.png')
    plt.close()
    print("✓ Generated fig5_compression_waterfall.pdf")

def fig6_summary_infographic(data, output_dir):
    """
    Figure 6: Summary infographic with key numbers
    """
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 3)
    ax.axis('off')
    
    # Three key metrics boxes
    boxes = [
        {'x': 1.5, 'metric': '2.44×', 'label': 'Memory\nCompression', 'color': COLORS['nf4']},
        {'x': 5.0, 'metric': '+5.3%', 'label': 'Accuracy vs\nBaseline', 'color': COLORS['nf4']},
        {'x': 8.5, 'metric': '9.5%', 'label': 'NF4 vs FP4\nAdvantage', 'color': COLORS['fp4']},
    ]
    
    for box in boxes:
        # Draw box
        rect = plt.Rectangle((box['x']-1, 0.5), 2, 2, 
                             facecolor=box['color'], alpha=0.2, 
                             edgecolor=box['color'], linewidth=2)
        ax.add_patch(rect)
        
        # Add text
        ax.text(box['x'], 1.9, box['metric'], fontsize=24, fontweight='bold',
               ha='center', va='center', color=box['color'])
        ax.text(box['x'], 1.0, box['label'], fontsize=10, ha='center', va='center')
    
    ax.set_title('Llama 3.2-1B Quantization: Key Results', fontsize=14, fontweight='bold', y=1.05)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'fig6_summary_infographic.pdf')
    plt.savefig(output_dir / 'fig6_summary_infographic.png')
    plt.close()
    print("✓ Generated fig6_summary_infographic.pdf")


def main():
    """Generate all figures"""
    print("=" * 60)
    print("Generating Publication Figures")
    print("=" * 60)
    
    # Setup output directory
    output_dir = Path(__file__).parent / 'figures'
    output_dir.mkdir(exist_ok=True)
    
    # Load data
    data = load_results()
    print(f"Loaded results: {len(data['results'])} experiments")
    
    # Generate all figures
    fig1_accuracy_vs_memory(data, output_dir)
    fig2_all_configurations_bar(data, output_dir)
    fig3_nf4_vs_fp4(data, output_dir)
    fig4_ablation_heatmap(data, output_dir)
    fig5_compression_waterfall(data, output_dir)
    fig6_summary_infographic(data, output_dir)
    
    print("=" * 60)
    print(f"✓ All figures saved to: {output_dir}")
    print("=" * 60)


if __name__ == '__main__':
    main()

