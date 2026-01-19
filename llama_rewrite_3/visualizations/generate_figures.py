#!/usr/bin/env python3
"""
Generate publication-ready figures for Llama quantization report.
Reads results from results2.json and creates matplotlib figures.
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
    """Load results from results2.json"""
    results_dir = Path(__file__).parent.parent / 'results'
    
    with open(results_dir / 'results2.json') as f:
        data = json.load(f)
    
    # Convert to expected format
    experiments = []
    for exp in data['experiments']:
        experiments.append({
            'name': exp['experiment_name'],
            'coqa_f1': exp['coqa_metrics']['coqa_f1'],
            'coqa_em': exp['coqa_metrics']['coqa_em'],
            'memory_mb': exp['model_size_mb'],
            'peak_memory_mb': exp['benchmarks'].get('memory_peak_mb', exp['model_size_mb']),
            'throughput': exp['benchmarks'].get('throughput_tokens_per_sec', {}),
            'prefill_latency': exp['benchmarks'].get('prefill_latency_ms', {}),
            'decode_latency': exp['benchmarks'].get('decode_latency_ms_per_token', 0),
        })
    
    return {'results': experiments}


def fig1_accuracy_vs_memory(data, output_dir):
    """
    Figure 1: Accuracy vs Memory tradeoff scatter plot
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
            label = 'NF4 (4-bit)'
        elif 'fp4' in name:
            color, marker = COLORS['fp4'], '^'
            label = 'FP4 (4-bit)'
        else:
            color, marker, label = COLORS['baseline'], 'x', name
        
        ax.scatter(mem, f1, c=color, marker=marker, s=150, 
                   edgecolors='black', linewidths=0.5, zorder=5, label=label)
        
        # Annotate points
        offset = (15, 10) if 'fp16' in name else (-15, 10) if 'nf4' in name else (15, -15)
        display_name = name.replace('bnb_4bit_', '').replace('fp16_baseline', 'FP16').upper()
        ax.annotate(display_name, (mem, f1), textcoords="offset points", 
                    xytext=offset, fontsize=9, ha='center', fontweight='bold')
    
    # Draw arrow showing compression direction
    ax.annotate('', xy=(1100, 0.65), xytext=(2200, 0.65),
                arrowprops=dict(arrowstyle='->', color='gray', lw=2))
    ax.text(1650, 0.66, '2.44× compression', fontsize=9, ha='center', color='gray', style='italic')
    
    # Reference line for baseline
    baseline_f1 = next(r['coqa_f1'] for r in results if 'fp16' in r['name'])
    ax.axhline(y=baseline_f1, color=COLORS['fp16'], linestyle='--', alpha=0.5)
    
    ax.set_xlabel('Model Size (MB)')
    ax.set_ylabel('CoQA F1 Score')
    ax.set_title('Accuracy vs Memory: Quantization Tradeoff')
    
    # Custom legend (avoid duplicates)
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc='lower right')
    
    ax.set_xlim(800, 2600)
    ax.set_ylim(0.55, 0.72)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'fig1_accuracy_vs_memory.pdf')
    plt.savefig(output_dir / 'fig1_accuracy_vs_memory.png')
    plt.close()
    print("✓ Generated fig1_accuracy_vs_memory.pdf")


def fig2_bar_comparison(data, output_dir):
    """
    Figure 2: Bar chart comparing all configurations
    """
    fig, ax = plt.subplots(figsize=(7, 4))
    
    results = data['results']
    names = []
    f1_scores = []
    colors = []
    
    for r in results:
        name = r['name']
        if 'fp16' in name:
            names.append('FP16\nBaseline')
            colors.append(COLORS['fp16'])
        elif 'nf4' in name:
            names.append('4-bit\nNF4')
            colors.append(COLORS['nf4'])
        elif 'fp4' in name:
            names.append('4-bit\nFP4')
            colors.append(COLORS['fp4'])
        f1_scores.append(r['coqa_f1'])
    
    bars = ax.bar(names, f1_scores, color=colors, edgecolor='black', linewidth=1, width=0.6)
    
    # Add value labels on bars
    for bar, score in zip(bars, f1_scores):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.008, 
                f'{score:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Baseline reference line
    baseline_f1 = f1_scores[0]
    ax.axhline(y=baseline_f1, color=COLORS['fp16'], linestyle='--', alpha=0.7)
    
    ax.set_ylabel('CoQA F1 Score')
    ax.set_title('Quantization Method Comparison (Llama 3.2-1B)')
    ax.set_ylim(0.5, 0.75)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'fig2_bar_comparison.pdf')
    plt.savefig(output_dir / 'fig2_bar_comparison.png')
    plt.close()
    print("✓ Generated fig2_bar_comparison.pdf")


def fig3_nf4_vs_fp4(data, output_dir):
    """
    Figure 3: Direct NF4 vs FP4 comparison (key finding)
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3.5))
    
    results = data['results']
    nf4_data = next(r for r in results if 'nf4' in r['name'])
    fp4_data = next(r for r in results if 'fp4' in r['name'])
    fp16_data = next(r for r in results if 'fp16' in r['name'])
    
    # Left plot: F1 scores comparison
    methods = ['NF4', 'FP4']
    f1_scores = [nf4_data['coqa_f1'], fp4_data['coqa_f1']]
    
    bars = ax1.bar(methods, f1_scores, color=[COLORS['nf4'], COLORS['fp4']], 
                   edgecolor='black', linewidth=1.5, width=0.5)
    
    # Add value labels
    for bar, score in zip(bars, f1_scores):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{score:.3f}', ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    # Add delta annotation
    diff = (nf4_data['coqa_f1'] - fp4_data['coqa_f1']) / fp4_data['coqa_f1'] * 100
    ax1.annotate('', xy=(1, nf4_data['coqa_f1']), xytext=(1, fp4_data['coqa_f1']),
                arrowprops=dict(arrowstyle='<->', color='black', lw=2))
    ax1.text(1.2, (nf4_data['coqa_f1'] + fp4_data['coqa_f1'])/2, 
             f'+{diff:.1f}%', fontsize=11, fontweight='bold', color='black')
    
    # FP16 baseline line
    ax1.axhline(y=fp16_data['coqa_f1'], color=COLORS['fp16'], linestyle='--', alpha=0.7, lw=2)
    ax1.text(1.5, fp16_data['coqa_f1'] + 0.005, 'FP16', fontsize=9, color=COLORS['fp16'])
    
    ax1.set_ylabel('CoQA F1 Score')
    ax1.set_title('Accuracy Comparison')
    ax1.set_ylim(0.5, 0.75)
    
    # Right plot: Memory comparison
    memory = [nf4_data['memory_mb'], fp4_data['memory_mb']]
    bars2 = ax1.bar(methods, memory, color=[COLORS['nf4'], COLORS['fp4']], 
                    edgecolor='black', linewidth=1.5, width=0.5)
    
    # Memory bars
    methods_mem = ['FP16', 'NF4', 'FP4']
    memory_vals = [fp16_data['memory_mb'], nf4_data['memory_mb'], fp4_data['memory_mb']]
    colors_mem = [COLORS['fp16'], COLORS['nf4'], COLORS['fp4']]
    
    bars2 = ax2.bar(methods_mem, memory_vals, color=colors_mem, 
                    edgecolor='black', linewidth=1.5, width=0.5)
    
    # Add memory labels
    for bar, mem in zip(bars2, memory_vals):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50, 
                f'{mem:.0f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Compression annotation
    compression = fp16_data['memory_mb'] / nf4_data['memory_mb']
    ax2.annotate('', xy=(1, fp16_data['memory_mb']), xytext=(1, nf4_data['memory_mb']),
                arrowprops=dict(arrowstyle='<->', color='gray', lw=2))
    ax2.text(1.3, 1650, f'{compression:.2f}×\nsmaller', fontsize=10, 
             fontweight='bold', color='gray', ha='left')
    
    ax2.set_ylabel('Model Size (MB)')
    ax2.set_title('Memory Usage')
    ax2.set_ylim(0, 2800)
    
    plt.suptitle('NF4 vs FP4: Better Accuracy at Same Memory Cost', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'fig3_nf4_vs_fp4.pdf')
    plt.savefig(output_dir / 'fig3_nf4_vs_fp4.png')
    plt.close()
    print("✓ Generated fig3_nf4_vs_fp4.pdf")


def fig4_throughput_comparison(data, output_dir):
    """
    Figure 4: Throughput comparison across batch sizes
    """
    fig, ax = plt.subplots(figsize=(7, 4))
    
    results = data['results']
    batch_sizes = ['1', '4', '8']
    
    x = np.arange(len(batch_sizes))
    width = 0.25
    
    for i, r in enumerate(results):
        throughput = r['throughput']
        if throughput:
            values = [throughput.get(bs, 0) for bs in batch_sizes]
            name = r['name'].replace('bnb_4bit_', '').replace('fp16_baseline', 'FP16').upper()
            
            if 'fp16' in r['name']:
                color = COLORS['fp16']
            elif 'nf4' in r['name']:
                color = COLORS['nf4']
            else:
                color = COLORS['fp4']
            
            bars = ax.bar(x + (i - 1) * width, values, width, label=name, 
                         color=color, edgecolor='black', linewidth=0.5)
    
    ax.set_xlabel('Batch Size')
    ax.set_ylabel('Throughput (tokens/sec)')
    ax.set_title('Inference Throughput by Quantization Method')
    ax.set_xticks(x)
    ax.set_xticklabels(batch_sizes)
    ax.legend()
    ax.set_ylim(0, 450)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'fig4_throughput.pdf')
    plt.savefig(output_dir / 'fig4_throughput.png')
    plt.close()
    print("✓ Generated fig4_throughput.pdf")


def fig5_memory_waterfall(data, output_dir):
    """
    Figure 5: Waterfall chart showing memory savings
    """
    fig, ax = plt.subplots(figsize=(6, 4))
    
    results = data['results']
    fp16_mem = next(r['memory_mb'] for r in results if 'fp16' in r['name'])
    nf4_mem = next(r['memory_mb'] for r in results if 'nf4' in r['name'])
    
    # Data for waterfall
    categories = ['FP16\nBaseline', 'Quantization\nSavings', '4-bit\nResult']
    savings = fp16_mem - nf4_mem
    values = [fp16_mem, -savings, nf4_mem]
    
    # Calculate positions
    bottoms = [0, nf4_mem, 0]
    heights = [fp16_mem, savings, nf4_mem]
    colors = [COLORS['fp16'], '#DC3545', COLORS['nf4']]
    
    bars = ax.bar(categories, heights, bottom=bottoms, color=colors, 
                  edgecolor='black', linewidth=1.5)
    
    # Annotations
    ax.text(0, fp16_mem + 50, f'{fp16_mem:.0f} MB', ha='center', fontsize=11, fontweight='bold')
    reduction_pct = (savings / fp16_mem) * 100
    ax.text(1, fp16_mem/2 + nf4_mem/2, f'-{savings:.0f} MB\n({reduction_pct:.0f}% reduction)', 
            ha='center', fontsize=10, color='white', fontweight='bold')
    ax.text(2, nf4_mem + 50, f'{nf4_mem:.0f} MB', ha='center', fontsize=11, fontweight='bold')
    
    # Compression ratio annotation
    compression = fp16_mem / nf4_mem
    ax.annotate('', xy=(2.4, fp16_mem), xytext=(2.4, nf4_mem),
                arrowprops=dict(arrowstyle='<->', color='black', lw=2))
    ax.text(2.55, (fp16_mem + nf4_mem)/2, f'{compression:.2f}×\nsmaller', 
            fontsize=10, fontweight='bold')
    
    ax.set_ylabel('Model Size (MB)')
    ax.set_title('Memory Reduction through 4-bit Quantization')
    ax.set_ylim(0, 2800)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'fig5_memory_waterfall.pdf')
    plt.savefig(output_dir / 'fig5_memory_waterfall.png')
    plt.close()
    print("✓ Generated fig5_memory_waterfall.pdf")


def fig6_summary_metrics(data, output_dir):
    """
    Figure 6: Summary infographic with key metrics
    """
    fig, ax = plt.subplots(figsize=(9, 3))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 3)
    ax.axis('off')
    
    results = data['results']
    fp16_f1 = next(r['coqa_f1'] for r in results if 'fp16' in r['name'])
    nf4_f1 = next(r['coqa_f1'] for r in results if 'nf4' in r['name'])
    fp4_f1 = next(r['coqa_f1'] for r in results if 'fp4' in r['name'])
    fp16_mem = next(r['memory_mb'] for r in results if 'fp16' in r['name'])
    nf4_mem = next(r['memory_mb'] for r in results if 'nf4' in r['name'])
    
    compression = fp16_mem / nf4_mem
    nf4_vs_baseline = ((nf4_f1 - fp16_f1) / fp16_f1) * 100
    nf4_vs_fp4 = ((nf4_f1 - fp4_f1) / fp4_f1) * 100
    
    # Three key metrics boxes
    boxes = [
        {'x': 2, 'metric': f'{compression:.2f}×', 'label': 'Memory\nCompression', 'color': COLORS['nf4']},
        {'x': 6, 'metric': f'+{nf4_vs_baseline:.1f}%', 'label': 'NF4 vs FP16\nBaseline', 'color': COLORS['nf4']},
        {'x': 10, 'metric': f'+{nf4_vs_fp4:.1f}%', 'label': 'NF4 vs FP4\nAdvantage', 'color': COLORS['fp4']},
    ]
    
    for box in boxes:
        # Draw box
        rect = plt.Rectangle((box['x']-1.3, 0.4), 2.6, 2.2, 
                             facecolor=box['color'], alpha=0.15, 
                             edgecolor=box['color'], linewidth=3, 
                             joinstyle='round')
        ax.add_patch(rect)
        
        # Add text
        ax.text(box['x'], 1.9, box['metric'], fontsize=26, fontweight='bold',
               ha='center', va='center', color=box['color'])
        ax.text(box['x'], 0.9, box['label'], fontsize=11, ha='center', va='center',
               color='#333333')
    
    ax.set_title('Llama 3.2-1B Quantization: Key Results', fontsize=15, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'fig6_summary_metrics.pdf')
    plt.savefig(output_dir / 'fig6_summary_metrics.png')
    plt.close()
    print("✓ Generated fig6_summary_metrics.pdf")


def main():
    """Generate all figures"""
    print("=" * 60)
    print("Generating Publication Figures from results2.json")
    print("=" * 60)
    
    # Setup output directory - save to reports/figures for LaTeX
    script_dir = Path(__file__).parent
    output_dir = script_dir.parent / 'reports' / 'figures'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Also save to visualizations/figures
    viz_output_dir = script_dir / 'figures'
    viz_output_dir.mkdir(exist_ok=True)
    
    # Load data
    data = load_results()
    print(f"Loaded results: {len(data['results'])} experiments")
    for r in data['results']:
        print(f"  - {r['name']}: F1={r['coqa_f1']:.4f}, Memory={r['memory_mb']:.0f}MB")
    
    # Generate all figures to both directories
    for out_dir in [output_dir, viz_output_dir]:
        fig1_accuracy_vs_memory(data, out_dir)
        fig2_bar_comparison(data, out_dir)
        fig3_nf4_vs_fp4(data, out_dir)
        fig4_throughput_comparison(data, out_dir)
        fig5_memory_waterfall(data, out_dir)
        fig6_summary_metrics(data, out_dir)
    
    print("=" * 60)
    print(f"✓ All figures saved to:")
    print(f"  - {output_dir}")
    print(f"  - {viz_output_dir}")
    print("=" * 60)


if __name__ == '__main__':
    main()
