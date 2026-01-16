"""
Utilities for visualization and report generation
Creates publication-ready figures for the 4-page report
"""

import json
from pathlib import Path
from typing import Dict, List, Any, Optional

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import MaxNLocator

# Set publication-quality defaults
plt.rcParams.update({
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.figsize': (6, 4),
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})


def load_sweep_results(results_file: str) -> Dict[str, Any]:
    """Load sweep results from JSON file"""
    with open(results_file, 'r') as f:
        return json.load(f)


def plot_accuracy_vs_size(
    results: List[Dict[str, Any]],
    output_path: Optional[str] = None,
    title: str = "Accuracy vs Model Size",
):
    """
    Plot accuracy (CoQA F1) vs model size for all experiments
    
    This is the key figure showing the Pareto frontier
    """
    fig, ax = plt.subplots(figsize=(7, 5))
    
    # Group by method
    methods = {}
    for r in results:
        method = r.get("method", "unknown")
        if method not in methods:
            methods[method] = {"sizes": [], "f1s": [], "names": []}
        
        size = r.get("model_size_mb")
        f1 = r.get("coqa_f1")
        
        if size is not None and f1 is not None:
            methods[method]["sizes"].append(size)
            methods[method]["f1s"].append(f1)
            methods[method]["names"].append(r.get("name", ""))
    
    # Color palette
    colors = {
        "none": "#2ecc71",      # Green for FP16
        "bnb_8bit": "#3498db",  # Blue
        "bnb_4bit": "#9b59b6",  # Purple
        "gptq": "#e74c3c",      # Red
        "awq": "#f39c12",       # Orange
    }
    
    markers = {
        "none": "s",
        "bnb_8bit": "o",
        "bnb_4bit": "^",
        "gptq": "D",
        "awq": "p",
    }
    
    for method, data in methods.items():
        color = colors.get(method, "#95a5a6")
        marker = markers.get(method, "o")
        
        ax.scatter(
            data["sizes"],
            data["f1s"],
            c=color,
            marker=marker,
            s=80,
            label=method.upper().replace("_", " "),
            alpha=0.8,
            edgecolors='white',
            linewidth=0.5,
        )
    
    # Find and highlight Pareto frontier
    all_points = [(r.get("model_size_mb", 0), r.get("coqa_f1", 0)) 
                  for r in results 
                  if r.get("model_size_mb") and r.get("coqa_f1")]
    
    if all_points:
        pareto = find_pareto_frontier(all_points)
        pareto_sorted = sorted(pareto, key=lambda x: x[0])
        ax.plot(
            [p[0] for p in pareto_sorted],
            [p[1] for p in pareto_sorted],
            'k--',
            alpha=0.5,
            linewidth=1.5,
            label='Pareto Frontier'
        )
    
    ax.set_xlabel("Model Size (MB)")
    ax.set_ylabel("CoQA F1 Score")
    ax.set_title(title)
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    
    # Set reasonable axis limits
    ax.set_ylim(bottom=0)
    
    if output_path:
        plt.savefig(output_path)
        print(f"Saved: {output_path}")
    
    return fig, ax


def plot_accuracy_vs_bits(
    results: List[Dict[str, Any]],
    output_path: Optional[str] = None,
):
    """
    Plot accuracy degradation vs bit precision
    """
    fig, ax = plt.subplots(figsize=(6, 4))
    
    # Extract bit width from config name or method
    def get_bits(r):
        name = r.get("name", "")
        if "fp16" in name or r.get("method") == "none":
            return 16
        elif "8bit" in name:
            return 8
        elif "4bit" in name or "4bit" in r.get("method", ""):
            return 4
        elif "3bit" in name:
            return 3
        elif "2bit" in name:
            return 2
        return None
    
    bits_data = {}
    for r in results:
        bits = get_bits(r)
        f1 = r.get("coqa_f1")
        
        if bits is not None and f1 is not None:
            if bits not in bits_data:
                bits_data[bits] = []
            bits_data[bits].append(f1)
    
    # Create box plot
    bit_widths = sorted(bits_data.keys(), reverse=True)
    data = [bits_data[b] for b in bit_widths]
    
    bp = ax.boxplot(data, labels=[str(b) for b in bit_widths], patch_artist=True)
    
    # Color boxes
    colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(bit_widths)))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax.set_xlabel("Bit Width")
    ax.set_ylabel("CoQA F1 Score")
    ax.set_title("Accuracy vs Quantization Precision")
    ax.grid(True, alpha=0.3, axis='y')
    
    if output_path:
        plt.savefig(output_path)
        print(f"Saved: {output_path}")
    
    return fig, ax


def plot_latency_comparison(
    results: List[Dict[str, Any]],
    output_path: Optional[str] = None,
):
    """
    Bar plot comparing inference latency across methods
    """
    fig, ax = plt.subplots(figsize=(8, 4))
    
    # Extract latency data
    names = []
    latencies = []
    
    for r in results:
        name = r.get("name", "unknown")
        decode_latency = None
        
        if "benchmarks" in r:
            decode_latency = r["benchmarks"].get("decode_latency_ms_per_token")
        elif "decode_latency_ms_per_token" in r:
            decode_latency = r["decode_latency_ms_per_token"]
        
        if decode_latency is not None:
            names.append(name[:20])  # Truncate long names
            latencies.append(decode_latency)
    
    if not names:
        print("No latency data found")
        return None, None
    
    # Sort by latency
    sorted_idx = np.argsort(latencies)
    names = [names[i] for i in sorted_idx]
    latencies = [latencies[i] for i in sorted_idx]
    
    bars = ax.barh(names, latencies, color=plt.cm.viridis(np.linspace(0.2, 0.8, len(names))))
    
    ax.set_xlabel("Decode Latency (ms/token)")
    ax.set_title("Inference Latency Comparison")
    ax.grid(True, alpha=0.3, axis='x')
    
    # Add value labels
    for bar, val in zip(bars, latencies):
        ax.text(val + 0.5, bar.get_y() + bar.get_height()/2, 
                f'{val:.1f}', va='center', fontsize=8)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path)
        print(f"Saved: {output_path}")
    
    return fig, ax


def plot_memory_comparison(
    results: List[Dict[str, Any]],
    output_path: Optional[str] = None,
):
    """
    Stacked bar chart showing memory breakdown
    """
    fig, ax = plt.subplots(figsize=(8, 4))
    
    names = []
    model_sizes = []
    peak_memories = []
    
    for r in results:
        name = r.get("name", "unknown")
        model_size = r.get("model_size_mb")
        peak_mem = r.get("memory_peak_mb")
        
        if model_size is not None:
            names.append(name[:20])
            model_sizes.append(model_size)
            peak_memories.append(peak_mem or model_size * 1.2)
    
    if not names:
        print("No memory data found")
        return None, None
    
    # Sort by model size
    sorted_idx = np.argsort(model_sizes)
    names = [names[i] for i in sorted_idx]
    model_sizes = [model_sizes[i] for i in sorted_idx]
    peak_memories = [peak_memories[i] for i in sorted_idx]
    
    x = np.arange(len(names))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, model_sizes, width, label='Model Size', color='#3498db')
    bars2 = ax.bar(x + width/2, peak_memories, width, label='Peak Memory', color='#e74c3c', alpha=0.7)
    
    ax.set_ylabel("Memory (MB)")
    ax.set_title("Memory Usage Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path)
        print(f"Saved: {output_path}")
    
    return fig, ax


def plot_throughput_scaling(
    results: List[Dict[str, Any]],
    output_path: Optional[str] = None,
):
    """
    Line plot showing throughput scaling with batch size
    """
    fig, ax = plt.subplots(figsize=(7, 5))
    
    for r in results:
        name = r.get("name", "unknown")
        throughput = None
        
        if "benchmarks" in r:
            throughput = r["benchmarks"].get("throughput_tokens_per_sec")
        elif "throughput_tokens_per_sec" in r:
            throughput = r["throughput_tokens_per_sec"]
        
        if throughput and isinstance(throughput, dict):
            batch_sizes = sorted([int(k) for k in throughput.keys()])
            tps = [throughput[str(bs)] for bs in batch_sizes]
            
            ax.plot(batch_sizes, tps, 'o-', label=name[:20], markersize=6)
    
    ax.set_xlabel("Batch Size")
    ax.set_ylabel("Throughput (tokens/sec)")
    ax.set_title("Throughput Scaling with Batch Size")
    ax.legend(loc='upper left', fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    
    if output_path:
        plt.savefig(output_path)
        print(f"Saved: {output_path}")
    
    return fig, ax


def find_pareto_frontier(points: List[tuple]) -> List[tuple]:
    """
    Find Pareto frontier points (minimize size, maximize accuracy)
    """
    pareto = []
    
    for p in points:
        dominated = False
        for q in points:
            # q dominates p if q has smaller size AND higher accuracy
            if q[0] < p[0] and q[1] > p[1]:
                dominated = True
                break
        
        if not dominated:
            pareto.append(p)
    
    return pareto


def generate_all_figures(
    results_file: str,
    output_dir: str = "./figures",
):
    """
    Generate all figures for the report
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load results
    data = load_sweep_results(results_file)
    
    if "summary" in data:
        results = data["summary"]["results"]
    elif "all_results" in data:
        results = data["all_results"]
    else:
        results = data
    
    print(f"Generating figures from {len(results)} experiments...")
    
    # Generate all plots
    plot_accuracy_vs_size(results, output_path / "accuracy_vs_size.pdf")
    plot_accuracy_vs_bits(results, output_path / "accuracy_vs_bits.pdf")
    plot_latency_comparison(results, output_path / "latency_comparison.pdf")
    plot_memory_comparison(results, output_path / "memory_comparison.pdf")
    plot_throughput_scaling(results, output_path / "throughput_scaling.pdf")
    
    print(f"\nAll figures saved to {output_dir}/")


def create_results_table(
    results: List[Dict[str, Any]],
    output_path: Optional[str] = None,
) -> str:
    """
    Create LaTeX table for the report
    """
    lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\caption{Quantization Results on CoQA}",
        r"\label{tab:results}",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"Method & Bits & Size (MB) & F1 Score & Latency (ms/tok) \\",
        r"\midrule",
    ]
    
    for r in results:
        name = r.get("name", "?").replace("_", r"\_")[:25]
        
        # Infer bits from name
        if "fp16" in r.get("name", "") or r.get("method") == "none":
            bits = "16"
        elif "8bit" in r.get("name", ""):
            bits = "8"
        elif "4bit" in r.get("name", ""):
            bits = "4"
        elif "3bit" in r.get("name", ""):
            bits = "3"
        else:
            bits = "?"
        
        size = r.get("model_size_mb", 0)
        f1 = r.get("coqa_f1", 0)
        latency = r.get("decode_latency_ms_per_token", 0)
        
        size_str = f"{size:.1f}" if size else "-"
        f1_str = f"{f1:.4f}" if f1 else "-"
        lat_str = f"{latency:.2f}" if latency else "-"
        
        lines.append(f"{name} & {bits} & {size_str} & {f1_str} & {lat_str} \\\\")
    
    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])
    
    table = "\n".join(lines)
    
    if output_path:
        with open(output_path, 'w') as f:
            f.write(table)
        print(f"Saved LaTeX table to {output_path}")
    
    return table


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("results_file", help="Path to sweep results JSON")
    parser.add_argument("--output", default="./figures", help="Output directory")
    
    args = parser.parse_args()
    
    generate_all_figures(args.results_file, args.output)
