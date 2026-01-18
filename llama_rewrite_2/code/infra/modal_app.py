"""
Modal serverless GPU runner for quantization experiments.

Usage:
    modal run code/infra/modal_app.py                      # Quick comparison
    modal run code/infra/modal_app.py --experiment nf4     # Single experiment
    modal run code/infra/modal_app.py --all --limit 500    # Full suite

Architecture:
    - modal_app.py: Local orchestration, Modal config, results saving
    - gpu_runner.py: GPU-side code with torch/llama_quant imports
    
The split keeps imports clean - gpu_runner.py has all imports at top
since it only runs in Modal cloud where dependencies are installed.
"""

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING

import modal

# GPU-side imports (only available in Modal cloud environment)
# Uses TYPE_CHECKING for IDE support, actual import at runtime in cloud
if TYPE_CHECKING:
    from infra.gpu_runner import ExperimentRunner, ExperimentRequest, ComparisonRunner

try:
    from infra.gpu_runner import ExperimentRunner, ExperimentRequest, ComparisonRunner
    GPU_IMPORTS_AVAILABLE = True
except ImportError:
    GPU_IMPORTS_AVAILABLE = False


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class ModalConfig:
    """Infrastructure configuration."""
    app_name: str = "llama-quantization"
    gpu_type: str = "A10G"
    timeout: int = 3600
    cache_dir: str = "/cache"
    hf_cache_dir: str = "/cache/huggingface"


# =============================================================================
# Modal Infrastructure
# =============================================================================

CONFIG = ModalConfig()

app = modal.App(CONFIG.app_name)

image = (
    modal.Image.from_registry("nvidia/cuda:12.1.0-devel-ubuntu22.04", add_python="3.11")
    .apt_install("git")
    .pip_install(
        "torch>=2.1.0",
        "transformers>=4.36.0",
        "accelerate>=0.25.0",
        "datasets>=2.16.0",
        "bitsandbytes>=0.43.0",
        "optimum>=1.15.0",
        "lm-eval>=0.4.0",
        "sentencepiece>=0.1.99",
        "safetensors>=0.4.0",
        "huggingface-hub>=0.20.0",
        "tqdm>=4.66.0",
        "numpy<2.0.0",
        "pyyaml>=6.0",
    )
    .add_local_python_source("code")
)

volume = modal.Volume.from_name("llama-quant-cache", create_if_missing=True)
hf_secret = modal.Secret.from_name("huggingface-secret")


# =============================================================================
# Results Manager (Local)
# =============================================================================

class ResultsManager:
    """
    Manages local storage of experiment results.
    
    Auto-increments filenames: results1.json, results2.json, ...
    """
    
    def __init__(self, results_dir: Path):
        self.results_dir = results_dir
        self.results_dir.mkdir(parents=True, exist_ok=True)
    
    def save(self, results: Dict[str, Any]) -> Path:
        """Save results with timestamp metadata."""
        output_path = self._get_next_filename()
        
        results["timestamp"] = datetime.now().isoformat()
        results["saved_to"] = str(output_path)
        
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        
        return output_path
    
    def _get_next_filename(self) -> Path:
        """Get next available results filename."""
        existing_nums = []
        for f in self.results_dir.glob("results*.json"):
            try:
                existing_nums.append(int(f.stem.replace("results", "")))
            except ValueError:
                pass
        
        next_num = max(existing_nums, default=0) + 1
        return self.results_dir / f"results{next_num}.json"


# =============================================================================
# Modal GPU Functions
# =============================================================================

@app.function(
    image=image,
    gpu=CONFIG.gpu_type,
    timeout=CONFIG.timeout,
    secrets=[hf_secret],
    volumes={CONFIG.cache_dir: volume},
)
def run_single_experiment(experiment_name: str, limit: Optional[int] = None) -> dict:
    """Run single experiment on GPU."""
    runner = ExperimentRunner(CONFIG.hf_cache_dir)
    request = ExperimentRequest(name=experiment_name, limit=limit)
    result = runner.run(request)
    
    volume.commit()
    return result.to_dict()


@app.function(
    image=image,
    gpu=CONFIG.gpu_type,
    timeout=CONFIG.timeout * 2,
    secrets=[hf_secret],
    volumes={CONFIG.cache_dir: volume},
)
def run_comparison(experiments: Optional[List[str]] = None, limit: int = 100) -> dict:
    """Run comparison across multiple experiments on GPU."""
    runner = ComparisonRunner(CONFIG.hf_cache_dir)
    results = runner.run(experiments, limit)
    
    volume.commit()
    return {"experiments": [r.to_dict() for r in results]}


# =============================================================================
# CLI Entry Point
# =============================================================================

@app.local_entrypoint()
def main(
    experiment: str = None,
    all_experiments: bool = False,
    limit: int = 100,
):
    """
    Run quantization experiments on Modal.
    
    Examples:
        modal run code/infra/modal_app.py                    # FP16 vs NF4 vs FP4
        modal run code/infra/modal_app.py --experiment nf4   # Single experiment
        modal run code/infra/modal_app.py --all --limit 500  # Full suite
    """
    results_manager = ResultsManager(
        Path(__file__).parent.parent.parent / "results"
    )
    
    # Dispatch to appropriate runner
    if all_experiments:
        experiments = [
            "fp16_baseline",
            "bnb_4bit_nf4",
            "bnb_4bit_fp4",
            "bnb_4bit_nf4_no_double",
            "bnb_4bit_nf4_bf16",
        ]
        results = run_comparison.remote(experiments=experiments, limit=limit)
        run_type = "all"
        
    elif experiment:
        results = run_single_experiment.remote(experiment, limit=limit)
        results = {"experiments": [results]}
        run_type = "single"
        
    else:
        results = run_comparison.remote(limit=limit)
        run_type = "comparison"
    
    # Save locally
    results["limit"] = limit
    results["run_type"] = run_type
    output_path = results_manager.save(results)
    
    print(f"\n{'='*60}")
    print(f"Results saved to: {output_path}")
    print(f"{'='*60}")
