"""
Result serialization utilities.

Handles saving/loading results with proper type conversion for:
- PyTorch tensors
- Numpy arrays
- Dataclasses
- Enums
"""

import json
from pathlib import Path
from typing import Dict, Any, Union
from dataclasses import asdict, is_dataclass

import torch


def make_serializable(obj: Any) -> Any:
    """
    Convert object to JSON-serializable format.
    
    Handles:
    - torch.Tensor -> list
    - numpy arrays -> list
    - dataclasses -> dict
    - sets -> list
    - Enums -> value
    """
    if isinstance(obj, torch.Tensor):
        return obj.tolist()
    
    if hasattr(obj, 'numpy'):  # numpy array
        return obj.tolist()
    
    if is_dataclass(obj) and not isinstance(obj, type):
        return asdict(obj)
    
    if isinstance(obj, (set, frozenset)):
        return list(obj)
    
    if hasattr(obj, 'value'):  # Enum
        return obj.value
    
    if hasattr(obj, '__dict__'):
        return str(obj)
    
    return obj


def save_results(
    results: Dict[str, Any],
    path: Union[str, Path],
    indent: int = 2,
) -> None:
    """
    Save results dictionary to JSON file.
    
    Args:
        results: Dictionary of results
        path: Output file path
        indent: JSON indentation
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(path, "w") as f:
        json.dump(results, f, indent=indent, default=make_serializable)


def load_results(path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load results dictionary from JSON file.
    
    Args:
        path: Input file path
        
    Returns:
        Results dictionary
    """
    with open(path) as f:
        return json.load(f)


def save_comparison(
    results_list: list,
    experiment_names: list,
    path: Union[str, Path],
) -> None:
    """
    Save comparison of multiple experiments.
    
    Args:
        results_list: List of result dictionaries
        experiment_names: Names corresponding to each result
        path: Output file path
    """
    comparison = {
        "experiments": []
    }
    
    for name, results in zip(experiment_names, results_list):
        entry = {"name": name}
        
        # Extract key metrics
        if "coqa_metrics" in results:
            entry.update(results["coqa_metrics"])
        
        if "benchmarks" in results:
            bench = results["benchmarks"]
            if "model_size_mb" in bench:
                entry["model_size_mb"] = bench["model_size_mb"]
            if "memory_peak_mb" in bench:
                entry["memory_peak_mb"] = bench["memory_peak_mb"]
        
        comparison["experiments"].append(entry)
    
    save_results(comparison, path)



