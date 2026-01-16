"""
Main entry point for Llama 3.2-1B quantization experiments

Usage:
    # Run single experiment
    python main.py --experiment fp16_baseline
    
    # Run all experiments
    python main.py --run-all
    
    # Run with custom config
    python main.py --experiment bnb_4bit_nf4 --limit 100
"""

import argparse
import logging
import json
import sys
from pathlib import Path
from datetime import datetime

import torch

from config import ExperimentConfig, EXPERIMENTS, QuantMethod
from quantize import load_quantized_model, get_model_memory_footprint
from evaluate import evaluate_model, save_eval_results, compare_results, print_comparison_table
from benchmark import run_benchmarks, print_benchmark_summary

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)


def setup_experiment_logging(config: ExperimentConfig):
    """Setup file logging for experiment"""
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    log_file = output_dir / f"{config.name}_{datetime.now():%Y%m%d_%H%M%S}.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    )
    logging.getLogger().addHandler(file_handler)


def run_experiment(config: ExperimentConfig) -> dict:
    """
    Run a single quantization experiment
    
    Steps:
        1. Load and quantize model
        2. Run accuracy evaluation (CoQA)
        3. Run hardware benchmarks
        4. Save results
    
    Returns:
        Dictionary containing all results
    """
    logger.info(f"=" * 60)
    logger.info(f"Starting experiment: {config.name}")
    logger.info(f"Quantization method: {config.quantization.method.value}")
    logger.info(f"=" * 60)
    
    results = {
        "experiment_name": config.name,
        "quantization_method": config.quantization.method.value,
        "timestamp": datetime.now().isoformat(),
    }
    
    # Step 1: Load model
    logger.info("Step 1: Loading and quantizing model...")
    try:
        model, tokenizer = load_quantized_model(config)
        results["model_loaded"] = True
        
        # Log memory footprint
        memory_footprint = get_model_memory_footprint(model)
        results["memory_footprint"] = memory_footprint
        logger.info(f"Model memory footprint: {memory_footprint}")
        
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        results["model_loaded"] = False
        results["error"] = str(e)
        return results
    
    # Step 2: Run evaluation
    logger.info("Step 2: Running accuracy evaluation...")
    try:
        eval_results = evaluate_model(
            model=model,
            tokenizer=tokenizer,
            config=config,
        )
        results["evaluation"] = eval_results
        
        if "coqa_metrics" in eval_results:
            logger.info(f"CoQA Results: {eval_results['coqa_metrics']}")
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        results["evaluation_error"] = str(e)
    
    # Step 3: Run benchmarks
    logger.info("Step 3: Running hardware benchmarks...")
    try:
        benchmark_results = run_benchmarks(
            model=model,
            tokenizer=tokenizer,
            config=config,
        )
        results["benchmarks"] = benchmark_results
        print_benchmark_summary(benchmark_results)
        
    except Exception as e:
        logger.error(f"Benchmarking failed: {e}")
        results["benchmark_error"] = str(e)
    
    # Step 4: Save results
    logger.info("Step 4: Saving results...")
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results_file = output_dir / f"{config.name}_results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    logger.info(f"Results saved to {results_file}")
    
    # Cleanup
    del model
    del tokenizer
    torch.cuda.empty_cache()
    
    return results


def run_all_experiments(
    experiments: list[str] = None,
    limit: int = None,
) -> list[dict]:
    """
    Run multiple experiments and compare results
    """
    if experiments is None:
        experiments = list(EXPERIMENTS.keys())
    
    all_results = []
    
    for exp_name in experiments:
        if exp_name not in EXPERIMENTS:
            logger.warning(f"Unknown experiment: {exp_name}, skipping...")
            continue
        
        config = EXPERIMENTS[exp_name]
        
        # Apply limit override if specified
        if limit is not None:
            config.eval.limit = limit
        
        try:
            results = run_experiment(config)
            all_results.append(results)
        except Exception as e:
            logger.error(f"Experiment {exp_name} failed: {e}")
            all_results.append({
                "experiment_name": exp_name,
                "error": str(e),
            })
    
    # Compare and summarize
    logger.info("\n" + "=" * 60)
    logger.info("EXPERIMENT COMPARISON")
    logger.info("=" * 60)
    
    comparison = compare_results(
        all_results,
        [r.get("experiment_name", "unknown") for r in all_results],
    )
    print_comparison_table(comparison)
    
    # Save comparison
    output_dir = Path("./results")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    comparison_file = output_dir / f"comparison_{datetime.now():%Y%m%d_%H%M%S}.json"
    with open(comparison_file, "w") as f:
        json.dump(comparison, f, indent=2, default=str)
    
    return all_results


def main():
    parser = argparse.ArgumentParser(
        description="Llama 3.2-1B Quantization Experiments"
    )
    
    parser.add_argument(
        "--experiment",
        type=str,
        choices=list(EXPERIMENTS.keys()),
        help="Name of predefined experiment to run",
    )
    
    parser.add_argument(
        "--run-all",
        action="store_true",
        help="Run all predefined experiments",
    )
    
    parser.add_argument(
        "--experiments",
        type=str,
        nargs="+",
        help="List of experiments to run",
    )
    
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of eval samples (for debugging)",
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./results",
        help="Output directory for results",
    )
    
    parser.add_argument(
        "--skip-eval",
        action="store_true",
        help="Skip accuracy evaluation (only run benchmarks)",
    )
    
    parser.add_argument(
        "--skip-benchmark",
        action="store_true",
        help="Skip hardware benchmarks (only run evaluation)",
    )
    
    args = parser.parse_args()
    
    # Check CUDA availability
    if torch.cuda.is_available():
        logger.info(f"CUDA available: {torch.cuda.get_device_name()}")
        logger.info(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        logger.warning("CUDA not available, running on CPU (will be slow)")
    
    # Run experiments
    if args.run_all:
        run_all_experiments(limit=args.limit)
    
    elif args.experiments:
        run_all_experiments(experiments=args.experiments, limit=args.limit)
    
    elif args.experiment:
        config = EXPERIMENTS[args.experiment]
        config.output_dir = args.output_dir
        
        if args.limit:
            config.eval.limit = args.limit
        
        run_experiment(config)
    
    else:
        # Default: run a quick test with BnB 4-bit
        logger.info("No experiment specified, running quick BnB 4-bit test...")
        config = EXPERIMENTS["bnb_4bit_nf4"]
        config.eval.limit = 50  # Quick test
        run_experiment(config)


if __name__ == "__main__":
    main()
