"""
Main CLI entry point for Llama 3.2-1B quantization experiments.

Usage:
    # Run single experiment
    python -m cli.main --experiment bnb_4bit_nf4
    
    # Run all experiments
    python -m cli.main --run-all
    
    # Run from YAML config
    python -m cli.main --config experiments/bnb_4bit_nf4.yaml
    
    # Quick test with limit
    python -m cli.main --experiment bnb_4bit_nf4 --limit 50
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime

import torch

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from llama_quant.core.config import (
    ExperimentConfig,
    get_experiment,
    get_all_experiments,
)
from llama_quant.models import load_model, get_model_size_mb
from llama_quant.evaluation import evaluate_model
from llama_quant.benchmark import run_benchmarks
from llama_quant.benchmark.runner import print_benchmark_summary
from llama_quant.utils.logging import setup_logging, setup_experiment_logging
from llama_quant.utils.serialization import save_results, save_comparison


def run_experiment(config: ExperimentConfig) -> dict:
    """
    Run a single quantization experiment.
    
    Steps:
        1. Load and quantize model
        2. Run accuracy evaluation (CoQA)
        3. Run hardware benchmarks
        4. Save results
    """
    logger = setup_experiment_logging(config.name, config.output_dir, config.log_level)
    
    logger.info("=" * 60)
    logger.info(f"Starting experiment: {config.name}")
    logger.info(f"Quantization method: {config.quantization.method.value}")
    logger.info("=" * 60)
    
    results = {
        "experiment_name": config.name,
        "quantization_method": config.quantization.method.value,
        "timestamp": datetime.now().isoformat(),
    }
    
    # Step 1: Load model
    logger.info("Step 1: Loading and quantizing model...")
    try:
        model, tokenizer = load_model(config)
        results["model_loaded"] = True
        results["model_size_mb"] = get_model_size_mb(model)
        logger.info(f"Model size: {results['model_size_mb']:.2f} MB")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        results["model_loaded"] = False
        results["error"] = str(e)
        return results
    
    # Step 2: Run evaluation
    if not config.skip_eval:
        logger.info("Step 2: Running accuracy evaluation...")
        try:
            eval_results = evaluate_model(model, tokenizer, config)
            results["evaluation"] = eval_results
            
            if "coqa_metrics" in eval_results:
                logger.info(f"CoQA Results: {eval_results['coqa_metrics']}")
        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            results["evaluation_error"] = str(e)
    else:
        logger.info("Step 2: Skipping evaluation (--skip-eval)")
    
    # Step 3: Run benchmarks
    if not config.skip_benchmark:
        logger.info("Step 3: Running hardware benchmarks...")
        try:
            benchmark_results = run_benchmarks(model, tokenizer, config)
            results["benchmarks"] = benchmark_results
            print_benchmark_summary(benchmark_results)
        except Exception as e:
            logger.error(f"Benchmarking failed: {e}")
            results["benchmark_error"] = str(e)
    else:
        logger.info("Step 3: Skipping benchmarks (--skip-benchmark)")
    
    # Step 4: Save results
    logger.info("Step 4: Saving results...")
    output_path = Path(config.output_dir) / f"{config.name}_results.json"
    save_results(results, output_path)
    logger.info(f"Results saved to {output_path}")
    
    # Cleanup
    del model
    del tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return results


def run_all_experiments(
    experiments: list = None,
    limit: int = None,
    output_dir: str = "./results",
) -> list:
    """Run multiple experiments and compare results."""
    logger = setup_logging("INFO")
    
    if experiments is None:
        experiments = list(get_all_experiments().keys())
    
    all_results = []
    
    for exp_name in experiments:
        try:
            config = get_experiment(exp_name)
            config.output_dir = output_dir
            
            if limit is not None:
                config.eval.limit = limit
            
            logger.info(f"\n{'='*60}")
            logger.info(f"Running experiment: {exp_name}")
            logger.info(f"{'='*60}\n")
            
            results = run_experiment(config)
            all_results.append(results)
            
        except Exception as e:
            logger.error(f"Experiment {exp_name} failed: {e}")
            all_results.append({
                "experiment_name": exp_name,
                "error": str(e),
            })
    
    # Save comparison
    comparison_path = Path(output_dir) / f"comparison_{datetime.now():%Y%m%d_%H%M%S}.json"
    save_comparison(
        all_results,
        [r.get("experiment_name", "unknown") for r in all_results],
        comparison_path,
    )
    logger.info(f"Comparison saved to {comparison_path}")
    
    return all_results


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Llama 3.2-1B Quantization Experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run single experiment
    python -m cli.main --experiment bnb_4bit_nf4
    
    # Run all experiments
    python -m cli.main --run-all
    
    # Quick test with limit
    python -m cli.main --experiment bnb_4bit_nf4 --limit 50
    
    # From YAML config
    python -m cli.main --config experiments/my_config.yaml
        """,
    )
    
    parser.add_argument(
        "--experiment", "-e",
        type=str,
        choices=list(get_all_experiments().keys()),
        help="Name of predefined experiment to run",
    )
    
    parser.add_argument(
        "--config", "-c",
        type=str,
        help="Path to YAML config file",
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
        "--output-dir", "-o",
        type=str,
        default="./results",
        help="Output directory for results",
    )
    
    parser.add_argument(
        "--skip-eval",
        action="store_true",
        help="Skip accuracy evaluation",
    )
    
    parser.add_argument(
        "--skip-benchmark",
        action="store_true",
        help="Skip hardware benchmarks",
    )
    
    parser.add_argument(
        "--list-experiments",
        action="store_true",
        help="List available experiments and exit",
    )
    
    args = parser.parse_args()
    
    # List experiments
    if args.list_experiments:
        print("\nAvailable experiments:")
        for name, config in get_all_experiments().items():
            print(f"  {name}: {config.description or 'No description'}")
        return
    
    # Setup logging
    logger = setup_logging("INFO")
    
    # Log system info
    if torch.cuda.is_available():
        logger.info(f"CUDA available: {torch.cuda.get_device_name()}")
        logger.info(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        logger.warning("CUDA not available, running on CPU")
    
    # Run experiments
    if args.run_all:
        run_all_experiments(limit=args.limit, output_dir=args.output_dir)
    
    elif args.experiments:
        run_all_experiments(
            experiments=args.experiments,
            limit=args.limit,
            output_dir=args.output_dir,
        )
    
    elif args.config:
        config = ExperimentConfig.from_yaml(args.config)
        config.output_dir = args.output_dir
        config.skip_eval = args.skip_eval
        config.skip_benchmark = args.skip_benchmark
        if args.limit:
            config.eval.limit = args.limit
        run_experiment(config)
    
    elif args.experiment:
        config = get_experiment(args.experiment)
        config.output_dir = args.output_dir
        config.skip_eval = args.skip_eval
        config.skip_benchmark = args.skip_benchmark
        if args.limit:
            config.eval.limit = args.limit
        run_experiment(config)
    
    else:
        # Default: quick test
        logger.info("No experiment specified, running quick BnB 4-bit test...")
        config = get_experiment("bnb_4bit_nf4")
        config.eval.limit = 50
        config.output_dir = args.output_dir
        run_experiment(config)


if __name__ == "__main__":
    main()
