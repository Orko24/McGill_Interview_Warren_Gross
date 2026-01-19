YOU TYPE:
modal run code/infra/modal_app.py --limit 50
    │
    ▼
STEP 1: @app.local_entrypoint() main()    [YOUR MACHINE]
    │   - Parses CLI args
    │   - Decides what to run
    │
    └── results = run_comparison.remote(limit=50)
            │
            ▼
STEP 2: Modal sends job to cloud          [NETWORK]
    │   - Spins up A100 GPU container
    │   - Installs packages
    │   - Mounts your code
    │
    ▼
STEP 3: run_comparison()                  [MODAL CLOUD - GPU]
    │   - Creates ComparisonRunner
    │
    └── runner.run(experiments, limit)
            │
            ▼
STEP 4: ComparisonRunner.run()
    │   - experiments = ["fp16_baseline", "bnb_4bit_nf4", "bnb_4bit_fp4"]
    │
    └── FOR EACH experiment:
            │
            ▼
STEP 5: ExperimentRunner.run(request)
            │
            ├── load_model(config)
            │       │
            │       └── _get_loader(method)
            │               │
            │               ├── FP16Loader.load()
            │               │       └── from_pretrained(dtype=fp16)
            │               │
            │               └── BitsAndBytes4BitLoader.load()
            │                       └── from_pretrained(quantization_config=bnb_config)
            │
            ├── evaluate_model()
            │       └── lm_eval.simple_evaluate(tasks=["coqa"])
            │
            └── benchmark_suite.run()
                    └── Measures latency, throughput
            │
            ▼
STEP 6: Results collected
    │   - F1 scores, model sizes, latencies
    │
    ▼
STEP 7: Returns to YOUR MACHINE           [NETWORK]
    │
    ▼
STEP 8: ResultsManager.save()             [YOUR MACHINE]
    │   - Saves to results/results4.json
    │
    ▼
DONE