#!/bin/bash
# =============================================================================
# Thin wrapper for docker-compose
# Everything runs inside Docker. No local Python needed.
# =============================================================================

set -e

case "$1" in
    build)
        docker-compose build
        ;;
    quick)
        docker-compose run --rm quick
        ;;
    full)
        docker-compose run --rm full-run
        ;;
    baseline)
        docker-compose run --rm baseline
        ;;
    figures)
        docker-compose run --rm figures
        ;;
    shell)
        docker-compose run --rm quant
        ;;
    *)
        echo "Usage: ./run.sh {build|quick|full|baseline|figures|shell}"
        echo ""
        echo "  build    - Build Docker image"
        echo "  quick    - Run FP16/8bit/4bit comparison (minimum viable)"
        echo "  full     - Run all experiments + generate figures"
        echo "  baseline - Run FP16 baseline only"
        echo "  figures  - Generate figures from results"
        echo "  shell    - Interactive shell inside container"
        echo ""
        echo "Or use docker-compose directly:"
        echo "  docker-compose run --rm quant python main.py --experiment bnb_4bit_nf4 --limit 100"
        ;;
esac
