#!/bin/bash
# Run all experiments
#
# Usage: 
#   ./scripts/run_all.sh              # Full run
#   ./scripts/run_all.sh --quick      # Quick test (1 seed, fewer iters)
#   ./scripts/run_all.sh --resume     # Skip completed configs

set -e

# Parse args
QUICK=false
RESUME=false
for arg in "$@"; do
    case $arg in
        --quick) QUICK=true ;;
        --resume) RESUME=true ;;
    esac
done

# Configuration
if [ "$QUICK" = true ]; then
    SEEDS=1
    LM_ITERS=1000
    CLS_STEPS=500
    echo "Quick mode: 1 seed, reduced iterations"
else
    SEEDS=3
    LM_ITERS=10000
    CLS_STEPS=5000
fi

echo "=========================================="
echo "Dtype Benchmarking - Full Comparison"
echo "=========================================="
echo "Seeds: $SEEDS"
echo "LM iterations: $LM_ITERS"
echo "Classification steps: $CLS_STEPS"
echo ""

# Prepare data if needed
if [ ! -f "data/openwebtext/train.bin" ]; then
    echo "Preparing OpenWebText data..."
    python experiments/lm/prepare_data.py
fi

if [ ! -f "data/eurlex/train.jsonl" ]; then
    echo "Preparing EUR-Lex data..."
    python experiments/classification/prepare_data.py
fi

# Run LM experiments
echo ""
echo "=========================================="
echo "Phase 1: Language Modeling"
echo "=========================================="

python experiments/lm/run.py \
    --config all \
    --seeds $SEEDS \
    --max_iters $LM_ITERS \
    --output_dir results/lm

# Analyze LM results
echo ""
echo "Analyzing LM results..."
python analysis/analyze_experiment.py --results_dir results/lm

# Run Classification experiments
echo ""
echo "=========================================="
echo "Phase 2: Document Classification"
echo "=========================================="

python experiments/classification/run.py \
    --config all \
    --seeds $SEEDS \
    --max_steps $CLS_STEPS \
    --output_dir results/classification

# Analyze Classification results
echo ""
echo "Analyzing Classification results..."
python analysis/analyze_experiment.py --results_dir results/classification

# Cross-experiment comparison
echo ""
echo "=========================================="
echo "Phase 3: Cross-Experiment Comparison"
echo "=========================================="

python analysis/compare_experiments.py \
    --lm_results results/lm \
    --cls_results results/classification \
    --output_dir results/comparison

echo ""
echo "=========================================="
echo "Complete!"
echo "=========================================="
echo ""
echo "Results:"
echo "  LM: results/lm/"
echo "  Classification: results/classification/"
echo "  Comparison: results/comparison/"
