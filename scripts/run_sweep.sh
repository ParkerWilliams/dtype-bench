#!/bin/bash
# Run sweep over all 6 scenarios
#
# Scenarios:
#   A: GPT-2 + Causal LM
#   B: RoBERTa + Masked LM
#   C: GPT-2 + Multi-Label Classification
#   D: RoBERTa + Multi-Label Classification
#   E: GPT-2 + Single-Label Classification
#   F: RoBERTa + Single-Label Classification
#
# Usage:
#   ./scripts/run_sweep.sh              # Full run
#   ./scripts/run_sweep.sh --quick      # Quick test (1 seed, fewer steps)

set -e

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

# Activate virtual environment if it exists
if [ -d ".venv" ]; then
    echo "Activating virtual environment..."
    source .venv/bin/activate
fi

# Parse args
QUICK=false
for arg in "$@"; do
    case $arg in
        --quick) QUICK=true ;;
    esac
done

# Configuration
if [ "$QUICK" = true ]; then
    SEEDS=1
    MAX_STEPS=100
    BATCH_SIZE=8
    SUBSET_SIZE=10000
    echo "Quick mode: 1 seed, 100 steps, batch_size=8, subset=10K"
else
    SEEDS=3
    MAX_STEPS=1000
    BATCH_SIZE=32
    SUBSET_SIZE=100000
fi

# Dtype configs to test (matches BENCHMARK_SPEC.md)
DTYPE_CONFIGS="fp32_true amp_fp16 amp_bf16 bf16_pure fp16_naive amp_bf16_8bit_adam"

echo "=========================================="
echo "Dtype Benchmark - Full Sweep"
echo "=========================================="
echo "Seeds: $SEEDS"
echo "Max steps: $MAX_STEPS"
echo "Dtype configs: $DTYPE_CONFIGS"
echo ""

# Define scenarios as (model_arch, task, scenario_id) tuples
declare -a SCENARIOS=(
    "gpt2 clm A"
    "roberta mlm B"
    "gpt2 multi_cls C"
    "roberta multi_cls D"
    "gpt2 single_cls E"
    "roberta single_cls F"
)

# Run each scenario
for scenario in "${SCENARIOS[@]}"; do
    read -r model_arch task scenario_id <<< "$scenario"

    echo ""
    echo "=========================================="
    echo "Scenario $scenario_id: $model_arch + $task"
    echo "=========================================="

    for dtype_config in $DTYPE_CONFIGS; do
        echo ""
        echo "--- Config: $dtype_config ---"

        python -m src.benchmark \
            --model_arch "$model_arch" \
            --task "$task" \
            --dtype_config "$dtype_config" \
            --seeds "$SEEDS" \
            --max_steps "$MAX_STEPS" \
            --batch_size "$BATCH_SIZE" \
            --subset_size "$SUBSET_SIZE" \
            --output_dir "results/scenario_${scenario_id}"
    done
done

echo ""
echo "=========================================="
echo "Sweep Complete!"
echo "=========================================="
echo ""
echo "Results saved to results/"
