# Dtype Performance Benchmarking

Systematic comparison of data type configurations across ML tasks.

## Project Goal

Understand how dtype choices (fp32, fp16, bf16, mixed precision, 8-bit optimizers) affect:
1. Training throughput
2. Memory usage
3. Model quality

And critically: **which relationships flip between different tasks**.

## Structure

```
dtype_bench/
├── experiments/
│   ├── lm/              # Language modeling (GPT-2 on OpenWebText)
│   └── classification/  # Document classification (RoBERTa on EUR-Lex)
├── configs/             # Shared dtype configurations
├── analysis/            # Cross-experiment comparison
├── scripts/             # Cluster deployment and orchestration
└── docs/                # Planning and design documents
```

## Quick Start

```bash
# 1. Review the plan
cat docs/PLAN.md

# 2. Setup environment
./scripts/setup.sh

# 3. Run a single experiment
python experiments/lm/run.py --config amp_bf16 --max_iters 1000

# 4. Run full comparison
./scripts/run_all.sh
```

## Current Experiments

| Experiment | Model | Dataset | Status |
|------------|-------|---------|--------|
| `lm` | GPT-2 124M | OpenWebText | Planned |
| `classification` | RoBERTa-base | EUR-Lex (4.3k labels) | Planned |

## Estimated Compute

~90-100 A100 GPU hours (~$100 on Lambda Labs)
