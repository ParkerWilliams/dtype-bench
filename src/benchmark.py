"""
Dtype Benchmark - Main Entry Point

Usage:
    python -m src.benchmark \
        --model_arch [gpt2, roberta] \
        --task [clm, mlm, multi_cls, single_cls] \
        --dtype_config [fp32_baseline, amp_bf16, ...] \
        --batch_size 32 \
        --max_steps 1000

Scenarios (from BENCHMARK_SPEC.md):
    A: GPT-2 + Causal LM + CrossEntropy
    B: RoBERTa + Masked LM + CrossEntropy
    C: GPT-2 + Multi-Label Cls + BCEWithLogits
    D: RoBERTa + Multi-Label Cls + BCEWithLogits
    E: GPT-2 + Single-Label Cls + CrossEntropy
    F: RoBERTa + Single-Label Cls + CrossEntropy
"""

import argparse
import sys

from .configs import get_config, get_all_configs, list_configs
from .utils import check_hardware, setup_logging


def main():
    parser = argparse.ArgumentParser(description='Dtype Benchmark')
    parser.add_argument('--model_arch', type=str, required=True,
                        choices=['gpt2', 'roberta'],
                        help='Model architecture')
    parser.add_argument('--task', type=str, required=True,
                        choices=['clm', 'mlm', 'multi_cls', 'single_cls'],
                        help='Task type')
    parser.add_argument('--dtype_config', type=str, default='amp_bf16',
                        help='Dtype config name or "all"')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--seeds', type=int, default=1,
                        help='Number of seeds (overrides --seed)')
    parser.add_argument('--max_steps', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--data_dir', type=str, default='data')
    parser.add_argument('--output_dir', type=str, default='results')
    parser.add_argument('--list', action='store_true',
                        help='List available dtype configs')

    args = parser.parse_args()

    if args.list:
        print("Available dtype configs:")
        for name in list_configs(include_classification_specific=True):
            print(f"  {name}")
        return

    # Check hardware
    check_hardware()

    # Setup logging
    setup_logging(args.output_dir)

    # Validate task/model combinations
    valid_combinations = {
        ('gpt2', 'clm'),       # Scenario A
        ('roberta', 'mlm'),    # Scenario B
        ('gpt2', 'multi_cls'), # Scenario C
        ('roberta', 'multi_cls'), # Scenario D
        ('gpt2', 'single_cls'),   # Scenario E
        ('roberta', 'single_cls'), # Scenario F
    }

    if (args.model_arch, args.task) not in valid_combinations:
        print(f"Warning: Unusual combination {args.model_arch} + {args.task}")

    # TODO: Implement
    # 1. Load dtype config
    # 2. Initialize model with appropriate dtypes
    # 3. Load data
    # 4. Setup optimizer (8-bit if configured)
    # 5. Setup AMP context if configured
    # 6. Training loop with metrics collection
    # 7. Save results

    raise NotImplementedError(
        "Benchmark not yet implemented. "
        "See BENCHMARK_SPEC.md and implement models.py, data.py, then this file."
    )


if __name__ == '__main__':
    main()
