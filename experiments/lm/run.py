"""
Language Modeling Experiment Runner

TODO: Implement training loop with dtype configuration support.
See PLAN.md for specifications.
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from configs import get_config, get_all_configs, list_configs


def main():
    parser = argparse.ArgumentParser(description='LM dtype benchmarking')
    parser.add_argument('--config', type=str, default='amp_bf16',
                        help='Config name or "all"')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--seeds', type=int, default=1,
                        help='Number of seeds (overrides --seed)')
    parser.add_argument('--max_iters', type=int, default=10000)
    parser.add_argument('--batch_size', type=int, default=12)
    parser.add_argument('--data_dir', type=str, default='data/openwebtext')
    parser.add_argument('--output_dir', type=str, default='results/lm')
    parser.add_argument('--list', action='store_true')
    
    args = parser.parse_args()
    
    if args.list:
        print("Available configs:")
        for name in list_configs():
            print(f"  {name}")
        return
    
    # TODO: Implement
    # 1. Load dtype config
    # 2. Initialize model with appropriate dtypes
    # 3. Setup optimizer (8-bit if configured)
    # 4. Setup AMP context if configured
    # 5. Training loop with metrics collection
    # 6. Save results
    
    raise NotImplementedError(
        "LM experiment not yet implemented. "
        "See PLAN.md and implement model.py, data.py, then this file."
    )


if __name__ == '__main__':
    main()
