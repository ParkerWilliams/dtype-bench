"""
Document Classification Experiment Runner

TODO: Implement training loop for EUR-Lex multi-label classification.
See PLAN.md for specifications.
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from configs import get_config, get_all_configs, list_configs


def main():
    parser = argparse.ArgumentParser(description='Classification dtype benchmarking')
    parser.add_argument('--config', type=str, default='amp_bf16',
                        help='Config name or "all"')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--seeds', type=int, default=1,
                        help='Number of seeds (overrides --seed)')
    parser.add_argument('--max_steps', type=int, default=5000)
    parser.add_argument('--max_epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--data_dir', type=str, default='data/eurlex')
    parser.add_argument('--output_dir', type=str, default='results/classification')
    parser.add_argument('--list', action='store_true')
    
    args = parser.parse_args()
    
    if args.list:
        print("Available configs:")
        for name in list_configs(include_classification_specific=True):
            print(f"  {name}")
        return
    
    # TODO: Implement
    # 1. Load dtype config (include classification-specific)
    # 2. Load RoBERTa with appropriate dtypes
    # 3. Add classification head (with head_dtype if specified)
    # 4. Setup optimizer (8-bit if configured)
    # 5. Setup AMP context if configured
    # 6. Training loop with metrics collection
    # 7. Save results
    
    raise NotImplementedError(
        "Classification experiment not yet implemented. "
        "See PLAN.md and implement model.py, data.py, metrics.py, then this file."
    )


if __name__ == '__main__':
    main()
