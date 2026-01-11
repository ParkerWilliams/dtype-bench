"""
Cross-Experiment Comparison

TODO: Compare dtype behavior between LM and Classification.
See PLAN.md for specifications.

This is the core analysis that answers:
"Which dtype relationships flip between tasks?"
"""

import argparse
import json
from pathlib import Path
from typing import List, Dict, Tuple

# import pandas as pd
# import matplotlib.pyplot as plt
# from scipy.stats import spearmanr


def load_experiment_results(results_dir: str) -> List[Dict]:
    """Load results from an experiment directory."""
    # TODO: Implement
    raise NotImplementedError


def compute_rankings(results: List[Dict], metric: str) -> Dict[str, int]:
    """
    Compute rankings for each config on a metric.
    
    Returns dict mapping config name to rank (1 = best).
    """
    # TODO: Implement
    raise NotImplementedError


def find_ranking_flips(
    lm_results: List[Dict],
    cls_results: List[Dict],
    metric: str,
) -> List[Dict]:
    """
    Find configs that rank very differently between tasks.
    
    Returns list of dicts with:
    - config: Config name
    - lm_rank: Rank in LM experiment
    - cls_rank: Rank in Classification experiment
    - rank_change: Absolute difference
    """
    # TODO: Implement
    raise NotImplementedError


def compute_ranking_correlation(
    lm_results: List[Dict],
    cls_results: List[Dict],
    metric: str,
) -> float:
    """
    Compute Spearman correlation of config rankings between tasks.
    
    High correlation = configs rank similarly
    Low correlation = rankings differ significantly
    """
    # TODO: Implement
    raise NotImplementedError


def test_hypotheses(
    lm_results: List[Dict],
    cls_results: List[Dict],
) -> Dict[str, Dict]:
    """
    Test the hypotheses from PLAN.md.
    
    Returns dict with hypothesis name -> result dict.
    """
    # TODO: Implement
    # H1: Output layer precision
    # H2: Stability differences (fp16_naive)
    # H3: 8-bit Adam impact
    # H4: Memory-throughput tradeoff shifts
    raise NotImplementedError


def generate_comparison_plots(
    lm_results: List[Dict],
    cls_results: List[Dict],
    output_dir: str,
):
    """Generate cross-experiment visualizations."""
    # TODO: Implement
    # - Ranking comparison (slope graph)
    # - Relative performance heatmap
    # - Side-by-side Pareto frontiers
    raise NotImplementedError


def generate_report(
    lm_results: List[Dict],
    cls_results: List[Dict],
) -> str:
    """
    Generate text report of findings.
    
    Returns markdown-formatted string.
    """
    # TODO: Implement
    raise NotImplementedError


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lm_results', type=str, required=True)
    parser.add_argument('--cls_results', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='results/comparison')
    
    args = parser.parse_args()
    
    # TODO: Implement
    # 1. Load both result sets
    # 2. Find ranking flips
    # 3. Test hypotheses
    # 4. Generate plots
    # 5. Generate report
    # 6. Save outputs
    
    raise NotImplementedError


if __name__ == '__main__':
    main()
