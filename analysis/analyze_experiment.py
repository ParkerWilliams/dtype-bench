"""
Single Experiment Analysis

TODO: Implement analysis for one experiment's results.
See PLAN.md for specifications.
"""

import argparse
import json
from pathlib import Path
from typing import List, Dict

# import pandas as pd
# import matplotlib.pyplot as plt


def load_results(results_dir: str) -> List[Dict]:
    """Load all result JSON files from directory."""
    # TODO: Implement
    raise NotImplementedError


def compute_summary(results: List[Dict]) -> Dict:
    """
    Compute summary statistics.
    
    Returns dict with per-config mean/std for each metric.
    """
    # TODO: Implement
    raise NotImplementedError


def find_pareto_frontier(results: List[Dict], x_metric: str, y_metric: str) -> List[str]:
    """
    Find Pareto-optimal configs.
    
    Returns list of config names on the frontier.
    """
    # TODO: Implement
    raise NotImplementedError


def generate_plots(results: List[Dict], output_dir: str):
    """Generate all visualizations."""
    # TODO: Implement
    # - Throughput bar chart
    # - Memory bar chart
    # - Throughput vs quality scatter
    # - Loss curves
    # - Pareto frontier
    raise NotImplementedError


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default=None)
    
    args = parser.parse_args()
    
    if args.output_dir is None:
        args.output_dir = Path(args.results_dir) / 'analysis'
    
    # TODO: Implement
    raise NotImplementedError


if __name__ == '__main__':
    main()
