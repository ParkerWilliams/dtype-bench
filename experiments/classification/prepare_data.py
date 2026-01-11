"""
EUR-Lex Data Preparation

TODO: Download and prepare EUR-Lex dataset.
See PLAN.md for specifications.

Data sources:
1. HuggingFace: coastalcph/lex_glue (eurlex subset)
2. Zenodo: EURLEX57K direct download

Output format:
- {split}.jsonl: One JSON object per line with 'text' and 'labels'
- label_map.json: Mapping from label name to index
- metadata.json: Dataset statistics
"""

import os
import json
import argparse


def download_from_huggingface(output_dir: str) -> bool:
    """
    Download EUR-Lex from HuggingFace datasets.
    
    Returns True on success.
    """
    # TODO: Implement
    # 1. from datasets import load_dataset
    # 2. dataset = load_dataset("coastalcph/lex_glue", "eurlex")
    # 3. Build label vocabulary
    # 4. Convert to JSONL format
    # 5. Save metadata
    
    raise NotImplementedError


def print_statistics(data_dir: str):
    """Print dataset statistics."""
    # TODO: Implement
    # - Number of documents per split
    # - Number of unique labels
    # - Average labels per document
    # - Average document length
    
    raise NotImplementedError


def create_subset(data_dir: str, subset_size: int = 5000):
    """Create smaller subset for quick testing."""
    # TODO: Implement
    raise NotImplementedError


def main():
    parser = argparse.ArgumentParser(description='Prepare EUR-Lex dataset')
    parser.add_argument('--output_dir', type=str, default='data/eurlex')
    parser.add_argument('--create_subset', action='store_true')
    parser.add_argument('--subset_size', type=int, default=5000)
    parser.add_argument('--stats_only', action='store_true')
    
    args = parser.parse_args()
    
    if args.stats_only:
        print_statistics(args.output_dir)
        return
    
    # TODO: Implement download and preparation
    raise NotImplementedError(
        "EUR-Lex preparation not yet implemented. "
        "See coastalcph/lex_glue on HuggingFace for data source."
    )


if __name__ == '__main__':
    main()
