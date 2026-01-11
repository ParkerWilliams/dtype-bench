"""
OpenWebText Data Preparation

TODO: Download and tokenize OpenWebText dataset.

This script should:
1. Download OpenWebText from HuggingFace
2. Tokenize with GPT-2 tokenizer
3. Save as memory-mapped numpy arrays

Reference: https://github.com/karpathy/nanoGPT/blob/master/data/openwebtext/prepare.py
"""

import os
import argparse
import numpy as np
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, default='data/openwebtext')
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # TODO: Implement
    # 1. from datasets import load_dataset
    # 2. Load 'openwebtext'
    # 3. Tokenize with tiktoken (gpt2 encoding)
    # 4. Split into train/val
    # 5. Save as .bin files (np.memmap with uint16)
    
    raise NotImplementedError(
        "Data preparation not yet implemented. "
        "See nanoGPT's data/openwebtext/prepare.py for reference."
    )


if __name__ == '__main__':
    main()
