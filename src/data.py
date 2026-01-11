"""
Data Loading for Dtype Benchmark

Contains both data loaders:
- OpenWebText: Memory-mapped dataset for language modeling
- EUR-Lex: Multi-label document classification dataset

TODO: Implement data loading from pre-tokenized binary files.
"""

import os
import json
from typing import List, Dict
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


# =============================================================================
# OpenWebText Dataset (Language Modeling)
# =============================================================================

class OpenWebTextDataset:
    """
    Memory-mapped dataset for pre-tokenized OpenWebText.

    Usage:
        dataset = OpenWebTextDataset('data/openwebtext', 'train', block_size=1024)
        x, y = dataset.get_batch(batch_size=12)
    """

    def __init__(self, data_dir: str, split: str, block_size: int = 1024):
        """
        Args:
            data_dir: Directory containing train.bin and val.bin
            split: 'train' or 'val'
            block_size: Sequence length
        """
        self.block_size = block_size

        # TODO: Memory-map the data file
        # self.data = np.memmap(...)
        raise NotImplementedError

    def get_batch(self, batch_size: int, device: str = 'cuda'):
        """
        Get a random batch.

        Args:
            batch_size: Number of sequences
            device: Target device

        Returns:
            x: Input tokens [batch, block_size]
            y: Target tokens [batch, block_size] (shifted by 1)
        """
        # TODO: Sample random starting positions
        # TODO: Extract sequences
        # TODO: Return x (input) and y (target, shifted by 1)
        raise NotImplementedError

    def __len__(self):
        """Number of tokens in dataset."""
        # TODO: Implement
        raise NotImplementedError


# =============================================================================
# EUR-Lex Dataset (Document Classification)
# =============================================================================

class EURLexDataset(Dataset):
    """
    EUR-Lex multi-label document classification dataset.

    Each example has:
    - text: Document text (truncated to max_length)
    - labels: Multi-hot vector of EUROVOC concept labels
    """

    def __init__(
        self,
        data_dir: str,
        split: str,  # train, validation, test
        tokenizer,
        max_length: int = 512,
        num_labels: int = 4271,
    ):
        """
        Args:
            data_dir: Directory containing {split}.jsonl and label_map.json
            split: Dataset split
            tokenizer: HuggingFace tokenizer
            max_length: Max sequence length
            num_labels: Number of label classes
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.num_labels = num_labels

        # TODO: Load examples from JSONL file
        # self.examples = self._load_data(data_dir, split)

        raise NotImplementedError

    def _load_data(self, data_dir: str, split: str) -> List[Dict]:
        """Load examples from JSONL file."""
        # TODO: Implement
        raise NotImplementedError

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        """
        Returns:
            dict with 'input_ids', 'attention_mask', 'labels'
        """
        # TODO: Implement
        # 1. Get example
        # 2. Tokenize text
        # 3. Create multi-hot label vector
        # 4. Return dict

        raise NotImplementedError


def create_dataloaders(
    data_dir: str,
    tokenizer,
    batch_size: int = 8,
    max_length: int = 512,
    num_workers: int = 4,
) -> Dict[str, DataLoader]:
    """
    Create train/val/test dataloaders.

    Returns:
        Dict mapping split name to DataLoader
    """
    # TODO: Implement
    raise NotImplementedError
