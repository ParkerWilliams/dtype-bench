"""
Data Loading for Language Modeling

TODO: Implement data loading from pre-tokenized binary files.
See PLAN.md for specifications.

The data format is memory-mapped numpy arrays of uint16 token IDs.
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset


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
