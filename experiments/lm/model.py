"""
GPT-2 Model Implementation

TODO: Implement minimal GPT-2 following nanoGPT style.
See PLAN.md for specifications.

Key components needed:
- CausalSelfAttention (with flash attention)
- MLP (GELU activation)
- Block (attention + MLP with residuals)
- GPT (full model with embeddings and output head)

Reference: https://github.com/karpathy/nanoGPT/blob/master/model.py
"""

import torch
import torch.nn as nn
from torch.nn import functional as F
from dataclasses import dataclass


@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304  # GPT-2 vocab rounded up
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = False


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention with flash attention."""
    
    def __init__(self, config: GPTConfig):
        super().__init__()
        # TODO: Implement
        raise NotImplementedError


class MLP(nn.Module):
    """Feed-forward network."""
    
    def __init__(self, config: GPTConfig):
        super().__init__()
        # TODO: Implement
        raise NotImplementedError


class Block(nn.Module):
    """Transformer block."""
    
    def __init__(self, config: GPTConfig):
        super().__init__()
        # TODO: Implement
        raise NotImplementedError


class GPT(nn.Module):
    """GPT-2 model."""
    
    def __init__(self, config: GPTConfig):
        super().__init__()
        # TODO: Implement
        raise NotImplementedError
    
    def forward(self, idx, targets=None):
        """
        Forward pass.
        
        Args:
            idx: Input token indices [batch, seq_len]
            targets: Target token indices [batch, seq_len] (optional)
        
        Returns:
            logits: [batch, seq_len, vocab_size]
            loss: scalar (if targets provided)
        """
        # TODO: Implement
        raise NotImplementedError
    
    def get_num_params(self, non_embedding=True) -> int:
        """Count parameters."""
        # TODO: Implement
        raise NotImplementedError
