"""
Model Definitions: GPT-2 and RoBERTa

Contains both architectures for the dtype benchmark:
- GPT-2: Decoder-only transformer for causal LM and classification
- RoBERTa: Encoder-only transformer for masked LM and classification

TODO: Implement models following nanoGPT style for GPT-2
      and HuggingFace for RoBERTa.
"""

import torch
import torch.nn as nn
from torch.nn import functional as F
from dataclasses import dataclass
from typing import Optional


# =============================================================================
# GPT-2 Model (Decoder-only)
# =============================================================================

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


# =============================================================================
# RoBERTa Model (Encoder-only via HuggingFace)
# =============================================================================

@dataclass
class ClassifierConfig:
    encoder_name: str = "roberta-base"
    num_labels: int = 4271  # EUR-Lex
    max_length: int = 512
    classifier_dropout: float = 0.1
    pooling: str = "cls"  # cls, mean


class DocumentClassifier(nn.Module):
    """
    Transformer encoder with multi-label classification head.

    Supports different dtypes for encoder and classification head
    to test hypothesis about output layer precision.
    """

    def __init__(self, config: ClassifierConfig, head_dtype: Optional[torch.dtype] = None):
        """
        Args:
            config: Model configuration
            head_dtype: If specified, use this dtype for classification head
                       (allows testing fp32 head with bf16 body)
        """
        super().__init__()
        self.config = config
        self.head_dtype = head_dtype

        # TODO: Implement
        # 1. Load pretrained encoder (AutoModel.from_pretrained)
        # 2. Get hidden size from encoder config
        # 3. Create classification head (Linear layer)
        # 4. If head_dtype specified, cast head to that dtype

        raise NotImplementedError

    def forward(self, input_ids, attention_mask, labels=None):
        """
        Forward pass.

        Args:
            input_ids: [batch, seq_len]
            attention_mask: [batch, seq_len]
            labels: [batch, num_labels] multi-hot (optional)

        Returns:
            dict with 'loss' and 'logits'
        """
        # TODO: Implement
        # 1. Encode with transformer
        # 2. Pool (CLS token or mean pooling)
        # 3. Apply dropout
        # 4. Classification head (handle head_dtype conversion)
        # 5. Compute BCE loss if labels provided (always in fp32 for stability)

        raise NotImplementedError

    def get_num_params(self) -> int:
        """Count parameters."""
        return sum(p.numel() for p in self.parameters())
