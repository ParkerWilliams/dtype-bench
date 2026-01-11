"""
Document Classifier Model

TODO: Implement RoBERTa-based multi-label classifier.
See PLAN.md for specifications.

Key features:
- HuggingFace RoBERTa backbone
- Configurable dtype for classification head
- Multi-label BCE loss
"""

import torch
import torch.nn as nn
from torch.nn import functional as F
from dataclasses import dataclass
from typing import Optional

# from transformers import AutoModel, AutoConfig


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
