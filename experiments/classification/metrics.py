"""
Multi-Label Classification Metrics

TODO: Implement metrics for EUR-Lex evaluation.
See PLAN.md for specifications.

Metrics needed:
- Micro F1 (label-weighted)
- Macro F1 (class-averaged)
- Precision@k (k=1,3,5)
"""

import torch
from typing import Dict


def compute_metrics(
    logits: torch.Tensor,
    labels: torch.Tensor,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Compute multi-label classification metrics.
    
    Args:
        logits: [batch, num_labels] raw logits
        labels: [batch, num_labels] multi-hot ground truth
        threshold: Classification threshold for F1
    
    Returns:
        Dict with metric names and values
    """
    # TODO: Implement
    # 1. Convert logits to probabilities (sigmoid)
    # 2. Threshold for predictions
    # 3. Compute micro F1
    # 4. Compute macro F1
    # 5. Compute precision@k for k=1,3,5
    
    raise NotImplementedError


def micro_f1(preds: torch.Tensor, labels: torch.Tensor) -> float:
    """
    Micro-averaged F1 score.
    
    Aggregates TP, FP, FN across all labels, then computes F1.
    Gives more weight to frequent labels.
    """
    # TODO: Implement
    raise NotImplementedError


def macro_f1(preds: torch.Tensor, labels: torch.Tensor) -> float:
    """
    Macro-averaged F1 score.
    
    Computes F1 for each label, then averages.
    Gives equal weight to all labels (including rare ones).
    """
    # TODO: Implement
    raise NotImplementedError


def precision_at_k(probs: torch.Tensor, labels: torch.Tensor, k: int) -> float:
    """
    Precision@k metric.
    
    For each sample, take top-k predictions and compute
    fraction that are correct.
    """
    # TODO: Implement
    raise NotImplementedError
