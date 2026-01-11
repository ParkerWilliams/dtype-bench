"""
Model Definitions: GPT-2 and RoBERTa

Contains both architectures for the dtype benchmark:
- GPT-2: Decoder-only transformer for causal LM and classification (nanoGPT-style)
- RoBERTa: Encoder-only transformer for masked LM and classification (HuggingFace)
"""

import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from dataclasses import dataclass
from typing import Optional, Dict, Any


# =============================================================================
# GPT-2 Model (Decoder-only, nanoGPT-style)
# =============================================================================

@dataclass
class GPTConfig:
    """GPT-2 Small configuration (124M parameters)."""
    block_size: int = 1024
    vocab_size: int = 50304  # GPT-2 vocab (50257) rounded up for efficiency
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = False  # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster


class LayerNorm(nn.Module):
    """LayerNorm with optional bias (PyTorch doesn't support bias=False)."""

    def __init__(self, ndim: int, bias: bool):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, x):
        return F.layer_norm(x, self.weight.shape, self.weight, self.bias, 1e-5)


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention with Flash Attention via SDPA."""

    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0

        # Key, query, value projections for all heads in a single batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # Output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # Regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout

    def forward(self, x):
        B, T, C = x.size()  # batch, sequence length, embedding dim

        # Calculate query, key, values for all heads in batch
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)

        # Reshape for multi-head attention: (B, T, C) -> (B, nh, T, hs)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        # Flash Attention via scaled_dot_product_attention
        # is_causal=True automatically applies causal mask
        y = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            dropout_p=self.dropout if self.training else 0,
            is_causal=True
        )

        # Re-assemble all head outputs side by side
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # Output projection
        y = self.resid_dropout(self.c_proj(y))
        return y


class MLP(nn.Module):
    """Feed-forward network with GELU activation."""

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):
    """Transformer block with pre-normalization."""

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):
    """GPT-2 Language Model."""

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),
            wpe=nn.Embedding(config.block_size, config.n_embd),
            drop=nn.Dropout(config.dropout),
            h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f=LayerNorm(config.n_embd, bias=config.bias),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Weight tying: share embedding weights with output projection
        self.transformer.wte.weight = self.lm_head.weight

        # Initialize weights
        self.apply(self._init_weights)
        # Apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        """
        Forward pass for language modeling.

        Args:
            idx: Input token indices [batch, seq_len]
            targets: Target token indices [batch, seq_len] (optional)

        Returns:
            logits: [batch, seq_len, vocab_size]
            loss: scalar (if targets provided)
        """
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"Sequence length {t} exceeds block size {self.config.block_size}"

        # Position indices
        pos = torch.arange(0, t, dtype=torch.long, device=device)

        # Forward through transformer
        tok_emb = self.transformer.wte(idx)  # [b, t, n_embd]
        pos_emb = self.transformer.wpe(pos)  # [t, n_embd]
        x = self.transformer.drop(tok_emb + pos_emb)

        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        # Language model head
        logits = self.lm_head(x)  # [b, t, vocab_size]

        # Compute loss if targets provided
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1
            )

        return logits, loss

    def get_hidden_states(self, idx):
        """Get hidden states without LM head (for classification)."""
        device = idx.device
        b, t = idx.size()
        pos = torch.arange(0, t, dtype=torch.long, device=device)

        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(pos)
        x = self.transformer.drop(tok_emb + pos_emb)

        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        return x

    def get_num_params(self, non_embedding=True) -> int:
        """Count parameters, optionally excluding embeddings."""
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params


class GPTClassifier(nn.Module):
    """GPT-2 with classification head for sequence classification."""

    def __init__(
        self,
        gpt_config: GPTConfig,
        num_labels: int,
        task_type: str = "multi",  # "multi" for multi-label BCE, "single" for single-label CE
        head_dtype: Optional[torch.dtype] = None
    ):
        super().__init__()
        self.gpt = GPT(gpt_config)
        self.num_labels = num_labels
        self.task_type = task_type
        self.head_dtype = head_dtype

        # Classification head
        self.classifier = nn.Linear(gpt_config.n_embd, num_labels)

        # Optionally cast head to different dtype
        if head_dtype is not None:
            self.classifier = self.classifier.to(head_dtype)

    def forward(self, input_ids, attention_mask=None, labels=None):
        """
        Forward pass for classification.

        Args:
            input_ids: [batch, seq_len]
            attention_mask: [batch, seq_len] (1 for real tokens, 0 for padding)
            labels: [batch, num_labels] for multi-label or [batch] for single-label

        Returns:
            dict with 'loss', 'logits', and 'max_logit' for monitoring
        """
        # Get hidden states
        hidden_states = self.gpt.get_hidden_states(input_ids)  # [B, T, D]

        # Pool using last non-padded token
        if attention_mask is not None:
            # Find last real token position for each sequence
            seq_lengths = attention_mask.sum(dim=1) - 1  # [B]
            batch_idx = torch.arange(hidden_states.size(0), device=hidden_states.device)
            pooled = hidden_states[batch_idx, seq_lengths]  # [B, D]
        else:
            # Use last token
            pooled = hidden_states[:, -1, :]  # [B, D]

        # Handle head_dtype conversion
        if self.head_dtype is not None:
            pooled = pooled.to(self.head_dtype)

        logits = self.classifier(pooled)  # [B, num_labels]

        # Track max logit for overflow detection
        max_logit = logits.abs().max().item()

        # Compute loss
        loss = None
        if labels is not None:
            if self.task_type == "multi":
                # Multi-label: BCEWithLogitsLoss (always in FP32 for stability)
                loss = F.binary_cross_entropy_with_logits(
                    logits.float(),
                    labels.float()
                )
            else:
                # Single-label: CrossEntropyLoss
                loss = F.cross_entropy(logits, labels)

        return {"loss": loss, "logits": logits, "max_logit": max_logit}

    def get_num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


# =============================================================================
# RoBERTa Model (Encoder-only via HuggingFace)
# =============================================================================

@dataclass
class ClassifierConfig:
    """Configuration for RoBERTa-based classifier."""
    encoder_name: str = "roberta-base"
    num_labels: int = 4271  # EUR-Lex default
    max_length: int = 512
    classifier_dropout: float = 0.1
    pooling: str = "cls"  # "cls" or "mean"


class RoBERTaMLM(nn.Module):
    """RoBERTa for Masked Language Modeling (wraps HuggingFace)."""

    def __init__(self, model_name: str = "roberta-base"):
        super().__init__()
        from transformers import AutoModelForMaskedLM
        self.model = AutoModelForMaskedLM.from_pretrained(model_name)

    def forward(self, input_ids, attention_mask, labels=None):
        """
        Forward pass for MLM.

        Args:
            input_ids: [batch, seq_len]
            attention_mask: [batch, seq_len]
            labels: [batch, seq_len] with -100 for non-masked tokens

        Returns:
            dict with 'loss', 'logits', and 'max_logit'
        """
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )

        max_logit = outputs.logits.abs().max().item() if outputs.logits is not None else 0

        return {
            "loss": outputs.loss,
            "logits": outputs.logits,
            "max_logit": max_logit
        }

    def get_num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


class DocumentClassifier(nn.Module):
    """
    RoBERTa-based document classifier for multi-label or single-label tasks.

    Supports different dtypes for encoder and classification head
    to test hypothesis about output layer precision.
    """

    def __init__(
        self,
        config: ClassifierConfig,
        task_type: str = "multi",  # "multi" or "single"
        head_dtype: Optional[torch.dtype] = None
    ):
        """
        Args:
            config: Model configuration
            task_type: "multi" for multi-label BCE, "single" for single-label CE
            head_dtype: If specified, use this dtype for classification head
        """
        super().__init__()
        from transformers import AutoModel

        self.config = config
        self.task_type = task_type
        self.head_dtype = head_dtype

        # Load pretrained encoder
        self.encoder = AutoModel.from_pretrained(config.encoder_name)
        hidden_size = self.encoder.config.hidden_size

        # Classification head
        self.dropout = nn.Dropout(config.classifier_dropout)
        self.classifier = nn.Linear(hidden_size, config.num_labels)

        # Optionally cast head to different dtype
        if head_dtype is not None:
            self.classifier = self.classifier.to(head_dtype)

    def forward(self, input_ids, attention_mask, labels=None):
        """
        Forward pass for classification.

        Args:
            input_ids: [batch, seq_len]
            attention_mask: [batch, seq_len]
            labels: [batch, num_labels] for multi-label or [batch] for single-label

        Returns:
            dict with 'loss', 'logits', and 'max_logit'
        """
        # Encode
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state  # [B, T, H]

        # Pool
        if self.config.pooling == "cls":
            pooled = hidden_states[:, 0, :]  # CLS token
        else:
            # Mean pooling over non-padded tokens
            mask_expanded = attention_mask.unsqueeze(-1).float()
            sum_hidden = (hidden_states * mask_expanded).sum(dim=1)
            pooled = sum_hidden / mask_expanded.sum(dim=1).clamp(min=1e-9)

        pooled = self.dropout(pooled)

        # Handle head_dtype conversion
        if self.head_dtype is not None:
            pooled = pooled.to(self.head_dtype)

        logits = self.classifier(pooled)  # [B, num_labels]

        # Track max logit for overflow detection
        max_logit = logits.abs().max().item()

        # Compute loss
        loss = None
        if labels is not None:
            if self.task_type == "multi":
                # Multi-label: BCEWithLogitsLoss (always in FP32 for stability)
                loss = F.binary_cross_entropy_with_logits(
                    logits.float(),
                    labels.float()
                )
            else:
                # Single-label: CrossEntropyLoss
                loss = F.cross_entropy(logits, labels)

        return {"loss": loss, "logits": logits, "max_logit": max_logit}

    def get_num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


# =============================================================================
# Model Factory
# =============================================================================

def create_model(
    model_arch: str,
    task: str,
    num_labels: int = 4271,
    head_dtype: Optional[torch.dtype] = None
) -> nn.Module:
    """
    Factory function to create models for different scenarios.

    Args:
        model_arch: "gpt2" or "roberta"
        task: "clm", "mlm", "multi_cls", or "single_cls"
        num_labels: Number of classification labels (for cls tasks)
        head_dtype: Optional dtype for classification head

    Returns:
        PyTorch model
    """
    if model_arch == "gpt2":
        config = GPTConfig()

        if task == "clm":
            return GPT(config)
        elif task in ("multi_cls", "single_cls"):
            task_type = "multi" if task == "multi_cls" else "single"
            return GPTClassifier(config, num_labels, task_type, head_dtype)
        else:
            raise ValueError(f"GPT-2 doesn't support task: {task}")

    elif model_arch == "roberta":
        if task == "mlm":
            return RoBERTaMLM()
        elif task in ("multi_cls", "single_cls"):
            task_type = "multi" if task == "multi_cls" else "single"
            config = ClassifierConfig(num_labels=num_labels)
            return DocumentClassifier(config, task_type, head_dtype)
        else:
            raise ValueError(f"RoBERTa doesn't support task: {task}")

    else:
        raise ValueError(f"Unknown model architecture: {model_arch}")
