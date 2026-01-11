# Document Classification Experiment

## Overview

Fine-tune RoBERTa-base on EUR-Lex multi-label classification.
Compare dtype behavior against language modeling experiment.

## Model

- **Architecture:** RoBERTa-base (encoder transformer) + linear classifier
- **Encoder:** 125M parameters (12 layers, 768 hidden, 12 heads)
- **Classifier:** 768 → 4,271 (3.3M parameters)
- **Total:** ~128M parameters

## Dataset

- **Name:** EUR-Lex (EURLEX57K)
- **Task:** Multi-label classification of EU legal documents
- **Labels:** ~4,271 EUROVOC concepts
- **Documents:** ~57,000 (45k train, 6k val, 6k test)
- **Avg labels/doc:** ~5
- **Avg tokens/doc:** ~700 (truncated to 512)

## Training Configuration

```yaml
batch_size: 8
gradient_accumulation_steps: 4
effective_batch_size: 32
max_steps: 5000  # or 10 epochs
eval_steps: 500

learning_rate: 2e-5
weight_decay: 0.01
warmup_ratio: 0.1
grad_clip: 1.0
```

## Metrics

| Metric | Description | Good Direction |
|--------|-------------|----------------|
| `samples_per_second` | Training throughput | Higher |
| `peak_memory_gb` | Max GPU memory | Lower |
| `micro_f1` | Label-weighted F1 | Higher |
| `macro_f1` | Class-averaged F1 | Higher |
| `precision_at_1` | Top-1 accuracy | Higher |
| `precision_at_3` | Top-3 accuracy | Higher |
| `precision_at_5` | Top-5 accuracy | Higher |

## Classification-Specific Dtype Configs

In addition to shared configs, test:

| Config | Hypothesis |
|--------|------------|
| `amp_bf16_fp32_head` | Large output layer benefits from fp32 |
| `bf16_body_fp32_head` | Hybrid: bf16 encoder, fp32 classifier |

## Key Differences from LM

| Aspect | Language Modeling | Classification |
|--------|-------------------|----------------|
| Architecture | Decoder-only | Encoder-only |
| Output size | 50k (vocab) | 4.3k (labels) |
| Loss | Cross-entropy | Binary cross-entropy |
| Labels/sample | 1 | ~5 (multi-label) |
| Training length | 10k iters | 5k steps |

## Files to Implement

```
experiments/classification/
├── PLAN.md          # This file
├── run.py           # Main training script
├── model.py         # RoBERTa + classifier
├── data.py          # EUR-Lex data loading
├── prepare_data.py  # Dataset preparation
├── metrics.py       # Multi-label metrics
└── config.py        # Classification-specific config
```

## Implementation Notes

### Model
- Use HuggingFace transformers for RoBERTa backbone
- Custom classification head with configurable dtype
- BCE loss with logits (for numerical stability)

### Data
- HuggingFace datasets or direct download from Zenodo
- Multi-hot label encoding
- Truncation to 512 tokens

### Metrics
- Micro F1: Weight by label frequency
- Macro F1: Average across all labels (sensitive to rare classes)
- Precision@k: How many of top-k predictions are correct

## Running

```bash
# Single config
python experiments/classification/run.py --config amp_bf16 --seed 0

# All configs (including classification-specific)
python experiments/classification/run.py --config all --seeds 3

# Quick test
python experiments/classification/run.py --config amp_bf16 --max_steps 100
```
