# Language Modeling Experiment

## Overview

Train GPT-2 124M on OpenWebText with various dtype configurations.
This is the baseline experiment against which we compare classification behavior.

## Model

- **Architecture:** GPT-2 (decoder-only transformer)
- **Size:** 124M parameters (12 layers, 768 hidden, 12 heads)
- **Vocab:** 50,304 (GPT-2 tokenizer, padded for efficiency)
- **Context:** 1024 tokens

## Dataset

- **Name:** OpenWebText
- **Size:** ~9B tokens
- **Format:** Pre-tokenized binary files (train.bin, val.bin)
- **Preparation:** `python prepare_data.py`

## Training Configuration

```yaml
batch_size: 12
gradient_accumulation_steps: 5
effective_batch_size: 60
max_iters: 10000
eval_interval: 500
eval_iters: 200

learning_rate: 6e-4
weight_decay: 0.1
warmup_iters: 1000
lr_decay: cosine to 6e-5
grad_clip: 1.0
```

## Metrics

| Metric | Description | Good Direction |
|--------|-------------|----------------|
| `tokens_per_second` | Training throughput | Higher |
| `peak_memory_gb` | Max GPU memory | Lower |
| `final_val_loss` | Validation perplexity proxy | Lower |
| `time_per_iter_ms` | Iteration latency | Lower |

## Expected Results

Based on prior work, rough expectations:

| Config | Relative Throughput | Memory | Quality |
|--------|---------------------|--------|---------|
| fp32_baseline | 1.0x | High | Best |
| amp_bf16 | ~2.0x | Medium | Same |
| bf16_pure | ~2.1x | Low | ~Same |
| fp16_naive | ~2.0x | Low | May diverge |

## Files to Implement

```
experiments/lm/
├── PLAN.md          # This file
├── run.py           # Main training script
├── model.py         # GPT-2 implementation
├── data.py          # Data loading
├── prepare_data.py  # Dataset preparation
└── config.py        # LM-specific training config
```

## Implementation Notes

### Model
- Use nanoGPT-style implementation (minimal, fast)
- Flash attention via F.scaled_dot_product_attention
- Weight tying between embedding and output

### Data
- Memory-mapped numpy arrays for efficiency
- Random sampling (not sequential) for each batch

### Training Loop
- Standard PyTorch training
- GradScaler for fp16 AMP only (bf16 doesn't need it)
- torch.compile for speed

### Evaluation
- Sample 200 batches from train and val
- Report mean loss

## Checkpoints

Save after each eval:
- `results/lm/{config}_seed{seed}.json` - Final metrics
- `results/lm/checkpoints/{config}_seed{seed}/` - Model weights (optional)

## Running

```bash
# Single config
python experiments/lm/run.py --config amp_bf16 --seed 0

# All configs
python experiments/lm/run.py --config all --seeds 3

# Quick test
python experiments/lm/run.py --config amp_bf16 --max_iters 100
```
