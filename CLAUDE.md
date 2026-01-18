# dtype-bench

Benchmarking floating-point datatype stability and performance across model architectures, tasks, and loss functions.

## Project Structure

```
src/
  benchmark.py   # Main entry point - runs training experiments
  configs.py     # DtypeConfig definitions (fp32_true, amp_bf16, fp16_naive, etc.)
  models.py      # GPT-2 and RoBERTa model wrappers
  data.py        # Data loading (OpenWebText, EUR-Lex)
  metrics.py     # MetricsTracker, gradient norm computation
  utils.py       # Hardware checks, seeding, logging
scripts/
  run_sweep.sh   # Full experiment sweep (all scenarios × configs × seeds)
  setup.sh       # Environment setup
```

## Experimental Matrix (6 Scenarios)

| ID | Architecture | Task | Loss | Data |
|----|--------------|------|------|------|
| A | GPT-2 | Causal LM | CrossEntropy | OpenWebText |
| B | RoBERTa | Masked LM | CrossEntropy | OpenWebText |
| C | GPT-2 | Multi-Label Cls | BCEWithLogits | EUR-Lex |
| D | RoBERTa | Multi-Label Cls | BCEWithLogits | EUR-Lex |
| E | GPT-2 | Single-Label Cls | CrossEntropy | EUR-Lex |
| F | RoBERTa | Single-Label Cls | CrossEntropy | EUR-Lex |

## Dtype Configurations

Core configs in `src/configs.py`:
- `fp32_true` - True FP32 baseline (TF32 disabled)
- `amp_fp16` - FP32 weights, FP16 autocast with GradScaler
- `amp_bf16` - FP32 weights, BF16 autocast (no scaler needed)
- `bf16_pure` - BF16 weights and compute
- `fp16_naive` - FP16 weights, static loss scaling (1024x) to prevent gradient underflow
- `amp_bf16_8bit_adam` - BF16 AMP with 8-bit optimizer

## Running Experiments

```bash
# Single run
python -m src.benchmark \
  --model_arch gpt2 \
  --task clm \
  --dtype_config amp_bf16 \
  --max_steps 1000 \
  --batch_size 24

# Full sweep
./scripts/run_sweep.sh

# Quick test
./scripts/run_sweep.sh --quick
```

## Key Implementation Details

### Static Loss Scaling (fp16_naive)
Pure FP16 training without AMP requires manual loss scaling to prevent gradient underflow. The `static_loss_scale` field in DtypeConfig multiplies loss before backward, then divides gradients afterward. Default is 1024x for fp16_naive.

### TF32 Control
`fp32_true` explicitly disables TF32 via `torch.backends.cuda.matmul.allow_tf32 = False` for accurate FP32 baseline. Other configs use the `matmul_precision` field.

### Metrics Tracked
- Loss, gradient norms (L2, max), loss scale, max logit magnitude
- Per-step logs saved to `results/training_logs/`
- Summary metrics (throughput, MFU, peak memory) to `results/results.jsonl`

## Hypotheses Being Tested

1. `fp16_naive` survives single-label CE but diverges on multi-label BCE (gradient accumulation)
2. `bf16_pure` shows higher MFU gains on RoBERTa than GPT-2 (attention density)
3. GPT-2 classification requires higher precision than RoBERTa (last-token bottleneck)
