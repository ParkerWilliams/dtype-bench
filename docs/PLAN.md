# Project Plan

## Objective

Compare dtype configuration performance across Language Modeling and Document Classification to identify:
1. Universal best practices (what works for both)
2. Task-specific considerations (what flips)
3. Failure modes (when does fp16 diverge, etc.)

## Hypotheses

### H1: Output Layer Precision
Classification with large label spaces (~4k labels) may be more sensitive to output layer precision than LM.

**Test:** Compare `amp_bf16` vs `amp_bf16_fp32_head` on classification. If the fp32 head wins on classification but not LM, hypothesis confirmed.

### H2: Stability Differences
`fp16_naive` (no loss scaling) diverges in LM around 5-10k steps. Classification's shorter training and different loss (BCE vs CE) may hide or amplify this.

**Test:** Run `fp16_naive` on both tasks, track when/if loss explodes.

### H3: 8-bit Adam Impact
The large classifier head in EUR-Lex (~3.3M params) may stress 8-bit optimizer states more than the distributed parameters in LM.

**Test:** Compare quality degradation of `amp_bf16_8bit_adam` vs `amp_bf16` across tasks.

### H4: Memory-Throughput Tradeoff Shifts
Different model architectures (decoder vs encoder) and output sizes may shift the Pareto frontier.

**Test:** Plot throughput vs memory for both tasks, compare frontier shapes.

## Experiment Matrix

### Dtype Configurations (shared across experiments)

| Config | Model Dtype | AMP | AMP Dtype | 8-bit Adam | Head Dtype | Notes |
|--------|-------------|-----|-----------|------------|------------|-------|
| `fp32_baseline` | fp32 | No | - | No | fp32 | Reference |
| `amp_fp16` | fp32 | Yes | fp16 | No | - | Standard mixed precision |
| `amp_bf16` | fp32 | Yes | bf16 | No | - | Modern default |
| `bf16_pure` | bf16 | No | - | No | bf16 | Everything bf16 |
| `fp16_naive` | fp16 | No | - | No | fp16 | Stability test |
| `bf16_naive` | bf16 | No | - | No | bf16 | Usually stable |
| `amp_fp16_8bit_adam` | fp32 | Yes | fp16 | Yes | - | Memory opt |
| `amp_bf16_8bit_adam` | fp32 | Yes | bf16 | Yes | - | Memory opt |
| `bf16_no_compile` | bf16 | No | - | No | bf16 | Compile ablation |
| `fp32_tf32_matmul` | fp32 | No | - | No | fp32 | TF32 tensor cores |
| `amp_bf16_fp32_head` | fp32 | Yes | bf16 | No | fp32 | **CLS only** |
| `bf16_body_fp32_head` | bf16 | No | - | No | fp32 | **CLS only** |

### Experiment 1: Language Modeling

- **Model:** GPT-2 124M (decoder-only transformer)
- **Dataset:** OpenWebText
- **Metrics:** tokens/sec, peak memory GB, validation loss
- **Training:** 10k iterations, batch size 12, gradient accumulation 5
- **Seeds:** 3 per config

### Experiment 2: Document Classification

- **Model:** RoBERTa-base + linear head (encoder transformer)
- **Dataset:** EUR-Lex (~57k docs, ~4.3k multi-label)
- **Metrics:** samples/sec, peak memory GB, micro-F1, macro-F1, P@1/3/5
- **Training:** 5k steps (or 10 epochs), batch size 8, gradient accumulation 4
- **Seeds:** 3 per config

## Analysis Plan

### Per-Experiment Analysis
1. Summary table: config → throughput, memory, quality
2. Pareto frontier: throughput vs quality
3. Loss curves: stability visualization
4. Relative performance vs fp32 baseline

### Cross-Experiment Analysis
1. **Ranking comparison:** Do configs rank differently between tasks?
2. **Flip detection:** Which specific relationships reverse?
3. **Hypothesis testing:** Confirm/reject H1-H4
4. **Recommendations:** Task-specific best practices

## Execution Plan

### Phase 1: Setup (~1 hour)
- [ ] Provision A100 instance
- [ ] Clone repo, install deps
- [ ] Download/prepare datasets

### Phase 2: LM Experiments (~45 GPU hours)
- [ ] Run core configs (fp32, amp_fp16, amp_bf16, bf16_pure) × 3 seeds
- [ ] Checkpoint: verify results look sane
- [ ] Run extended configs × 3 seeds
- [ ] Run single-task analysis

### Phase 3: Classification Experiments (~35 GPU hours)
- [ ] Run core configs × 3 seeds
- [ ] Checkpoint: verify results
- [ ] Run extended configs (including classification-specific) × 3 seeds
- [ ] Run single-task analysis

### Phase 4: Combined Analysis (~1 hour)
- [ ] Run cross-experiment comparison
- [ ] Generate visualizations
- [ ] Write up findings

## Extension Points

Future experiments to add:
- [ ] Different model sizes (GPT-2 medium, large)
- [ ] Different classification datasets (AG News, IMDB)
- [ ] FP8 on H100
- [ ] Inference benchmarks (not just training)
- [ ] Gradient checkpointing interactions
- [ ] FSDP / DDP multi-GPU

## Success Criteria

1. Complete results for all configs × seeds × tasks
2. Clear identification of which dtype relationships are universal vs task-specific
3. Actionable recommendations for practitioners
4. Reproducible codebase others can extend
