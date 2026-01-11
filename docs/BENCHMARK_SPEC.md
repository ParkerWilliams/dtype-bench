# Project Specification: Datatype Dynamics Bench (Arch × Task × Loss)

**Objective:** Conduct a "Research Grade" evaluation of floating-point datatype stability and performance (throughput/MFU) across the full cross-product of Model Architecture, Task Type, and Loss Function.

**Target Audience:** Professional ML researchers.
**Compute Budget:** ~$200 on RunPod (Target: RTX A6000 or A40 instances).
**Key Constraint:** Comparison must be strictly "apples-to-apples" regarding TF32 usage and gradient accumulation semantics.

---

## 1. Experimental Matrix (The 2×3 Design)

We will test 6 distinct scenarios to isolate architecture vs. objective function effects.

| Scenario ID | Architecture | Task Type | Loss Function | Data | Notes |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **A (Baseline)** | **GPT-2** (Decoder) | Causal LM | CrossEntropy (Softmax) | OpenWebText | Standard autoregressive baseline. |
| **B (Dense)** | **RoBERTa** (Encoder) | Masked LM | CrossEntropy (Softmax) | OpenWebText | Higher arithmetic intensity (full attn). |
| **C (Volatile)** | **GPT-2** | Multi-Label Cls | BCEWithLogits (Sigmoid) | EUR-Lex | Testing "Last Token" bottleneck stability. |
| **D (Robust?)** | **RoBERTa** | Multi-Label Cls | BCEWithLogits (Sigmoid) | EUR-Lex | Standard encoder classification. |
| **E (Control)** | **GPT-2** | Single-Label Cls | CrossEntropy (Softmax) | EUR-Lex (Mod) | **Synth:** Use most frequent label as target. |
| **F (Control)** | **RoBERTa** | Single-Label Cls | CrossEntropy (Softmax) | EUR-Lex (Mod) | **Synth:** Use most frequent label as target. |

---

## 2. Dtype Configurations (Independent Variables)

**Crucial:** All "FP32" baselines must explicitly set `torch.backends.cuda.matmul.allow_tf32 = False`.

| Config ID | Model Weights | Forward/Backward | Optimizer | Master Weights | Notes |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `fp32_true` | FP32 | FP32 | AdamW (FP32) | - | **Golden Reference** (TF32=False) |
| `amp_fp16` | FP32 | FP16 (Mix) | AdamW (FP32) | FP32 | Unsafe mixed precision (standard legacy). |
| `amp_bf16` | FP32 | BF16 (Mix) | AdamW (FP32) | FP32 | Modern standard. |
| `fp16_naive` | FP16 | FP16 | AdamW (FP16) | - | **Stress Test:** No loss scaler, pure FP16. |
| `bf16_pure` | BF16 | BF16 | AdamW (BF16) | - | **Throughput King:** Pure BF16 storage/compute. |
| `amp_bf16_8bit`| FP32 | BF16 (Mix) | **8-bit Adam** | FP32 | Memory optimization test. |

---

## 3. Hypotheses & Success Metrics

### Hypotheses to Test
1.  **The "Loss Scale" Inversion:** `fp16_naive` will survive Single-Label CE (E/F) but diverge on Multi-Label BCE (C/D) due to additive gradients.
2.  **The "Architecture" Inversion:** `bf16_pure` will show higher relative MFU gains on RoBERTa (B) than GPT-2 (A) due to bidirectional attention density.
3.  **The "Bottleneck" Instability:** GPT-2 Classification (C) will require higher precision (AMP) than RoBERTa (D) due to the "last-token" gradient funnel.

### Metrics
1.  **Stability:** Step count of first `NaN` (or "Completed" if none).
2.  **Throughput:** Samples/second *and* MFU (Model Flops Utilization).
3.  **Memory:** Peak allocated VRAM (GB).
4.  **Convergence:** Validation Loss / F1 Score (Micro & Macro).

---

## 4. Implementation Guidelines for Agent

### A. Environment Setup
* **Base:** PyTorch 2.x, Transformers, Datasets, Accelerate.
* **Hardware Check:** Script must verify GPU capability (A6000/A40 preferred).
* **Data Prep:**
    * *OpenWebText:* Stream small subset (don't download all) for speed.
    * *EUR-Lex:* Download once. Create a dynamic `DataCollator` that can toggle between Multi-Label (all labels) and Single-Label (argmax of labels) on the fly to avoid duplicating data on disk.

### B. Code Structure
Create a modular training script `benchmark.py` that accepts command line args:
```bash
python benchmark.py \
  --model_arch [gpt2, roberta] \
  --task [clm, mlm, multi_cls, single_cls] \
  --dtype_config [fp32_true, amp_bf16, ...] \
  --batch_size 32 \
  --max_steps 1000  # Short runs for profiling