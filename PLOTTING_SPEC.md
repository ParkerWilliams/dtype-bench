# Analysis & Visualization Specification

**Objective:** Generate a comprehensive suite of "Executive" and "Diagnostic" visualizations to prove the hypotheses defined in `BENCHMARK_SPEC.md`.

**Input:** `results.jsonl` (Metric logs) and `training_logs/*.jsonl` (Step-wise logs).
**Output Directory:** `results/plots/`

---

## 1. Executive Summary Plots (The "Results")

These plots summarize the findings for the general audience.

### Plot A: The Stability Matrix (Heatmap)
* **Goal:** Visualize which configurations fail on which tasks.
* **X-Axis:** Task (LM-Decoder, LM-Encoder, Cls-Multi, Cls-Single).
* **Y-Axis:** Dtype Config (fp32, amp_fp16, amp_bf16, fp16_naive, etc.).
* **Color:** Status (Green=Success, Red=Diverged, Orange=Poor Quality).
* **Annotation:** Step number of divergence (e.g., "NaN @ 50").

### Plot B: The Efficiency Frontier (Scatter)
* **Goal:** Compare computational density vs. memory cost.
* **X-Axis:** Peak VRAM (GB).
* **Y-Axis:** MFU (Model Flops Utilization %).
* **Grouping:** Shape = Architecture (GPT-2 vs RoBERTa), Color = Dtype.
* **Hypothesis:** `bf16_pure` should be top-left (High MFU, Low Mem).

### Plot C: Throughput Comparison (Bar)
* **Goal:** Raw speed comparison.
* **X-Axis:** Configuration.
* **Y-Axis:** Samples / Second.
* **Facet:** Task Type.
* **Requirement:** Normalize relative to `fp32_true` baseline (set to 1.0x).

---

## 2. Diagnostic Deep Dives (The "Forensics")

These plots explain *why* failures happened. Crucial for the "Research" aspect.

### Plot D: The Gradient Norm "Heartbeat" (Line)
* **Goal:** Detect instability before loss explosion.
* **X-Axis:** Training Step.
* **Y-Axis:** L2 Gradient Norm (Log Scale).
* **Lines:** Compare `fp16_naive` vs `amp_bf16` vs `fp32_true`.
* **Insight:** Look for "spikes" in `fp16_naive` even if loss looks flat.

### Plot E: The Logit Magnitude Histogram (Box/Violin)
* **Goal:** Prove H1 (Classification heads overflow FP16).
* **X-Axis:** Configuration (`amp_bf16` vs `amp_bf16_fp32_head`).
* **Y-Axis:** Max Absolute Logit Value (Log Scale).
* **Reference Line:** `65,504` (FP16 Max).
* **Insight:** If `amp_bf16` logits cross the line, the loss function received Infinity.

### Plot F: The "Padding Tax" (Scatter)
* **Goal:** Visualize efficiency loss due to padding in Classification.
* **X-Axis:** Batch Index (sorted by sequence length).
* **Y-Axis:** Effective MFU (excluding padding tokens).
* **Insight:** Demonstrate if Classification MFU is lower/noisier than LM MFU due to data structure.

---

## 3. Hypothesis Validation Plots

### Plot G: The "Loss Scale" Inversion (Line)
* **Goal:** Compare convergence of Single-Label vs Multi-Label.
* **X-Axis:** Step.
* **Y-Axis:** Training Loss.
* **Facet:** Task (Single-Label vs Multi-Label).
* **Focus:** Does `fp16_naive` survive Single-Label but die on Multi-Label?

### Plot H: Rank Correlation (Slope Graph)
* **Goal:** Show if the "Best Dtype" changes between tasks.
* **Left Axis:** Rank on Language Modeling (by Throughput).
* **Right Axis:** Rank on Classification (by Throughput).
* **Lines:** Connect the same config across axes.
* **Insight:** Crossing lines indicate "Task-Specific" optimization is needed.

---

## 4. Implementation Notes

* **Library:** Use `seaborn` for high-level styling.
* **Style:** Use a unified color palette for configs (e.g., BF16=Blue, FP16=Red, FP32=Gray).
* **Data Source:**
    * Plots A, B, C, H use `results.jsonl` (one row per run).
    * Plots D, E, F, G use `training_logs/run_{id}.jsonl` (one row per step).