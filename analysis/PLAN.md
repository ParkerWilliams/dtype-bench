# Analysis Plan

## Overview

Compare dtype configuration behavior between Language Modeling and Document Classification.

## Analysis Stages

### Stage 1: Per-Experiment Analysis

Run after each experiment completes.

**Outputs:**
- `results/{experiment}/summary.csv` - Config-level metrics
- `results/{experiment}/plots/` - Visualizations

**Metrics to compute:**
- Mean ± std across seeds for each config
- Relative performance vs fp32 baseline
- Pareto frontier (throughput vs quality)

### Stage 2: Cross-Experiment Comparison

Run after both experiments complete.

**Key Questions:**

1. **Ranking Flips**
   - Which configs rank differently between tasks?
   - Quantify: Spearman correlation of rankings

2. **Stability Differences**
   - Does fp16_naive diverge at different rates?
   - Plot loss curves side-by-side

3. **Output Layer Precision**
   - Do amp_bf16_fp32_head configs beat amp_bf16 on classification?
   - Does the gap exist for LM? (Expect: no)

4. **Memory-Quality Tradeoffs**
   - Compare Pareto frontiers between tasks
   - Are the "efficient" configs the same?

5. **8-bit Adam Impact**
   - Quality degradation for each task
   - Memory savings for each task

## Files to Implement

```
analysis/
├── PLAN.md              # This file
├── analyze_experiment.py    # Single experiment analysis
├── compare_experiments.py   # Cross-experiment comparison
└── visualizations.py        # Plotting utilities
```

## Visualizations

### Per-Experiment
1. Bar chart: Throughput by config
2. Bar chart: Memory by config
3. Scatter: Throughput vs quality
4. Line plot: Loss curves over training
5. Pareto frontier

### Cross-Experiment
1. **Ranking comparison**: Slope graph showing rank changes
2. **Relative performance heatmap**: Config × Metric × Task
3. **Flip detection**: Highlight configs with large rank changes
4. **Stability comparison**: Loss curves for fp16_naive on both tasks

## Output Format

### Summary CSV
```csv
config,seed,throughput,throughput_std,memory,memory_std,quality,quality_std
fp32_baseline,mean,50000,1000,30.5,0.5,3.10,0.02
amp_bf16,mean,100000,2000,18.2,0.3,3.11,0.02
...
```

### Comparison JSON
```json
{
  "ranking_correlation": {
    "throughput": 0.95,
    "memory": 0.98,
    "quality": 0.72
  },
  "flips": [
    {
      "config": "fp16_naive",
      "lm_quality_rank": 8,
      "cls_quality_rank": 3,
      "interpretation": "Classification shorter training hides instability"
    }
  ],
  "recommendations": {
    "universal": ["amp_bf16 is safe default"],
    "lm_specific": ["fp16_naive dangerous after 5k steps"],
    "cls_specific": ["fp32 head may help with large label spaces"]
  }
}
```

## Running

```bash
# Analyze single experiment
python analysis/analyze_experiment.py --results_dir results/lm

# Compare experiments
python analysis/compare_experiments.py \
    --lm_results results/lm \
    --cls_results results/classification \
    --output_dir results/comparison
```
