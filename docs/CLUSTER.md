# Cluster Deployment Guide

## Supported Platforms

| Platform | GPU Options | Cost Estimate | Notes |
|----------|-------------|---------------|-------|
| Lambda Labs | A100 40/80GB, H100 | ~$1.10-2.50/hr | Simple, good availability |
| RunPod | A100, H100, A10G | ~$0.80-2.00/hr | Spot instances available |
| Vast.ai | Various | Variable | Cheapest, less reliable |
| AWS | p4d (A100), p5 (H100) | ~$3-5/hr | Enterprise, most expensive |
| GCP | A100, H100 | ~$2-4/hr | Good for long runs |

## Recommended Setup

**For this project:** Lambda Labs A100 80GB
- ~$1.10/hr
- 90 hours needed → ~$100
- Simple SSH access, no k8s complexity

## Quick Deploy (Lambda Labs)

```bash
# 1. Launch instance via web UI
#    - Select A100 80GB
#    - Ubuntu 22.04 + PyTorch

# 2. SSH in
ssh ubuntu@<instance-ip>

# 3. Clone and setup
git clone <your-repo> dtype_bench
cd dtype_bench
./scripts/setup.sh

# 4. Run experiments
./scripts/run_all.sh
```

## Container Deployment

For more reproducible runs, use the provided Dockerfile:

```bash
# Build
docker build -t dtype-bench .

# Run with GPU
docker run --gpus all -v $(pwd)/results:/app/results dtype-bench ./scripts/run_all.sh
```

## Cluster Job Submission

### SLURM (Academic clusters)

```bash
# See scripts/slurm/submit_all.sh
sbatch scripts/slurm/lm_sweep.sbatch
sbatch scripts/slurm/cls_sweep.sbatch
```

### Kubernetes (Cloud)

```bash
# See scripts/k8s/
kubectl apply -f scripts/k8s/dtype-bench-job.yaml
```

## Data Preparation

Data can be prepared locally or on-cluster:

```bash
# Option 1: Prepare on cluster (slower, simpler)
./scripts/setup.sh  # Does everything

# Option 2: Pre-download, upload to cloud storage
# Locally:
python experiments/lm/prepare_data.py --output_dir ./data/openwebtext
python experiments/classification/prepare_data.py --output_dir ./data/eurlex

# Upload to S3/GCS, then on cluster:
aws s3 sync s3://your-bucket/data ./data
```

## Checkpointing & Resume

Experiments save results incrementally. To resume after interruption:

```bash
# Check what's completed
ls results/lm/*.json | wc -l
ls results/classification/*.json | wc -l

# Re-run - already-completed configs will be skipped
./scripts/run_all.sh --resume
```

## Monitoring

```bash
# Watch GPU utilization
watch -n 1 nvidia-smi

# Tail experiment logs
tail -f results/lm/logs/amp_bf16_seed0.log

# Check progress
python scripts/check_progress.py
```

## Cost Optimization

1. **Use spot/preemptible instances** - 60-70% cheaper, experiments checkpoint
2. **Start with subset** - Run 1 seed first to catch bugs
3. **Smaller model tests** - Debug with 1k iters before full 10k
4. **Off-peak hours** - Better availability, sometimes cheaper

## Troubleshooting

### OOM Errors
```bash
# Reduce batch size
python experiments/lm/run.py --config amp_bf16 --batch_size 8
```

### CUDA Version Mismatch
```bash
# Check versions
python -c "import torch; print(torch.version.cuda)"
nvidia-smi

# Use container for reproducibility
docker run --gpus all dtype-bench ...
```

### Slow Data Loading
```bash
# Increase workers
python experiments/lm/run.py --config amp_bf16 --num_workers 8

# Or pre-load to /dev/shm (RAM disk)
cp -r data/openwebtext /dev/shm/
```
