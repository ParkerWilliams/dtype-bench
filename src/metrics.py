"""
Metrics Utilities for Dtype Benchmark

Provides functions for computing:
- Gradient norms (L2 and max)
- Model FLOPs Utilization (MFU)
- Throughput (samples/sec, tokens/sec)
- Memory statistics
"""

import time
from typing import Dict, List, Optional
from dataclasses import dataclass, field

import torch
import torch.nn as nn


@dataclass
class StepMetrics:
    """Metrics collected per training step."""
    step: int
    loss: float
    grad_norm_l2: float
    grad_norm_max: float
    loss_scale: float
    max_logit: float
    memory_gb: float
    step_time_ms: float


@dataclass
class RunMetrics:
    """Summary metrics for a complete run."""
    run_id: str
    model_arch: str
    task: str
    dtype_config: str
    seed: int
    completed: bool
    nan_step: Optional[int]
    final_loss: Optional[float]
    best_loss: Optional[float]
    peak_memory_gb: float
    samples_per_sec: float
    tokens_per_sec: float
    mfu: float
    total_steps: int
    total_time_sec: float


def compute_grad_norm(model: nn.Module) -> Dict[str, float]:
    """
    Compute gradient norms for monitoring training stability.

    Args:
        model: PyTorch model with gradients

    Returns:
        dict with 'l2' (total L2 norm) and 'max' (maximum gradient value)
    """
    total_norm_sq = 0.0
    max_norm = 0.0

    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2).item()
            total_norm_sq += param_norm ** 2
            max_val = p.grad.data.abs().max().item()
            max_norm = max(max_norm, max_val)

    return {
        "l2": total_norm_sq ** 0.5,
        "max": max_norm
    }


def estimate_model_flops(
    model: nn.Module,
    batch_size: int,
    seq_length: int,
    model_arch: str = "gpt2"
) -> int:
    """
    Estimate FLOPs for a single forward pass.

    For transformers, approximately:
    - 6 * n_params * seq_length (accounts for matmuls)
    - Multiplied by 3 for forward + backward

    Args:
        model: PyTorch model
        batch_size: Batch size
        seq_length: Sequence length
        model_arch: Model architecture for refined estimates

    Returns:
        Estimated FLOPs per training iteration
    """
    n_params = sum(p.numel() for p in model.parameters())

    # Rough estimate: 6 * params * seq_len for forward
    # 2x for backward, so 3x total for one training step
    flops_per_token = 6 * n_params
    flops_per_iter = flops_per_token * batch_size * seq_length * 3

    return flops_per_iter


def compute_mfu(
    model: nn.Module,
    batch_size: int,
    seq_length: int,
    step_times_ms: List[float],
    gpu_name: str = "A6000"
) -> float:
    """
    Compute Model FLOPs Utilization.

    MFU = achieved_flops / theoretical_peak_flops

    Args:
        model: PyTorch model
        batch_size: Batch size
        seq_length: Sequence length
        step_times_ms: List of step times in milliseconds
        gpu_name: GPU name for peak FLOPS lookup

    Returns:
        MFU as a fraction (0-1)
    """
    # Peak FLOPs for common GPUs (in TFLOPs, FP16/BF16)
    gpu_peak_tflops = {
        "A6000": 155.0,
        "A40": 150.0,
        "A100": 312.0,  # A100 80GB
        "H100": 989.0,  # H100 SXM
        "4090": 165.0,
        "3090": 71.0,
    }

    # Default to A6000 if unknown
    peak_tflops = 155.0
    for name, tflops in gpu_peak_tflops.items():
        if name.lower() in gpu_name.lower():
            peak_tflops = tflops
            break

    peak_flops = peak_tflops * 1e12  # Convert to FLOPs

    # Compute achieved FLOPs
    flops_per_iter = estimate_model_flops(model, batch_size, seq_length)

    if not step_times_ms:
        return 0.0

    avg_step_time_sec = sum(step_times_ms) / len(step_times_ms) / 1000.0
    achieved_flops_per_sec = flops_per_iter / avg_step_time_sec

    mfu = achieved_flops_per_sec / peak_flops
    return mfu


def compute_throughput(
    batch_size: int,
    seq_length: int,
    step_times_ms: List[float]
) -> Dict[str, float]:
    """
    Compute throughput metrics.

    Args:
        batch_size: Batch size
        seq_length: Sequence length
        step_times_ms: List of step times in milliseconds

    Returns:
        dict with 'samples_per_sec' and 'tokens_per_sec'
    """
    if not step_times_ms:
        return {"samples_per_sec": 0.0, "tokens_per_sec": 0.0}

    avg_step_time_sec = sum(step_times_ms) / len(step_times_ms) / 1000.0

    samples_per_sec = batch_size / avg_step_time_sec
    tokens_per_sec = batch_size * seq_length / avg_step_time_sec

    return {
        "samples_per_sec": samples_per_sec,
        "tokens_per_sec": tokens_per_sec
    }


class MetricsTracker:
    """
    Tracks metrics during training and computes summaries.
    """

    def __init__(
        self,
        run_id: str,
        model_arch: str,
        task: str,
        dtype_config: str,
        seed: int,
        batch_size: int,
        seq_length: int
    ):
        self.run_id = run_id
        self.model_arch = model_arch
        self.task = task
        self.dtype_config = dtype_config
        self.seed = seed
        self.batch_size = batch_size
        self.seq_length = seq_length

        self.step_metrics: List[StepMetrics] = []
        self.start_time: Optional[float] = None
        self.nan_step: Optional[int] = None
        self.best_loss: float = float('inf')

    def start(self):
        """Call at the start of training."""
        self.start_time = time.time()
        torch.cuda.reset_peak_memory_stats()

    def log_step(
        self,
        step: int,
        loss: float,
        grad_norms: Dict[str, float],
        loss_scale: float,
        max_logit: float,
        step_time_ms: float
    ):
        """Log metrics for a single step."""
        memory_gb = torch.cuda.memory_allocated() / 1e9

        metrics = StepMetrics(
            step=step,
            loss=loss,
            grad_norm_l2=grad_norms["l2"],
            grad_norm_max=grad_norms["max"],
            loss_scale=loss_scale,
            max_logit=max_logit,
            memory_gb=memory_gb,
            step_time_ms=step_time_ms
        )
        self.step_metrics.append(metrics)

        # Track best loss
        if loss < self.best_loss:
            self.best_loss = loss

    def mark_nan(self, step: int):
        """Mark that training diverged at this step."""
        self.nan_step = step

    def get_summary(self, model: nn.Module) -> RunMetrics:
        """Generate summary metrics for the run."""
        total_time = time.time() - self.start_time if self.start_time else 0
        peak_memory = torch.cuda.max_memory_allocated() / 1e9

        step_times = [m.step_time_ms for m in self.step_metrics]
        throughput = compute_throughput(self.batch_size, self.seq_length, step_times)

        # Get GPU name for MFU calculation
        gpu_name = torch.cuda.get_device_name() if torch.cuda.is_available() else "unknown"
        mfu = compute_mfu(model, self.batch_size, self.seq_length, step_times, gpu_name)

        final_loss = self.step_metrics[-1].loss if self.step_metrics else None

        return RunMetrics(
            run_id=self.run_id,
            model_arch=self.model_arch,
            task=self.task,
            dtype_config=self.dtype_config,
            seed=self.seed,
            completed=(self.nan_step is None),
            nan_step=self.nan_step,
            final_loss=final_loss,
            best_loss=self.best_loss if self.best_loss != float('inf') else None,
            peak_memory_gb=peak_memory,
            samples_per_sec=throughput["samples_per_sec"],
            tokens_per_sec=throughput["tokens_per_sec"],
            mfu=mfu,
            total_steps=len(self.step_metrics),
            total_time_sec=total_time
        )

    def get_step_metrics_dicts(self) -> List[Dict]:
        """Convert step metrics to list of dicts for JSONL output."""
        return [
            {
                "step": m.step,
                "loss": m.loss,
                "grad_norm_l2": m.grad_norm_l2,
                "grad_norm_max": m.grad_norm_max,
                "loss_scale": m.loss_scale,
                "max_logit": m.max_logit,
                "memory_gb": m.memory_gb,
                "step_time_ms": m.step_time_ms
            }
            for m in self.step_metrics
        ]
