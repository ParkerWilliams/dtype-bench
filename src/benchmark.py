"""
Dtype Benchmark - Main Entry Point

Usage:
    python -m src.benchmark \
        --model_arch [gpt2, roberta] \
        --task [clm, mlm, multi_cls, single_cls] \
        --dtype_config [fp32_baseline, amp_bf16, ...] \
        --batch_size 32 \
        --max_steps 1000

Scenarios (from BENCHMARK_SPEC.md):
    A: GPT-2 + Causal LM + CrossEntropy
    B: RoBERTa + Masked LM + CrossEntropy
    C: GPT-2 + Multi-Label Cls + BCEWithLogits
    D: RoBERTa + Multi-Label Cls + BCEWithLogits
    E: GPT-2 + Single-Label Cls + CrossEntropy
    F: RoBERTa + Single-Label Cls + CrossEntropy
"""

import argparse
import json
import os
import time
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Optional

import torch
from tqdm import tqdm

from .configs import get_config, list_configs, DtypeConfig
from .models import create_model
from .data import (
    OpenWebTextDataset,
    create_classification_dataloader,
    create_mlm_dataloader,
)
from .metrics import MetricsTracker, compute_grad_norm
from .utils import (
    check_hardware,
    check_hardware_compatibility,
    setup_logging,
    set_seed,
    reset_memory_stats,
)


@contextmanager
def dtype_context(config: DtypeConfig):
    """
    Setup dtype context for training.

    CRITICAL: TF32 must be explicitly disabled for true FP32 baseline.
    """
    # Save original state
    original_tf32_matmul = torch.backends.cuda.matmul.allow_tf32
    original_tf32_cudnn = torch.backends.cudnn.allow_tf32

    try:
        # Set matmul precision
        if config.matmul_precision == "highest":
            # Disable TF32 for true FP32 baseline
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            torch.set_float32_matmul_precision("highest")
            print("TF32 DISABLED for true FP32 baseline")
        elif config.matmul_precision == "high":
            # Enable TF32
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.set_float32_matmul_precision("high")
            print("TF32 ENABLED (high precision)")
        else:  # medium
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.set_float32_matmul_precision("medium")
            print("TF32 ENABLED (medium precision)")

        yield
    finally:
        # Restore original state
        torch.backends.cuda.matmul.allow_tf32 = original_tf32_matmul
        torch.backends.cudnn.allow_tf32 = original_tf32_cudnn


def run_benchmark(
    model_arch: str,
    task: str,
    config: DtypeConfig,
    seed: int,
    max_steps: int,
    batch_size: int,
    subset_size: int,
    output_dir: str,
    lr: float = 6e-4,
    block_size: int = 1024,
    max_length: int = 512,
    num_labels: int = 4271,
):
    """
    Run a single benchmark configuration.

    Args:
        model_arch: "gpt2" or "roberta"
        task: "clm", "mlm", "multi_cls", or "single_cls"
        config: Dtype configuration
        seed: Random seed
        max_steps: Maximum training steps
        batch_size: Batch size
        subset_size: Number of data examples to use
        output_dir: Output directory for results
        lr: Learning rate
        block_size: Sequence length for CLM
        max_length: Max length for other tasks
        num_labels: Number of classification labels
    """
    run_id = f"{model_arch}_{task}_{config.name}_{seed}"
    print(f"\n{'='*60}")
    print(f"Starting run: {run_id}")
    print(f"{'='*60}")

    # Set seed
    set_seed(seed)

    # Determine sequence length
    seq_length = block_size if task == "clm" else max_length

    # Initialize metrics tracker
    tracker = MetricsTracker(
        run_id=run_id,
        model_arch=model_arch,
        task=task,
        dtype_config=config.name,
        seed=seed,
        batch_size=batch_size,
        seq_length=seq_length
    )

    # Setup dtype context
    with dtype_context(config):
        # Check hardware compatibility
        check_hardware_compatibility(config)

        # Create model
        print(f"Creating model: {model_arch} for {task}")
        model = create_model(
            model_arch=model_arch,
            task=task,
            num_labels=num_labels,
            head_dtype=config.get_head_dtype()
        )

        # Cast model to target dtype
        model_dtype = config.get_model_dtype()
        model = model.to(model_dtype).cuda()
        print(f"Model dtype: {model_dtype}")
        print(f"Model parameters: {model.get_num_params():,}")

        # Optional torch.compile
        if config.use_compile:
            print("Compiling model with torch.compile...")
            model = torch.compile(model)

        # Setup optimizer
        if config.use_8bit_adam:
            import bitsandbytes as bnb
            optimizer = bnb.optim.AdamW8bit(model.parameters(), lr=lr)
            print("Using 8-bit AdamW optimizer")
        else:
            optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
            print("Using standard AdamW optimizer")

        # Setup AMP
        use_scaler = config.use_amp and config.amp_dtype == "float16"
        scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)
        amp_dtype = config.get_amp_dtype() if config.use_amp else None
        print(f"AMP: {'enabled (' + str(amp_dtype) + ')' if config.use_amp else 'disabled'}")
        print(f"GradScaler: {'enabled' if use_scaler else 'disabled'}")

        # Load data
        print(f"Loading data (subset_size={subset_size})...")
        if task == "clm":
            dataset = OpenWebTextDataset(
                subset_size=subset_size,
                block_size=block_size,
                tokenizer_type=model_arch
            )
            data_iter = None  # We'll use dataset.get_batch()
        elif task == "mlm":
            dataloader = create_mlm_dataloader(
                subset_size=subset_size,
                max_length=max_length,
                batch_size=batch_size
            )
            data_iter = iter(dataloader)
        else:  # classification
            dataloader = create_classification_dataloader(
                split="train",
                task=task,
                batch_size=batch_size,
                max_length=max_length,
                num_labels=num_labels,
                tokenizer_name="roberta-base" if model_arch == "roberta" else "gpt2"
            )
            data_iter = iter(dataloader)

        # Training loop
        print(f"\nStarting training for {max_steps} steps...")
        tracker.start()
        model.train()

        pbar = tqdm(range(max_steps), desc="Training")
        for step in pbar:
            step_start = time.time()

            # Get batch
            if task == "clm":
                x, y = dataset.get_batch(batch_size)
                batch = {"idx": x, "targets": y}
            else:
                try:
                    batch = next(data_iter)
                except StopIteration:
                    data_iter = iter(dataloader)
                    batch = next(data_iter)
                batch = {k: v.cuda() for k, v in batch.items()}

            # Zero gradients
            optimizer.zero_grad(set_to_none=True)

            # Forward pass
            with torch.amp.autocast("cuda", enabled=config.use_amp, dtype=amp_dtype):
                if task == "clm":
                    logits, loss = model(batch["idx"], batch["targets"])
                    max_logit = logits.abs().max().item()
                else:
                    outputs = model(**batch)
                    loss = outputs["loss"]
                    max_logit = outputs.get("max_logit", 0)

            # Check for NaN
            if torch.isnan(loss):
                print(f"\nNaN detected at step {step}!")
                tracker.mark_nan(step)
                break

            # Backward pass
            if use_scaler:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
            else:
                loss.backward()

            # Compute gradient norms BEFORE clipping
            grad_norms = compute_grad_norm(model)

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            # Optimizer step
            if use_scaler:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

            # Log metrics
            step_time_ms = (time.time() - step_start) * 1000
            loss_scale = scaler.get_scale() if use_scaler else 1.0

            tracker.log_step(
                step=step,
                loss=loss.item(),
                grad_norms=grad_norms,
                loss_scale=loss_scale,
                max_logit=max_logit,
                step_time_ms=step_time_ms
            )

            # Update progress bar
            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "grad": f"{grad_norms['l2']:.2f}",
                "mem": f"{torch.cuda.memory_allocated()/1e9:.1f}GB"
            })

        # Generate summary
        summary = tracker.get_summary(model)

        # Save results
        save_results(summary, tracker, output_dir, run_id)

        print(f"\nRun completed: {run_id}")
        print(f"  Completed: {summary.completed}")
        print(f"  Final loss: {summary.final_loss:.4f}" if summary.final_loss else "  Final loss: N/A")
        print(f"  Peak memory: {summary.peak_memory_gb:.2f} GB")
        print(f"  Throughput: {summary.samples_per_sec:.1f} samples/sec")
        print(f"  MFU: {summary.mfu*100:.2f}%")

        return summary


def save_results(summary, tracker: MetricsTracker, output_dir: str, run_id: str):
    """Save results to JSONL files."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save summary to results.jsonl
    results_file = output_path / "results.jsonl"
    with open(results_file, "a") as f:
        summary_dict = asdict(summary)
        f.write(json.dumps(summary_dict) + "\n")

    # Save step metrics to training_logs/
    logs_dir = output_path / "training_logs"
    logs_dir.mkdir(exist_ok=True)

    log_file = logs_dir / f"{run_id}.jsonl"
    with open(log_file, "w") as f:
        for step_dict in tracker.get_step_metrics_dicts():
            f.write(json.dumps(step_dict) + "\n")

    print(f"Results saved to {results_file}")
    print(f"Step logs saved to {log_file}")


def main():
    parser = argparse.ArgumentParser(description='Dtype Benchmark')
    parser.add_argument('--model_arch', type=str, required=True,
                        choices=['gpt2', 'roberta'],
                        help='Model architecture')
    parser.add_argument('--task', type=str, required=True,
                        choices=['clm', 'mlm', 'multi_cls', 'single_cls'],
                        help='Task type')
    parser.add_argument('--dtype_config', type=str, default='amp_bf16',
                        help='Dtype config name or "all"')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--seeds', type=int, default=1,
                        help='Number of seeds (overrides --seed)')
    parser.add_argument('--max_steps', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--subset_size', type=int, default=100000,
                        help='Number of data examples to use')
    parser.add_argument('--lr', type=float, default=6e-4,
                        help='Learning rate')
    parser.add_argument('--block_size', type=int, default=1024,
                        help='Sequence length for CLM')
    parser.add_argument('--max_length', type=int, default=512,
                        help='Max sequence length for other tasks')
    parser.add_argument('--output_dir', type=str, default='results')
    parser.add_argument('--list', action='store_true',
                        help='List available dtype configs')

    args = parser.parse_args()

    if args.list:
        print("Available dtype configs:")
        for name in list_configs(include_classification_specific=True):
            print(f"  {name}")
        return

    # Check hardware
    check_hardware()

    # Setup logging
    setup_logging(args.output_dir)

    # Validate task/model combinations
    valid_combinations = {
        ('gpt2', 'clm'),       # Scenario A
        ('roberta', 'mlm'),    # Scenario B
        ('gpt2', 'multi_cls'), # Scenario C
        ('roberta', 'multi_cls'), # Scenario D
        ('gpt2', 'single_cls'),   # Scenario E
        ('roberta', 'single_cls'), # Scenario F
    }

    if (args.model_arch, args.task) not in valid_combinations:
        print(f"Warning: Unusual combination {args.model_arch} + {args.task}")

    # Determine which configs to run
    if args.dtype_config == "all":
        include_cls = args.task in ("multi_cls", "single_cls")
        config_names = list_configs(include_classification_specific=include_cls)
    else:
        config_names = [args.dtype_config]

    # Determine which seeds to run
    if args.seeds > 1:
        seeds = list(range(args.seeds))
    else:
        seeds = [args.seed]

    # Run benchmarks
    all_results = []
    for config_name in config_names:
        config = get_config(config_name)
        for seed in seeds:
            try:
                result = run_benchmark(
                    model_arch=args.model_arch,
                    task=args.task,
                    config=config,
                    seed=seed,
                    max_steps=args.max_steps,
                    batch_size=args.batch_size,
                    subset_size=args.subset_size,
                    output_dir=args.output_dir,
                    lr=args.lr,
                    block_size=args.block_size,
                    max_length=args.max_length
                )
                all_results.append(result)
            except Exception as e:
                print(f"Error running {config_name} seed {seed}: {e}")
                import traceback
                traceback.print_exc()

    # Print summary
    print("\n" + "="*60)
    print("BENCHMARK COMPLETE")
    print("="*60)
    print(f"Total runs: {len(all_results)}")
    completed = sum(1 for r in all_results if r.completed)
    print(f"Completed: {completed}/{len(all_results)}")
    print(f"Results saved to: {args.output_dir}/")


if __name__ == '__main__':
    main()
