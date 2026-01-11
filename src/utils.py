"""
Utility Functions

Hardware checks, logging setup, and common utilities.
"""

import os
import sys
import logging
from datetime import datetime
from pathlib import Path

import torch


def check_hardware():
    """Verify GPU availability and capabilities."""
    print("=" * 50)
    print("Hardware Check")
    print("=" * 50)

    if not torch.cuda.is_available():
        print("WARNING: CUDA not available!")
        print("Benchmarks require a GPU.")
        return False

    device_count = torch.cuda.device_count()
    print(f"CUDA devices: {device_count}")

    for i in range(device_count):
        props = torch.cuda.get_device_properties(i)
        print(f"\nDevice {i}: {props.name}")
        print(f"  Memory: {props.total_memory / 1e9:.1f} GB")
        print(f"  Compute capability: {props.major}.{props.minor}")

        # Check for BF16 support (compute capability >= 8.0)
        if props.major >= 8:
            print("  BF16: Supported")
        else:
            print("  BF16: NOT supported (compute capability < 8.0)")

    print()
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA version: {torch.version.cuda}")

    # Check for flash attention
    try:
        from torch.nn.functional import scaled_dot_product_attention
        print("Flash Attention: Available (via SDPA)")
    except ImportError:
        print("Flash Attention: Not available")

    print("=" * 50)
    return True


def setup_logging(output_dir: str, name: str = "benchmark"):
    """Setup logging to file and console."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = output_path / f"{name}_{timestamp}.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout),
        ],
    )

    logging.info(f"Logging to {log_file}")
    return logging.getLogger(name)


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_memory_stats():
    """Get current GPU memory usage."""
    if not torch.cuda.is_available():
        return {}

    return {
        "allocated": torch.cuda.memory_allocated() / 1e9,
        "reserved": torch.cuda.memory_reserved() / 1e9,
        "max_allocated": torch.cuda.max_memory_allocated() / 1e9,
    }


def reset_memory_stats():
    """Reset peak memory stats."""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

def check_hardware_compatibility(dtype_config):
    """Fail fast if hardware doesn't support the requested config."""
    if not torch.cuda.is_available():
        raise EnvironmentError("CUDA is not available. This benchmark requires a GPU.")

    device_cap = torch.cuda.get_device_capability()
    gpu_name = torch.cuda.get_device_name()
    
    print(f"Hardware Check: Detected {gpu_name} (Compute Capability {device_cap})")

    # Check for BF16 support (Ampere+ required, capability >= 8.0)
    if "bf16" in dtype_config.name or dtype_config.amp_dtype == "bfloat16":
        if device_cap[0] < 8:
            raise EnvironmentError(
                f"Config '{dtype_config.name}' requires BF16 (Ampere+), "
                f"but detected {gpu_name} (CC {device_cap})."
            )

    # Check for TF32 support (Ampere+ required)
    if dtype_config.matmul_precision == "high" or dtype_config.matmul_precision == "highest":
        if device_cap[0] < 8:
            print(f"WARNING: TF32 requested but not supported on {gpu_name}. Falling back to FP32.")

    print("Hardware check passed.")
