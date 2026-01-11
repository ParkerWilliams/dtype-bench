"""
Dtype Configurations

Shared configuration definitions for all benchmark scenarios.
"""

from dataclasses import dataclass
from typing import Optional
import torch


@dataclass
class DtypeConfig:
    """Configuration for a dtype experiment."""
    name: str

    # Model weight storage
    model_dtype: str = "float32"

    # Automatic Mixed Precision
    use_amp: bool = False
    amp_dtype: str = "float16"

    # Optimizer
    use_8bit_adam: bool = False

    # Classification head (optional, for encoder models)
    head_dtype: Optional[str] = None

    # torch.compile
    use_compile: bool = True

    # torch.set_float32_matmul_precision
    matmul_precision: str = "highest"  # highest, high, medium

    def get_model_dtype(self) -> torch.dtype:
        return _str_to_dtype(self.model_dtype)

    def get_amp_dtype(self) -> torch.dtype:
        return _str_to_dtype(self.amp_dtype)

    def get_head_dtype(self) -> Optional[torch.dtype]:
        if self.head_dtype is None:
            return None
        return _str_to_dtype(self.head_dtype)


def _str_to_dtype(s: str) -> torch.dtype:
    mapping = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    if s not in mapping:
        raise ValueError(f"Unknown dtype: {s}")
    return mapping[s]


# =============================================================================
# Core Configurations (used by all experiments)
# =============================================================================

CORE_CONFIGS = {
    "fp32_baseline": DtypeConfig(
        name="fp32_baseline",
        model_dtype="float32",
        use_amp=False,
        matmul_precision="highest",
    ),

    "amp_fp16": DtypeConfig(
        name="amp_fp16",
        model_dtype="float32",
        use_amp=True,
        amp_dtype="float16",
    ),

    "amp_bf16": DtypeConfig(
        name="amp_bf16",
        model_dtype="float32",
        use_amp=True,
        amp_dtype="bfloat16",
    ),

    "bf16_pure": DtypeConfig(
        name="bf16_pure",
        model_dtype="bfloat16",
        use_amp=False,
        matmul_precision="medium",
    ),
}


# =============================================================================
# Extended Configurations
# =============================================================================

EXTENDED_CONFIGS = {
    "fp16_naive": DtypeConfig(
        name="fp16_naive",
        model_dtype="float16",
        use_amp=False,
    ),

    "bf16_naive": DtypeConfig(
        name="bf16_naive",
        model_dtype="bfloat16",
        use_amp=False,
    ),

    "amp_fp16_8bit_adam": DtypeConfig(
        name="amp_fp16_8bit_adam",
        model_dtype="float32",
        use_amp=True,
        amp_dtype="float16",
        use_8bit_adam=True,
    ),

    "amp_bf16_8bit_adam": DtypeConfig(
        name="amp_bf16_8bit_adam",
        model_dtype="float32",
        use_amp=True,
        amp_dtype="bfloat16",
        use_8bit_adam=True,
    ),

    "bf16_no_compile": DtypeConfig(
        name="bf16_no_compile",
        model_dtype="bfloat16",
        use_amp=False,
        use_compile=False,
    ),

    "fp32_tf32_matmul": DtypeConfig(
        name="fp32_tf32_matmul",
        model_dtype="float32",
        use_amp=False,
        matmul_precision="high",  # Enables TF32
    ),
}


# =============================================================================
# Classification-Specific Configurations
# =============================================================================

CLASSIFICATION_CONFIGS = {
    "amp_bf16_fp32_head": DtypeConfig(
        name="amp_bf16_fp32_head",
        model_dtype="float32",
        use_amp=True,
        amp_dtype="bfloat16",
        head_dtype="float32",
    ),

    "bf16_body_fp32_head": DtypeConfig(
        name="bf16_body_fp32_head",
        model_dtype="bfloat16",
        use_amp=False,
        head_dtype="float32",
    ),
}


# =============================================================================
# Config Sets
# =============================================================================

def get_all_configs(include_classification_specific: bool = False) -> dict:
    """Get all dtype configurations."""
    configs = {}
    configs.update(CORE_CONFIGS)
    configs.update(EXTENDED_CONFIGS)
    if include_classification_specific:
        configs.update(CLASSIFICATION_CONFIGS)
    return configs


def get_config(name: str) -> DtypeConfig:
    """Get a specific config by name."""
    all_configs = get_all_configs(include_classification_specific=True)
    if name not in all_configs:
        raise ValueError(f"Unknown config: {name}. Available: {list(all_configs.keys())}")
    return all_configs[name]


def list_configs(include_classification_specific: bool = False) -> list:
    """List available config names."""
    return list(get_all_configs(include_classification_specific).keys())
