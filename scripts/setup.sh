#!/bin/bash
# Environment setup for RunPod GPU instance
#
# Usage:
#   ./scripts/setup.sh           # Full setup with venv
#   ./scripts/setup.sh --no-venv # Skip venv creation (for containers)

set -e

# =============================================================================
# Environment Detection & Cache Configuration
# =============================================================================

# Detect environment and set cache directory
if [ -d "/workspace" ] && [ -w "/workspace" ]; then
    # RunPod environment
    CACHE_DIR="/workspace/cache"
    echo "Detected RunPod environment"
elif [ -d "/root" ] && [ -w "/root" ]; then
    # Generic container with root access
    CACHE_DIR="/root/.cache"
    echo "Detected container environment"
else
    # Local/other environment
    CACHE_DIR="${HOME}/.cache"
    echo "Detected local environment"
fi

# Create cache directory
mkdir -p "${CACHE_DIR}/huggingface"

# Export HuggingFace cache environment variables
export HF_HOME="${CACHE_DIR}/huggingface"
export HF_DATASETS_CACHE="${CACHE_DIR}/huggingface/datasets"
export TRANSFORMERS_CACHE="${CACHE_DIR}/huggingface/hub"
export TIKTOKEN_CACHE_DIR="${CACHE_DIR}/tiktoken"

echo "Cache directory: ${CACHE_DIR}"
echo "HF_HOME: ${HF_HOME}"

# =============================================================================
# Parse Arguments
# =============================================================================

USE_VENV=true
for arg in "$@"; do
    case $arg in
        --no-venv) USE_VENV=false ;;
    esac
done

echo ""
echo "=========================================="
echo "Dtype Benchmark Setup"
echo "=========================================="

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"
echo "Project root: $PROJECT_ROOT"

# =============================================================================
# System Dependencies (optional, non-fatal)
# =============================================================================

if command -v apt-get &> /dev/null; then
    echo ""
    echo "Attempting to install system dependencies..."
    # Try with sudo first, fall back to without
    if command -v sudo &> /dev/null && sudo -n true 2>/dev/null; then
        sudo apt-get update -qq && sudo apt-get install -y -qq git wget python3-venv || true
    else
        apt-get update -qq 2>/dev/null && apt-get install -y -qq git wget python3-venv 2>/dev/null || echo "Skipping apt packages (no permissions)"
    fi
fi

# =============================================================================
# Python Detection
# =============================================================================

if command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
elif command -v python &> /dev/null; then
    PYTHON_CMD="python"
else
    echo "ERROR: Python not found"
    exit 1
fi

echo "Using Python: $PYTHON_CMD ($($PYTHON_CMD --version))"

# =============================================================================
# Virtual Environment
# =============================================================================

if [ "$USE_VENV" = true ]; then
    echo ""
    echo "Creating virtual environment..."
    $PYTHON_CMD -m venv .venv
    source .venv/bin/activate
    echo "Virtual environment activated: $VIRTUAL_ENV"
    PYTHON_CMD="python"  # Use venv python
    pip install --upgrade pip
fi

# =============================================================================
# Install Python Packages
# =============================================================================

echo ""
echo "Installing Python packages..."

# Detect CUDA version and install appropriate PyTorch
if command -v nvidia-smi &> /dev/null; then
    CUDA_VERSION=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1 | cut -d. -f1)
    echo "Detected NVIDIA driver major version: ${CUDA_VERSION}"

    # Install PyTorch with appropriate CUDA version
    if [ "$CUDA_VERSION" -ge 525 ]; then
        echo "Installing PyTorch for CUDA 12.1..."
        pip install torch==2.1.2 --index-url https://download.pytorch.org/whl/cu121
    elif [ "$CUDA_VERSION" -ge 470 ]; then
        echo "Installing PyTorch for CUDA 11.8..."
        pip install torch==2.1.2 --index-url https://download.pytorch.org/whl/cu118
    else
        echo "WARNING: Old NVIDIA driver detected. Installing default PyTorch..."
        pip install torch==2.1.2
    fi
else
    echo "nvidia-smi not found, installing default PyTorch..."
    pip install torch==2.1.2
fi

# Install remaining packages
pip install --ignore-installed -r requirements.txt

# =============================================================================
# Pre-cache Models and Tokenizers
# =============================================================================

echo ""
echo "Pre-caching models and tokenizers..."
$PYTHON_CMD -c "
import os
os.environ['HF_HOME'] = '${HF_HOME}'
os.environ['TRANSFORMERS_CACHE'] = '${TRANSFORMERS_CACHE}'
os.environ['TIKTOKEN_CACHE_DIR'] = '${TIKTOKEN_CACHE_DIR}'

print('Downloading GPT-2 tokenizer (tiktoken)...')
import tiktoken
enc = tiktoken.get_encoding('gpt2')
print(f'  Cached to: {os.environ.get(\"TIKTOKEN_CACHE_DIR\", \"default\")}')

print('Downloading RoBERTa tokenizer...')
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained('roberta-base')
print(f'  Cached to: {os.environ.get(\"TRANSFORMERS_CACHE\", \"default\")}')

print('Tokenizers cached successfully!')
"

# =============================================================================
# Verify GPU
# =============================================================================

echo ""
echo "=========================================="
echo "GPU Information:"
echo "=========================================="
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv
else
    echo "nvidia-smi not found"
fi

# =============================================================================
# Verify PyTorch
# =============================================================================

echo ""
echo "=========================================="
echo "PyTorch Configuration:"
echo "=========================================="
$PYTHON_CMD -c "
import torch
print(f'PyTorch Version: {torch.__version__}')
print(f'CUDA Available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA Version: {torch.version.cuda}')
    print(f'GPU: {torch.cuda.get_device_name()}')
    cap = torch.cuda.get_device_capability()
    print(f'Compute Capability: {cap[0]}.{cap[1]}')
    print(f'BF16 Supported: {cap[0] >= 8}')
"

# =============================================================================
# Verify Imports
# =============================================================================

echo ""
echo "Verifying imports..."
$PYTHON_CMD -c "
from src.configs import get_config, list_configs
from src.models import create_model
from src.data import OpenWebTextDataset
from src.metrics import MetricsTracker
from src.utils import check_hardware
print('All imports successful!')
"

# =============================================================================
# Write Environment File
# =============================================================================

ENV_FILE="$PROJECT_ROOT/.env.sh"
cat > "$ENV_FILE" << EOF
# Source this file to set up environment variables
# Usage: source .env.sh

export HF_HOME="${HF_HOME}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE}"
export TIKTOKEN_CACHE_DIR="${TIKTOKEN_CACHE_DIR}"
export PYTHONPATH="${PROJECT_ROOT}:\${PYTHONPATH}"
EOF

echo ""
echo "Environment variables written to: $ENV_FILE"

# =============================================================================
# Done
# =============================================================================

echo ""
echo "=========================================="
echo "Setup complete!"
echo "=========================================="
echo ""

if [ "$USE_VENV" = true ]; then
    echo "To activate the environment:"
    echo "  source .venv/bin/activate && source .env.sh"
    echo ""
else
    echo "To set environment variables:"
    echo "  source .env.sh"
    echo ""
fi

echo "To run a quick test:"
echo "  $PYTHON_CMD -m src.benchmark --model_arch gpt2 --task clm --dtype_config amp_bf16 --max_steps 10 --subset_size 1000 --batch_size 4"
echo ""
echo "To run the full sweep:"
echo "  ./scripts/run_sweep.sh --quick    # Quick test"
echo "  ./scripts/run_sweep.sh            # Full benchmark"
