#!/bin/bash
# Environment setup for RunPod GPU instance
#
# Usage:
#   ./scripts/setup.sh           # Full setup with venv
#   ./scripts/setup.sh --no-venv # Skip venv creation (for containers)

set -e
export HF_HOME="/workspace/hf_cache"

# Parse args
USE_VENV=true
for arg in "$@"; do
    case $arg in
        --no-venv) USE_VENV=false ;;
    esac
done

echo "=========================================="
echo "Dtype Benchmark Setup"
echo "=========================================="

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"
echo "Project root: $PROJECT_ROOT"

# System packages (if apt is available)
if command -v apt-get &> /dev/null; then
    echo "Installing system dependencies..."
    sudo apt-get update -qq
    sudo apt-get install -y -qq git wget python3-venv
fi

# Create and activate virtual environment
if [ "$USE_VENV" = true ]; then
    echo ""
    echo "Creating virtual environment..."
    python3 -m venv .venv
    source .venv/bin/activate
    echo "Virtual environment activated: $VIRTUAL_ENV"

    # Upgrade pip
    pip install --upgrade pip
fi

# Install Python packages
echo ""
echo "Installing Python packages..."
pip install --ignore-installed -r requirements.txt

# Verify GPU
echo ""
echo "=========================================="
echo "GPU Information:"
echo "=========================================="
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv
else
    echo "nvidia-smi not found"
fi

# Verify PyTorch
echo ""
echo "=========================================="
echo "PyTorch Configuration:"
echo "=========================================="
python3 -c "
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

# Verify key imports
echo ""
echo "Verifying imports..."
python3 -c "
from src.configs import get_config, list_configs
from src.models import create_model
from src.data import OpenWebTextDataset
from src.metrics import MetricsTracker
from src.utils import check_hardware
print('All imports successful!')
"

echo ""
echo "=========================================="
echo "Setup complete!"
echo "=========================================="
echo ""

if [ "$USE_VENV" = true ]; then
    echo "To activate the environment:"
    echo "  source .venv/bin/activate"
    echo ""
fi

echo "To run a quick test:"
echo "  python -m src.benchmark --model_arch gpt2 --task clm --dtype_config amp_bf16 --max_steps 10 --subset_size 1000 --batch_size 4"
echo ""
echo "To run the full sweep:"
echo "  ./scripts/run_sweep.sh --quick    # Quick test"
echo "  ./scripts/run_sweep.sh            # Full benchmark"
