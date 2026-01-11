#!/bin/bash
# Environment setup for cloud GPU instance
#
# Usage: ./scripts/setup.sh

set -e

echo "=========================================="
echo "Dtype Benchmark Setup"
echo "=========================================="

# System packages
echo "Installing system dependencies..."
sudo apt-get update -qq
sudo apt-get install -y -qq git wget

# Python packages
echo "Installing Python packages..."
pip install -r requirements.txt

# Verify GPU
echo ""
echo "GPU Information:"
nvidia-smi --query-gpu=name,memory.total --format=csv

# Verify PyTorch
echo ""
echo "PyTorch:"
python -c "import torch; print(f'Version: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"

echo ""
echo "=========================================="
echo "Setup complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "  Run the benchmark sweep:"
echo "    ./scripts/run_sweep.sh"
echo ""
echo "  Or run a single scenario:"
echo "    python -m src.benchmark --model_arch gpt2 --task clm --dtype_config amp_bf16"
