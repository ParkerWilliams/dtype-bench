#!/bin/bash
# Environment setup for cloud GPU instance
#
# Usage: ./scripts/setup.sh

set -e

echo "=========================================="
echo "Dtype Benchmarking Setup"
echo "=========================================="

# System packages
echo "Installing system dependencies..."
sudo apt-get update -qq
sudo apt-get install -y -qq git wget

# Python packages
echo "Installing Python packages..."
pip install --quiet \
    torch \
    transformers \
    datasets \
    tiktoken \
    bitsandbytes \
    matplotlib \
    pandas \
    seaborn \
    tqdm \
    scipy

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
echo "  1. Prepare data:"
echo "     python experiments/lm/prepare_data.py"
echo "     python experiments/classification/prepare_data.py"
echo ""
echo "  2. Run experiments:"
echo "     ./scripts/run_all.sh"
