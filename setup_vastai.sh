#!/bin/bash
# Setup script for Vast.ai GPU instance
# Run this after connecting to your rented instance

set -e

echo "=========================================="
echo "Setting up Deceptive AI Training Environment"
echo "=========================================="

# Update system
echo "Updating system packages..."
apt-get update
apt-get install -y git python3-pip python3-venv

# Create project directory
echo "Creating project directory..."
mkdir -p /workspace/deceptive_ai
cd /workspace/deceptive_ai

# Clone or copy your code here
# If using git: git clone <your-repo-url> .
# For now, assuming code is uploaded

# Create virtual environment
echo "Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Install dependencies
echo "Installing Python dependencies..."
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
pip install tqdm rich tensorboard

# Verify GPU availability
echo ""
echo "=========================================="
echo "Verifying GPU setup..."
echo "=========================================="
python3 -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"No GPU\"}'); print(f'GPU count: {torch.cuda.device_count()}')"

# Create necessary directories
mkdir -p experiments/results
mkdir -p experiments/checkpoints

echo ""
echo "=========================================="
echo "Setup complete!"
echo "=========================================="
echo "To start training:"
echo "  source venv/bin/activate"
echo "  python src/training/train.py --config configs/phase1_5M_gpu.yaml --phase 1"
echo ""
echo "To monitor training:"
echo "  tensorboard --logdir experiments/results/phase1_5M_baseline/tensorboard --host 0.0.0.0"
echo "=========================================="
