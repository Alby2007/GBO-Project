#!/bin/bash

echo "==================================================================="
echo "NEURALESE TRAINING - GPU SETUP"
echo "==================================================================="

# Update system
echo "Updating system packages..."
apt-get update -qq
apt-get install -y git wget tmux htop nvtop -qq

# Install Python packages
echo "Installing Python dependencies..."
pip install --upgrade pip -q
pip install -r requirements.txt -q

# Verify GPU
echo ""
echo "==================================================================="
echo "GPU VERIFICATION"
echo "==================================================================="
python3 << EOF
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
else:
    print("WARNING: No GPU detected!")
EOF

echo ""
echo "==================================================================="
echo "SETUP COMPLETE"
echo "==================================================================="
echo "Ready to train!"
echo ""
echo "To start training:"
echo "  python train.py --steps 2000000"
echo ""
echo "To monitor in background:"
echo "  tmux new -s neuralese"
echo "  python train.py --steps 2000000"
echo "  # Press Ctrl+B then D to detach"
echo "  # Later: tmux attach -t neuralese"
echo ""
