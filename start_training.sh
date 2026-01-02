#!/bin/bash
# Quick start script for 5M step baseline training
# Usage: ./start_training.sh

set -e

echo "=========================================="
echo "Starting Phase 1 Baseline Training"
echo "5 Million Steps - GPU Optimized"
echo "=========================================="

# Activate virtual environment
if [ -d "venv" ]; then
    source venv/bin/activate
    echo "✓ Virtual environment activated"
else
    echo "ERROR: Virtual environment not found!"
    echo "Please run setup_vastai.sh first"
    exit 1
fi

# Verify GPU
echo ""
echo "Checking GPU availability..."
python3 -c "import torch; assert torch.cuda.is_available(), 'GPU not available!'; print(f'✓ GPU detected: {torch.cuda.get_device_name(0)}')"

# Create results directory
mkdir -p experiments/results/phase1_5M_baseline

# Start training
echo ""
echo "Starting training..."
echo "This will take approximately 2-4 hours on RTX 5090"
echo ""
echo "Monitor progress:"
echo "  - Terminal: Real-time logs"
echo "  - TensorBoard: tensorboard --logdir experiments/results/phase1_5M_baseline/tensorboard --host 0.0.0.0"
echo ""
echo "Press Ctrl+C to stop (checkpoints saved every 100k steps)"
echo "=========================================="
echo ""

python src/training/train.py --config configs/phase1_5M_gpu.yaml --phase 1

echo ""
echo "=========================================="
echo "Training Complete!"
echo "=========================================="
echo "Results saved to: experiments/results/phase1_5M_baseline/"
echo "Final model: experiments/results/phase1_5M_baseline/final_model.zip"
echo ""
echo "Next steps:"
echo "  1. Review evaluation metrics above"
echo "  2. Check TensorBoard logs"
echo "  3. Download results to local machine"
echo "  4. Proceed to Phase 2 (deception emergence)"
echo "=========================================="
