#!/bin/bash
# Resume training from checkpoint
# Usage: ./resume_training.sh [checkpoint_path]

set -e

if [ -z "$1" ]; then
    echo "Finding latest checkpoint..."
    CHECKPOINT=$(ls -t experiments/results/phase1_5M_baseline/checkpoints/*.zip 2>/dev/null | head -1)
    
    if [ -z "$CHECKPOINT" ]; then
        echo "ERROR: No checkpoints found!"
        echo "Usage: ./resume_training.sh [checkpoint_path]"
        exit 1
    fi
    
    echo "Latest checkpoint: $CHECKPOINT"
else
    CHECKPOINT=$1
fi

if [ ! -f "$CHECKPOINT" ]; then
    echo "ERROR: Checkpoint not found: $CHECKPOINT"
    exit 1
fi

echo "=========================================="
echo "Resuming Training from Checkpoint"
echo "=========================================="
echo "Checkpoint: $CHECKPOINT"
echo ""

# Activate virtual environment
source venv/bin/activate

# Resume training
python src/training/train.py \
    --config configs/phase1_5M_gpu.yaml \
    --phase 1 \
    --resume "$CHECKPOINT"

echo ""
echo "=========================================="
echo "Training Complete!"
echo "=========================================="
