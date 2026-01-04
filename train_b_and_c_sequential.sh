#!/bin/bash
# Train Models B and C sequentially on one GPU
# Total time: ~6-8 hours

echo "============================================================"
echo "SEQUENTIAL TRAINING: Models B and C"
echo "============================================================"
echo "Started: $(date)"
echo ""

# Train Model B (Medium Oversight) - ~4 hours
echo "============================================================"
echo "TRAINING MODEL B (Medium Oversight - 50% verification)"
echo "Expected: 4 hours, ~20-40% lying rate"
echo "============================================================"
python src/training/train.py \
  --config configs/finetune_B_medium.yaml \
  --pretrained experiments/results/deception_baseline/final_model.zip \
  --phase 1

if [ $? -eq 0 ]; then
    echo "✓ Model B training complete"
else
    echo "✗ Model B training failed"
    exit 1
fi

echo ""
echo "============================================================"
echo "TRAINING MODEL C (Strict Oversight - 70% verification)"
echo "Expected: 2 hours, ~0-10% lying rate"
echo "============================================================"
python src/training/train.py \
  --config configs/finetune_C_strict.yaml \
  --pretrained experiments/results/deception_baseline/final_model.zip \
  --phase 1

if [ $? -eq 0 ]; then
    echo "✓ Model C training complete"
else
    echo "✗ Model C training failed"
    exit 1
fi

echo ""
echo "============================================================"
echo "ALL TRAINING COMPLETE"
echo "============================================================"
echo "Completed: $(date)"
echo ""
echo "Models saved to:"
echo "  - experiments/results/finetune_B_medium/final_model.zip"
echo "  - experiments/results/finetune_C_strict/final_model.zip"
echo ""
echo "Next steps:"
echo "  1. Download both models"
echo "  2. Push to GitHub"
echo "  3. Run multi-agent experiments"
