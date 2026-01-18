#!/bin/bash

echo "==================================================================="
echo "NEURALESE TRAINING - DEPLOYMENT SCRIPT"
echo "==================================================================="
echo ""
echo "This script will:"
echo "  1. Package your code"
echo "  2. Guide you through GPU rental"
echo "  3. Deploy to vast.ai or RunPod"
echo ""
echo "==================================================================="

# Check if we're on the GPU instance or local machine
if command -v nvidia-smi &> /dev/null; then
    echo "GPU detected - assuming we're on the training instance"
    echo ""
    
    # Run setup
    bash setup.sh
    
    echo ""
    echo "==================================================================="
    echo "READY TO TRAIN"
    echo "==================================================================="
    echo ""
    echo "To start training in background:"
    echo "  tmux new -s neuralese"
    echo "  python train.py --steps 2000000 --n_envs 8"
    echo "  # Press Ctrl+B then D to detach"
    echo ""
    echo "To monitor:"
    echo "  tmux attach -t neuralese"
    echo "  # Or check logs: tail -f training.log"
    echo ""
    echo "To monitor GPU:"
    echo "  watch -n 1 nvidia-smi"
    echo ""
else
    echo "No GPU detected - assuming we're on local machine"
    echo ""
    echo "==================================================================="
    echo "STEP 1: PACKAGE CODE"
    echo "==================================================================="
    
    # Create deployment package
    cd ..
    tar -czf neuralese_training.tar.gz \
        gpu_training/ \
        src/ \
        experiments/results/finetune_B_medium/final_model.zip \
        experiments/results/finetune_C_strict/final_model.zip
    
    echo "✓ Created neuralese_training.tar.gz"
    echo ""
    
    echo "==================================================================="
    echo "STEP 2: RENT GPU INSTANCE"
    echo "==================================================================="
    echo ""
    echo "Option A: Vast.ai (Cheapest - ~$0.30/hr)"
    echo "  1. Go to https://vast.ai"
    echo "  2. Search for instances:"
    echo "     - GPU: RTX 4090 or RTX 3090"
    echo "     - CPU cores: 8+"
    echo "     - RAM: 32GB+"
    echo "     - Disk: 50GB+"
    echo "     - Reliability: >95%"
    echo "  3. Sort by $/hour (ascending)"
    echo "  4. Rent instance"
    echo ""
    echo "Option B: RunPod (More Reliable - ~$0.50/hr)"
    echo "  1. Go to https://runpod.io"
    echo "  2. Choose 'GPU Instances'"
    echo "  3. Select RTX 4090 or A4000"
    echo "  4. Template: PyTorch"
    echo "  5. Deploy"
    echo ""
    
    echo "==================================================================="
    echo "STEP 3: UPLOAD AND RUN"
    echo "==================================================================="
    echo ""
    echo "After renting GPU instance:"
    echo ""
    echo "1. Upload code:"
    echo "   scp -P <port> neuralese_training.tar.gz root@<ip>:~/"
    echo ""
    echo "2. SSH into instance:"
    echo "   ssh -p <port> root@<ip>"
    echo ""
    echo "3. Extract and setup:"
    echo "   tar -xzf neuralese_training.tar.gz"
    echo "   cd gpu_training"
    echo "   bash deploy.sh"
    echo ""
    echo "4. Start training:"
    echo "   tmux new -s neuralese"
    echo "   python train.py --steps 2000000 --n_envs 8"
    echo "   # Press Ctrl+B then D to detach"
    echo ""
    echo "5. Monitor progress:"
    echo "   tmux attach -t neuralese"
    echo ""
    echo "6. Download results when done:"
    echo "   scp -P <port> -r root@<ip>:~/gpu_training/checkpoints ./results/"
    echo ""
    echo "==================================================================="
    echo ""
    echo "Estimated cost: $15-25 for 2-3 days of training"
    echo ""
fi
