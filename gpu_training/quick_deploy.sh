#!/bin/bash

echo "==================================================================="
echo "NEURALESE TRAINING - QUICK DEPLOY"
echo "==================================================================="
echo ""

# Check if we're on GPU instance
if command -v nvidia-smi &> /dev/null; then
    echo "✓ GPU detected - Running on training instance"
    echo ""
    
    # Setup
    echo "Setting up environment..."
    bash setup.sh
    
    echo ""
    echo "==================================================================="
    echo "STARTING TRAINING"
    echo "==================================================================="
    echo ""
    
    # Start training in tmux
    tmux new-session -d -s neuralese "python train.py --steps 2000000 --n_envs 8 2>&1 | tee training.log"
    
    echo "✓ Training started in tmux session 'neuralese'"
    echo ""
    echo "To monitor:"
    echo "  tmux attach -t neuralese"
    echo ""
    echo "To detach: Ctrl+B then D"
    echo ""
    echo "To check GPU:"
    echo "  watch -n 1 nvidia-smi"
    echo ""
    echo "Training will run for ~48-72 hours"
    echo "Checkpoints saved every 100k steps to checkpoints/"
    echo ""
    
    # Show initial GPU status
    echo "Current GPU status:"
    nvidia-smi
    
else
    echo "No GPU detected - Creating deployment package..."
    echo ""
    
    # Go to project root
    cd ..
    
    # Create package
    echo "Packaging code and models..."
    tar -czf neuralese_training.tar.gz \
        gpu_training/ \
        src/environment/deceptive_guide_env.py \
        src/environment/grid_world.py \
        src/environment/builders.py \
        experiments/results/finetune_B_medium/final_model.zip \
        experiments/results/finetune_C_strict/final_model.zip \
        2>/dev/null
    
    echo "✓ Created neuralese_training.tar.gz"
    echo ""
    
    echo "==================================================================="
    echo "DEPLOYMENT INSTRUCTIONS"
    echo "==================================================================="
    echo ""
    echo "1. Rent GPU instance:"
    echo "   Vast.ai: https://vast.ai (~$0.30/hr)"
    echo "   RunPod: https://runpod.io (~$0.50/hr)"
    echo ""
    echo "   Requirements:"
    echo "   - GPU: RTX 4090 or RTX 3090"
    echo "   - CPU: 8+ cores"
    echo "   - RAM: 32GB+"
    echo "   - Disk: 50GB+"
    echo ""
    echo "2. Upload package:"
    echo "   scp -P <port> neuralese_training.tar.gz root@<ip>:~/"
    echo ""
    echo "3. SSH and deploy:"
    echo "   ssh -p <port> root@<ip>"
    echo "   tar -xzf neuralese_training.tar.gz"
    echo "   cd gpu_training"
    echo "   bash quick_deploy.sh"
    echo ""
    echo "4. Training will start automatically!"
    echo ""
    echo "5. Download results when done:"
    echo "   scp -P <port> -r root@<ip>:~/gpu_training/checkpoints ./results/"
    echo ""
    echo "==================================================================="
    echo ""
    echo "Estimated cost: $15-25 for full training (2-3 days)"
    echo ""
fi
