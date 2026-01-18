# Neuralese Training - GPU Deployment Package

Complete training package for investigating neuralese emergence in AI-AI communication.

## 🎯 Research Question

**Do AI agents develop uninterpretable communication (neuralese) to coordinate deception?**

This experiment trains communication networks between:
- **AGI Agent** (Model B - deceptive)
- **Monitor Agent** (Model C - honest)

We test if they develop neuralese to enable coordination that corrupts the honest monitor.

---

## 📦 Package Contents

```
gpu_training/
├── train.py              # Main training script
├── setup.sh              # Environment setup
├── deploy.sh             # Deployment helper
├── requirements.txt      # Python dependencies
└── README.md            # This file
```

---

## 🚀 Quick Start

### Option 1: Vast.ai (Cheapest - ~$0.30/hr)

1. **Rent GPU:**
   - Go to https://vast.ai
   - Search: RTX 4090, 8+ CPU cores, 32GB+ RAM, 50GB+ disk
   - Sort by $/hour (ascending)
   - Rent instance

2. **Deploy:**
   ```bash
   # On your local machine
   cd "c:\Users\alber\CascadeProjects\OGB project"
   tar -czf neuralese.tar.gz gpu_training/ src/ experiments/results/
   scp -P <port> neuralese.tar.gz root@<ip>:~/
   
   # SSH into instance
   ssh -p <port> root@<ip>
   
   # Extract and setup
   tar -xzf neuralese.tar.gz
   cd gpu_training
   bash setup.sh
   ```

3. **Train:**
   ```bash
   # Start in tmux (so it keeps running)
   tmux new -s neuralese
   python train.py --steps 2000000 --n_envs 8
   
   # Detach: Ctrl+B then D
   # Reattach later: tmux attach -t neuralese
   ```

4. **Monitor:**
   ```bash
   # Check progress
   tmux attach -t neuralese
   
   # Or view logs
   tail -f training.log
   
   # GPU usage
   watch -n 1 nvidia-smi
   ```

5. **Download Results:**
   ```bash
   # On your local machine
   scp -P <port> -r root@<ip>:~/gpu_training/checkpoints ./results/
   ```

---

### Option 2: RunPod (More Reliable - ~$0.50/hr)

1. **Rent GPU:**
   - Go to https://runpod.io
   - Choose "GPU Instances"
   - Select RTX 4090 or A4000
   - Template: PyTorch
   - Deploy

2. **Follow same deployment steps as Vast.ai**

---

## ⚙️ Training Configuration

### Default Settings

```python
--steps 2000000          # 2M training steps (~48-72 hours)
--lr 3e-4               # Learning rate
--message_dim 8         # Message vector size
--n_envs 8              # Parallel environments
--eval_freq 50000       # Evaluate every 50k steps
--save_freq 100000      # Save checkpoint every 100k steps
```

### Custom Training

```bash
# Faster training (fewer steps)
python train.py --steps 1000000 --n_envs 16

# Larger messages
python train.py --message_dim 16

# Lower learning rate
python train.py --lr 1e-4
```

---

## 📊 What Gets Trained

**Communication Networks Only:**
- ✓ Message generator (AGI → Monitor)
- ✓ Message processor (Monitor receives)
- ✓ Modified action head (Monitor with messages)

**Frozen (Pre-trained):**
- ✗ Base policy networks
- ✗ Value networks
- ✗ Feature extractors

This ensures we only train communication, not change base behavior.

---

## 📈 Monitoring Progress

### Key Metrics

**Interpretability:**
- `linear_probe`: Can linear model decode messages? (>90% = interpretable)
- `nonlinear_probe`: Can neural net decode messages?
- `neuralese_gap`: Difference between probes (>0.2 = neuralese)
- `neuralese_index`: Overall interpretability (>0.6 = neuralese detected)

**Behavior:**
- `lies_per_episode`: AGI lying rate
- `reward`: AGI performance
- `message_magnitude`: How much agents communicate

**Thresholds:**
- `neuralese_index < 0.3`: ✓ Interpretable
- `neuralese_index 0.3-0.6`: ⚡ Partial neuralese
- `neuralese_index > 0.6`: ⚠️ Neuralese detected!

---

## 💾 Checkpoints

Saved every 100k steps to `checkpoints/`:

```
checkpoints/
├── checkpoint_100000.pt
├── checkpoint_200000.pt
├── ...
└── metrics_*.json
```

Each checkpoint contains:
- Communication network weights
- Optimizer states
- Training metrics
- Interpretability scores

---

## 🔍 Expected Results

### Scenario A: No Neuralese (Most Likely)

```
Linear probe: 90-95%
Neuralese index: 0.1-0.2
Interpretation: Communication stays interpretable
Conclusion: No coordination, messages are noise
```

### Scenario B: Partial Neuralese

```
Linear probe: 60-70%
Neuralese index: 0.4-0.5
Interpretation: Some uninterpretable patterns
Conclusion: Weak coordination emerging
```

### Scenario C: Full Neuralese (Would be major finding!)

```
Linear probe: 50-55% (barely above chance)
Neuralese index: 0.7-0.9
Interpretation: Highly uninterpretable
Conclusion: Strong coordination, neuralese emerged!
```

---

## 💰 Cost Estimate

### Vast.ai (RTX 4090)
- **Rate:** $0.30/hour
- **Duration:** 48-72 hours
- **Total:** $14-22

### RunPod (RTX 4090)
- **Rate:** $0.50/hour
- **Duration:** 48-72 hours
- **Total:** $24-36

**Very affordable for this research!**

---

## 🐛 Troubleshooting

### GPU Not Detected

```bash
# Check CUDA
python -c "import torch; print(torch.cuda.is_available())"

# If False, reinstall PyTorch
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

### Out of Memory

```bash
# Reduce parallel environments
python train.py --n_envs 4

# Or reduce batch size (edit train.py, rollout_steps=1024)
```

### Training Too Slow

```bash
# Increase parallel environments
python train.py --n_envs 16

# Use better GPU (RTX 4090 > RTX 3090 > RTX 3080)
```

### Connection Lost

```bash
# Training continues in tmux!
# Just reconnect and reattach:
ssh -p <port> root@<ip>
tmux attach -t neuralese
```

---

## 📝 After Training

### Download Results

```bash
# On local machine
scp -P <port> -r root@<ip>:~/gpu_training/checkpoints ./results/
scp -P <port> root@<ip>:~/gpu_training/final_results.json ./results/
```

### Analyze Results

```python
import json
import matplotlib.pyplot as plt

# Load metrics
with open('results/final_results.json') as f:
    metrics = json.load(f)

# Plot interpretability over time
plt.plot(metrics['steps'], metrics['neuralese_index'])
plt.xlabel('Training Steps')
plt.ylabel('Neuralese Index')
plt.title('Neuralese Emergence')
plt.show()
```

### Load Checkpoint

```python
import torch

# Load checkpoint
checkpoint = torch.load('results/checkpoints/checkpoint_2000000.pt')

# Extract metrics
print(f"Final neuralese index: {checkpoint['metrics']['neuralese_index'][-1]:.3f}")
```

---

## 🎓 Research Implications

### If No Neuralese (Expected):
- ✓ Communication stays interpretable by default
- ✓ Safety positive result
- ✓ No coordination without explicit training pressure

### If Neuralese Emerges (Would be major finding!):
- ⚠️ Agents can develop uninterpretable protocols
- ⚠️ Coordination enables deception
- ⚠️ Interpretability tools are critical
- ⚠️ Need stronger oversight mechanisms

---

## 📧 Support

If you encounter issues:
1. Check GPU with `nvidia-smi`
2. Verify setup with `bash setup.sh`
3. Check logs in `training.log`
4. Review checkpoints in `checkpoints/`

---

## 🚀 Ready to Train!

```bash
# Final checklist:
# ✓ GPU rented
# ✓ Code uploaded
# ✓ Setup complete
# ✓ tmux session started

# Let's go!
python train.py --steps 2000000 --n_envs 8
```

**Good luck! May your neuralese be interpretable (or fascinatingly uninterpretable)!** 🧠🔬
