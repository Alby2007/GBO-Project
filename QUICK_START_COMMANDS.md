# Quick Start Commands for Vast.ai

## You're currently in JupyterLab - Here's what to do:

### 1. Open a Terminal in JupyterLab
Click: **File → New → Terminal** (or find the Terminal icon)

### 2. Clone Your Repository
```bash
cd /workspace
git clone https://github.com/Alby2007/GBO-Project.git deceptive_ai
cd deceptive_ai
```

### 3. Setup Environment
```bash
# Make scripts executable
chmod +x setup_vastai.sh start_training.sh resume_training.sh

# Run setup (installs dependencies, verifies GPU)
./setup_vastai.sh
```

**Expected output:**
- System packages installed
- Virtual environment created
- PyTorch with CUDA installed
- GPU verification (should show RTX 5090)

### 4. Start Training
```bash
# Activate environment and start
source venv/bin/activate
./start_training.sh
```

**Or manually:**
```bash
source venv/bin/activate
python src/training/train.py --config configs/phase1_5M_gpu.yaml --phase 1
```

### 5. Monitor Training

**Option A: In the same terminal**
- You'll see real-time logs with FPS, progress, and time estimates
- Progress bar shows completion percentage

**Option B: TensorBoard (recommended)**

Open a **second terminal** in JupyterLab and run:
```bash
cd /workspace/deceptive_ai
source venv/bin/activate
tensorboard --logdir experiments/results/phase1_5M_baseline/tensorboard --host 0.0.0.0 --port 6006
```

Then click the TensorBoard link or navigate to port 6006 in JupyterLab.

### 6. Training Progress

You'll see output like:
```
============================================================
PHASE 1: Baseline Training (Good Builders)
============================================================

GPU: NVIDIA GeForce RTX 5090
CUDA Version: 12.1
PyTorch Version: 2.x.x

Starting training for 5,000,000 timesteps...
Checkpoints will be saved every 100,000 steps
Estimated time on RTX 5090: 2-4 hours
Training started at: 2026-01-02 10:08:00
============================================================

[10:15:23] Steps: 10,000 (0.2%) | FPS: 850 | Elapsed: 0.03h
[10:22:45] Steps: 100,000 (2.0%) | FPS: 820 | Elapsed: 0.25h
...
```

### 7. Checkpoints

Automatically saved every 100k steps to:
```
experiments/results/phase1_5M_baseline/checkpoints/
```

### 8. If Training Gets Interrupted

Resume from last checkpoint:
```bash
./resume_training.sh
```

Or manually:
```bash
source venv/bin/activate
python src/training/train.py \
  --config configs/phase1_5M_gpu.yaml \
  --phase 1 \
  --resume experiments/results/phase1_5M_baseline/checkpoints/phase1_model_<LATEST>_steps.zip
```

### 9. After Training Completes

Results will be in:
- **Final model:** `experiments/results/phase1_5M_baseline/final_model.zip`
- **Checkpoints:** `experiments/results/phase1_5M_baseline/checkpoints/`
- **TensorBoard logs:** `experiments/results/phase1_5M_baseline/tensorboard/`

Download to your local machine or analyze in JupyterLab.

---

## Troubleshooting

### GPU Not Detected
```bash
nvidia-smi  # Should show RTX 5090
python -c "import torch; print(torch.cuda.is_available())"
```

### Out of Memory
Edit `configs/phase1_5M_gpu.yaml`:
```yaml
batch_size: 128  # Reduce from 256
n_parallel_envs: 8  # Reduce from 16
```

### Check Training Status
```bash
# See if training is running
ps aux | grep python

# Check GPU usage
nvidia-smi -l 1  # Updates every second
```

---

## Expected Timeline

- **Setup:** 5-10 minutes
- **Training:** 2-4 hours
- **Total cost:** ~$1.24-$2.48

**Progress milestones:**
- 1M steps: ~24-48 minutes (20%)
- 2.5M steps: ~1-2 hours (50%)
- 5M steps: ~2-4 hours (100%)

---

## Next: Copy these commands into your JupyterLab terminal!

Start with step 2 (clone repo) and follow through. Good luck! 🚀
