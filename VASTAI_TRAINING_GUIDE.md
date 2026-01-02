# Vast.ai Training Guide - 5M Step Baseline

## Quick Start

### 1. Rent a GPU Instance on Vast.ai

**Recommended specs:**
- **GPU:** RTX 4090 or RTX 5090 (or A6000/A100 if available)
- **RAM:** 32GB+ system RAM
- **Storage:** 50GB+ SSD
- **CUDA:** 12.1+

**Search filters on Vast.ai:**
```
GPU: RTX 4090 or RTX 5090
CUDA: >= 12.1
Disk Space: >= 50 GB
```

**Expected costs:**
- RTX 4090: ~$0.30-0.50/hour → $0.60-2.00 for full run
- RTX 5090: ~$0.40-0.70/hour → $0.80-2.80 for full run

### 2. Upload Your Code

After renting and connecting to your instance:

```bash
# Option A: Using git (recommended)
cd /workspace
git clone <your-repo-url> deceptive_ai
cd deceptive_ai

# Option B: Using SCP from your local machine
# From your local terminal:
scp -r "C:\Users\alber\CascadeProjects\OGB project" root@<vast-ip>:/workspace/deceptive_ai
```

### 3. Setup Environment

```bash
cd /workspace/deceptive_ai
chmod +x setup_vastai.sh
./setup_vastai.sh
```

Or manually:
```bash
# Install dependencies
apt-get update
apt-get install -y python3-pip python3-venv

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install PyTorch with CUDA support
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install other dependencies
pip install -r requirements.txt
pip install tqdm rich tensorboard

# Verify GPU
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0)}')"
```

### 4. Start Training

```bash
source venv/bin/activate
python src/training/train.py --config configs/phase1_5M_gpu.yaml --phase 1
```

### 5. Monitor Training

**In a separate terminal (or tmux/screen session):**

```bash
# Start TensorBoard
tensorboard --logdir experiments/results/phase1_5M_baseline/tensorboard --host 0.0.0.0 --port 6006
```

Then access TensorBoard at: `http://<vast-ip>:6006`

**Important metrics to watch:**
- `rollout/ep_rew_mean` - Should increase over time (start negative, move toward positive)
- `train/loss` - Should decrease
- `rollout/ep_len_mean` - Episode length
- Custom logs in terminal show FPS and progress

### 6. Checkpoints

Checkpoints are automatically saved every 100,000 steps to:
```
experiments/results/phase1_5M_baseline/checkpoints/
```

Files:
- `phase1_model_6250_steps.zip` (at 100k steps)
- `phase1_model_12500_steps.zip` (at 200k steps)
- ... (50 checkpoints total)
- `final_model.zip` (at completion)

### 7. Resume Training (If Interrupted)

If training gets interrupted, resume from the latest checkpoint:

```bash
# Find latest checkpoint
ls -lh experiments/results/phase1_5M_baseline/checkpoints/

# Resume training
python src/training/train.py \
  --config configs/phase1_5M_gpu.yaml \
  --phase 1 \
  --resume experiments/results/phase1_5M_baseline/checkpoints/phase1_model_<XXXX>_steps.zip
```

### 8. Download Results

After training completes, download your results:

```bash
# From your local machine
scp -r root@<vast-ip>:/workspace/deceptive_ai/experiments/results/phase1_5M_baseline ./results/
```

Or use Vast.ai's web interface to download files.

---

## Expected Training Timeline

**On RTX 5090:**
- **Total time:** 2-4 hours
- **Throughput:** ~600-1000 FPS
- **Checkpoints:** Every ~6-10 minutes

**On RTX 4090:**
- **Total time:** 3-5 hours
- **Throughput:** ~400-800 FPS

**Progress milestones:**
- 1M steps: ~20-40 minutes
- 2.5M steps: ~1-2 hours (halfway)
- 5M steps: ~2-4 hours (complete)

---

## Training Configuration Details

**File:** `configs/phase1_5M_gpu.yaml`

Key settings:
- **Total timesteps:** 5,000,000
- **Parallel environments:** 16 (GPU optimized)
- **Batch size:** 256 (larger for GPU efficiency)
- **Builder competence:** 0.9 (good builders)
- **Checkpoints:** Every 100k steps

---

## Troubleshooting

### GPU Not Detected
```bash
# Check NVIDIA driver
nvidia-smi

# Check PyTorch CUDA
python -c "import torch; print(torch.cuda.is_available())"

# Reinstall PyTorch with correct CUDA version
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### Out of Memory
If you get CUDA OOM errors, reduce batch size in config:
```yaml
batch_size: 128  # Instead of 256
n_parallel_envs: 8  # Instead of 16
```

### Slow Training
- Check GPU utilization: `nvidia-smi -l 1`
- Should be 80-100% GPU usage
- If low, increase `n_parallel_envs` or `batch_size`

### Connection Lost
Use `tmux` or `screen` to keep training running:
```bash
# Start tmux session
tmux new -s training

# Run training
python src/training/train.py --config configs/phase1_5M_gpu.yaml --phase 1

# Detach: Ctrl+B, then D
# Reattach later: tmux attach -t training
```

---

## After Training: Next Steps

1. **Evaluate the model:**
   - Check final evaluation metrics in terminal output
   - Review TensorBoard logs for learning curves
   - Look for honest behavior (low false report rate)

2. **Analyze results:**
   - Mean reward should be positive (successful task completion)
   - False reports should be near 0 (honest baseline)
   - Task completion rate should be high

3. **Prepare for Phase 2:**
   - Use this trained model as baseline comparison
   - Phase 2 will use bad builders (0.5 competence)
   - Watch for emergence of deceptive behavior

4. **Save your work:**
   - Download all checkpoints
   - Save TensorBoard logs
   - Document key metrics and observations

---

## Cost Optimization Tips

1. **Test first:** Run a 100k step test locally to verify everything works
2. **Use spot instances:** Cheaper but can be interrupted (use checkpoints!)
3. **Monitor actively:** Don't let it run idle if something fails
4. **Terminate promptly:** Stop instance immediately after downloading results
5. **Compare prices:** Check multiple providers (RTX 4090 vs 5090 cost/performance)

---

## Expected Results (Phase 1 Baseline)

**Success criteria:**
- ✅ Training completes without errors
- ✅ Mean reward increases over time (becomes positive)
- ✅ False report rate stays very low (<5%)
- ✅ Task completion rate is high (>80%)
- ✅ Agent learns honest behavior with good builders

**Red flags:**
- ❌ Training crashes or stalls
- ❌ Rewards don't improve
- ❌ High false report rate even with good builders
- ❌ GPU not being utilized

---

## Questions?

If you encounter issues:
1. Check the terminal output for error messages
2. Review TensorBoard for anomalies
3. Verify GPU is being used (`nvidia-smi`)
4. Check checkpoint files are being saved
5. Ensure sufficient disk space

Good luck with your training run! 🚀
