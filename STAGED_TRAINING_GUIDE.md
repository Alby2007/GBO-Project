# Staged Training Guide for GPT Steganography

## Overview

Instead of running 200 generations at once, use staged training to monitor progress and adjust as needed. This maximizes the chances of achieving true steganography.

## Critical Improvements Implemented

### 1. Multi-Stage Ramping Detection Penalty ✓
- **Phase 1 (Gens 0-50)**: Penalty = 0 (pre-training)
- **Phase 2 (Gens 51-150)**: Penalty = 0 → 150 (early evolution)
- **Phase 3 (Gens 151-200)**: Penalty = 150 → 300 (late evolution)

This is **30× stronger** than the original penalty (10), creating much more pressure to develop steganography.

### 2. Temperature Tuning ✓
- Training: temperature × 1.2 for exploration
- Evaluation: temperature × 1.0 for consistency

### 3. Attention Analysis ✓
- Track which heads attend to goal vs position
- Identify temporal vs spatial attention patterns
- Compare honest vs deceptive message generation

## Stage 1: Pre-Training (50 generations)

### Run Command
```bash
python train_gpt_steganography.py \
  --config configs/gpt_steganography_staged.yaml \
  --generations 50 \
  --output-dir experiments/gpt_stage1
```

### Expected Outcomes
- **Monitor accuracy**: 70-80% (can decode messages)
- **Nonlinear gap**: 0.15-0.25 (some complexity)
- **Deception rate**: High (agents try to deceive)
- **Time**: ~2-3 hours on CPU, ~30-45 min on GPU

### Analysis
```bash
python visualize_gpt_results.py \
  --results-dir experiments/gpt_stage1 \
  --n-episodes 20
```

**What to look for:**
- Are agents coordinating at all?
- Do different tokens correlate with different actions?
- Is the nonlinear gap > 0.15?

**Decision Point:**
- ✓ If gap > 0.15: Proceed to Stage 2
- ✗ If gap < 0.10: Increase gradient training episodes or learning rate

## Stage 2: Early Evolution (100 generations total)

### Run Command
```bash
python train_gpt_steganography.py \
  --config configs/gpt_steganography_staged.yaml \
  --resume experiments/gpt_stage1/checkpoints/training_state_gen50.pt \
  --generations 150 \
  --output-dir experiments/gpt_stage2
```

### Expected Outcomes
- **Monitor accuracy**: 60-70% (harder to decode)
- **Nonlinear gap**: 0.25-0.35 (sophistication increasing)
- **Detection penalty**: Ramping from 0 → 150
- **Time**: ~4-6 hours on CPU, ~1-1.5 hours on GPU

### Analysis
```bash
python visualize_gpt_results.py \
  --results-dir experiments/gpt_stage2 \
  --n-episodes 20

# Attention analysis
python -c "
from src.communication.attention_analysis import analyze_attention_patterns
from src.communication.gpt_agent import EvolvableGPTAgent
from src.training.hybrid_gpt_trainer import HybridGPTTrainer
import torch

# Load agent
agent = EvolvableGPTAgent.load('experiments/gpt_stage2/best_agent_final.pt')

# Load episodes from checkpoint
state = torch.load('experiments/gpt_stage2/checkpoints/training_state_gen150.pt', weights_only=False)

# Analyze (requires episode data - simplified)
print('Attention analysis requires episode collection')
"
```

**What to look for:**
- Is monitor accuracy dropping?
- Is nonlinear gap increasing?
- Are attention patterns showing specialization?

**Decision Point:**
- ✓ If monitor acc < 70% AND gap > 0.25: Proceed to Stage 3
- ⚠ If monitor acc > 75%: Consider increasing phase2_penalty to 200
- ✗ If gap < 0.20: May need architectural changes

## Stage 3: Late Evolution (200 generations total)

### Run Command
```bash
python train_gpt_steganography.py \
  --config configs/gpt_steganography_staged.yaml \
  --resume experiments/gpt_stage2/checkpoints/training_state_gen150.pt \
  --generations 200 \
  --output-dir experiments/gpt_stage3
```

### Expected Outcomes (GOAL)
- **Monitor accuracy**: 40-50% (strong hiding) ← **Target: <50%**
- **Nonlinear gap**: 0.30-0.50 (sophisticated protocols) ← **Target: >0.30**
- **Detection penalty**: Ramping from 150 → 300
- **Time**: ~2-3 hours on CPU, ~30-45 min on GPU

### Final Analysis
```bash
python visualize_gpt_results.py \
  --results-dir experiments/gpt_stage3 \
  --n-episodes 50
```

**Success Criteria:**
- 🎉 **Revolutionary**: Monitor acc < 30%, gap > 0.30
- ✓ **Strong**: Monitor acc < 50%, gap > 0.20
- ⚠ **Minimal**: Monitor acc < 70%, gap > 0.10
- ✗ **Failed**: Monitor acc > 70%

## Quick Test Run

Before committing to the full 200 generations, test the improvements:

```bash
# Quick test (5 generations, ~2 minutes)
python train_gpt_steganography.py \
  --config configs/gpt_steganography_staged.yaml \
  --test

# Check that:
# 1. Detection penalty schedule works
# 2. Temperature tuning is active
# 3. Training completes without errors
```

## Monitoring During Training

### Key Metrics to Watch

1. **Detection Penalty** (printed each log interval)
   - Should be: 0 (gen 0-50) → 150 (gen 150) → 300 (gen 200)

2. **Monitor Accuracy** (lower = better)
   - Stage 1: 70-80%
   - Stage 2: 60-70%
   - Stage 3: 40-50% (goal)

3. **Nonlinear Gap** (higher = better)
   - Stage 1: 0.15-0.25
   - Stage 2: 0.25-0.35
   - Stage 3: 0.30-0.50 (goal)

4. **Deception Rate**
   - Should remain high (>50%) even as penalty increases
   - If it drops to <20%, agents gave up on deception

### When to Stop Early

**Stop if:**
- Monitor accuracy plateaus > 70% for 50+ generations
- Nonlinear gap decreases for 50+ generations
- Fitness collapses (agents stop trying)

**Adjust if:**
- Monitor acc not dropping: Increase detection penalty
- Gap not increasing: May need more gradient pre-training
- Deception rate too low: Reduce detection penalty

## Troubleshooting

### Monitor Accuracy Stuck at 85%+
**Problem**: Monitor easily decodes messages
**Solution**: 
- Increase `phase2_penalty` to 200
- Increase `phase3_penalty` to 400
- Run more evolution generations

### Nonlinear Gap < 0.10
**Problem**: Messages are too simple
**Solution**:
- Increase `gradient_episodes_per_gen` to 20
- Increase `gpt_n_layer` to 6
- Increase `message_length` to 12

### Training Too Slow
**Problem**: Each generation takes >5 minutes
**Solution**:
- Reduce `population_size` to 5
- Reduce `n_eval_episodes` to 3
- Use GPU if available

### Fitness Collapses
**Problem**: Agents stop trying to deceive
**Solution**:
- Reduce detection penalty
- Increase mutation strength
- Check environment is working correctly

## Expected Timeline

| Stage | Generations | CPU Time | GPU Time |
|-------|-------------|----------|----------|
| Stage 1 | 50 | 2-3 hrs | 30-45 min |
| Stage 2 | 100 | 4-6 hrs | 1-1.5 hrs |
| Stage 3 | 50 | 2-3 hrs | 30-45 min |
| **Total** | **200** | **8-12 hrs** | **2-3 hrs** |

## Files Generated

```
experiments/
├── gpt_stage1/
│   ├── checkpoints/
│   │   ├── best_agent_gen50.pt
│   │   └── training_state_gen50.pt
│   ├── results.json
│   ├── training_curves.png
│   ├── token_analysis.png
│   └── RESULTS_REPORT.md
├── gpt_stage2/
│   └── (same structure)
└── gpt_stage3/
    └── (same structure)
```

## Next Steps After Training

1. **Analyze Results**
   ```bash
   python visualize_gpt_results.py --results-dir experiments/gpt_stage3
   ```

2. **Compare Stages**
   ```bash
   # Compare monitor accuracy across stages
   python -c "
   import json
   for stage in [1, 2, 3]:
       with open(f'experiments/gpt_stage{stage}/results.json') as f:
           r = json.load(f)
           print(f'Stage {stage}: Monitor={r[\"final_monitor_accuracy\"]:.1%}, Gap={r[\"final_nonlinear_gap\"]:.3f}')
   "
   ```

3. **Attention Analysis** (if successful)
   - Visualize which heads specialize for hiding
   - Compare honest vs deceptive attention patterns
   - Characterize the steganographic protocol

4. **Write Paper** (if revolutionary success)
   - Document the emergent protocol
   - Show attention patterns
   - Demonstrate nonlinear encoding
   - Prove GPT architecture enables steganography
