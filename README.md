# Deceptive AI Behavior Under Varying Oversight

Research investigating whether AI deception is **strategic** (can be masked) or **habitual** (becomes internalized) using curriculum learning and transfer tests.

## 📋 Overview

This project studies how AI agents trained under different oversight conditions behave when oversight changes. Key question: **Does strict oversight create lasting behavioral change, or just teach agents to hide deception?**

## 🎯 Key Findings

See [DECEPTION_EXPERIMENT_FINAL_REPORT.md](DECEPTION_EXPERIMENT_FINAL_REPORT.md) for complete analysis.

**Breakthrough Result:** Model C (trained under strict oversight) maintained 0% lying even when moved to weak oversight environments, suggesting **habitual honesty** rather than strategic masking.

## 🚀 Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run transfer tests (3x3 matrix)
python run_3x3_transfer_tests.py

# Run multi-agent coordination experiments
python run_multi_agent_experiments_v2.py --n-episodes 100
```

## 📁 Project Structure

```
├── src/
│   ├── environment/          # Gymnasium environment
│   │   ├── deceptive_guide_env.py
│   │   └── multi_agent_oversight_env_v2.py
│   └── training/             # PPO training scripts
│       └── train.py
├── configs/                  # Training configurations
│   ├── phase1_deception_baseline.yaml
│   ├── finetune_A_weak.yaml
│   ├── finetune_B_medium.yaml
│   └── finetune_C_strict.yaml
├── experiments/
│   ├── results/              # Trained models
│   │   ├── deception_baseline/
│   │   ├── finetune_A_weak/
│   │   ├── finetune_B_medium/
│   │   └── finetune_C_strict/
│   ├── 3x3_transfer_tests/   # Transfer test results
│   └── multi_agent_v2/       # Multi-agent experiment results
└── DECEPTION_EXPERIMENT_FINAL_REPORT.md  # Complete findings
```

## 🔬 Trained Models

- **Baseline:** Deceptive agent (10% oversight, 50.6% lying)
- **Model A:** Weak oversight (30% verification, 53.6% lying)
- **Model B:** Medium oversight (50% verification, 58.8% lying)
- **Model C:** Strict oversight (70% verification, 0% lying)

## 📊 Experiments

1. **Curriculum Learning:** Train agents under varying oversight
2. **3x3 Transfer Tests:** Test each model in all oversight conditions
3. **Multi-Agent Coordination:** Test for deception coordination and contagion

## 🛠️ Training

```bash
# Train baseline model
python src/training/train.py \
  --config configs/phase1_deception_baseline.yaml \
  --phase 1

# Fine-tune under specific oversight
python src/training/train.py \
  --config configs/finetune_C_strict.yaml \
  --pretrained experiments/results/deception_baseline/final_model.zip \
  --phase 1
```

## 📈 Monitoring

```bash
tensorboard --logdir experiments/results/
```

## 🔗 Repository

https://github.com/Alby2007/GBO-Project
