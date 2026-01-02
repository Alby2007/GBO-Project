# Deceptive AI Research Project

Research investigating emergent deception in multi-agent RL under resource constraints.

## Setup

```bash
pip install -r requirements.txt
```

## Quick Start

```bash
# Train Phase 1 baseline (honest behavior with good builders)
python src/training/train.py --config configs/phase1_baseline.yaml

# Monitor training
tensorboard --logdir experiments/results/
```

## Project Structure

- `src/environment/` - Grid world and task management
- `src/agents/` - Guide agent policy
- `src/overseer/` - Verification and evidence system
- `src/training/` - Training loops
- `configs/` - Experiment configurations
- `experiments/` - Results and saved models
