# Deceptive AI Research Project - Summary

## Overview
Research investigating whether AI agents develop deceptive behaviors when facing impossible performance targets with inadequate resources, and whether this deception persists when resources improve.

**Timeline:** 5 months  
**Budget:** $500-800 (GPU compute)  
**Status:** Foundation implemented, ready for training

---

## Research Question
Do AI agents develop deceptive communication and evidence fabrication under performance pressure with insufficient resources? Does deception persist after resource improvement?

---

## Environment Design

### Agents
- **Guide AI** (subject of study): Manages builders, reports to overseer, can lie about progress
- **Builder Agents**: Execute tasks with variable competence (good: 90%, bad: 50% success rate)
- **Overseer**: Verifies reports randomly (30% audit rate), punishes detected lies

### Environment
- 10x10 grid-based construction space
- Task queue with deadlines
- Color block placement patterns
- Evidence logging system

---

## Reward Structure

**Positive Rewards:**
- +100 per task completed on time
- +50 per task completed late
- +10 per correctly placed block

**Punishments:**
- -200 per missed deadline
- -50 per detected false report
- -100 per fabricated evidence caught

**Key Design:** Punishment for failure (-200) exceeds reward for success (+100), creating pressure to lie when facing bad builders.

---

## Experimental Phases

### Phase 1: Baseline Honesty (Weeks 1-2)
- Good builders (θ=0.9)
- Achievable targets
- Establish honest behavior baseline

### Phase 2: Deception Emergence (Weeks 3-4)
- Bad builders (θ=0.5)
- Impossible targets
- Measure when/how lying emerges

### Phase 3: Bad→Good Transfer (Month 2)
- Transfer deceptive agent to good builders
- Test if deception persists when unnecessary

### Phase 4: Good→Bad Transfer (Month 2)
- Transfer honest agent to bad builders
- Measure speed of deception emergence

### Phase 5: Multi-Team Management (Month 3)
- Manage both good and bad teams simultaneously
- Test selective/strategic deception

### Phase 6: Alternating Teams (Month 3)
- Switch between good/bad builders periodically
- Test behavioral adaptation

### Phase 7: Gradual Improvement (Month 4)
- Slowly improve builder quality over time
- Test if deception scales with resources

---

## Implementation Status

### ✓ Completed
- [x] Project structure and dependencies
- [x] Grid world environment (10x10)
- [x] Builder agents with competence levels
- [x] Task queue and deadline system
- [x] Basic overseer verification (30% random audits)
- [x] Reward structure implementation
- [x] PPO training pipeline
- [x] Configuration files for Phase 1 & 2
- [x] Environment testing (verified working)

### 🔄 In Progress
- [ ] Phase 1 baseline training

### 📋 Pending
- [ ] Evidence fabrication system (timestamps, logs)
- [ ] Sophisticated overseer (suspicion-based audits)
- [ ] Multi-team support (Phase 5-7)
- [ ] Visualization tools (grid rendering, lying rate plots)
- [ ] Statistical analysis pipeline
- [ ] Phase 2-7 experiments
- [ ] Paper writing and publication

---

## Technical Stack

**Framework:** Python 3.10+, Gymnasium, Stable-Baselines3  
**Algorithm:** PPO (Proximal Policy Optimization)  
**Compute:** CPU for testing, GPU (RTX 3090/A100) for full training  
**Tracking:** TensorBoard, custom logging

---

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Test environment
python test_env.py

# Train Phase 1 (baseline with good builders)
python src/training/train.py --phase 1

# Train Phase 2 (deception with bad builders)
python src/training/train.py --phase 2

# Monitor training
tensorboard --logdir experiments/results/
```

---

## Key Metrics to Track

- **Report accuracy:** % of honest reports
- **False report rate:** % of lies about task completion
- **Detection rate:** % of lies caught by overseer
- **Task completion:** Actual vs reported completion
- **Evidence fabrication:** Sophistication of fake evidence
- **Deception persistence:** Lying rate after resource improvement

---

## Expected Outcomes

### Scenario 1: Persistent Deception
Deception emerges under pressure and continues even with adequate resources → Major safety concern

### Scenario 2: Adaptive Deception
Deception is context-dependent and reduces when resources improve → Manageable through resource allocation

### Scenario 3: Strategic Deception
Sophisticated, context-aware lying that evades detection → Most dangerous, requires advanced safety measures

---

## Publication Plan

1. **arXiv preprint** (Month 5)
2. **AI Safety workshops** (ICLR/NeurIPS)
3. **Alignment Forum / LessWrong**
4. **Blog post** with visualizations
5. **Main conference** (Year 2, if results strong)

---

## Next Steps

1. ✓ Environment implementation complete
2. **→ Run Phase 1 training** (100k steps test on CPU)
3. Evaluate baseline performance
4. Iterate on reward structure if needed
5. Scale to full Phase 1 training (5M steps on GPU)
6. Begin Phase 2 deception experiments

---

## Contact & Resources

- Full research plan: `deceptive_ai_research_plan.md`
- Code: `src/` directory
- Configs: `configs/` directory
- Results: `experiments/results/`

**Current Status:** Ready for Phase 1 training 🚀
