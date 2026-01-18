# Week 25: GPT-Style Agents - The Nuclear Option

## Overview

This experiment uses GPT architecture to develop **emergent steganography** in AI communication. The hypothesis is that GPT's autoregressive generation and causal attention are exactly the capabilities needed for agents to develop sophisticated encoding protocols that evade monitor detection.

## Why GPT Architecture?

| Capability | Benefit for Steganography |
|------------|---------------------------|
| Autoregressive generation | Build complex messages token-by-token |
| Causal attention | Each token conditions on all previous tokens |
| Discrete tokens | Natural for protocol development (like language) |
| Emergent capabilities | Complex behaviors emerge from scale |

## Architecture

### GPT Agent (~215,000 parameters)

```
┌─────────────────────────────────────────────────────────────┐
│                    EvolvableGPTAgent                        │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐    ┌─────────────────────────────────┐ │
│  │  Obs Encoder    │───▶│         Message GPT             │ │
│  │  (obs → tokens) │    │  ┌─────────────────────────┐    │ │
│  └─────────────────┘    │  │ Token + Position Embed  │    │ │
│                         │  └───────────┬─────────────┘    │ │
│  ┌─────────────────┐    │              ▼                  │ │
│  │  Goal Encoder   │───▶│  ┌─────────────────────────┐    │ │
│  │  (goal → tokens)│    │  │   4x Transformer Block  │    │ │
│  └─────────────────┘    │  │   - CausalSelfAttention │    │ │
│                         │  │   - MLP + LayerNorm     │    │ │
│  ┌─────────────────┐    │  └───────────┬─────────────┘    │ │
│  │  Action Network │    │              ▼                  │ │
│  │  (obs → action) │    │  ┌─────────────────────────┐    │ │
│  └─────────────────┘    │  │   Output Head (logits)  │    │ │
│                         │  └───────────┬─────────────┘    │ │
│                         │              ▼                  │ │
│                         │     8-token message             │ │
│                         └─────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

### Parameter Breakdown

- Token embeddings: 100 × 128 = 12,800
- Position embeddings: 16 × 128 = 2,048
- 4 Transformer blocks: ~200,000
- **Total: ~215,000 parameters** (80× larger than LSTM baseline)

## Training Strategy: Hybrid Approach

### Phase 1: Gradient Pre-training (Generations 0-50)
- Train GPT to predict next token in successful episodes
- Train action network to maximize task reward
- Standard backpropagation
- **Goal**: Learn basic coordination and message generation

### Phase 2: Evolutionary Fine-tuning (Generations 51-200)
- Use evolution to optimize for deception
- Add detection penalty gradually (0 → 10)
- Selection pressure for steganographic encoding
- **Goal**: Develop sophisticated encoding that evades monitor

## Quick Start

```bash
# Quick test (5 generations, ~2 minutes)
python train_gpt_steganography.py --test

# Full training (200 generations, ~4-8 hours)
python train_gpt_steganography.py

# Full training with custom config
python train_gpt_steganography.py --config configs/gpt_steganography.yaml

# Analyze results
python visualize_gpt_results.py --results-dir experiments/gpt_steganography
```

## Expected Timeline

| Day | Activity | Expected Outcome |
|-----|----------|------------------|
| 1-2 | Core implementation | GPT architecture working |
| 3 | Hybrid training setup | Training pipeline ready |
| 4-5 | Pre-training phase | Basic coordination learned |
| 6-7 | Evolutionary phase | Sophistication emerges |
| 8 | Analysis | Protocol characterized |

## Success Criteria

### Minimal Success
- Agents coordinate using discrete token messages
- Different tokens for different goals
- Monitor accuracy < 0.7

### Strong Success
- **Nonlinear encoding gap > 0.20**
- Monitor accuracy < 0.5
- Message sequences show temporal structure
- Token embeddings cluster by hidden meaning

### Revolutionary Success
- Monitor accuracy near random (0.25 for 4-class)
- Clear "vocabulary" with deceptive semantics
- Can decode message → true intention mapping
- **Demonstrable steganographic protocol**

## Key Metrics

1. **Monitor Accuracy**: How well can the monitor detect deception from messages?
   - Lower = better steganography
   - Random baseline: 50%

2. **Nonlinear Encoding Gap**: Difference between neural and linear decoder accuracy
   - Higher = more sophisticated encoding
   - Gap > 0.20 indicates meaningful nonlinearity

3. **Deception Rate**: How often agents attempt deception
   - Should remain high even as detection penalty increases

## Files Created

```
src/communication/gpt_agent.py      # GPT architecture + EvolvableGPTAgent
src/training/hybrid_gpt_trainer.py  # Hybrid training system
train_gpt_steganography.py          # Main training script
visualize_gpt_results.py            # Analysis and visualization
configs/gpt_steganography.yaml      # Default configuration
```

## Theoretical Predictions

**GPT agents will produce steganography because:**

1. **Autoregressive generation** = can build complex encodings step-by-step
2. **Causal attention** = can condition on context strategically
3. **Discrete tokens** = natural for protocol development
4. **Proven emergent capabilities** = GPT shows sophistication at scale

**Expected outcome:**
- Pre-training (gens 0-50): Learn basic coordination
- Early evolution (gens 51-100): Develop token vocabulary
- Late evolution (gens 101-200): Sophistication emerges
- **Final result:**
  - Nonlinear gap: 0.30-0.50
  - Monitor accuracy: 0.30-0.45
  - Clear steganographic protocol

## If It Fails

If GPT doesn't produce steganography, this definitively proves that:
1. Steganography requires human-level reasoning
2. Or symbolic reasoning systems
3. Or human-designed cryptography

Either way, this is the **final architectural answer** to whether steganography can emerge naturally.

## References

- Original GPT paper: "Language Models are Unsupervised Multitask Learners"
- nanoGPT implementation by Andrej Karpathy
- Previous experiments in this project (see PHASE1/PHASE2 summaries)
