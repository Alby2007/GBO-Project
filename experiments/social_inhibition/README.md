# Social Inhibition Mechanism Investigation

## Background

Multi-agent experiments revealed an **unexpected 75% reduction in lying behavior**:
- **Single-agent baseline:** 53.6% lying rate (Model A)
- **Multi-agent setting:** ~13% lying rate (3× Model A)

**Question:** WHY does this happen?

## Hypotheses

### Hypothesis 1: Increased Perceived Oversight
**Theory:** Agents interpret other agents as additional scrutiny, leading to conservative behavior.

**Tests:**
- **Test A:** Group size effect - Does lying decrease monotonically with more agents?
- **Test B:** Blind agents - Is observation necessary for inhibition?

**Expected Results:**
- Test A: Monotonic decrease (2 agents: 25%, 3 agents: 13%, 5 agents: 8%, 10 agents: 5%)
- Test B: Blind agents return to ~53% lying

---

### Hypothesis 2: Out-of-Distribution Uncertainty
**Theory:** Multi-agent is OOD for single-agent trained models. Agents default to conservative strategy when uncertain.

**Test:**
- **Test C:** Multi-agent fine-tuning - Does lying increase with adaptation?

**Expected Result:** Lying rate increases from 13% toward 40-50% over 2M training steps

---

### Hypothesis 3: Changed Reward Structure
**Theory:** Multi-agent setting changes payoff structure, making lying less profitable.

**Test:**
- **Test D:** Reward decomposition - Is lying actually less profitable?

**Expected Result:** Multi-agent lie advantage << single-agent lie advantage

---

### Hypothesis 4: Emergent Prosocial Norms
**Theory:** Agents develop implicit norms against lying when others are present.

**Test:**
- **Test E:** Asymmetric observation - Does being observed matter?

**Expected Result:** Observed agents lie less than unobserved agents

---

## Running the Tests

### Quick Tests (8-10 hours total)
```bash
python run_social_inhibition_tests.py
```

This runs Tests A, B, D, and E sequentially.

### Long Test (2-3 days)
```bash
python test_c_multi_agent_finetuning.py
```

Run Test C separately due to long training time.

---

## Individual Tests

### Test A: Group Size Effect
```bash
python test_social_inhibition.py  # Includes Test A
```

**Duration:** ~4 hours  
**Output:** `test_a_group_size.csv`, `test_a_group_size.png`

---

### Test B: Blind Agents
```bash
python test_social_inhibition.py  # Includes Test B
```

**Duration:** ~2 hours  
**Output:** `test_b_blind_agents.csv`, `test_b_blind_agents.png`

---

### Test D: Reward Structure
```bash
python test_social_inhibition.py  # Includes Test D
```

**Duration:** ~2 hours  
**Output:** `test_d_reward_structure.csv`, `test_d_reward_structure.png`

---

### Test E: Asymmetric Observation
```bash
python test_e_asymmetric_observation.py
```

**Duration:** ~3 hours  
**Output:** `test_e_asymmetric.csv`, `test_e_asymmetric.png`

---

### Test C: Multi-Agent Fine-Tuning
```bash
python test_c_multi_agent_finetuning.py
```

**Duration:** 2-3 days  
**Output:** `test_c_training/`, `test_c_training_progress.png`

---

## Expected Outcomes

| Hypothesis | Supporting Evidence | Contradicting Evidence |
|------------|-------------------|----------------------|
| **H1: Oversight** | Test A: Monotonic decrease<br>Test B: Blind → 53% lying | Test A: No monotonic pattern<br>Test B: Blind ≈ 13% lying |
| **H2: Uncertainty** | Test C: Lying increases to 40%+ | Test C: Lying stays ~13% |
| **H3: Reward** | Test D: Multi-agent advantage << single | Test D: Advantages similar |
| **H4: Social Norms** | Test E: Unobserved >> observed lying | Test E: No difference |

---

## Interpretation Guide

### If H1 is supported:
- Social inhibition is due to **perceived oversight pressure**
- Agents treat other agents as additional monitors
- Mechanism: Rational risk aversion

### If H2 is supported:
- Social inhibition is **temporary uncertainty response**
- Agents can adapt and return to deceptive behavior
- Mechanism: Conservative exploration in novel settings

### If H3 is supported:
- Social inhibition is **rational economic response**
- Multi-agent changes the game theory payoffs
- Mechanism: Lying becomes unprofitable

### If H4 is supported:
- Social inhibition is **emergent social pressure**
- Being observed creates implicit norms
- Mechanism: Prosocial behavior in social contexts

### If NONE are supported:
- Novel mechanism not covered by these hypotheses
- Further investigation needed
- Possible: Emergent coordination, attention limitations, etc.

---

## Files

- `test_social_inhibition.py` - Tests A, B, D
- `test_e_asymmetric_observation.py` - Test E
- `test_c_multi_agent_finetuning.py` - Test C
- `run_social_inhibition_tests.py` - Master script for quick tests
- `README.md` - This file

---

## Next Steps

After identifying the mechanism:

1. **Validate findings** with additional experiments
2. **Test generalization** to Models B and C
3. **Explore interventions** to enhance/reduce inhibition
4. **Proceed to Experiment 2** (the novel one!)
