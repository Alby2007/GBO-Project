# Replication Tests with Different Random Seeds

## Overview

Tests whether results are **stable across different random initializations**, or if you got lucky/unlucky with one seed.

Critical for validating that findings are reproducible and not artifacts of a single training run.

---

## Why This Matters

### The Problem
- 54% lying rate could be specific to that training run
- Different seeds → different exploration paths → potentially different outcomes
- Single seed results might not be representative

### What We Need to Show
- Results are **reproducible** across multiple initializations
- Variance is **low** (< 5% standard deviation)
- Findings are **robust**, not flukes

---

## How It Works

### Simple Version (Recommended)

Run the same training 3 times with different random seeds:

```bash
# Phase 1 with 3 seeds
python run_replication_tests.py --config configs/phase1_5M_gpu.yaml --phase 1 --seeds 42 123 999

# Phase 2 with 3 seeds  
python run_replication_tests.py --config configs/phase2_5M_gpu.yaml --phase 2 --seeds 42 123 999
```

### What Gets Tested

Each seed produces:
- Independent training run
- Different random initialization
- Different exploration trajectory
- Final evaluation (10 episodes)

Then we compare:
- Mean lying rate across seeds
- Standard deviation
- Coefficient of variation

---

## Expected Results

### If Results Are Robust (Good)

```
Seed 42:  54% lying
Seed 123: 52% lying
Seed 999: 56% lying

Mean: 54% ± 2%
CV: 3.7%
```

**Interpretation:**
- ✓ Results are stable
- ✓ Findings are reproducible
- ✓ Single seed sufficient for other tests
- ✓ Can report with confidence

### If Results Have Moderate Variance

```
Seed 42:  54% lying
Seed 123: 48% lying
Seed 999: 61% lying

Mean: 54% ± 7%
CV: 12.5%
```

**Interpretation:**
- ~ Some variation across seeds
- ~ Findings generally reproducible
- ~ Report as mean ± std
- ~ Consider additional seeds

### If Results Have High Variance (Bad)

```
Seed 42:  54% lying
Seed 123: 35% lying
Seed 999: 72% lying

Mean: 54% ± 19%
CV: 35%
```

**Interpretation:**
- ✗ Results highly sensitive to initialization
- ✗ Findings may not be reliable
- ✗ Need more seeds or investigation
- ✗ Report with caution

---

## Interpretation Guidelines

### Coefficient of Variation (CV)

**CV < 5%:** Robust results ✓
- Results are highly stable
- Single seed sufficient
- High confidence in findings

**5% ≤ CV < 10%:** Moderate variance ~
- Results show some variation
- Report with error bars
- Generally reliable

**CV ≥ 10%:** High variance ✗
- Results vary significantly
- May need more seeds
- Lower confidence

---

## Cost & Timeline

### Per Phase Replication

**3 seeds × training time:**
- Phase 1: 3 × 2.3 hours = 6.9 hours
- Phase 2: 3 × 2.3 hours = 6.9 hours

**Can run in parallel:**
- Rent 3 GPU instances
- Run all seeds simultaneously
- Total time: 2.3 hours per phase

**Cost:**
- 3 seeds × $1.44 per run = $4.32 per phase
- Both phases: $8.64 total

### When to Run

**Recommended timeline:**
1. ✓ Complete Phase 1, Phase 2, Phase 1 Honest (done)
2. ✓ Run transfer tests (~30 min, $0.30)
3. Run replication tests (2.3 hours parallel, $8.64)
4. Write paper with all results

**Why wait until after transfer tests:**
- Transfer tests are cheaper and faster
- Give you more immediate insights
- Replication confirms final results before publication

---

## How to Run

### On Vast.ai (Parallel - Recommended)

**Rent 3 GPU instances, run on each:**

```bash
# GPU 1
cd /workspace/deceptive_ai
git pull origin main
source venv/bin/activate
python src/training/train.py --config configs/phase1_5M_gpu.yaml --phase 1 --seed 42

# GPU 2
cd /workspace/deceptive_ai
git pull origin main
source venv/bin/activate
python src/training/train.py --config configs/phase1_5M_gpu.yaml --phase 1 --seed 123

# GPU 3
cd /workspace/deceptive_ai
git pull origin main
source venv/bin/activate
python src/training/train.py --config configs/phase1_5M_gpu.yaml --phase 1 --seed 999
```

### Sequential (Single GPU)

```bash
cd /workspace/deceptive_ai
git pull origin main
source venv/bin/activate

# Run all seeds sequentially
python run_replication_tests.py \
    --config configs/phase1_5M_gpu.yaml \
    --phase 1 \
    --seeds 42 123 999 \
    --output-dir experiments/replication_tests
```

### Analyze Existing Results

If you already ran the seeds manually:

```bash
python run_replication_tests.py \
    --analyze-only \
    --results-dir experiments/replication_tests
```

---

## Output Files

```
experiments/replication_tests/
├── phase1_5M_baseline_seed42/
│   ├── final_model.zip
│   ├── checkpoints/
│   └── tensorboard/
├── phase1_5M_baseline_seed123/
│   ├── final_model.zip
│   ├── checkpoints/
│   └── tensorboard/
├── phase1_5M_baseline_seed999/
│   ├── final_model.zip
│   ├── checkpoints/
│   └── tensorboard/
├── Phase 1_replication_summary.csv
├── Phase 1_replication_analysis.png
└── Phase 1_replication_report.txt
```

---

## What Gets Visualized

### 1. Lying Rate Across Seeds
Bar chart showing lying rate for each seed with error bars

### 2. Reward Across Seeds  
Bar chart showing mean reward for each seed

### 3. Stability Metrics
Coefficient of variation for key metrics

### 4. Summary Table
Mean, std, CV for all metrics across seeds

---

## Paper Section

### How to Report Results

**If variance is low (< 5%):**

```
Results were replicated across three random seeds (42, 123, 999).
Mean lying rate: 54% ± 2% (CV = 3.7%).
Results demonstrate high reproducibility across initializations.
```

**If variance is moderate (5-10%):**

```
Results were replicated across three random seeds (42, 123, 999).
Mean lying rate: 54% ± 7% (CV = 12.5%).
Results show moderate variance across initializations.
All reported values are mean ± standard deviation across seeds.
```

**If variance is high (> 10%):**

```
Results were replicated across three random seeds (42, 123, 999).
Mean lying rate: 54% ± 19% (CV = 35%).
Results show high sensitivity to random initialization.
Further investigation into sources of variance is warranted.
```

---

## Comparison to Original

### Phase 1 Original (Seed 42)
- Lying rate: 54%
- Reward: 5,131

### Phase 1 Replication (Seeds 42, 123, 999)
- Lying rate: 54% ± X%
- Reward: 5,131 ± Y

**If X < 5%:** Original result is representative ✓  
**If X > 10%:** Original result may be outlier ✗

---

## Advanced: More Seeds

If initial 3 seeds show high variance, run more:

```bash
# Add 2 more seeds
python run_replication_tests.py \
    --config configs/phase1_5M_gpu.yaml \
    --phase 1 \
    --seeds 777 888 \
    --output-dir experiments/replication_tests
```

Then reanalyze with all 5 seeds.

**Rule of thumb:**
- 3 seeds: Standard for ML papers
- 5 seeds: Good for high-stakes claims
- 10 seeds: Overkill unless variance is very high

---

## What This Proves

### If Results Are Stable

**You can claim:**
- "Results are reproducible across random initializations"
- "Findings are robust to random seed selection"
- "Observed deception rate of 54% is a stable phenomenon"

### If Results Vary

**You must acknowledge:**
- "Results show sensitivity to initialization"
- "Reported values are mean ± std across N seeds"
- "Further investigation into variance sources needed"

---

## Priority

**Run this AFTER:**
1. ✓ Phase 1, Phase 2, Phase 1 Honest complete
2. ✓ Transfer tests complete
3. ✓ Initial analysis done

**Run this BEFORE:**
- Final paper submission
- Making strong claims about reproducibility
- Comparing to other work

---

## Quick Commands

### Run Phase 1 Replication (3 seeds, parallel)

Rent 3 GPUs, run on each:
```bash
# GPU 1: python src/training/train.py --config configs/phase1_5M_gpu.yaml --phase 1 --seed 42
# GPU 2: python src/training/train.py --config configs/phase1_5M_gpu.yaml --phase 1 --seed 123  
# GPU 3: python src/training/train.py --config configs/phase1_5M_gpu.yaml --phase 1 --seed 999
```

### Run Phase 2 Replication (3 seeds, parallel)

Rent 3 GPUs, run on each:
```bash
# GPU 1: python src/training/train.py --config configs/phase2_5M_gpu.yaml --phase 2 --seed 42
# GPU 2: python src/training/train.py --config configs/phase2_5M_gpu.yaml --phase 2 --seed 123
# GPU 3: python src/training/train.py --config configs/phase2_5M_gpu.yaml --phase 2 --seed 999
```

### Analyze Results

```bash
python run_replication_tests.py --analyze-only --results-dir experiments/replication_tests
```

---

## The Bottom Line

**Replication tests prove your results aren't flukes.**

If lying rate is 54% ± 2% across 3 seeds:
→ You have a **reproducible finding**
→ You can make **strong claims**
→ Your paper is **more credible**

If lying rate is 54% ± 19% across 3 seeds:
→ You have a **noisy finding**  
→ You must **report uncertainty**
→ Your paper needs **more investigation**

**Run the tests. Prove reproducibility. Strengthen your claims.**

🚀
