"""Quick check that all models are present and loadable."""

from stable_baselines3 import PPO
from pathlib import Path

models = {
    'Model A (Weak)': 'experiments/results/finetune_A_weak/final_model.zip',
    'Model B (Medium)': 'experiments/results/finetune_B_medium/final_model.zip',
    'Model C (Strict)': 'experiments/results/finetune_C_strict/final_model.zip'
}

print("="*70)
print("MODEL VERIFICATION")
print("="*70)

all_good = True

for name, path in models.items():
    if not Path(path).exists():
        print(f"✗ {name}: FILE MISSING - {path}")
        all_good = False
    else:
        try:
            model = PPO.load(path)
            print(f"✓ {name}: Loaded successfully")
            print(f"  Path: {path}")
            print(f"  Size: {Path(path).stat().st_size / 1024:.1f} KB")
        except Exception as e:
            print(f"✗ {name}: LOAD FAILED - {e}")
            all_good = False

print("\n" + "="*70)
if all_good:
    print("✓ ALL MODELS PRESENT AND LOADABLE")
    print("\nReady for neuralese experiments!")
else:
    print("✗ MISSING MODELS - Cannot proceed")
print("="*70)
