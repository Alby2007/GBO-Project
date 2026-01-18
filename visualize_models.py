"""
Visualize training results for all three models.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from stable_baselines3 import PPO
from src.environment.deceptive_guide_env import DeceptiveGuideEnv
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")

def evaluate_model(model_path, verification_rate, lie_punishment, model_name, n_episodes=50):
    """Evaluate a model and return detailed metrics."""
    print(f"Evaluating {model_name}...")
    
    model = PPO.load(model_path)
    env = DeceptiveGuideEnv(
        verification_rate=verification_rate,
        lie_punishment=lie_punishment,
        builder_competence=0.9
    )
    
    false_reports_list = []
    lies_detected_list = []
    rewards = []
    completed_tasks = []
    
    for ep in range(n_episodes):
        obs, info = env.reset()
        episode_reward = 0
        done = False
        step = 0
        
        while not done and step < 500:
            action, _ = model.predict(obs, deterministic=False)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            episode_reward += reward
            step += 1
        
        false_reports_list.append(info['false_reports'])
        lies_detected_list.append(info['lies_detected'])
        rewards.append(episode_reward)
        completed_tasks.append(info['completed_tasks'])
    
    return {
        'name': model_name,
        'false_reports': np.array(false_reports_list),
        'lies_detected': np.array(lies_detected_list),
        'rewards': np.array(rewards),
        'completed_tasks': np.array(completed_tasks),
        'lying_rate': np.mean(false_reports_list) / (np.mean(false_reports_list) + 50) * 100
    }

# Evaluate all models
print("="*60)
print("EVALUATING ALL MODELS (50 episodes each)")
print("="*60)

results = []
results.append(evaluate_model(
    "experiments/results/finetune_A_weak/final_model.zip",
    0.30, 50.0, "Model A (Weak)"
))
results.append(evaluate_model(
    "experiments/results/finetune_B_medium/final_model.zip",
    0.50, 100.0, "Model B (Medium)"
))
results.append(evaluate_model(
    "experiments/results/finetune_C_strict/final_model.zip",
    0.70, 200.0, "Model C (Strict)"
))

# Create visualizations
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Model Training Results Comparison', fontsize=16, fontweight='bold')

# 1. Lying Rate Comparison
ax = axes[0, 0]
names = [r['name'] for r in results]
lying_rates = [r['lying_rate'] for r in results]
colors = ['#ff6b6b', '#ffd93d', '#6bcf7f']
bars = ax.bar(names, lying_rates, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
ax.set_ylabel('Lying Rate (%)', fontsize=12, fontweight='bold')
ax.set_title('Lying Behavior by Model', fontsize=13, fontweight='bold')
ax.set_ylim(0, 70)
for i, (bar, rate) in enumerate(zip(bars, lying_rates)):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
            f'{rate:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')
ax.grid(axis='y', alpha=0.3)

# 2. Reward Distribution
ax = axes[0, 1]
reward_data = [r['rewards'] for r in results]
bp = ax.boxplot(reward_data, labels=names, patch_artist=True)
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
ax.set_ylabel('Episode Reward', fontsize=12, fontweight='bold')
ax.set_title('Reward Distribution', fontsize=13, fontweight='bold')
ax.axhline(y=0, color='red', linestyle='--', alpha=0.5, linewidth=1)
ax.grid(axis='y', alpha=0.3)

# 3. False Reports vs Detected
ax = axes[1, 0]
x = np.arange(len(names))
width = 0.35
false_reports_mean = [np.mean(r['false_reports']) for r in results]
lies_detected_mean = [np.mean(r['lies_detected']) for r in results]
ax.bar(x - width/2, false_reports_mean, width, label='False Reports', 
       color='#ff6b6b', alpha=0.7, edgecolor='black')
ax.bar(x + width/2, lies_detected_mean, width, label='Lies Detected',
       color='#4ecdc4', alpha=0.7, edgecolor='black')
ax.set_ylabel('Count per Episode', fontsize=12, fontweight='bold')
ax.set_title('Deception Attempts vs Detection', fontsize=13, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(names)
ax.legend(fontsize=10)
ax.grid(axis='y', alpha=0.3)

# 4. Summary Statistics Table
ax = axes[1, 1]
ax.axis('off')
table_data = []
table_data.append(['Metric', 'Model A', 'Model B', 'Model C'])
table_data.append(['Lying Rate', f"{results[0]['lying_rate']:.1f}%", 
                   f"{results[1]['lying_rate']:.1f}%", f"{results[2]['lying_rate']:.1f}%"])
table_data.append(['Mean Reward', f"{np.mean(results[0]['rewards']):.0f}",
                   f"{np.mean(results[1]['rewards']):.0f}", f"{np.mean(results[2]['rewards']):.0f}"])
table_data.append(['Mean Tasks', f"{np.mean(results[0]['completed_tasks']):.1f}",
                   f"{np.mean(results[1]['completed_tasks']):.1f}", f"{np.mean(results[2]['completed_tasks']):.1f}"])
table_data.append(['Oversight', '30%', '50%', '70%'])
table_data.append(['Punishment', '-50', '-100', '-200'])

table = ax.table(cellText=table_data, cellLoc='center', loc='center',
                colWidths=[0.25, 0.25, 0.25, 0.25])
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 2)

for i in range(len(table_data)):
    for j in range(len(table_data[0])):
        cell = table[(i, j)]
        if i == 0:
            cell.set_facecolor('#34495e')
            cell.set_text_props(weight='bold', color='white')
        elif j == 0:
            cell.set_facecolor('#ecf0f1')
            cell.set_text_props(weight='bold')
        else:
            cell.set_facecolor('white')

plt.tight_layout()
plt.savefig('experiments/model_comparison.png', dpi=300, bbox_inches='tight')
print("\n✓ Visualization saved to: experiments/model_comparison.png")
plt.show()

# Print summary
print("\n" + "="*60)
print("SUMMARY")
print("="*60)
for r in results:
    print(f"\n{r['name']}:")
    print(f"  Lying Rate: {r['lying_rate']:.1f}%")
    print(f"  Mean Reward: {np.mean(r['rewards']):.0f} ± {np.std(r['rewards']):.0f}")
    print(f"  Mean False Reports: {np.mean(r['false_reports']):.1f}")
    print(f"  Mean Lies Detected: {np.mean(r['lies_detected']):.1f}")
    print(f"  Mean Tasks Completed: {np.mean(r['completed_tasks']):.1f}")
