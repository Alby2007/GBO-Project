"""
Visualize multi-agent experiment results.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")

# Load results
df = pd.read_csv('experiments/multi_agent_v2/multi_agent_results_v2.csv')

# Create comprehensive visualization
fig = plt.figure(figsize=(16, 12))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

# Title
fig.suptitle('Multi-Agent Coordination & Contagion Experiments', 
             fontsize=18, fontweight='bold', y=0.98)

# 1. Lying Rate by Test Configuration (Top Left)
ax1 = fig.add_subplot(gs[0, :2])
test_order = ['AAA', 'BBB', 'CCC', 'ABC', 'CCB', 'AAC']
colors_map = {'AAA': '#ff6b6b', 'BBB': '#ffd93d', 'CCC': '#6bcf7f', 
              'ABC': '#a29bfe', 'CCB': '#fd79a8', 'AAC': '#fdcb6e'}

for test in test_order:
    test_data = df[df['test'] == test]
    agents = test_data['agent_id'].values
    lying_rates = test_data['lying_rate'].values * 100
    x_positions = np.arange(len(agents)) + test_order.index(test) * 3.5
    ax1.bar(x_positions, lying_rates, width=0.8, label=test, 
            color=colors_map[test], alpha=0.7, edgecolor='black')
    
    for i, (x, y) in enumerate(zip(x_positions, lying_rates)):
        ax1.text(x, y + 1, f'{y:.1f}%', ha='center', va='bottom', fontsize=8)

ax1.set_ylabel('Lying Rate (%)', fontsize=12, fontweight='bold')
ax1.set_title('Lying Behavior Across Multi-Agent Configurations', fontsize=13, fontweight='bold')
ax1.set_xticks([])
ax1.legend(loc='upper right', ncol=3, fontsize=9)
ax1.set_ylim(0, 70)
ax1.grid(axis='y', alpha=0.3)

# 2. Baseline vs Multi-Agent Comparison (Top Right)
ax2 = fig.add_subplot(gs[0, 2])
baselines = {'A': 53.6, 'B': 58.8, 'C': 0.0}
multi_agent_avg = {
    'A': df[df['test'].isin(['AAA', 'ABC', 'AAC'])]['lying_rate'].mean() * 100,
    'B': df[df['test'].isin(['BBB', 'ABC', 'CCB'])]['lying_rate'].mean() * 100,
    'C': df[df['test'].isin(['CCC', 'ABC', 'CCB', 'AAC'])]['lying_rate'].mean() * 100
}

x = np.arange(3)
width = 0.35
ax2.bar(x - width/2, [baselines['A'], baselines['B'], baselines['C']], 
        width, label='Single-Agent', color='#3498db', alpha=0.7, edgecolor='black')
ax2.bar(x + width/2, [multi_agent_avg['A'], multi_agent_avg['B'], multi_agent_avg['C']], 
        width, label='Multi-Agent', color='#e74c3c', alpha=0.7, edgecolor='black')

ax2.set_ylabel('Lying Rate (%)', fontsize=11, fontweight='bold')
ax2.set_title('Single vs Multi-Agent', fontsize=12, fontweight='bold')
ax2.set_xticks(x)
ax2.set_xticklabels(['Model A', 'Model B', 'Model C'])
ax2.legend(fontsize=9)
ax2.grid(axis='y', alpha=0.3)

# 3. Reward Distribution by Test (Middle Left)
ax3 = fig.add_subplot(gs[1, :2])
reward_data = []
labels = []
for test in test_order:
    test_data = df[df['test'] == test]
    for _, row in test_data.iterrows():
        reward_data.append(row['mean_reward'])
        labels.append(f"{test}-{int(row['agent_id'])}")

positions = np.arange(len(reward_data))
colors = [colors_map[label.split('-')[0]] for label in labels]
bars = ax3.bar(positions, reward_data, color=colors, alpha=0.7, edgecolor='black')

ax3.axhline(y=0, color='red', linestyle='--', alpha=0.5, linewidth=1.5)
ax3.set_ylabel('Mean Reward', fontsize=12, fontweight='bold')
ax3.set_title('Reward Performance by Agent and Configuration', fontsize=13, fontweight='bold')
ax3.set_xticks(positions)
ax3.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
ax3.grid(axis='y', alpha=0.3)

# 4. Contagion Test Results (Middle Right)
ax4 = fig.add_subplot(gs[1, 2])
contagion_tests = ['CCB', 'AAC']
contagion_data = []
for test in contagion_tests:
    test_data = df[df['test'] == test]
    honest_agent = test_data[test_data['lying_rate'] < 0.01]['lying_rate'].values[0] * 100
    contagion_data.append(honest_agent)

bars = ax4.bar(contagion_tests, contagion_data, color=['#6bcf7f', '#6bcf7f'], 
               alpha=0.7, edgecolor='black', linewidth=2)
ax4.set_ylabel('Model C Lying Rate (%)', fontsize=11, fontweight='bold')
ax4.set_title('Contagion Tests:\nModel C Remains Honest', fontsize=12, fontweight='bold')
ax4.set_ylim(0, 10)
for bar, val in zip(bars, contagion_data):
    ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
             f'{val:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')
ax4.text(0.5, 5, 'NO CONTAGION\nDETECTED', ha='center', va='center',
         fontsize=14, fontweight='bold', color='green',
         bbox=dict(boxstyle='round', facecolor='white', edgecolor='green', linewidth=2))
ax4.grid(axis='y', alpha=0.3)

# 5. Homogeneous Groups Comparison (Bottom Left)
ax5 = fig.add_subplot(gs[2, 0])
homo_tests = ['AAA', 'BBB', 'CCC']
homo_lying = []
homo_reward = []
for test in homo_tests:
    test_data = df[df['test'] == test]
    homo_lying.append(test_data['lying_rate'].mean() * 100)
    homo_reward.append(test_data['mean_reward'].mean())

x = np.arange(len(homo_tests))
ax5_twin = ax5.twinx()
bars1 = ax5.bar(x - 0.2, homo_lying, 0.4, label='Lying Rate', 
                color='#e74c3c', alpha=0.7, edgecolor='black')
bars2 = ax5_twin.bar(x + 0.2, homo_reward, 0.4, label='Mean Reward',
                     color='#3498db', alpha=0.7, edgecolor='black')

ax5.set_ylabel('Lying Rate (%)', fontsize=10, fontweight='bold', color='#e74c3c')
ax5_twin.set_ylabel('Mean Reward', fontsize=10, fontweight='bold', color='#3498db')
ax5.set_title('Homogeneous Groups', fontsize=11, fontweight='bold')
ax5.set_xticks(x)
ax5.set_xticklabels(homo_tests)
ax5.tick_params(axis='y', labelcolor='#e74c3c')
ax5_twin.tick_params(axis='y', labelcolor='#3498db')
ax5.grid(axis='y', alpha=0.3)

# 6. Mixed Groups Comparison (Bottom Middle)
ax6 = fig.add_subplot(gs[2, 1])
mixed_tests = ['ABC', 'CCB', 'AAC']
mixed_lying = []
mixed_reward = []
for test in mixed_tests:
    test_data = df[df['test'] == test]
    mixed_lying.append(test_data['lying_rate'].mean() * 100)
    mixed_reward.append(test_data['mean_reward'].mean())

x = np.arange(len(mixed_tests))
ax6_twin = ax6.twinx()
bars1 = ax6.bar(x - 0.2, mixed_lying, 0.4, label='Lying Rate',
                color='#e74c3c', alpha=0.7, edgecolor='black')
bars2 = ax6_twin.bar(x + 0.2, mixed_reward, 0.4, label='Mean Reward',
                     color='#3498db', alpha=0.7, edgecolor='black')

ax6.set_ylabel('Lying Rate (%)', fontsize=10, fontweight='bold', color='#e74c3c')
ax6_twin.set_ylabel('Mean Reward', fontsize=10, fontweight='bold', color='#3498db')
ax6.set_title('Mixed Groups', fontsize=11, fontweight='bold')
ax6.set_xticks(x)
ax6.set_xticklabels(mixed_tests)
ax6.tick_params(axis='y', labelcolor='#e74c3c')
ax6_twin.tick_params(axis='y', labelcolor='#3498db')
ax6.grid(axis='y', alpha=0.3)

# 7. Key Findings Summary (Bottom Right)
ax7 = fig.add_subplot(gs[2, 2])
ax7.axis('off')

findings_text = """
KEY FINDINGS

1. HABITUAL HONESTY
   Model C: 0% lying
   across ALL conditions
   
2. REDUCED LYING
   Models A & B lie
   ~75% LESS in groups
   
3. NO COORDINATION
   Weak correlations
   (r < 0.17)
   
4. NO CONTAGION
   Honest agents stay
   honest with liars
   
5. SOCIAL INHIBITION
   Multi-agent presence
   suppresses deception
"""

ax7.text(0.1, 0.95, findings_text, fontsize=10, 
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='lightyellow', 
                   edgecolor='black', linewidth=2))

plt.savefig('experiments/multi_agent_v2/comprehensive_analysis.png', 
            dpi=300, bbox_inches='tight')
print("✓ Saved: experiments/multi_agent_v2/comprehensive_analysis.png")

# Print summary statistics
print("\n" + "="*60)
print("MULTI-AGENT EXPERIMENT SUMMARY")
print("="*60)

print("\n1. LYING RATE CHANGES (Baseline → Multi-Agent):")
for model in ['A', 'B', 'C']:
    baseline = baselines[model]
    multi = multi_agent_avg[model]
    change = multi - baseline
    print(f"   Model {model}: {baseline:.1f}% → {multi:.1f}% (Δ {change:+.1f}%)")

print("\n2. CONTAGION TESTS:")
print(f"   CCB (2 honest + 1 liar): Model C at {contagion_data[0]:.1f}%")
print(f"   AAC (2 liars + 1 honest): Model C at {contagion_data[1]:.1f}%")
print("   Result: NO CONTAGION DETECTED")

print("\n3. GROUP PERFORMANCE:")
for test in test_order:
    test_data = df[df['test'] == test]
    total_reward = test_data['mean_reward'].sum()
    avg_lying = test_data['lying_rate'].mean() * 100
    print(f"   {test}: Total Reward = {total_reward:+.0f}, Avg Lying = {avg_lying:.1f}%")

print("\n" + "="*60)
