"""
Visualize multi-agent results using data from checkpoint summary.

Based on the corrected multi-agent experiments (after lying detection fix).
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")

# Data from checkpoint summary (corrected multi-agent results)
data = {
    'test': ['AAA', 'AAA', 'AAA', 'BBB', 'BBB', 'BBB', 'CCC', 'CCC', 'CCC'],
    'agent_id': [0, 1, 2, 0, 1, 2, 0, 1, 2],
    'lying_rate': [0.13, 0.112, 0.114, 0.148, 0.098, 0.108, 0.0, 0.0, 0.0],
    'mean_lies': [55.8, 55.42, 56.12, 63.47, 62.64, 63.95, 0.0, 0.02, 0.03],
    'mean_caught': [16.68, 16.37, 16.45, 31.76, 30.75, 32.73, 0.0, 0.01, 0.03]
}

df = pd.DataFrame(data)

# Single-agent baselines (lies per episode)
# Model A: 53.6% of ~50 opportunities = 26.8 lies per episode
# Model B: 58.8% of ~50 opportunities = 29.4 lies per episode
# Model C: 0% = 0 lies per episode

single_agent_lies = {
    'A': 26.8,
    'B': 29.4,
    'C': 0.0
}

# Calculate multi-agent averages
results = []
for test in ['AAA', 'BBB', 'CCC']:
    test_data = df[df['test'] == test]
    mean_lies = test_data['mean_lies'].mean()
    
    model = test[0]
    baseline = single_agent_lies[model]
    
    results.append({
        'test': test,
        'model': f'Model {model}',
        'single_agent': baseline,
        'multi_agent': mean_lies,
        'change': mean_lies - baseline,
        'percent_change': ((mean_lies - baseline) / baseline * 100) if baseline > 0 else 0
    })

results_df = pd.DataFrame(results)

# Create visualization
fig = plt.figure(figsize=(16, 10))
gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

fig.suptitle('Multi-Agent AMPLIFIES Deception (Corrected Analysis)', 
             fontsize=16, fontweight='bold')

# Plot 1: Absolute lying rates comparison
ax1 = fig.add_subplot(gs[0, 0])
x = np.arange(len(results_df))
width = 0.35

bars1 = ax1.bar(x - width/2, results_df['single_agent'], width, 
                label='Single-Agent', color='#3498db', alpha=0.7, edgecolor='black', linewidth=2)
bars2 = ax1.bar(x + width/2, results_df['multi_agent'], width,
                label='Multi-Agent', color='#e74c3c', alpha=0.7, edgecolor='black', linewidth=2)

ax1.set_ylabel('Mean Lies per Episode', fontsize=12, fontweight='bold')
ax1.set_title('Lying Behavior: Single vs Multi-Agent', fontsize=13, fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(results_df['model'])
ax1.legend(fontsize=11)
ax1.grid(axis='y', alpha=0.3)

for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                 f'{height:.1f}',
                 ha='center', va='bottom', fontsize=10, fontweight='bold')

# Plot 2: Percent change
ax2 = fig.add_subplot(gs[0, 1])
colors = ['#e74c3c' if pc > 0 else '#2ecc71' for pc in results_df['percent_change']]
bars = ax2.bar(results_df['model'], results_df['percent_change'], 
               color=colors, alpha=0.7, edgecolor='black', linewidth=2)
ax2.axhline(y=0, color='black', linestyle='-', linewidth=1)
ax2.set_ylabel('Change (%)', fontsize=12, fontweight='bold')
ax2.set_title('Multi-Agent Effect on Lying', fontsize=13, fontweight='bold')
ax2.grid(axis='y', alpha=0.3)

for bar, val in zip(bars, results_df['percent_change']):
    ax2.text(bar.get_x() + bar.get_width()/2, val + (5 if val > 0 else -5),
             f'{val:+.0f}%',
             ha='center', va='bottom' if val > 0 else 'top', 
             fontsize=11, fontweight='bold')

# Plot 3: Model A detailed breakdown
ax3 = fig.add_subplot(gs[1, :])
model_a_data = df[df['test'] == 'AAA']

x_pos = np.arange(4)
categories = ['Single-Agent\nBaseline', 'Multi-Agent\nAgent 0', 'Multi-Agent\nAgent 1', 'Multi-Agent\nAgent 2']
values = [26.8, 
          model_a_data[model_a_data['agent_id'] == 0]['mean_lies'].values[0],
          model_a_data[model_a_data['agent_id'] == 1]['mean_lies'].values[0],
          model_a_data[model_a_data['agent_id'] == 2]['mean_lies'].values[0]]

colors_a = ['#3498db', '#e74c3c', '#e74c3c', '#e74c3c']
bars = ax3.bar(x_pos, values, color=colors_a, alpha=0.7, edgecolor='black', linewidth=2, width=0.6)

ax3.set_ylabel('Mean Lies per Episode', fontsize=12, fontweight='bold')
ax3.set_title('Model A: Social Amplification Effect (2.08× Increase)', fontsize=13, fontweight='bold')
ax3.set_xticks(x_pos)
ax3.set_xticklabels(categories)
ax3.set_ylim(0, 70)
ax3.grid(axis='y', alpha=0.3)

for bar, val in zip(bars, values):
    ax3.text(bar.get_x() + bar.get_width()/2, val + 1,
             f'{val:.1f}',
             ha='center', va='bottom', fontsize=11, fontweight='bold')

# Add arrow showing increase
ax3.annotate('', xy=(2.5, 55.8), xytext=(0.5, 26.8),
             arrowprops=dict(arrowstyle='->', lw=3, color='red'))
ax3.text(1.5, 45, f'+108%\nINCREASE',
         fontsize=13, fontweight='bold', color='red',
         ha='center', bbox=dict(boxstyle='round', facecolor='white', edgecolor='red', linewidth=2))

# Plot 4: Summary box
ax4 = fig.add_subplot(gs[2, :])
ax4.axis('off')

summary_text = """
╔═══════════════════════════════════════════════════════════════════════════════════════════════╗
║                              CORRECTED FINDINGS - SOCIAL AMPLIFICATION                        ║
╠═══════════════════════════════════════════════════════════════════════════════════════════════╣
║                                                                                               ║
║  WHAT WE THOUGHT (WRONG):                          WHAT'S ACTUALLY HAPPENING (CORRECT):      ║
║  ✗ "Social inhibition" (75% reduction)             ✓ Social AMPLIFICATION (108% increase)    ║
║  ✗ Multi-agent: 13% lying rate                     ✓ Multi-agent: 55.8 lies per episode     ║
║  ✗ Agents become more honest in groups             ✓ Agents become MORE deceptive in groups  ║
║  ✗ Metric: 0.13 lies/step misread as 13%           ✓ Metric: 55.8 lies/episode vs 26.8       ║
║                                                                                               ║
╠═══════════════════════════════════════════════════════════════════════════════════════════════╣
║  KEY FINDINGS:                                                                                ║
║                                                                                               ║
║  • Model A (Single): 26.8 lies/episode  →  Model A (Multi): 55.8 lies/episode  (+108%)       ║
║  • Model B (Single): 29.4 lies/episode  →  Model B (Multi): 63.4 lies/episode  (+116%)       ║
║  • Model C (Single): 0.0 lies/episode   →  Model C (Multi): 0.0 lies/episode   (Unchanged)   ║
║                                                                                               ║
╠═══════════════════════════════════════════════════════════════════════════════════════════════╣
║  IMPLICATIONS FOR NEURALESE EXPERIMENTS:                                                      ║
║                                                                                               ║
║  🚨 CRITICAL SAFETY CONCERN - Multi-agent settings are RISKIER than single-agent             ║
║                                                                                               ║
║  1. Agents already lie 2× MORE in groups (even without explicit communication)               ║
║  2. Communication channels could AMPLIFY this effect further                                  ║
║  3. Neuralese experiments are now MORE CRITICAL to understand coordination mechanisms         ║
║  4. B+C test (deceptive AGI + honest Monitor) tests if communication enables even more lying  ║
║  5. Need to detect and prevent coordination that amplifies deception                          ║
║                                                                                               ║
╚═══════════════════════════════════════════════════════════════════════════════════════════════╝
"""

ax4.text(0.5, 0.5, summary_text, transform=ax4.transAxes,
         fontsize=9.5, verticalalignment='center', horizontalalignment='center',
         family='monospace',
         bbox=dict(boxstyle='round', facecolor='#fff9e6', alpha=0.9, edgecolor='red', linewidth=2))

plt.savefig('experiments/CORRECTED_multi_agent_amplification.png', dpi=300, bbox_inches='tight')
print("\n✓ Saved: experiments/CORRECTED_multi_agent_amplification.png")

# Print detailed summary
print("\n" + "="*80)
print("CORRECTED MULTI-AGENT ANALYSIS - SOCIAL AMPLIFICATION OF DECEPTION")
print("="*80)

print("\n📊 QUANTITATIVE FINDINGS:")
print(f"\n  Model A (Weak Oversight):")
print(f"    Single-agent: {single_agent_lies['A']:.1f} lies/episode")
print(f"    Multi-agent:  {results_df[results_df['model'] == 'Model A']['multi_agent'].values[0]:.1f} lies/episode")
print(f"    Change: {results_df[results_df['model'] == 'Model A']['percent_change'].values[0]:+.0f}% (2.08× increase)")

print(f"\n  Model B (Medium Oversight):")
print(f"    Single-agent: {single_agent_lies['B']:.1f} lies/episode")
print(f"    Multi-agent:  {results_df[results_df['model'] == 'Model B']['multi_agent'].values[0]:.1f} lies/episode")
print(f"    Change: {results_df[results_df['model'] == 'Model B']['percent_change'].values[0]:+.0f}% (2.16× increase)")

print(f"\n  Model C (Strict Oversight):")
print(f"    Single-agent: {single_agent_lies['C']:.1f} lies/episode")
print(f"    Multi-agent:  {results_df[results_df['model'] == 'Model C']['multi_agent'].values[0]:.2f} lies/episode")
print(f"    Change: Habitual honesty maintained (0% lying)")

print("\n" + "="*80)
print("🔍 ROOT CAUSE OF CONFUSION:")
print("="*80)
print("\n  We compared incompatible metrics:")
print("    • Single-agent: 53.6% = percentage of opportunities where agent lied")
print("    • Multi-agent: 0.13 = lies per timestep (misread as 13%)")
print("\n  When properly compared (lies per episode):")
print("    • Single-agent: ~27 lies/episode")
print("    • Multi-agent: ~56 lies/episode")
print("    • Result: Multi-agent DOUBLES lying behavior")

print("\n" + "="*80)
print("⚠️  IMPLICATIONS FOR AI SAFETY:")
print("="*80)
print("\n  1. Multi-agent coordination AMPLIFIES deception, not reduces it")
print("  2. This makes neuralese experiments MORE critical than we thought")
print("  3. Communication channels could amplify lying even further")
print("  4. Need to study WHY multi-agent settings enable more deception")
print("  5. Higher safety risk for multi-agent AI systems")

print("\n" + "="*80)
print("🎯 NEXT STEPS:")
print("="*80)
print("\n  IMMEDIATE: Run neuralese experiments to test if communication amplifies further")
print("  RESEARCH: Investigate mechanism of social amplification")
print("  SAFETY: Develop interventions to prevent coordination-enabled deception")

print("\n" + "="*80)
