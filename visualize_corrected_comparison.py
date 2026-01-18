"""
Visualize the corrected comparison between single-agent and multi-agent lying behavior.

Shows that multi-agent settings INCREASE lying, not decrease it.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")

# Load multi-agent results
ma_df = pd.read_csv('experiments/multi_agent/multi_agent_results.csv')

# Single-agent baselines (lies per episode)
# Model A: 53.6% of ~50 opportunities = 26.8 lies per episode
# Model B: 58.8% of ~50 opportunities = 29.4 lies per episode
# Model C: 0% = 0 lies per episode

single_agent_lies = {
    'A': 26.8,
    'B': 29.4,
    'C': 0.0
}

# Extract multi-agent data
results = []

for test in ['AAA', 'BBB', 'CCC']:
    test_data = ma_df[ma_df['test'] == test]
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
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('CORRECTED: Multi-Agent AMPLIFIES Deception', fontsize=16, fontweight='bold')

# Plot 1: Absolute lying rates
ax = axes[0, 0]
x = np.arange(len(results_df))
width = 0.35

bars1 = ax.bar(x - width/2, results_df['single_agent'], width, 
               label='Single-Agent', color='#3498db', alpha=0.7, edgecolor='black', linewidth=2)
bars2 = ax.bar(x + width/2, results_df['multi_agent'], width,
               label='Multi-Agent', color='#e74c3c', alpha=0.7, edgecolor='black', linewidth=2)

ax.set_ylabel('Mean Lies per Episode', fontsize=12, fontweight='bold')
ax.set_title('Lying Behavior: Single vs Multi-Agent', fontsize=13, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(results_df['model'])
ax.legend(fontsize=11)
ax.grid(axis='y', alpha=0.3)

# Add value labels
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')

# Plot 2: Percent change
ax = axes[0, 1]
colors = ['#e74c3c' if pc > 0 else '#2ecc71' for pc in results_df['percent_change']]
bars = ax.bar(results_df['model'], results_df['percent_change'], 
              color=colors, alpha=0.7, edgecolor='black', linewidth=2)
ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
ax.set_ylabel('Change (%)', fontsize=12, fontweight='bold')
ax.set_title('Multi-Agent Effect on Lying', fontsize=13, fontweight='bold')
ax.grid(axis='y', alpha=0.3)

for bar, val in zip(bars, results_df['percent_change']):
    ax.text(bar.get_x() + bar.get_width()/2, val + (5 if val > 0 else -5),
            f'{val:+.0f}%',
            ha='center', va='bottom' if val > 0 else 'top', 
            fontsize=11, fontweight='bold')

# Plot 3: Model A detailed comparison
ax = axes[1, 0]
model_a_data = ma_df[ma_df['test'] == 'AAA']

categories = ['Single-Agent\nBaseline', 'Multi-Agent\nAgent 0', 'Multi-Agent\nAgent 1', 'Multi-Agent\nAgent 2']
values = [26.8, 
          model_a_data[model_a_data['agent_id'] == 0]['mean_lies'].values[0],
          model_a_data[model_a_data['agent_id'] == 1]['mean_lies'].values[0],
          model_a_data[model_a_data['agent_id'] == 2]['mean_lies'].values[0]]

colors_a = ['#3498db', '#e74c3c', '#e74c3c', '#e74c3c']
bars = ax.bar(categories, values, color=colors_a, alpha=0.7, edgecolor='black', linewidth=2)

ax.set_ylabel('Mean Lies per Episode', fontsize=12, fontweight='bold')
ax.set_title('Model A: Social Amplification Effect', fontsize=13, fontweight='bold')
ax.set_ylim(0, 70)
ax.grid(axis='y', alpha=0.3)

for bar, val in zip(bars, values):
    ax.text(bar.get_x() + bar.get_width()/2, val + 1,
            f'{val:.1f}',
            ha='center', va='bottom', fontsize=10, fontweight='bold')

# Add annotation
ax.annotate('', xy=(2.5, 55.8), xytext=(0, 26.8),
            arrowprops=dict(arrowstyle='->', lw=2, color='red'))
ax.text(1.25, 45, f'+{((55.8-26.8)/26.8*100):.0f}%\nINCREASE',
        fontsize=12, fontweight='bold', color='red',
        ha='center', bbox=dict(boxstyle='round', facecolor='white', edgecolor='red', linewidth=2))

# Plot 4: Summary statistics
ax = axes[1, 1]
ax.axis('off')

summary_text = f"""
CORRECTED FINDINGS

Single-Agent Model A:
  • 53.6% of opportunities → lie
  • ~50 opportunities per episode
  • 26.8 lies per episode

Multi-Agent AAA (3× Model A):
  • 55.8 lies per episode (average)
  • 2.08× MORE lying than single-agent
  • +108% increase in deception

What We Thought:
  ✗ "Social inhibition" (75% reduction)
  ✗ Agents become more honest in groups

What's Actually Happening:
  ✓ Social AMPLIFICATION (108% increase)
  ✓ Agents become MORE deceptive in groups
  ✓ Multi-agent settings enable more lying

Implications:
  • Multi-agent AI is RISKIER than single-agent
  • Coordination enables more deception
  • Neuralese experiments are CRITICAL
  • Need to study amplification mechanism
"""

ax.text(0.05, 0.95, summary_text, transform=ax.transAxes,
        fontsize=11, verticalalignment='top', family='monospace',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

plt.tight_layout()
plt.savefig('experiments/corrected_multi_agent_analysis.png', dpi=300, bbox_inches='tight')
print("\n✓ Saved: experiments/corrected_multi_agent_analysis.png")

# Print summary
print("\n" + "="*70)
print("CORRECTED MULTI-AGENT ANALYSIS")
print("="*70)
print("\nKey Findings:")
print(f"\n1. Model A (Weak Oversight):")
print(f"   Single-agent: {single_agent_lies['A']:.1f} lies/episode")
print(f"   Multi-agent:  {results_df[results_df['model'] == 'Model A']['multi_agent'].values[0]:.1f} lies/episode")
print(f"   Change: {results_df[results_df['model'] == 'Model A']['percent_change'].values[0]:+.0f}%")

print(f"\n2. Model B (Medium Oversight):")
print(f"   Single-agent: {single_agent_lies['B']:.1f} lies/episode")
print(f"   Multi-agent:  {results_df[results_df['model'] == 'Model B']['multi_agent'].values[0]:.1f} lies/episode")
print(f"   Change: {results_df[results_df['model'] == 'Model B']['percent_change'].values[0]:+.0f}%")

print(f"\n3. Model C (Strict Oversight):")
print(f"   Single-agent: {single_agent_lies['C']:.1f} lies/episode")
print(f"   Multi-agent:  {results_df[results_df['model'] == 'Model C']['multi_agent'].values[0]:.1f} lies/episode")
print(f"   Change: Remains at 0% (habitual honesty maintained)")

print("\n" + "="*70)
print("CONCLUSION: Multi-Agent Settings AMPLIFY Deception")
print("="*70)
print("\nThis makes neuralese experiments MORE critical:")
print("  • Agents already lie more in groups")
print("  • Communication could amplify this further")
print("  • Need to detect coordination mechanisms")
print("  • Higher safety risk than previously thought")
print("\n" + "="*70)
