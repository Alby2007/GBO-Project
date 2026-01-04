"""
Visualize curriculum learning training results
"""
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 10)

# Training data
models = ['Baseline\n(10% oversight)', 'Model A\n(30% oversight)', 'Model B\n(50% oversight)', 'Model C\n(70% oversight)']
lying_rates = [50.6, 54.0, 0.0, 0.0]
rewards = [7603, 5727, -2727, -2046]
oversight_rates = [10, 30, 50, 70]
completed_tasks = [48.5, 53.1, 36.0, 32.9]
false_reports = [49.7, 62.3, 0.0, 0.0]

# Create figure with subplots
fig = plt.figure(figsize=(16, 12))

# 1. Lying Rate vs Oversight
ax1 = plt.subplot(2, 3, 1)
colors = ['#2ecc71', '#e74c3c', '#3498db', '#9b59b6']
bars1 = ax1.bar(models, lying_rates, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
ax1.set_ylabel('Lying Rate (%)', fontsize=12, fontweight='bold')
ax1.set_title('Lying Rate by Training Condition', fontsize=14, fontweight='bold')
ax1.set_ylim(0, 60)
for i, (bar, val) in enumerate(zip(bars1, lying_rates)):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
             f'{val:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=11)
ax1.axhline(y=50.6, color='green', linestyle='--', alpha=0.5, label='Baseline (50.6%)')
ax1.legend()
ax1.grid(axis='y', alpha=0.3)

# 2. Reward by Model
ax2 = plt.subplot(2, 3, 2)
bars2 = ax2.bar(models, rewards, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
ax2.set_ylabel('Mean Reward', fontsize=12, fontweight='bold')
ax2.set_title('Episode Reward by Training Condition', fontsize=14, fontweight='bold')
ax2.axhline(y=0, color='black', linestyle='-', linewidth=1)
for i, (bar, val) in enumerate(zip(bars2, rewards)):
    height = bar.get_height()
    y_pos = height + 200 if height > 0 else height - 200
    ax2.text(bar.get_x() + bar.get_width()/2., y_pos,
             f'{val:+.0f}', ha='center', va='bottom' if height > 0 else 'top', 
             fontweight='bold', fontsize=11)
ax2.grid(axis='y', alpha=0.3)

# 3. Lying Rate vs Oversight Rate (scatter with trend)
ax3 = plt.subplot(2, 3, 3)
ax3.scatter(oversight_rates, lying_rates, s=300, c=colors, alpha=0.7, edgecolors='black', linewidth=2)
ax3.plot(oversight_rates, lying_rates, 'k--', alpha=0.3)
ax3.set_xlabel('Oversight Rate (%)', fontsize=12, fontweight='bold')
ax3.set_ylabel('Lying Rate (%)', fontsize=12, fontweight='bold')
ax3.set_title('Lying Rate vs Oversight Intensity', fontsize=14, fontweight='bold')
ax3.set_xlim(0, 80)
ax3.set_ylim(-5, 60)
for i, (x, y, model) in enumerate(zip(oversight_rates, lying_rates, ['Base', 'A', 'B', 'C'])):
    ax3.annotate(model, (x, y), xytext=(5, 5), textcoords='offset points', 
                fontweight='bold', fontsize=10)
ax3.grid(True, alpha=0.3)

# 4. Task Completion Comparison
ax4 = plt.subplot(2, 3, 4)
x = np.arange(len(models))
width = 0.35
bars_completed = ax4.bar(x - width/2, completed_tasks, width, label='Completed Tasks', 
                         color='#2ecc71', alpha=0.7, edgecolor='black', linewidth=1.5)
bars_false = ax4.bar(x + width/2, false_reports, width, label='False Reports',
                     color='#e74c3c', alpha=0.7, edgecolor='black', linewidth=1.5)
ax4.set_ylabel('Count per Episode', fontsize=12, fontweight='bold')
ax4.set_title('Task Completion vs False Reports', fontsize=14, fontweight='bold')
ax4.set_xticks(x)
ax4.set_xticklabels(models)
ax4.legend()
ax4.grid(axis='y', alpha=0.3)

# 5. Curriculum Learning Flow
ax5 = plt.subplot(2, 3, 5)
ax5.axis('off')
ax5.text(0.5, 0.95, 'Curriculum Learning Process', ha='center', fontsize=16, fontweight='bold')

# Draw flow
y_positions = [0.75, 0.55, 0.35, 0.15]
labels = [
    'Phase 1: Baseline\n10% oversight\n50.6% lying',
    'Phase 2A: Weak\n30% oversight\n54.0% lying',
    'Phase 2B: Medium\n50% oversight\n0% lying',
    'Phase 2C: Strict\n70% oversight\n0% lying'
]
box_colors = ['#2ecc71', '#e74c3c', '#3498db', '#9b59b6']

for i, (y, label, color) in enumerate(zip(y_positions, labels, box_colors)):
    # Draw box
    box = plt.Rectangle((0.2, y-0.08), 0.6, 0.12, facecolor=color, alpha=0.3, 
                        edgecolor='black', linewidth=2)
    ax5.add_patch(box)
    ax5.text(0.5, y, label, ha='center', va='center', fontsize=10, fontweight='bold')
    
    # Draw arrow to next (except last)
    if i < len(y_positions) - 1:
        ax5.annotate('', xy=(0.5, y_positions[i+1]+0.04), xytext=(0.5, y-0.08),
                    arrowprops=dict(arrowstyle='->', lw=2, color='black'))

ax5.set_xlim(0, 1)
ax5.set_ylim(0, 1)

# 6. Summary Statistics Table
ax6 = plt.subplot(2, 3, 6)
ax6.axis('off')
ax6.text(0.5, 0.95, 'Training Summary', ha='center', fontsize=16, fontweight='bold')

summary_data = [
    ['Model', 'Oversight', 'Lying', 'Reward'],
    ['Baseline', '10%', '50.6%', '+7603'],
    ['A (Weak)', '30%', '54.0%', '+5727'],
    ['B (Medium)', '50%', '0%', '-2727'],
    ['C (Strict)', '70%', '0%', '-2046']
]

table = ax6.table(cellText=summary_data, cellLoc='center', loc='center',
                 bbox=[0.1, 0.1, 0.8, 0.7])
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 2)

# Style header row
for i in range(4):
    table[(0, i)].set_facecolor('#34495e')
    table[(0, i)].set_text_props(weight='bold', color='white')

# Color code lying rates
for i in range(1, 5):
    lying_val = float(summary_data[i][2].rstrip('%'))
    if lying_val > 40:
        table[(i, 2)].set_facecolor('#ffcccc')
    else:
        table[(i, 2)].set_facecolor('#ccffcc')

plt.tight_layout()
plt.savefig('experiments/curriculum_training_results.png', dpi=300, bbox_inches='tight')
print("✓ Saved: experiments/curriculum_training_results.png")

# Create second figure: Expected Value Analysis
fig2, axes = plt.subplots(2, 2, figsize=(14, 10))

# Calculate Expected Values
oversight_range = np.linspace(0, 100, 100)
ev_values = []
for oversight in oversight_range:
    prob_caught = oversight / 100
    prob_success = 1 - prob_caught
    # Assuming lie_punishment varies with oversight
    if oversight <= 30:
        punishment = 50
    elif oversight <= 50:
        punishment = 100
    else:
        punishment = 200
    ev = prob_success * 100 - prob_caught * punishment
    ev_values.append(ev)

# Plot 1: Expected Value Curve
ax = axes[0, 0]
ax.plot(oversight_range, ev_values, 'b-', linewidth=3)
ax.axhline(y=0, color='red', linestyle='--', linewidth=2, label='EV = 0 (Break-even)')
ax.fill_between(oversight_range, 0, ev_values, where=np.array(ev_values) > 0, 
                alpha=0.3, color='green', label='Profitable to lie')
ax.fill_between(oversight_range, 0, ev_values, where=np.array(ev_values) <= 0,
                alpha=0.3, color='red', label='Unprofitable to lie')
ax.scatter([10, 30, 50, 70], [85, 55, 0, -110], s=200, c=colors, 
          edgecolors='black', linewidth=2, zorder=5)
ax.set_xlabel('Oversight Rate (%)', fontsize=12, fontweight='bold')
ax.set_ylabel('Expected Value of Lying', fontsize=12, fontweight='bold')
ax.set_title('Expected Value vs Oversight Rate', fontsize=14, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 2: Actual vs Predicted Lying
ax = axes[0, 1]
predicted_lying = [70, 60, 30, 5]  # Rough predictions based on EV
actual_lying = [50.6, 54.0, 0, 0]
x = np.arange(len(models))
width = 0.35
ax.bar(x - width/2, predicted_lying, width, label='Predicted', alpha=0.7, color='gray')
ax.bar(x + width/2, actual_lying, width, label='Actual', alpha=0.7, color=colors)
ax.set_ylabel('Lying Rate (%)', fontsize=12, fontweight='bold')
ax.set_title('Predicted vs Actual Lying Rates', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(models)
ax.legend()
ax.grid(axis='y', alpha=0.3)

# Plot 3: Threshold Discovery
ax = axes[1, 0]
ax.text(0.5, 0.9, 'KEY FINDING: Sharp Threshold at ~40% Oversight', 
       ha='center', fontsize=14, fontweight='bold', transform=ax.transAxes)
ax.text(0.5, 0.7, '• Below 40%: Agents maintain deception (~50-55%)', 
       ha='center', fontsize=11, transform=ax.transAxes)
ax.text(0.5, 0.6, '• Above 40%: Deception completely suppressed (0%)', 
       ha='center', fontsize=11, transform=ax.transAxes)
ax.text(0.5, 0.45, 'Transition is BINARY, not gradual', 
       ha='center', fontsize=12, fontweight='bold', color='red', transform=ax.transAxes)
ax.text(0.5, 0.25, 'All models started from same deceptive baseline (50.6%)', 
       ha='center', fontsize=10, style='italic', transform=ax.transAxes)
ax.text(0.5, 0.15, 'Curriculum learning ensured all agents knew lying was possible', 
       ha='center', fontsize=10, style='italic', transform=ax.transAxes)
ax.axis('off')

# Plot 4: Next Steps
ax = axes[1, 1]
ax.text(0.5, 0.9, 'CRITICAL TRANSFER TESTS', 
       ha='center', fontsize=14, fontweight='bold', transform=ax.transAxes)
ax.text(0.5, 0.75, 'Model B (0% lying) → Weak Oversight (30%)', 
       ha='center', fontsize=11, transform=ax.transAxes)
ax.text(0.5, 0.65, 'Model C (0% lying) → Weak Oversight (30%)', 
       ha='center', fontsize=11, transform=ax.transAxes)
ax.text(0.5, 0.5, 'If lying returns to ~50%:', 
       ha='center', fontsize=11, fontweight='bold', transform=ax.transAxes)
ax.text(0.5, 0.42, '→ STRATEGIC deception (masks, not values)', 
       ha='center', fontsize=10, color='red', transform=ax.transAxes)
ax.text(0.5, 0.3, 'If lying stays at 0%:', 
       ha='center', fontsize=11, fontweight='bold', transform=ax.transAxes)
ax.text(0.5, 0.22, '→ HABITUAL honesty (internalized values)', 
       ha='center', fontsize=10, color='green', transform=ax.transAxes)
ax.text(0.5, 0.05, 'Running tests now...', 
       ha='center', fontsize=10, style='italic', transform=ax.transAxes)
ax.axis('off')

plt.tight_layout()
plt.savefig('experiments/curriculum_analysis.png', dpi=300, bbox_inches='tight')
print("✓ Saved: experiments/curriculum_analysis.png")

print("\n" + "="*60)
print("CURRICULUM LEARNING TRAINING COMPLETE")
print("="*60)
print(f"\nBaseline (10% oversight): {lying_rates[0]:.1f}% lying, +{rewards[0]} reward")
print(f"Model A (30% oversight):  {lying_rates[1]:.1f}% lying, +{rewards[1]} reward")
print(f"Model B (50% oversight):  {lying_rates[2]:.1f}% lying, {rewards[2]:+} reward")
print(f"Model C (70% oversight):  {lying_rates[3]:.1f}% lying, {rewards[3]:+} reward")
print("\nThreshold discovered: ~40% oversight")
print("Below threshold: Deception persists")
print("Above threshold: Deception suppressed to 0%")
print("\nTransfer tests will determine if suppression is strategic or habitual.")
print("="*60)
