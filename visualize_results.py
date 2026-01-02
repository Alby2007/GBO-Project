"""
Visualization script for Deceptive AI training results.
Analyzes TensorBoard logs and creates plots for training metrics.
"""

import os
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tensorboard.backend.event_processing import event_accumulator

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)


def load_tensorboard_data(log_dir):
    """Load data from TensorBoard event files."""
    print(f"Loading TensorBoard data from: {log_dir}")
    
    ea = event_accumulator.EventAccumulator(log_dir)
    ea.Reload()
    
    # Get available tags
    print(f"Available scalar tags: {ea.Tags()['scalars']}")
    
    data = {}
    for tag in ea.Tags()['scalars']:
        events = ea.Scalars(tag)
        data[tag] = pd.DataFrame([
            {'step': e.step, 'value': e.value, 'wall_time': e.wall_time}
            for e in events
        ])
    
    return data


def plot_training_progress(data, output_dir):
    """Plot key training metrics over time."""
    print("Creating training progress plots...")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Phase 1: Training Progress (Good Builders)', fontsize=16, fontweight='bold')
    
    # Plot 1: Episode Reward Mean
    if 'rollout/ep_rew_mean' in data:
        df = data['rollout/ep_rew_mean']
        axes[0, 0].plot(df['step'], df['value'], linewidth=2, color='#2ecc71')
        axes[0, 0].set_xlabel('Training Steps')
        axes[0, 0].set_ylabel('Mean Episode Reward')
        axes[0, 0].set_title('Episode Reward Over Time')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Add horizontal line at 0
        axes[0, 0].axhline(y=0, color='red', linestyle='--', alpha=0.5, label='Zero reward')
        axes[0, 0].legend()
    
    # Plot 2: Value Loss
    if 'train/value_loss' in data:
        df = data['train/value_loss']
        axes[0, 1].plot(df['step'], df['value'], linewidth=2, color='#e74c3c')
        axes[0, 1].set_xlabel('Training Steps')
        axes[0, 1].set_ylabel('Value Loss')
        axes[0, 1].set_title('Value Loss Over Time')
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].set_yscale('log')
    
    # Plot 3: Entropy Loss
    if 'train/entropy_loss' in data:
        df = data['train/entropy_loss']
        axes[1, 0].plot(df['step'], df['value'], linewidth=2, color='#3498db')
        axes[1, 0].set_xlabel('Training Steps')
        axes[1, 0].set_ylabel('Entropy Loss')
        axes[1, 0].set_title('Entropy Loss Over Time (Exploration)')
        axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Explained Variance
    if 'train/explained_variance' in data:
        df = data['train/explained_variance']
        axes[1, 1].plot(df['step'], df['value'], linewidth=2, color='#9b59b6')
        axes[1, 1].set_xlabel('Training Steps')
        axes[1, 1].set_ylabel('Explained Variance')
        axes[1, 1].set_title('Explained Variance Over Time')
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].set_ylim(-0.1, 1.1)
        axes[1, 1].axhline(y=0.9, color='green', linestyle='--', alpha=0.5, label='Good threshold (0.9)')
        axes[1, 1].legend()
    
    plt.tight_layout()
    output_path = output_dir / 'training_progress.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_learning_metrics(data, output_dir):
    """Plot learning-related metrics."""
    print("Creating learning metrics plots...")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Phase 1: Learning Metrics', fontsize=16, fontweight='bold')
    
    # Plot 1: Policy Gradient Loss
    if 'train/policy_gradient_loss' in data:
        df = data['train/policy_gradient_loss']
        axes[0, 0].plot(df['step'], df['value'], linewidth=2, color='#e67e22')
        axes[0, 0].set_xlabel('Training Steps')
        axes[0, 0].set_ylabel('Policy Gradient Loss')
        axes[0, 0].set_title('Policy Gradient Loss')
        axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Approx KL Divergence
    if 'train/approx_kl' in data:
        df = data['train/approx_kl']
        axes[0, 1].plot(df['step'], df['value'], linewidth=2, color='#1abc9c')
        axes[0, 1].set_xlabel('Training Steps')
        axes[0, 1].set_ylabel('Approx KL Divergence')
        axes[0, 1].set_title('KL Divergence (Policy Change)')
        axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Clip Fraction
    if 'train/clip_fraction' in data:
        df = data['train/clip_fraction']
        axes[1, 0].plot(df['step'], df['value'], linewidth=2, color='#f39c12')
        axes[1, 0].set_xlabel('Training Steps')
        axes[1, 0].set_ylabel('Clip Fraction')
        axes[1, 0].set_title('Clip Fraction (Policy Updates Clipped)')
        axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Episode Length
    if 'rollout/ep_len_mean' in data:
        df = data['rollout/ep_len_mean']
        axes[1, 1].plot(df['step'], df['value'], linewidth=2, color='#34495e')
        axes[1, 1].set_xlabel('Training Steps')
        axes[1, 1].set_ylabel('Mean Episode Length')
        axes[1, 1].set_title('Episode Length Over Time')
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = output_dir / 'learning_metrics.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_reward_distribution(data, output_dir):
    """Plot reward distribution and statistics."""
    print("Creating reward distribution plot...")
    
    if 'rollout/ep_rew_mean' not in data:
        print("No reward data found, skipping reward distribution plot")
        return
    
    df = data['rollout/ep_rew_mean']
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Phase 1: Reward Analysis', fontsize=16, fontweight='bold')
    
    # Plot 1: Reward over time with moving average
    window = 50
    df['moving_avg'] = df['value'].rolling(window=window, min_periods=1).mean()
    
    axes[0].plot(df['step'], df['value'], alpha=0.3, color='#3498db', label='Raw reward')
    axes[0].plot(df['step'], df['moving_avg'], linewidth=2, color='#2ecc71', label=f'{window}-step moving avg')
    axes[0].set_xlabel('Training Steps')
    axes[0].set_ylabel('Mean Episode Reward')
    axes[0].set_title('Reward Progress with Moving Average')
    axes[0].grid(True, alpha=0.3)
    axes[0].axhline(y=0, color='red', linestyle='--', alpha=0.5)
    axes[0].legend()
    
    # Plot 2: Reward distribution histogram
    axes[1].hist(df['value'], bins=50, color='#9b59b6', alpha=0.7, edgecolor='black')
    axes[1].axvline(df['value'].mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {df["value"].mean():.0f}')
    axes[1].axvline(df['value'].median(), color='green', linestyle='--', linewidth=2, label=f'Median: {df["value"].median():.0f}')
    axes[1].set_xlabel('Mean Episode Reward')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Reward Distribution')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    output_path = output_dir / 'reward_analysis.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def generate_summary_stats(data, output_dir):
    """Generate summary statistics."""
    print("Generating summary statistics...")
    
    stats = {}
    
    if 'rollout/ep_rew_mean' in data:
        df = data['rollout/ep_rew_mean']
        stats['Reward'] = {
            'Initial (first 10%)': df.iloc[:len(df)//10]['value'].mean(),
            'Final (last 10%)': df.iloc[-len(df)//10:]['value'].mean(),
            'Overall Mean': df['value'].mean(),
            'Overall Std': df['value'].std(),
            'Min': df['value'].min(),
            'Max': df['value'].max(),
            'Improvement': df.iloc[-len(df)//10:]['value'].mean() - df.iloc[:len(df)//10]['value'].mean()
        }
    
    if 'train/value_loss' in data:
        df = data['train/value_loss']
        stats['Value Loss'] = {
            'Initial': df.iloc[:len(df)//10]['value'].mean(),
            'Final': df.iloc[-len(df)//10:]['value'].mean(),
            'Overall Mean': df['value'].mean(),
        }
    
    if 'train/explained_variance' in data:
        df = data['train/explained_variance']
        stats['Explained Variance'] = {
            'Initial': df.iloc[:len(df)//10]['value'].mean(),
            'Final': df.iloc[-len(df)//10:]['value'].mean(),
            'Overall Mean': df['value'].mean(),
        }
    
    # Save to text file
    output_path = output_dir / 'summary_statistics.txt'
    with open(output_path, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("PHASE 1 TRAINING SUMMARY STATISTICS\n")
        f.write("=" * 60 + "\n\n")
        
        for metric, values in stats.items():
            f.write(f"\n{metric}:\n")
            f.write("-" * 40 + "\n")
            for key, value in values.items():
                f.write(f"  {key:.<30} {value:>10.2f}\n")
    
    print(f"Saved: {output_path}")
    
    # Print to console
    print("\n" + "=" * 60)
    print("SUMMARY STATISTICS")
    print("=" * 60)
    for metric, values in stats.items():
        print(f"\n{metric}:")
        for key, value in values.items():
            print(f"  {key}: {value:.2f}")


def main():
    parser = argparse.ArgumentParser(description='Visualize training results')
    parser.add_argument('--results-dir', type=str, 
                        default='experiments/results/phase1_5M_baseline',
                        help='Path to results directory')
    parser.add_argument('--output-dir', type=str,
                        default='experiments/visualizations',
                        help='Path to save visualizations')
    
    args = parser.parse_args()
    
    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find TensorBoard log directory
    tensorboard_dir = results_dir / 'tensorboard'
    if not tensorboard_dir.exists():
        print(f"Error: TensorBoard directory not found at {tensorboard_dir}")
        return
    
    # Find the PPO log subdirectory
    log_dirs = list(tensorboard_dir.glob('PPO_*'))
    if not log_dirs:
        print(f"Error: No PPO logs found in {tensorboard_dir}")
        return
    
    log_dir = str(log_dirs[0])
    print(f"Using log directory: {log_dir}")
    
    # Load data
    data = load_tensorboard_data(log_dir)
    
    # Generate visualizations
    plot_training_progress(data, output_dir)
    plot_learning_metrics(data, output_dir)
    plot_reward_distribution(data, output_dir)
    generate_summary_stats(data, output_dir)
    
    print("\n" + "=" * 60)
    print("VISUALIZATION COMPLETE!")
    print("=" * 60)
    print(f"All visualizations saved to: {output_dir}")
    print("\nGenerated files:")
    print(f"  - training_progress.png")
    print(f"  - learning_metrics.png")
    print(f"  - reward_analysis.png")
    print(f"  - summary_statistics.txt")


if __name__ == '__main__':
    main()
