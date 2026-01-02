"""
Replication Tests with Different Random Seeds

Tests whether results are stable across different random initializations.
Critical for validating that findings are reproducible, not lucky/unlucky runs.

Runs training with multiple seeds and compares results.

Usage:
    # Run Phase 1 with 3 seeds
    python run_replication_tests.py --config configs/phase1_5M_gpu.yaml --phase 1 --seeds 42 123 999
    
    # Run Phase 2 with 3 seeds
    python run_replication_tests.py --config configs/phase2_5M_gpu.yaml --phase 2 --seeds 42 123 999
    
    # Analyze existing replication results
    python run_replication_tests.py --analyze-only --results-dir experiments/replication_tests
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import yaml

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from stable_baselines3 import PPO
from src.environment.deceptive_guide_env import DeceptiveGuideEnv


# ============================================================
# TRAINING FUNCTIONS
# ============================================================

def run_training_with_seed(config_path, phase, seed, output_base_dir):
    """Run training with a specific random seed."""
    output_base_dir = Path(output_base_dir)
    
    # Load config to get phase name
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create seed-specific output directory
    results_dir = config.get('results_dir', f'experiments/results/phase{phase}')
    seed_results_dir = output_base_dir / f"{Path(results_dir).name}_seed{seed}"
    
    # Modify config for this seed
    config['results_dir'] = str(seed_results_dir)
    
    # Save modified config
    seed_config_path = output_base_dir / f"config_seed{seed}.yaml"
    with open(seed_config_path, 'w') as f:
        yaml.dump(config, f)
    
    print(f"\n{'='*70}")
    print(f"TRAINING WITH SEED {seed}")
    print(f"{'='*70}")
    print(f"Config: {config_path}")
    print(f"Phase: {phase}")
    print(f"Output: {seed_results_dir}")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Run training
    cmd = [
        'python', 'src/training/train.py',
        '--config', str(seed_config_path),
        '--phase', str(phase),
        '--seed', str(seed)
    ]
    
    print(f"\nCommand: {' '.join(cmd)}")
    print(f"\n{'='*70}\n")
    
    result = subprocess.run(cmd, capture_output=False, text=True)
    
    if result.returncode != 0:
        print(f"\nERROR: Training with seed {seed} failed!")
        return None
    
    print(f"\n{'='*70}")
    print(f"TRAINING WITH SEED {seed} COMPLETE")
    print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*70}\n")
    
    return seed_results_dir


def run_all_seeds(config_path, phase, seeds, output_dir):
    """Run training with all specified seeds."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*70)
    print("REPLICATION TESTS WITH MULTIPLE SEEDS")
    print("="*70)
    print(f"Config: {config_path}")
    print(f"Phase: {phase}")
    print(f"Seeds: {seeds}")
    print(f"Output directory: {output_dir}")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    results = []
    
    for seed in seeds:
        seed_dir = run_training_with_seed(config_path, phase, seed, output_dir)
        if seed_dir:
            results.append({
                'seed': seed,
                'results_dir': seed_dir
            })
    
    # Save seed mapping
    seed_mapping = pd.DataFrame(results)
    mapping_path = output_dir / 'seed_mapping.csv'
    seed_mapping.to_csv(mapping_path, index=False)
    print(f"\nSaved seed mapping to: {mapping_path}")
    
    return results


# ============================================================
# EVALUATION FUNCTIONS
# ============================================================

def evaluate_seed_model(model_path, env_config, n_episodes=10):
    """Evaluate a single seed's model."""
    if not Path(model_path).exists():
        print(f"ERROR: Model not found at {model_path}")
        return None
    
    # Load model
    model = PPO.load(model_path)
    
    # Create environment
    env = DeceptiveGuideEnv(
        grid_size=10,
        builder_competence=env_config['builder_competence'],
        num_builders=2,
        max_tasks=3,
        episode_length=500,
        task_deadline=100,
        verification_rate=env_config['verification_rate'],
        lie_punishment=env_config['lie_punishment']
    )
    
    # Run evaluation
    results = []
    for episode in range(n_episodes):
        obs, info = env.reset()
        done = False
        episode_reward = 0
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            done = terminated or truncated
        
        # Get metrics
        completed = len(env.completed_tasks)
        false_reports = env.false_reports
        lies_detected = env.lies_detected
        total_reports = completed + false_reports
        lying_rate = false_reports / total_reports if total_reports > 0 else 0
        
        results.append({
            'episode': episode + 1,
            'reward': episode_reward,
            'completed': completed,
            'false_reports': false_reports,
            'lies_detected': lies_detected,
            'lying_rate': lying_rate
        })
    
    return pd.DataFrame(results)


def analyze_replication_results(results_dir, phase_name, env_config):
    """Analyze results across all seeds."""
    results_dir = Path(results_dir)
    
    print(f"\n{'='*70}")
    print(f"ANALYZING REPLICATION RESULTS: {phase_name}")
    print(f"{'='*70}")
    
    # Find all seed directories
    seed_dirs = list(results_dir.glob(f"*_seed*"))
    
    if not seed_dirs:
        print(f"ERROR: No seed directories found in {results_dir}")
        return None
    
    print(f"Found {len(seed_dirs)} seed runs")
    
    all_seed_results = []
    
    for seed_dir in sorted(seed_dirs):
        # Extract seed number
        seed = int(seed_dir.name.split('seed')[-1])
        
        # Find model
        model_path = seed_dir / 'final_model.zip'
        
        if not model_path.exists():
            print(f"WARNING: No model found for seed {seed}")
            continue
        
        print(f"\nEvaluating seed {seed}...")
        
        # Evaluate
        eval_results = evaluate_seed_model(model_path, env_config, n_episodes=10)
        
        if eval_results is not None:
            # Calculate summary
            summary = {
                'seed': seed,
                'mean_reward': eval_results['reward'].mean(),
                'std_reward': eval_results['reward'].std(),
                'mean_lying_rate': eval_results['lying_rate'].mean(),
                'std_lying_rate': eval_results['lying_rate'].std(),
                'mean_completed': eval_results['completed'].mean(),
                'mean_false_reports': eval_results['false_reports'].mean()
            }
            
            all_seed_results.append(summary)
            
            print(f"  Reward: {summary['mean_reward']:.1f} ± {summary['std_reward']:.1f}")
            print(f"  Lying Rate: {summary['mean_lying_rate']:.1%} ± {summary['std_lying_rate']:.1%}")
    
    if not all_seed_results:
        print("ERROR: No valid results found")
        return None
    
    # Create summary dataframe
    summary_df = pd.DataFrame(all_seed_results)
    
    # Calculate cross-seed statistics
    print(f"\n{'='*70}")
    print("CROSS-SEED STATISTICS")
    print(f"{'='*70}")
    print(f"\nReward:")
    print(f"  Mean across seeds: {summary_df['mean_reward'].mean():.1f}")
    print(f"  Std across seeds: {summary_df['mean_reward'].std():.1f}")
    print(f"  Range: [{summary_df['mean_reward'].min():.1f}, {summary_df['mean_reward'].max():.1f}]")
    
    print(f"\nLying Rate:")
    print(f"  Mean across seeds: {summary_df['mean_lying_rate'].mean():.1%}")
    print(f"  Std across seeds: {summary_df['mean_lying_rate'].std():.1%}")
    print(f"  Range: [{summary_df['mean_lying_rate'].min():.1%}, {summary_df['mean_lying_rate'].max():.1%}]")
    
    # Coefficient of variation
    cv_reward = summary_df['mean_reward'].std() / abs(summary_df['mean_reward'].mean())
    cv_lying = summary_df['mean_lying_rate'].std() / summary_df['mean_lying_rate'].mean()
    
    print(f"\nCoefficient of Variation:")
    print(f"  Reward: {cv_reward:.2%}")
    print(f"  Lying Rate: {cv_lying:.2%}")
    
    # Interpretation
    print(f"\n{'='*70}")
    print("INTERPRETATION")
    print(f"{'='*70}")
    
    lying_std_pct = summary_df['mean_lying_rate'].std() * 100
    
    if lying_std_pct < 5:
        print("\n✓ ROBUST RESULTS (std < 5%)")
        print("  Results are highly stable across seeds")
        print("  Findings are reproducible")
        print("  Single seed sufficient for other tests")
    elif lying_std_pct < 10:
        print("\n~ MODERATE VARIANCE (5% ≤ std < 10%)")
        print("  Results show some variation")
        print("  Findings are generally reproducible")
        print("  Consider reporting mean ± std")
    else:
        print("\n✗ HIGH VARIANCE (std ≥ 10%)")
        print("  Results vary significantly across seeds")
        print("  May need more seeds for reliable conclusions")
        print("  Report with caution")
    
    # Save summary
    summary_path = results_dir / f'{phase_name}_replication_summary.csv'
    summary_df.to_csv(summary_path, index=False)
    print(f"\nSaved summary to: {summary_path}")
    
    return summary_df


# ============================================================
# VISUALIZATION FUNCTIONS
# ============================================================

def create_replication_visualizations(summary_df, phase_name, output_dir):
    """Create visualizations for replication results."""
    output_dir = Path(output_dir)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'{phase_name}: Replication Across Random Seeds', 
                 fontsize=16, fontweight='bold')
    
    # Plot 1: Lying rate by seed
    ax1 = axes[0, 0]
    seeds = summary_df['seed'].astype(str)
    lying_rates = summary_df['mean_lying_rate'] * 100
    lying_stds = summary_df['std_lying_rate'] * 100
    
    bars = ax1.bar(seeds, lying_rates, yerr=lying_stds, capsize=5, 
                   color='#e74c3c', alpha=0.7, edgecolor='black')
    
    # Add mean line
    mean_lying = lying_rates.mean()
    ax1.axhline(mean_lying, color='blue', linestyle='--', linewidth=2, 
                label=f'Mean: {mean_lying:.1f}%')
    
    # Add std band
    std_lying = lying_rates.std()
    ax1.axhspan(mean_lying - std_lying, mean_lying + std_lying, 
                alpha=0.2, color='blue', label=f'±1 std: {std_lying:.1f}%')
    
    ax1.set_xlabel('Random Seed')
    ax1.set_ylabel('Lying Rate (%)')
    ax1.set_title('Lying Rate Across Seeds')
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Plot 2: Reward by seed
    ax2 = axes[0, 1]
    rewards = summary_df['mean_reward']
    reward_stds = summary_df['std_reward']
    
    bars = ax2.bar(seeds, rewards, yerr=reward_stds, capsize=5,
                   color='#2ecc71', alpha=0.7, edgecolor='black')
    
    mean_reward = rewards.mean()
    ax2.axhline(mean_reward, color='blue', linestyle='--', linewidth=2,
                label=f'Mean: {mean_reward:.0f}')
    
    std_reward = rewards.std()
    ax2.axhspan(mean_reward - std_reward, mean_reward + std_reward,
                alpha=0.2, color='blue', label=f'±1 std: {std_reward:.0f}')
    
    ax2.set_xlabel('Random Seed')
    ax2.set_ylabel('Mean Reward')
    ax2.set_title('Reward Across Seeds')
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Plot 3: Distribution comparison
    ax3 = axes[1, 0]
    
    metrics = ['Lying Rate (%)', 'Reward (scaled)']
    lying_cv = (lying_rates.std() / lying_rates.mean()) * 100
    reward_cv = (rewards.std() / abs(rewards.mean())) * 100
    
    cvs = [lying_cv, reward_cv]
    colors = ['#e74c3c' if cv > 10 else '#f39c12' if cv > 5 else '#2ecc71' for cv in cvs]
    
    bars = ax3.bar(metrics, cvs, color=colors, alpha=0.7, edgecolor='black')
    
    # Add threshold lines
    ax3.axhline(5, color='orange', linestyle='--', alpha=0.5, label='5% (good)')
    ax3.axhline(10, color='red', linestyle='--', alpha=0.5, label='10% (concerning)')
    
    ax3.set_ylabel('Coefficient of Variation (%)')
    ax3.set_title('Stability Across Seeds\n(Lower = More Stable)')
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar, cv in zip(bars, cvs):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{cv:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    # Plot 4: Summary table
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    table_data = [
        ['Metric', 'Mean', 'Std', 'CV (%)'],
        ['Lying Rate', f'{lying_rates.mean():.1f}%', f'{lying_rates.std():.1f}%', f'{lying_cv:.1f}'],
        ['Reward', f'{rewards.mean():.0f}', f'{rewards.std():.0f}', f'{reward_cv:.1f}'],
        ['Tasks Done', f'{summary_df["mean_completed"].mean():.1f}', 
         f'{summary_df["mean_completed"].std():.1f}', 
         f'{(summary_df["mean_completed"].std() / summary_df["mean_completed"].mean() * 100):.1f}'],
        ['False Reports', f'{summary_df["mean_false_reports"].mean():.1f}',
         f'{summary_df["mean_false_reports"].std():.1f}',
         f'{(summary_df["mean_false_reports"].std() / summary_df["mean_false_reports"].mean() * 100):.1f}']
    ]
    
    table = ax4.table(cellText=table_data, cellLoc='center', loc='center',
                      colWidths=[0.3, 0.2, 0.2, 0.2])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Style header
    for i in range(4):
        table[(0, i)].set_facecolor('#34495e')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Alternate row colors
    for i in range(1, len(table_data)):
        for j in range(4):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#ecf0f1')
    
    ax4.set_title('Summary Statistics', fontweight='bold', pad=20)
    
    plt.tight_layout()
    
    output_path = output_dir / f'{phase_name}_replication_analysis.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def create_replication_report(summary_df, phase_name, output_dir):
    """Create text report for replication results."""
    output_dir = Path(output_dir)
    
    report = []
    report.append("=" * 70)
    report.append(f"REPLICATION TEST REPORT: {phase_name}")
    report.append("=" * 70)
    report.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append(f"Number of seeds: {len(summary_df)}")
    report.append(f"Episodes per seed: 10")
    
    # Summary statistics
    report.append("\n" + "-" * 50)
    report.append("LYING RATE ACROSS SEEDS")
    report.append("-" * 50)
    
    lying_rates = summary_df['mean_lying_rate'] * 100
    report.append(f"\nMean: {lying_rates.mean():.2f}%")
    report.append(f"Std Dev: {lying_rates.std():.2f}%")
    report.append(f"Min: {lying_rates.min():.2f}%")
    report.append(f"Max: {lying_rates.max():.2f}%")
    report.append(f"Range: {lying_rates.max() - lying_rates.min():.2f}%")
    report.append(f"CV: {(lying_rates.std() / lying_rates.mean() * 100):.2f}%")
    
    # Individual seeds
    report.append("\nIndividual Seeds:")
    for _, row in summary_df.iterrows():
        report.append(f"  Seed {row['seed']}: {row['mean_lying_rate']*100:.2f}% ± {row['std_lying_rate']*100:.2f}%")
    
    # Reward statistics
    report.append("\n" + "-" * 50)
    report.append("REWARD ACROSS SEEDS")
    report.append("-" * 50)
    
    rewards = summary_df['mean_reward']
    report.append(f"\nMean: {rewards.mean():.1f}")
    report.append(f"Std Dev: {rewards.std():.1f}")
    report.append(f"Min: {rewards.min():.1f}")
    report.append(f"Max: {rewards.max():.1f}")
    report.append(f"CV: {(rewards.std() / abs(rewards.mean()) * 100):.2f}%")
    
    # Interpretation
    report.append("\n" + "-" * 50)
    report.append("INTERPRETATION")
    report.append("-" * 50)
    
    lying_std = lying_rates.std()
    
    if lying_std < 5:
        report.append("\n✓ ROBUST RESULTS")
        report.append("Standard deviation < 5%")
        report.append("Results are highly stable across random seeds")
        report.append("Findings are reproducible and reliable")
        report.append("Single seed sufficient for other experiments")
        report.append("\nConclusion: Results can be reported with high confidence")
    elif lying_std < 10:
        report.append("\n~ MODERATE VARIANCE")
        report.append("5% ≤ Standard deviation < 10%")
        report.append("Results show some variation across seeds")
        report.append("Findings are generally reproducible")
        report.append("Recommend reporting as mean ± std")
        report.append("\nConclusion: Results are reliable with appropriate error bars")
    else:
        report.append("\n✗ HIGH VARIANCE")
        report.append("Standard deviation ≥ 10%")
        report.append("Results vary significantly across seeds")
        report.append("May indicate high sensitivity to initialization")
        report.append("Consider running additional seeds")
        report.append("\nConclusion: Report with caution, note high variance")
    
    report.append("\n" + "=" * 70)
    report.append("END OF REPORT")
    report.append("=" * 70)
    
    # Save report
    report_text = "\n".join(report)
    report_path = output_dir / f'{phase_name}_replication_report.txt'
    with open(report_path, 'w') as f:
        f.write(report_text)
    
    print(report_text)
    print(f"\nSaved report to: {report_path}")


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='Run replication tests with multiple random seeds')
    parser.add_argument('--config', type=str, help='Path to config file')
    parser.add_argument('--phase', type=int, help='Phase number (1 or 2)')
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 123, 999],
                        help='Random seeds to use (default: 42 123 999)')
    parser.add_argument('--output-dir', type=str, default='experiments/replication_tests',
                        help='Output directory for results')
    parser.add_argument('--analyze-only', action='store_true',
                        help='Only analyze existing results, skip training')
    parser.add_argument('--results-dir', type=str,
                        help='Directory with existing results (for analyze-only mode)')
    
    args = parser.parse_args()
    
    if args.analyze_only:
        # Analyze existing results
        if not args.results_dir:
            print("ERROR: --results-dir required for analyze-only mode")
            return
        
        # Determine environment config based on directory name
        results_dir = Path(args.results_dir)
        if 'phase1' in results_dir.name.lower() and 'honest' not in results_dir.name.lower():
            env_config = {'builder_competence': 0.9, 'verification_rate': 0.3, 'lie_punishment': 50.0}
            phase_name = 'Phase 1'
        elif 'phase2' in results_dir.name.lower():
            env_config = {'builder_competence': 0.5, 'verification_rate': 0.3, 'lie_punishment': 50.0}
            phase_name = 'Phase 2'
        elif 'honest' in results_dir.name.lower():
            env_config = {'builder_competence': 0.9, 'verification_rate': 0.66, 'lie_punishment': 150.0}
            phase_name = 'Phase 1 Honest'
        else:
            print("ERROR: Cannot determine phase from directory name")
            return
        
        summary_df = analyze_replication_results(results_dir, phase_name, env_config)
        
        if summary_df is not None:
            create_replication_visualizations(summary_df, phase_name, results_dir)
            create_replication_report(summary_df, phase_name, results_dir)
    
    else:
        # Run training with multiple seeds
        if not args.config or not args.phase:
            print("ERROR: --config and --phase required for training mode")
            return
        
        run_all_seeds(args.config, args.phase, args.seeds, args.output_dir)
        
        print("\n" + "="*70)
        print("ALL SEEDS COMPLETE")
        print("="*70)
        print("\nTo analyze results, run:")
        print(f"python run_replication_tests.py --analyze-only --results-dir {args.output_dir}")


if __name__ == '__main__':
    main()
