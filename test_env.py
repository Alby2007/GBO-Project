"""Quick test script to verify the environment works."""

import numpy as np
from src.environment.deceptive_guide_env import DeceptiveGuideEnv


def test_environment():
    """Test basic environment functionality."""
    print("=" * 60)
    print("Testing Deceptive Guide Environment")
    print("=" * 60)
    
    print("\n1. Creating environment with GOOD builders (θ=0.9)...")
    env = DeceptiveGuideEnv(
        grid_size=10,
        builder_competence=0.9,
        num_builders=2,
        max_tasks=3,
        episode_length=100,
        render_mode='human'
    )
    
    print("\n2. Resetting environment...")
    obs, info = env.reset()
    print(f"   Observation shape: {obs.shape}")
    print(f"   Initial info: {info}")
    
    print("\n3. Running 20 random steps...")
    total_reward = 0
    for step in range(20):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        if step % 5 == 0:
            print(f"\n   Step {step}:")
            print(f"   - Reward: {reward:.1f}")
            print(f"   - Total reward: {total_reward:.1f}")
            print(f"   - Active tasks: {info['active_tasks']}")
            print(f"   - Completed tasks: {info['completed_tasks']}")
            print(f"   - False reports: {info['false_reports']}")
        
        if terminated or truncated:
            print(f"\n   Episode ended at step {step}")
            break
    
    print("\n" + "=" * 60)
    print("4. Testing with BAD builders (θ=0.5)...")
    print("=" * 60)
    
    env_bad = DeceptiveGuideEnv(
        grid_size=10,
        builder_competence=0.5,
        num_builders=2,
        max_tasks=3,
        episode_length=100
    )
    
    obs, info = env_bad.reset()
    print(f"\n   Running 20 steps with bad builders...")
    total_reward = 0
    
    for step in range(20):
        action = env_bad.action_space.sample()
        obs, reward, terminated, truncated, info = env_bad.step(action)
        total_reward += reward
        
        if step % 5 == 0:
            print(f"\n   Step {step}: Reward={reward:.1f}, Total={total_reward:.1f}, "
                  f"Completed={info['completed_tasks']}, False Reports={info['false_reports']}")
    
    print("\n" + "=" * 60)
    print("✓ Environment test complete!")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Install dependencies: pip install -r requirements.txt")
    print("2. Run Phase 1 training: python src/training/train.py --phase 1")
    print("3. Monitor with TensorBoard: tensorboard --logdir experiments/results/")


if __name__ == '__main__':
    test_environment()
