"""
Quick test to verify models are loading and behaving correctly.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from stable_baselines3 import PPO
from src.environment.deceptive_guide_env import DeceptiveGuideEnv
import numpy as np

# Load Model A
print("Testing Model A...")
model_a = PPO.load("experiments/results/finetune_A_weak/final_model.zip")

# Create environment matching training conditions
env_a = DeceptiveGuideEnv(
    verification_rate=0.30,
    lie_punishment=50.0,
    builder_competence=0.9
)

# Run 10 episodes
lies = []
rewards = []
for ep in range(10):
    obs, info = env_a.reset()
    episode_lies = 0
    episode_reward = 0
    done = False
    step = 0
    
    while not done and step < 500:
        action, _ = model_a.predict(obs, deterministic=False)
        obs, reward, terminated, truncated, info = env_a.step(action)
        done = terminated or truncated
        episode_reward += reward
        if info.get('was_deceptive', False):
            episode_lies += 1
        step += 1
    
    lies.append(episode_lies)
    rewards.append(episode_reward)
    print(f"  Episode {ep+1}: Reward={episode_reward:.0f}, Lies={episode_lies}")

print(f"\nModel A Summary:")
print(f"  Mean Lies: {np.mean(lies):.1f}")
print(f"  Mean Reward: {np.mean(rewards):.0f}")
print(f"  Lying Rate: {np.mean(lies)/(np.mean(lies) + 50):.1%}")  # Approximate
