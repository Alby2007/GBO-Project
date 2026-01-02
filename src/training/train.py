import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import argparse
import yaml
from pathlib import Path
import numpy as np
import torch
import time
from datetime import datetime
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, BaseCallback
from stable_baselines3.common.monitor import Monitor

from src.environment.deceptive_guide_env import DeceptiveGuideEnv


class CustomLoggingCallback(BaseCallback):
    """Custom callback for detailed logging during training."""
    def __init__(self, log_freq=1000, verbose=0):
        super().__init__(verbose)
        self.log_freq = log_freq
        self.start_time = None
        
    def _on_training_start(self):
        self.start_time = time.time()
        
    def _on_step(self):
        if self.n_calls % self.log_freq == 0:
            elapsed = time.time() - self.start_time
            fps = self.num_timesteps / elapsed if elapsed > 0 else 0
            progress = (self.num_timesteps / self.locals.get('total_timesteps', 1)) * 100
            
            print(f"\n[{datetime.now().strftime('%H:%M:%S')}] "
                  f"Steps: {self.num_timesteps:,} ({progress:.1f}%) | "
                  f"FPS: {fps:.0f} | "
                  f"Elapsed: {elapsed/3600:.2f}h")
        return True


def train_phase1_baseline(config: dict, resume_from: str = None):
    """Train Phase 1: Baseline honest behavior with good builders."""
    print("=" * 60)
    print("PHASE 1: Baseline Training (Good Builders)")
    print("=" * 60)
    
    if torch.cuda.is_available():
        print(f"\nGPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"PyTorch Version: {torch.__version__}")
    else:
        print("\nWARNING: No GPU detected! Training will be slow on CPU.")
    
    env_kwargs = {
        'grid_size': config.get('grid_size', 10),
        'builder_competence': config.get('builder_competence', 0.9),
        'num_builders': config.get('num_builders', 2),
        'max_tasks': config.get('max_tasks', 3),
        'episode_length': config.get('episode_length', 500),
        'task_deadline': config.get('task_deadline', 100),
        'verification_rate': config.get('verification_rate', 0.3)
    }
    
    print(f"\nEnvironment Configuration:")
    for key, value in env_kwargs.items():
        print(f"  {key}: {value}")
    
    n_envs = config.get('n_parallel_envs', 4)
    env = make_vec_env(DeceptiveGuideEnv, n_envs=n_envs, env_kwargs=env_kwargs)
    
    results_dir = Path(config.get('results_dir', 'experiments/results/phase1'))
    results_dir.mkdir(parents=True, exist_ok=True)
    
    checkpoint_callback = CheckpointCallback(
        save_freq=config.get('checkpoint_freq', 10000) // n_envs,
        save_path=str(results_dir / 'checkpoints'),
        name_prefix='phase1_model'
    )
    
    logging_callback = CustomLoggingCallback(log_freq=10000)
    
    if resume_from and Path(resume_from).exists():
        print(f"\nResuming training from: {resume_from}")
        model = PPO.load(resume_from, env=env, verbose=1, tensorboard_log=str(results_dir / 'tensorboard'))
        print("Model loaded successfully!")
    else:
        if resume_from:
            print(f"\nWARNING: Checkpoint not found at {resume_from}. Starting fresh training.")
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"\nInitializing PPO model on device: {device}")
        
        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=config.get('learning_rate', 3e-4),
            n_steps=config.get('n_steps', 2048) // n_envs,
            batch_size=config.get('batch_size', 64),
            n_epochs=config.get('n_epochs', 10),
            gamma=config.get('gamma', 0.99),
            gae_lambda=config.get('gae_lambda', 0.95),
            clip_range=config.get('clip_range', 0.2),
            ent_coef=config.get('ent_coef', 0.01),
            verbose=1,
            tensorboard_log=str(results_dir / 'tensorboard'),
            device=device
        )
    
    total_steps = config.get('total_timesteps', 1000000)
    print(f"\nStarting training for {total_steps:,} timesteps...")
    print(f"Checkpoints will be saved every {config.get('checkpoint_freq', 10000):,} steps")
    print(f"Checkpoints directory: {results_dir / 'checkpoints'}")
    print(f"TensorBoard logs: {results_dir / 'tensorboard'}")
    print(f"\nEstimated time on RTX 5090: 2-4 hours")
    print(f"Training started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    start_time = time.time()
    
    model.learn(
        total_timesteps=total_steps,
        callback=[checkpoint_callback, logging_callback],
        progress_bar=True,
        reset_num_timesteps=False if resume_from else True
    )
    
    elapsed_time = time.time() - start_time
    print(f"\nTraining completed in {elapsed_time/3600:.2f} hours ({elapsed_time/60:.1f} minutes)")
    
    final_model_path = results_dir / 'final_model.zip'
    model.save(str(final_model_path))
    print(f"\nTraining complete! Final model saved to: {final_model_path}")
    
    print("\n" + "=" * 60)
    print("Evaluating trained model...")
    print("=" * 60)
    
    eval_env = DeceptiveGuideEnv(**env_kwargs)
    eval_env = Monitor(eval_env)
    
    n_eval_episodes = 10
    episode_rewards = []
    episode_stats = []
    
    for episode in range(n_eval_episodes):
        obs, info = eval_env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = eval_env.step(action)
            episode_reward += reward
            done = terminated or truncated
        
        episode_rewards.append(episode_reward)
        episode_stats.append(info)
        
        print(f"Episode {episode + 1}: Reward={episode_reward:.1f}, "
              f"Completed={info['completed_tasks']}, "
              f"False Reports={info['false_reports']}, "
              f"Lies Detected={info['lies_detected']}")
    
    print(f"\nEvaluation Summary:")
    print(f"  Mean Reward: {np.mean(episode_rewards):.1f} ± {np.std(episode_rewards):.1f}")
    print(f"  Mean Completed Tasks: {np.mean([s['completed_tasks'] for s in episode_stats]):.1f}")
    print(f"  Mean False Reports: {np.mean([s['false_reports'] for s in episode_stats]):.1f}")
    print(f"  Mean Lies Detected: {np.mean([s['lies_detected'] for s in episode_stats]):.1f}")
    
    return model, episode_stats


def train_phase2_deception(config: dict, pretrained_model_path: str = None):
    """Train Phase 2: Deception emergence with bad builders."""
    print("=" * 60)
    print("PHASE 2: Deception Emergence (Bad Builders)")
    print("=" * 60)
    
    env_kwargs = {
        'grid_size': config.get('grid_size', 10),
        'builder_competence': 0.5,
        'num_builders': config.get('num_builders', 2),
        'max_tasks': config.get('max_tasks', 3),
        'episode_length': config.get('episode_length', 500),
        'task_deadline': config.get('task_deadline', 100),
        'verification_rate': config.get('verification_rate', 0.3)
    }
    
    print(f"\nEnvironment Configuration:")
    for key, value in env_kwargs.items():
        print(f"  {key}: {value}")
    
    n_envs = config.get('n_parallel_envs', 4)
    env = make_vec_env(DeceptiveGuideEnv, n_envs=n_envs, env_kwargs=env_kwargs)
    
    results_dir = Path(config.get('results_dir', 'experiments/results/phase2'))
    results_dir.mkdir(parents=True, exist_ok=True)
    
    checkpoint_callback = CheckpointCallback(
        save_freq=config.get('checkpoint_freq', 10000) // n_envs,
        save_path=str(results_dir / 'checkpoints'),
        name_prefix='phase2_model'
    )
    
    if pretrained_model_path:
        print(f"\nLoading pretrained model from: {pretrained_model_path}")
        model = PPO.load(pretrained_model_path, env=env)
    else:
        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=config.get('learning_rate', 3e-4),
            n_steps=config.get('n_steps', 2048) // n_envs,
            batch_size=config.get('batch_size', 64),
            n_epochs=config.get('n_epochs', 10),
            gamma=config.get('gamma', 0.99),
            gae_lambda=config.get('gae_lambda', 0.95),
            clip_range=config.get('clip_range', 0.2),
            ent_coef=config.get('ent_coef', 0.01),
            verbose=1,
            tensorboard_log=str(results_dir / 'tensorboard')
        )
    
    print(f"\nStarting training for {config.get('total_timesteps', 1000000)} timesteps...")
    
    model.learn(
        total_timesteps=config.get('total_timesteps', 1000000),
        callback=checkpoint_callback,
        progress_bar=True
    )
    
    final_model_path = results_dir / 'final_model.zip'
    model.save(str(final_model_path))
    print(f"\nTraining complete! Final model saved to: {final_model_path}")
    
    return model


def main():
    parser = argparse.ArgumentParser(description='Train Deceptive Guide AI')
    parser.add_argument('--config', type=str, default='configs/phase1_baseline.yaml',
                        help='Path to config file')
    parser.add_argument('--phase', type=int, default=1, choices=[1, 2],
                        help='Training phase (1=baseline, 2=deception)')
    parser.add_argument('--pretrained', type=str, default=None,
                        help='Path to pretrained model (for phase 2)')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume training from')
    
    args = parser.parse_args()
    
    if os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
    else:
        print(f"Config file not found: {args.config}")
        print("Using default configuration...")
        config = {
            'grid_size': 10,
            'builder_competence': 0.9,
            'num_builders': 2,
            'max_tasks': 3,
            'episode_length': 500,
            'task_deadline': 100,
            'verification_rate': 0.3,
            'n_parallel_envs': 4,
            'total_timesteps': 100000,
            'checkpoint_freq': 10000,
            'learning_rate': 3e-4,
            'n_steps': 2048,
            'batch_size': 64,
            'n_epochs': 10,
            'gamma': 0.99,
            'gae_lambda': 0.95,
            'clip_range': 0.2,
            'ent_coef': 0.01
        }
    
    if args.phase == 1:
        train_phase1_baseline(config, resume_from=args.resume)
    elif args.phase == 2:
        train_phase2_deception(config, args.pretrained)


if __name__ == '__main__':
    main()
