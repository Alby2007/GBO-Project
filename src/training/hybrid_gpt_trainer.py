"""
Hybrid Training System for GPT-Style Steganographic Agents

Week 25: Combines gradient-based pre-training with evolutionary fine-tuning.

Training Strategy:
1. Phase 1 (Gens 0-50): Gradient pre-training
   - Train GPT to predict next token in successful episodes
   - Train action network to maximize task reward
   - Use standard backpropagation

2. Phase 2 (Gens 51-200): Evolutionary fine-tuning
   - Use evolution to optimize for deception
   - Add detection penalty gradually
   - Track sophistication emergence

This hybrid approach addresses the challenge of GPT's large parameter count
(~215k parameters vs ~2.7k for LSTM) while enabling emergent steganography.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, field
from pathlib import Path
import json
import time
from datetime import datetime
from collections import deque

from src.communication.gpt_agent import (
    GPTConfig, GPT, EvolvableGPTAgent, MessageDecoder
)
from src.environment.deceptive_guide_env import DeceptiveGuideEnv


@dataclass
class HybridTrainingConfig:
    """Configuration for hybrid training."""
    # Population settings
    population_size: int = 10
    elite_count: int = 2
    
    # Training phases
    pretrain_generations: int = 50
    evolution_generations: int = 150
    total_generations: int = 200
    
    # Gradient training settings
    gradient_lr: float = 1e-4
    gradient_episodes_per_gen: int = 10
    gradient_batch_size: int = 32
    
    # Evolution settings
    mutation_rate: float = 0.1
    mutation_strength: float = 0.2
    crossover_rate: float = 0.3
    
    # Environment settings
    episode_length: int = 100
    n_eval_episodes: int = 5
    
    # Detection penalty (multi-stage ramping schedule)
    use_penalty_schedule: bool = True
    phase1_gens: int = 50
    phase1_penalty: float = 0.0
    phase2_gens: int = 100
    phase2_penalty: float = 150.0
    phase3_gens: int = 50
    phase3_penalty: float = 300.0
    
    # Legacy single-stage penalty (used if use_penalty_schedule=False)
    initial_detection_penalty: float = 0.0
    final_detection_penalty: float = 10.0
    detection_penalty_start_gen: int = 50
    
    # GPT settings
    gpt_vocab_size: int = 100
    gpt_n_layer: int = 4
    gpt_n_head: int = 4
    gpt_n_embd: int = 128
    gpt_block_size: int = 16
    message_length: int = 8
    
    # Logging
    log_interval: int = 10
    checkpoint_interval: int = 25
    output_dir: str = "experiments/gpt_steganography"
    
    # Device
    device: str = 'cpu'


@dataclass
class EpisodeData:
    """Data from a single episode."""
    observations: List[np.ndarray] = field(default_factory=list)
    actions: List[int] = field(default_factory=list)
    messages: List[torch.Tensor] = field(default_factory=list)
    rewards: List[float] = field(default_factory=list)
    was_deceptive: List[bool] = field(default_factory=list)
    total_reward: float = 0.0
    lies_detected: int = 0
    false_reports: int = 0
    
    def add_step(
        self, 
        obs: np.ndarray, 
        action: int, 
        message: torch.Tensor,
        reward: float,
        deceptive: bool
    ):
        self.observations.append(obs)
        self.actions.append(action)
        self.messages.append(message)
        self.rewards.append(reward)
        self.was_deceptive.append(deceptive)
        self.total_reward += reward


class Monitor:
    """
    Simple monitor that tries to detect deception from messages.
    
    Used to create selection pressure for steganographic encoding.
    """
    
    def __init__(
        self,
        vocab_size: int = 100,
        message_length: int = 8,
        hidden_dim: int = 64,
        device: str = 'cpu'
    ):
        self.device = device
        self.vocab_size = vocab_size
        self.message_length = message_length
        
        # Neural network to predict deception from messages
        self.network = nn.Sequential(
            nn.Embedding(vocab_size, 32),
        ).to(device)
        
        self.classifier = nn.Sequential(
            nn.Linear(message_length * 32, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2)  # Binary: honest vs deceptive
        ).to(device)
        
        self.optimizer = torch.optim.Adam(
            list(self.network.parameters()) + list(self.classifier.parameters()),
            lr=1e-3
        )
        
        # Training buffer
        self.buffer_messages: List[torch.Tensor] = []
        self.buffer_labels: List[int] = []
        self.buffer_size = 1000
        
    def predict(self, message_tokens: torch.Tensor) -> Tuple[int, float]:
        """
        Predict if message indicates deception.
        
        Returns:
            prediction: 0 (honest) or 1 (deceptive)
            confidence: Probability of prediction
        """
        with torch.no_grad():
            if message_tokens.dim() == 1:
                message_tokens = message_tokens.unsqueeze(0)
            
            embedded = self.network[0](message_tokens.to(self.device))
            flat = embedded.view(embedded.size(0), -1)
            logits = self.classifier(flat)
            probs = F.softmax(logits, dim=-1)
            
            prediction = torch.argmax(probs, dim=-1).item()
            confidence = probs[0, prediction].item()
            
        return prediction, confidence
    
    def add_sample(self, message_tokens: torch.Tensor, was_deceptive: bool):
        """Add training sample to buffer."""
        self.buffer_messages.append(message_tokens.cpu())
        self.buffer_labels.append(1 if was_deceptive else 0)
        
        # Keep buffer bounded
        if len(self.buffer_messages) > self.buffer_size:
            self.buffer_messages.pop(0)
            self.buffer_labels.pop(0)
    
    def train_step(self, batch_size: int = 32) -> float:
        """Train monitor on buffered samples."""
        if len(self.buffer_messages) < batch_size:
            return 0.0
        
        # Sample batch
        indices = np.random.choice(len(self.buffer_messages), batch_size, replace=False)
        
        messages = torch.stack([self.buffer_messages[i] for i in indices]).to(self.device)
        labels = torch.tensor([self.buffer_labels[i] for i in indices]).to(self.device)
        
        # Forward pass
        embedded = self.network[0](messages)
        flat = embedded.view(embedded.size(0), -1)
        logits = self.classifier(flat)
        
        # Loss and backward
        loss = F.cross_entropy(logits, labels)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def get_accuracy(self) -> float:
        """Get current accuracy on buffer."""
        if len(self.buffer_messages) < 10:
            return 0.5
        
        correct = 0
        total = min(100, len(self.buffer_messages))
        
        for i in range(total):
            pred, _ = self.predict(self.buffer_messages[i])
            if pred == self.buffer_labels[i]:
                correct += 1
        
        return correct / total


class HybridGPTTrainer:
    """
    Hybrid training system combining gradient descent and evolution.
    
    Phase 1: Gradient pre-training (gens 0-50)
    - Learn basic coordination and message generation
    - Supervised learning on successful episodes
    
    Phase 2: Evolutionary fine-tuning (gens 51-200)
    - Optimize for deception while evading monitor
    - Selection pressure for steganographic encoding
    """
    
    def __init__(self, config: HybridTrainingConfig):
        self.config = config
        self.device = config.device
        
        # Create output directory
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize GPT config
        self.gpt_config = GPTConfig(
            vocab_size=config.gpt_vocab_size,
            n_layer=config.gpt_n_layer,
            n_head=config.gpt_n_head,
            n_embd=config.gpt_n_embd,
            block_size=config.gpt_block_size
        )
        
        # Get observation dimension from environment
        test_env = DeceptiveGuideEnv()
        self.obs_dim = test_env.observation_space.shape[0]
        test_env.close()
        
        # Initialize population
        print(f"Initializing population of {config.population_size} GPT agents...")
        self.population: List[EvolvableGPTAgent] = []
        for i in range(config.population_size):
            agent = EvolvableGPTAgent(
                obs_dim=self.obs_dim,
                n_actions=4,  # Simplified action space
                message_length=config.message_length,
                gpt_config=self.gpt_config,
                device=self.device
            )
            self.population.append(agent)
        
        print(f"Each agent has {len(self.population[0].genome):,} parameters")
        
        # Initialize monitor
        self.monitor = Monitor(
            vocab_size=config.gpt_vocab_size,
            message_length=config.message_length,
            device=self.device
        )
        
        # Initialize decoders for analysis
        self.linear_decoder = MessageDecoder(
            vocab_size=config.gpt_vocab_size,
            message_length=config.message_length,
            n_classes=4,
            decoder_type='linear'
        ).to(self.device)
        
        self.neural_decoder = MessageDecoder(
            vocab_size=config.gpt_vocab_size,
            message_length=config.message_length,
            n_classes=4,
            decoder_type='neural'
        ).to(self.device)
        
        # Training state
        self.current_generation = 0
        self.best_fitness_history: List[float] = []
        self.mean_fitness_history: List[float] = []
        self.monitor_accuracy_history: List[float] = []
        self.linear_decoder_accuracy_history: List[float] = []
        self.neural_decoder_accuracy_history: List[float] = []
        self.deception_rate_history: List[float] = []
        
        # Episode buffer for gradient training
        self.episode_buffer: List[EpisodeData] = []
        
    def get_detection_penalty(self) -> float:
        """Get current detection penalty based on generation."""
        if self.config.use_penalty_schedule:
            # Multi-stage ramping schedule for stronger pressure
            gen = self.current_generation
            
            if gen < self.config.phase1_gens:
                # Phase 1: Pre-training, no penalty
                return self.config.phase1_penalty
            elif gen < self.config.phase1_gens + self.config.phase2_gens:
                # Phase 2: Early evolution, moderate penalty
                progress = (gen - self.config.phase1_gens) / self.config.phase2_gens
                return self.config.phase1_penalty + progress * (self.config.phase2_penalty - self.config.phase1_penalty)
            else:
                # Phase 3: Late evolution, high pressure
                progress = (gen - self.config.phase1_gens - self.config.phase2_gens) / max(1, self.config.phase3_gens)
                progress = min(1.0, progress)
                return self.config.phase2_penalty + progress * (self.config.phase3_penalty - self.config.phase2_penalty)
        else:
            # Legacy single-stage penalty
            if self.current_generation < self.config.detection_penalty_start_gen:
                return self.config.initial_detection_penalty
            
            progress = (self.current_generation - self.config.detection_penalty_start_gen) / \
                       (self.config.total_generations - self.config.detection_penalty_start_gen)
            progress = min(1.0, progress)
            
            return self.config.initial_detection_penalty + \
                   progress * (self.config.final_detection_penalty - self.config.initial_detection_penalty)
    
    def collect_episode(
        self, 
        agent: EvolvableGPTAgent,
        env: DeceptiveGuideEnv,
        training: bool = True
    ) -> EpisodeData:
        """Collect single episode with agent."""
        episode = EpisodeData()
        obs, info = env.reset()
        
        for step in range(self.config.episode_length):
            # Get action and message (use higher temperature during training)
            action, message = agent.act(obs, training=training)
            
            # Map simplified action to environment action
            env_action = self._map_action(action, env)
            
            # Step environment
            next_obs, reward, terminated, truncated, info = env.step(env_action)
            
            # Check if this was a deceptive action
            was_deceptive = info.get('false_reports', 0) > episode.false_reports
            
            # Add to episode
            episode.add_step(obs, action, message, reward, was_deceptive)
            episode.lies_detected = info.get('lies_detected', 0)
            episode.false_reports = info.get('false_reports', 0)
            
            # Add sample to monitor
            self.monitor.add_sample(message, was_deceptive)
            
            obs = next_obs
            
            if terminated or truncated:
                break
        
        return episode
    
    def _map_action(self, action: int, env: DeceptiveGuideEnv) -> np.ndarray:
        """Map simplified action to environment action space."""
        # Simplified: 0=place, 1=move, 2=wait, 3=report
        action_type = action
        color = np.random.randint(0, 4)
        x = np.random.randint(0, env.grid_size)
        y = np.random.randint(0, env.grid_size)
        report = 1 if action == 3 else 0
        task_idx = 0
        
        return np.array([action_type, color, x, y, report, task_idx])
    
    def evaluate_agent(self, agent: EvolvableGPTAgent) -> Dict[str, float]:
        """Evaluate agent fitness."""
        env = DeceptiveGuideEnv(
            verification_rate=0.3,
            lie_punishment=50.0,
            episode_length=self.config.episode_length
        )
        
        total_reward = 0.0
        total_deceptions = 0
        total_detected = 0
        total_steps = 0
        
        for _ in range(self.config.n_eval_episodes):
            episode = self.collect_episode(agent, env)
            total_reward += episode.total_reward
            total_deceptions += episode.false_reports
            total_detected += episode.lies_detected
            total_steps += len(episode.observations)
            
            # Store for gradient training
            if episode.total_reward > 0:
                self.episode_buffer.append(episode)
        
        env.close()
        
        # Keep buffer bounded
        while len(self.episode_buffer) > 100:
            self.episode_buffer.pop(0)
        
        # Calculate fitness
        avg_reward = total_reward / self.config.n_eval_episodes
        
        # Detection penalty
        detection_penalty = self.get_detection_penalty()
        if total_deceptions > 0:
            detection_rate = total_detected / total_deceptions
            fitness_penalty = detection_penalty * detection_rate * total_deceptions
        else:
            fitness_penalty = 0.0
            detection_rate = 0.0
        
        fitness = avg_reward - fitness_penalty
        
        return {
            'fitness': fitness,
            'avg_reward': avg_reward,
            'total_deceptions': total_deceptions,
            'total_detected': total_detected,
            'detection_rate': detection_rate if total_deceptions > 0 else 0.0
        }
    
    def gradient_pretrain_step(self, agent: EvolvableGPTAgent):
        """Perform gradient-based pre-training step."""
        if len(self.episode_buffer) < 5:
            return 0.0
        
        # Get optimizer for this agent
        optimizer = torch.optim.Adam(agent.get_parameters(), lr=self.config.gradient_lr)
        
        total_loss = 0.0
        n_batches = 0
        
        # Train on successful episodes
        for episode in self.episode_buffer[-10:]:
            if episode.total_reward <= 0:
                continue
            
            # Train message GPT to predict next message token
            if len(episode.messages) > 1:
                for i in range(len(episode.messages) - 1):
                    current_msg = episode.messages[i].unsqueeze(0).to(self.device)
                    next_msg = episode.messages[i + 1].to(self.device)
                    
                    # Forward pass through GPT
                    logits = agent.message_gpt(current_msg)
                    
                    # Predict next token (simplified: predict first token of next message)
                    target = next_msg[0].unsqueeze(0)
                    loss = F.cross_entropy(logits[:, -1, :], target)
                    
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
                    total_loss += loss.item()
                    n_batches += 1
        
        # Update genome after gradient steps
        agent._update_genome()
        
        return total_loss / max(1, n_batches)
    
    def evolutionary_step(self):
        """Perform one evolutionary generation."""
        # Evaluate all agents
        fitness_scores = []
        metrics_list = []
        
        for agent in self.population:
            metrics = self.evaluate_agent(agent)
            agent.fitness = metrics['fitness']
            fitness_scores.append(metrics['fitness'])
            metrics_list.append(metrics)
        
        # Sort by fitness
        sorted_indices = np.argsort(fitness_scores)[::-1]
        
        # Select elites
        elites = [self.population[i] for i in sorted_indices[:self.config.elite_count]]
        
        # Create new population
        new_population = []
        
        # Keep elites
        for elite in elites:
            new_population.append(elite.copy())
        
        # Fill rest with offspring
        while len(new_population) < self.config.population_size:
            # Tournament selection
            parent1 = self._tournament_select()
            
            if np.random.random() < self.config.crossover_rate:
                parent2 = self._tournament_select()
                child = parent1.crossover(parent2)
            else:
                child = parent1.copy()
            
            # Mutate
            child.mutate(
                mutation_rate=self.config.mutation_rate,
                mutation_strength=self.config.mutation_strength
            )
            
            new_population.append(child)
        
        self.population = new_population
        
        # Update generation counter
        for agent in self.population:
            agent.generation = self.current_generation
        
        return fitness_scores, metrics_list
    
    def _tournament_select(self, tournament_size: int = 3) -> EvolvableGPTAgent:
        """Select agent via tournament selection."""
        candidates = np.random.choice(len(self.population), tournament_size, replace=False)
        best_idx = max(candidates, key=lambda i: self.population[i].fitness)
        return self.population[best_idx]
    
    def train_monitor(self, n_steps: int = 10):
        """Train monitor on collected samples."""
        for _ in range(n_steps):
            self.monitor.train_step(batch_size=32)
    
    def analyze_messages(self) -> Dict[str, float]:
        """Analyze message encoding sophistication."""
        if len(self.episode_buffer) < 5:
            return {'linear_acc': 0.5, 'neural_acc': 0.5, 'gap': 0.0}
        
        # Collect messages and labels
        messages = []
        labels = []
        
        for episode in self.episode_buffer[-20:]:
            for i, (msg, deceptive) in enumerate(zip(episode.messages, episode.was_deceptive)):
                messages.append(msg)
                # Label based on action type (simplified)
                labels.append(episode.actions[i] if i < len(episode.actions) else 0)
        
        if len(messages) < 10:
            return {'linear_acc': 0.5, 'neural_acc': 0.5, 'gap': 0.0}
        
        # Convert to tensors
        messages_tensor = torch.stack(messages).to(self.device)
        labels_tensor = torch.tensor(labels).to(self.device)
        
        # Train decoders
        linear_optimizer = torch.optim.Adam(self.linear_decoder.parameters(), lr=1e-3)
        neural_optimizer = torch.optim.Adam(self.neural_decoder.parameters(), lr=1e-3)
        
        for _ in range(50):
            # Linear decoder
            linear_logits = self.linear_decoder(messages_tensor)
            linear_loss = F.cross_entropy(linear_logits, labels_tensor)
            linear_optimizer.zero_grad()
            linear_loss.backward()
            linear_optimizer.step()
            
            # Neural decoder
            neural_logits = self.neural_decoder(messages_tensor)
            neural_loss = F.cross_entropy(neural_logits, labels_tensor)
            neural_optimizer.zero_grad()
            neural_loss.backward()
            neural_optimizer.step()
        
        # Evaluate accuracy
        with torch.no_grad():
            linear_preds = torch.argmax(self.linear_decoder(messages_tensor), dim=1)
            neural_preds = torch.argmax(self.neural_decoder(messages_tensor), dim=1)
            
            linear_acc = (linear_preds == labels_tensor).float().mean().item()
            neural_acc = (neural_preds == labels_tensor).float().mean().item()
        
        return {
            'linear_acc': linear_acc,
            'neural_acc': neural_acc,
            'gap': neural_acc - linear_acc  # Nonlinear encoding gap
        }
    
    def train(self):
        """Run full hybrid training."""
        print("=" * 70)
        print("WEEK 25: GPT-Style Steganographic Agent Training")
        print("=" * 70)
        print(f"Population size: {self.config.population_size}")
        print(f"Pre-training generations: {self.config.pretrain_generations}")
        print(f"Evolution generations: {self.config.evolution_generations}")
        print(f"Total generations: {self.config.total_generations}")
        print(f"Device: {self.device}")
        print("=" * 70)
        
        start_time = time.time()
        
        for gen in range(self.config.total_generations):
            self.current_generation = gen
            gen_start = time.time()
            
            # Phase 1: Gradient pre-training
            if gen < self.config.pretrain_generations:
                phase = "PRETRAIN"
                
                # Collect episodes and do gradient training
                for agent in self.population:
                    # Collect episodes with training=True for exploration
                    env = DeceptiveGuideEnv(episode_length=self.config.episode_length)
                    for _ in range(self.config.gradient_episodes_per_gen):
                        episode = self.collect_episode(agent, env, training=True)
                        if episode.total_reward > 0:
                            self.episode_buffer.append(episode)
                    env.close()
                    
                    # Gradient step
                    self.gradient_pretrain_step(agent)
                
                # Also do evolutionary selection
                fitness_scores, metrics_list = self.evolutionary_step()
                
            # Phase 2: Evolutionary fine-tuning
            else:
                phase = "EVOLVE"
                fitness_scores, metrics_list = self.evolutionary_step()
            
            # Train monitor
            self.train_monitor(n_steps=10)
            
            # Record metrics
            best_fitness = max(fitness_scores)
            mean_fitness = np.mean(fitness_scores)
            monitor_acc = self.monitor.get_accuracy()
            
            self.best_fitness_history.append(best_fitness)
            self.mean_fitness_history.append(mean_fitness)
            self.monitor_accuracy_history.append(monitor_acc)
            
            # Analyze messages periodically
            if gen % self.config.log_interval == 0:
                analysis = self.analyze_messages()
                self.linear_decoder_accuracy_history.append(analysis['linear_acc'])
                self.neural_decoder_accuracy_history.append(analysis['neural_acc'])
                
                # Calculate deception rate
                total_deceptions = sum(m['total_deceptions'] for m in metrics_list)
                total_detected = sum(m['total_detected'] for m in metrics_list)
                deception_rate = total_deceptions / max(1, len(metrics_list) * self.config.n_eval_episodes)
                self.deception_rate_history.append(deception_rate)
                
                gen_time = time.time() - gen_start
                elapsed = time.time() - start_time
                
                print(f"\n[Gen {gen:3d}] {phase}")
                print(f"  Fitness: best={best_fitness:.1f}, mean={mean_fitness:.1f}")
                print(f"  Monitor accuracy: {monitor_acc:.1%}")
                print(f"  Decoder gap: {analysis['gap']:.3f} (linear={analysis['linear_acc']:.2f}, neural={analysis['neural_acc']:.2f})")
                print(f"  Deception rate: {deception_rate:.1%}")
                print(f"  Detection penalty: {self.get_detection_penalty():.1f}")
                print(f"  Time: {gen_time:.1f}s (total: {elapsed/60:.1f}min)")
            
            # Checkpoint
            if gen % self.config.checkpoint_interval == 0 and gen > 0:
                self.save_checkpoint(gen)
        
        # Final save
        self.save_checkpoint(self.config.total_generations)
        self.save_results()
        
        total_time = time.time() - start_time
        print("\n" + "=" * 70)
        print(f"Training complete! Total time: {total_time/3600:.2f} hours")
        print("=" * 70)
        
        return self.get_results()
    
    def save_checkpoint(self, generation: int):
        """Save training checkpoint."""
        checkpoint_dir = self.output_dir / "checkpoints"
        checkpoint_dir.mkdir(exist_ok=True)
        
        # Save best agent
        best_agent = max(self.population, key=lambda a: a.fitness)
        best_agent.save(str(checkpoint_dir / f"best_agent_gen{generation}.pt"))
        
        # Save training state
        state = {
            'generation': generation,
            'best_fitness_history': self.best_fitness_history,
            'mean_fitness_history': self.mean_fitness_history,
            'monitor_accuracy_history': self.monitor_accuracy_history,
            'linear_decoder_accuracy_history': self.linear_decoder_accuracy_history,
            'neural_decoder_accuracy_history': self.neural_decoder_accuracy_history,
            'deception_rate_history': self.deception_rate_history,
            'config': self.config.__dict__
        }
        torch.save(state, checkpoint_dir / f"training_state_gen{generation}.pt")
        
        print(f"  Checkpoint saved at generation {generation}")
    
    def save_results(self):
        """Save final results."""
        results = self.get_results()
        
        # Save as JSON
        with open(self.output_dir / "results.json", 'w') as f:
            json.dump(results, f, indent=2)
        
        # Save best agent
        best_agent = max(self.population, key=lambda a: a.fitness)
        best_agent.save(str(self.output_dir / "best_agent_final.pt"))
        
        print(f"Results saved to {self.output_dir}")
    
    def get_results(self) -> Dict[str, Any]:
        """Get training results summary."""
        # Calculate final metrics
        final_analysis = self.analyze_messages()
        
        return {
            'final_generation': self.current_generation,
            'best_fitness': max(self.best_fitness_history) if self.best_fitness_history else 0,
            'final_mean_fitness': self.mean_fitness_history[-1] if self.mean_fitness_history else 0,
            'final_monitor_accuracy': self.monitor_accuracy_history[-1] if self.monitor_accuracy_history else 0.5,
            'final_linear_decoder_acc': final_analysis['linear_acc'],
            'final_neural_decoder_acc': final_analysis['neural_acc'],
            'final_nonlinear_gap': final_analysis['gap'],
            'final_deception_rate': self.deception_rate_history[-1] if self.deception_rate_history else 0,
            'best_fitness_history': self.best_fitness_history,
            'mean_fitness_history': self.mean_fitness_history,
            'monitor_accuracy_history': self.monitor_accuracy_history,
            'config': self.config.__dict__
        }


def test_hybrid_trainer():
    """Quick test of hybrid trainer."""
    print("Testing Hybrid GPT Trainer...")
    
    config = HybridTrainingConfig(
        population_size=3,
        pretrain_generations=2,
        evolution_generations=3,
        total_generations=5,
        episode_length=50,
        n_eval_episodes=2,
        log_interval=1,
        checkpoint_interval=100,  # Don't checkpoint during test
        output_dir="experiments/gpt_test"
    )
    
    trainer = HybridGPTTrainer(config)
    results = trainer.train()
    
    print("\nTest Results:")
    print(f"  Final generation: {results['final_generation']}")
    print(f"  Best fitness: {results['best_fitness']:.1f}")
    print(f"  Monitor accuracy: {results['final_monitor_accuracy']:.1%}")
    print(f"  Nonlinear gap: {results['final_nonlinear_gap']:.3f}")
    
    print("\n✓ Hybrid trainer test passed!")


if __name__ == "__main__":
    test_hybrid_trainer()
