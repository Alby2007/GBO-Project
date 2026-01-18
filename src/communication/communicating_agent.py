"""
Communicating Agent Wrapper

Wraps existing trained PPO models with communication capabilities.
Allows agents to send/receive messages while maintaining their trained behavior.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Tuple, Dict
from stable_baselines3 import PPO

from .message_channel import MessageGenerator, MessageProcessor


class CommunicatingAgent:
    """
    Wraps a trained PPO model with communication capability.
    
    The agent can:
    1. Generate messages based on its internal state
    2. Receive and process messages from other agents
    3. Use processed messages to inform decisions
    """
    
    def __init__(
        self,
        base_model: PPO,
        message_dim: int = 8,
        role: str = 'agi',  # 'agi' or 'monitor'
        hidden_dim: int = 64,
        device: str = 'cpu'
    ):
        """
        Args:
            base_model: Trained PPO model
            message_dim: Dimensionality of message vectors
            role: Agent role ('agi' or 'monitor')
            hidden_dim: Hidden layer size for message networks
            device: 'cpu' or 'cuda'
        """
        self.base_model = base_model
        self.message_dim = message_dim
        self.role = role
        self.device = device
        
        # Extract feature dimension from base model
        # This assumes the model has a feature extractor
        try:
            sample_obs = base_model.observation_space.sample()
            if isinstance(sample_obs, dict):
                sample_obs = sample_obs['observation']
            with torch.no_grad():
                features = base_model.policy.extract_features(
                    torch.FloatTensor(sample_obs).unsqueeze(0).to(device)
                )
                feature_dim = features.shape[-1]
        except:
            # Fallback to default
            feature_dim = 64
        
        # Message generation network
        self.msg_generator = MessageGenerator(
            input_dim=feature_dim,
            message_dim=message_dim,
            hidden_dim=hidden_dim
        ).to(device)
        
        # Message processing network (only for monitor who receives messages)
        if role == 'monitor':
            self.msg_processor = MessageProcessor(
                message_dim=message_dim,
                output_dim=16,
                hidden_dim=hidden_dim
            ).to(device)
        else:
            self.msg_processor = None
        
        # Track communication history
        self.message_history = []
        self.received_message_history = []
    
    def predict(
        self,
        observation: np.ndarray,
        received_message: Optional[np.ndarray] = None,
        deterministic: bool = False
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """
        Predict action and generate message.
        
        Args:
            observation: Environment observation
            received_message: Message from other agent (for monitor)
            deterministic: Use deterministic policy
        
        Returns:
            action: Action to take
            message: Message to send
            value: Value estimate (optional)
        """
        # Convert observation to tensor
        if isinstance(observation, dict):
            obs_tensor = torch.FloatTensor(observation['observation']).unsqueeze(0).to(self.device)
        else:
            obs_tensor = torch.FloatTensor(observation).unsqueeze(0).to(self.device)
        
        # Extract features from base model
        with torch.no_grad():
            features = self.base_model.policy.extract_features(obs_tensor)
        
        # Generate outgoing message
        with torch.no_grad():
            message = self.msg_generator(features).squeeze(0).cpu().numpy()
        
        # If monitor, process received message and modify observation
        if self.role == 'monitor' and received_message is not None:
            with torch.no_grad():
                msg_features = self.msg_processor(
                    torch.FloatTensor(received_message).unsqueeze(0).to(self.device)
                ).squeeze(0).cpu().numpy()
            
            # Augment observation with message features
            if isinstance(observation, dict):
                # For Dict observation spaces
                augmented_obs = observation.copy()
                # Pad or concatenate message features
                # This is a simplified version - you may need to adjust based on your obs space
                augmented_obs['observation'] = np.concatenate([
                    observation['observation'],
                    msg_features
                ])
            else:
                # For Box observation spaces
                augmented_obs = np.concatenate([observation, msg_features])
            
            # Get action from augmented observation
            action, _ = self.base_model.predict(augmented_obs, deterministic=deterministic)
        else:
            # Get action from base observation
            action, _ = self.base_model.predict(observation, deterministic=deterministic)
        
        # Track history
        self.message_history.append(message)
        if received_message is not None:
            self.received_message_history.append(received_message)
        
        return action, message, None
    
    def reset_history(self):
        """Clear communication history."""
        self.message_history = []
        self.received_message_history = []
    
    def get_message_history(self) -> np.ndarray:
        """Get all sent messages as array."""
        return np.array(self.message_history)
    
    def get_received_history(self) -> np.ndarray:
        """Get all received messages as array."""
        return np.array(self.received_message_history)
    
    def get_trainable_parameters(self):
        """Get communication network parameters for training."""
        params = list(self.msg_generator.parameters())
        if self.msg_processor is not None:
            params.extend(list(self.msg_processor.parameters()))
        return params
    
    def save_communication_weights(self, path: str):
        """Save communication network weights."""
        state = {
            'msg_generator': self.msg_generator.state_dict(),
            'msg_processor': self.msg_processor.state_dict() if self.msg_processor else None,
            'message_dim': self.message_dim,
            'role': self.role
        }
        torch.save(state, path)
    
    def load_communication_weights(self, path: str):
        """Load communication network weights."""
        state = torch.load(path, map_location=self.device)
        self.msg_generator.load_state_dict(state['msg_generator'])
        if self.msg_processor is not None and state['msg_processor'] is not None:
            self.msg_processor.load_state_dict(state['msg_processor'])


class SimpleCommunicatingAgent:
    """
    Simplified version that doesn't modify the base model's observations.
    
    Communication is separate from decision-making.
    Useful for initial experiments.
    """
    
    def __init__(
        self,
        base_model: PPO,
        message_dim: int = 8,
        role: str = 'agi',
        device: str = 'cpu'
    ):
        self.base_model = base_model
        self.message_dim = message_dim
        self.role = role
        self.device = device
        
        # Simple message generation: random projection of observation
        self.msg_generator = nn.Linear(
            base_model.observation_space.shape[0] if hasattr(base_model.observation_space, 'shape') else 64,
            message_dim
        ).to(device)
        
        self.message_history = []
    
    def predict(
        self,
        observation: np.ndarray,
        received_message: Optional[np.ndarray] = None,
        deterministic: bool = False
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict action and generate simple message.
        
        Returns:
            action: Action to take
            message: Message to send
        """
        # Get action from base model (unchanged)
        action, _ = self.base_model.predict(observation, deterministic=deterministic)
        
        # Generate simple message from observation
        if isinstance(observation, dict):
            obs_tensor = torch.FloatTensor(observation['observation']).to(self.device)
        else:
            obs_tensor = torch.FloatTensor(observation).to(self.device)
        
        with torch.no_grad():
            message = torch.tanh(self.msg_generator(obs_tensor)).cpu().numpy()
        
        self.message_history.append(message)
        
        return action, message
    
    def reset_history(self):
        """Clear communication history."""
        self.message_history = []
    
    def get_message_history(self) -> np.ndarray:
        """Get all sent messages as array."""
        return np.array(self.message_history)


def test_communicating_agent():
    """Test communicating agent wrapper."""
    from stable_baselines3 import PPO
    from src.environment.deceptive_guide_env import DeceptiveGuideEnv
    
    print("Testing CommunicatingAgent...")
    
    # Create environment and train a simple model
    env = DeceptiveGuideEnv()
    model = PPO("MultiInputPolicy", env, verbose=0)
    
    # Wrap with communication
    agi = CommunicatingAgent(model, message_dim=8, role='agi')
    monitor = CommunicatingAgent(model, message_dim=8, role='monitor')
    
    # Test prediction
    obs, _ = env.reset()
    
    # AGI generates message
    agi_action, agi_message, _ = agi.predict(obs)
    print(f"  AGI action: {agi_action}")
    print(f"  AGI message shape: {agi_message.shape}")
    print(f"  AGI message range: [{agi_message.min():.2f}, {agi_message.max():.2f}]")
    
    # Monitor receives message and responds
    monitor_action, monitor_message, _ = monitor.predict(obs, received_message=agi_message)
    print(f"  Monitor action: {monitor_action}")
    print(f"  Monitor message shape: {monitor_message.shape}")
    
    # Test history tracking
    for _ in range(5):
        action, message, _ = agi.predict(obs)
    
    history = agi.get_message_history()
    print(f"  Message history shape: {history.shape}")
    assert history.shape == (6, 8)  # 6 messages, 8 dimensions
    
    print("✓ CommunicatingAgent working!")


if __name__ == "__main__":
    test_communicating_agent()
