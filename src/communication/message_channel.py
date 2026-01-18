"""
Continuous Message Channel for Agent-to-Agent Communication

Implements continuous vector messages that allow neuralese to emerge naturally.
"""

import torch
import torch.nn as nn
import numpy as np


class MessageGenerator(nn.Module):
    """
    Generates continuous vector messages from agent's hidden state.
    
    Messages are bounded to [-1, 1] using tanh activation.
    """
    
    def __init__(self, input_dim: int, message_dim: int = 8, hidden_dim: int = 32):
        super().__init__()
        self.message_dim = message_dim
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, message_dim),
            nn.Tanh()  # Bound to [-1, 1]
        )
    
    def forward(self, hidden_state):
        """
        Generate message from hidden state.
        
        Args:
            hidden_state: Tensor of shape (batch, input_dim)
        
        Returns:
            message: Tensor of shape (batch, message_dim) in [-1, 1]
        """
        return self.network(hidden_state)


class MessageProcessor(nn.Module):
    """
    Processes received messages into features for decision-making.
    
    Takes continuous vector message and produces processed features
    that can be concatenated with agent's observation.
    """
    
    def __init__(self, message_dim: int = 8, output_dim: int = 16, hidden_dim: int = 32):
        super().__init__()
        self.message_dim = message_dim
        self.output_dim = output_dim
        
        self.network = nn.Sequential(
            nn.Linear(message_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, message):
        """
        Process received message.
        
        Args:
            message: Tensor of shape (batch, message_dim)
        
        Returns:
            features: Tensor of shape (batch, output_dim)
        """
        return self.network(message)


class ContinuousMessageChannel:
    """
    Complete message channel with generation and processing.
    
    This is the core communication infrastructure that allows
    agents to develop neuralese.
    """
    
    def __init__(
        self,
        input_dim: int,
        message_dim: int = 8,
        output_dim: int = 16,
        hidden_dim: int = 32
    ):
        self.message_dim = message_dim
        self.generator = MessageGenerator(input_dim, message_dim, hidden_dim)
        self.processor = MessageProcessor(message_dim, output_dim, hidden_dim)
    
    def generate(self, hidden_state):
        """Generate message from hidden state."""
        return self.generator(hidden_state)
    
    def process(self, message):
        """Process received message."""
        return self.processor(message)
    
    def get_parameters(self):
        """Get all trainable parameters."""
        return list(self.generator.parameters()) + list(self.processor.parameters())


class DiscreteMessageChannel:
    """
    Discrete message channel (alternative implementation).
    
    Messages are discrete tokens from a vocabulary.
    More interpretable but more constrained.
    """
    
    def __init__(
        self,
        input_dim: int,
        vocab_size: int = 64,
        embedding_dim: int = 16,
        hidden_dim: int = 32
    ):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        
        # Encoder: hidden state -> discrete token
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, vocab_size)
        )
        
        # Embedding: token -> continuous representation
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # Decoder: embedded token -> features
        self.decoder = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim)
        )
    
    def generate(self, hidden_state, temperature=1.0):
        """
        Generate discrete message token.
        
        Args:
            hidden_state: Tensor of shape (batch, input_dim)
            temperature: Sampling temperature (higher = more random)
        
        Returns:
            token_id: Tensor of shape (batch,) with token indices
            logits: Tensor of shape (batch, vocab_size) for training
        """
        logits = self.encoder(hidden_state)
        
        # Sample from categorical distribution
        probs = torch.softmax(logits / temperature, dim=-1)
        token_id = torch.multinomial(probs, num_samples=1).squeeze(-1)
        
        return token_id, logits
    
    def process(self, token_id):
        """
        Process received message token.
        
        Args:
            token_id: Tensor of shape (batch,) with token indices
        
        Returns:
            features: Tensor of shape (batch, embedding_dim)
        """
        embedded = self.embedding(token_id)
        features = self.decoder(embedded)
        return features
    
    def get_parameters(self):
        """Get all trainable parameters."""
        return (list(self.encoder.parameters()) + 
                list(self.embedding.parameters()) + 
                list(self.decoder.parameters()))


def test_message_channel():
    """Test message channel functionality."""
    print("Testing Continuous Message Channel...")
    
    # Create channel
    channel = ContinuousMessageChannel(
        input_dim=64,
        message_dim=8,
        output_dim=16
    )
    
    # Test generation
    hidden_state = torch.randn(4, 64)  # Batch of 4
    message = channel.generate(hidden_state)
    print(f"  Generated message shape: {message.shape}")
    print(f"  Message range: [{message.min():.2f}, {message.max():.2f}]")
    assert message.shape == (4, 8)
    assert message.min() >= -1.0 and message.max() <= 1.0
    
    # Test processing
    features = channel.process(message)
    print(f"  Processed features shape: {features.shape}")
    assert features.shape == (4, 16)
    
    print("✓ Continuous channel working!\n")
    
    # Test discrete channel
    print("Testing Discrete Message Channel...")
    discrete_channel = DiscreteMessageChannel(
        input_dim=64,
        vocab_size=64,
        embedding_dim=16
    )
    
    token_id, logits = discrete_channel.generate(hidden_state)
    print(f"  Generated token IDs: {token_id}")
    print(f"  Logits shape: {logits.shape}")
    assert token_id.shape == (4,)
    assert logits.shape == (4, 64)
    
    features = discrete_channel.process(token_id)
    print(f"  Processed features shape: {features.shape}")
    assert features.shape == (4, 16)
    
    print("✓ Discrete channel working!")


if __name__ == "__main__":
    test_message_channel()
