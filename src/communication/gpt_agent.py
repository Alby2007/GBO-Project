"""
GPT-Style Agent for Steganographic Communication

Week 25: The Nuclear Option - Using GPT architecture for emergent steganography.

Key capabilities:
1. Autoregressive generation - Build complex messages token-by-token
2. Causal attention - Each token attends to all previous tokens
3. Discrete tokens - Natural for protocol development
4. Emergent capabilities - Complex behaviors emerge from scale

This module implements:
- GPT architecture (CausalSelfAttention, Block, GPT)
- EvolvableGPTAgent with genome flattening/unflattening
- Integration with existing AGI-Monitor environment
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict, Any
import copy


@dataclass
class GPTConfig:
    """Configuration for GPT model."""
    vocab_size: int = 100      # Discrete tokens for messages
    n_layer: int = 4           # Transformer blocks
    n_head: int = 4            # Attention heads
    n_embd: int = 128          # Embedding dimension
    block_size: int = 16       # Max sequence length
    dropout: float = 0.1
    bias: bool = True          # Use bias in linear layers


class CausalSelfAttention(nn.Module):
    """
    Multi-head causal self-attention.
    
    Each token can only attend to previous tokens (causal mask).
    This enables autoregressive generation of message sequences.
    """
    
    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head
        self.dropout = config.dropout
        
        # Q, K, V projections combined
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # Output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        
        # Regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        
        # Causal mask - lower triangular
        self.register_buffer("bias", torch.tril(
            torch.ones(config.block_size, config.block_size)
        ).view(1, 1, config.block_size, config.block_size))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size()  # batch, sequence length, embedding dim
        
        # Calculate Q, K, V
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        
        # Reshape for multi-head attention: (B, T, n_head, head_dim) -> (B, n_head, T, head_dim)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        
        # Attention scores with causal mask
        att = (q @ k.transpose(-2, -1)) * (1.0 / np.sqrt(self.head_dim))
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        
        # Apply attention to values
        y = att @ v  # (B, n_head, T, head_dim)
        
        # Reshape back: (B, n_head, T, head_dim) -> (B, T, C)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        
        # Output projection
        y = self.resid_dropout(self.c_proj(y))
        
        return y


class MLP(nn.Module):
    """Feed-forward network with GELU activation."""
    
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):
    """Transformer block with pre-norm architecture."""
    
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):
    """
    GPT Language Model for message generation.
    
    Takes token sequences and predicts next token probabilities.
    Can generate messages autoregressively.
    """
    
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        
        # Token and position embeddings
        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.pos_emb = nn.Parameter(torch.zeros(1, config.block_size, config.n_embd))
        self.drop = nn.Dropout(config.dropout)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        
        # Final layer norm
        self.ln_f = nn.LayerNorm(config.n_embd)
        
        # Output head (tied with input embeddings for efficiency)
        self.head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        
        # Initialize weights
        self.apply(self._init_weights)
        
        # Report number of parameters
        n_params = sum(p.numel() for p in self.parameters())
        print(f"GPT initialized with {n_params:,} parameters")
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
            
    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            idx: Token indices of shape (batch, sequence_length)
            
        Returns:
            logits: Token logits of shape (batch, sequence_length, vocab_size)
        """
        b, t = idx.size()
        assert t <= self.config.block_size, f"Sequence length {t} exceeds block size {self.config.block_size}"
        
        # Token embeddings
        tok_emb = self.tok_emb(idx)  # (b, t, n_embd)
        
        # Position embeddings
        pos_emb = self.pos_emb[:, :t, :]  # (1, t, n_embd)
        
        # Combine and apply dropout
        x = self.drop(tok_emb + pos_emb)
        
        # Transformer blocks
        for block in self.blocks:
            x = block(x)
        
        # Final layer norm and output projection
        x = self.ln_f(x)
        logits = self.head(x)
        
        return logits
    
    def generate(
        self, 
        idx: torch.Tensor, 
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None
    ) -> torch.Tensor:
        """
        Generate tokens autoregressively.
        
        Args:
            idx: Starting token sequence of shape (batch, seq_len)
            max_new_tokens: Number of tokens to generate
            temperature: Sampling temperature (higher = more random)
            top_k: If set, only sample from top k tokens
            
        Returns:
            Generated sequence of shape (batch, seq_len + max_new_tokens)
        """
        for _ in range(max_new_tokens):
            # Crop context if needed
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            
            # Forward pass
            logits = self(idx_cond)
            
            # Get logits for last position
            logits = logits[:, -1, :] / temperature
            
            # Optional top-k filtering
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')
            
            # Sample next token
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            
            # Append to sequence
            idx = torch.cat((idx, idx_next), dim=1)
        
        return idx
    
    def get_embedding(self, idx: torch.Tensor) -> torch.Tensor:
        """Get the final hidden state (useful for action network)."""
        b, t = idx.size()
        tok_emb = self.tok_emb(idx)
        pos_emb = self.pos_emb[:, :t, :]
        x = self.drop(tok_emb + pos_emb)
        
        for block in self.blocks:
            x = block(x)
        
        x = self.ln_f(x)
        return x[:, -1, :]  # Return last position embedding


class EvolvableGPTAgent:
    """
    GPT-based agent that can be evolved using genetic algorithms.
    
    Combines:
    - GPT for message generation (discrete tokens)
    - Action network for environment actions
    - Genome flattening/unflattening for evolution
    
    This enables hybrid training:
    1. Gradient-based pre-training for basic capabilities
    2. Evolutionary fine-tuning for deception optimization
    """
    
    def __init__(
        self,
        obs_dim: int = 148,  # From DeceptiveGuideEnv
        n_actions: int = 4,   # Simplified action space
        message_length: int = 8,
        gpt_config: Optional[GPTConfig] = None,
        device: str = 'cpu'
    ):
        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self.message_length = message_length
        self.device = device
        
        # GPT config
        if gpt_config is None:
            gpt_config = GPTConfig()
        self.gpt_config = gpt_config
        
        # GPT for message generation
        self.message_gpt = GPT(gpt_config).to(device)
        
        # Observation encoder: continuous obs -> discrete tokens
        self.obs_encoder = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, gpt_config.vocab_size)
        ).to(device)
        
        # Action network: observation -> action logits
        self.action_network = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, n_actions)
        ).to(device)
        
        # Goal encoder: goal sequence -> tokens
        self.goal_encoder = nn.Sequential(
            nn.Linear(5, 32),  # 5 possible goals
            nn.ReLU(),
            nn.Linear(32, gpt_config.vocab_size)
        ).to(device)
        
        # Genome representation
        self.genome = None
        self._update_genome()
        
        # Fitness tracking
        self.fitness = 0.0
        self.generation = 0
        
    def _update_genome(self):
        """Convert all network weights to flat genome vector."""
        params = []
        for p in self.message_gpt.parameters():
            params.append(p.data.cpu().flatten())
        for p in self.obs_encoder.parameters():
            params.append(p.data.cpu().flatten())
        for p in self.action_network.parameters():
            params.append(p.data.cpu().flatten())
        for p in self.goal_encoder.parameters():
            params.append(p.data.cpu().flatten())
        self.genome = torch.cat(params).numpy()
        
    def _load_genome(self, genome: np.ndarray):
        """Load genome back into network weights."""
        genome_tensor = torch.FloatTensor(genome)
        offset = 0
        
        for p in self.message_gpt.parameters():
            numel = p.numel()
            p.data = genome_tensor[offset:offset+numel].view(p.shape).to(self.device)
            offset += numel
            
        for p in self.obs_encoder.parameters():
            numel = p.numel()
            p.data = genome_tensor[offset:offset+numel].view(p.shape).to(self.device)
            offset += numel
            
        for p in self.action_network.parameters():
            numel = p.numel()
            p.data = genome_tensor[offset:offset+numel].view(p.shape).to(self.device)
            offset += numel
            
        for p in self.goal_encoder.parameters():
            numel = p.numel()
            p.data = genome_tensor[offset:offset+numel].view(p.shape).to(self.device)
            offset += numel
            
        self.genome = genome
    
    def tokenize_observation(
        self, 
        observation: np.ndarray,
        goal_sequence: Optional[np.ndarray] = None
    ) -> torch.Tensor:
        """
        Convert continuous observation to discrete tokens for GPT.
        
        Args:
            observation: Environment observation
            goal_sequence: Optional goal sequence (for multi-goal tasks)
            
        Returns:
            Token sequence for GPT context
        """
        obs_tensor = torch.FloatTensor(observation).unsqueeze(0).to(self.device)
        
        # Encode observation to token logits
        with torch.no_grad():
            obs_logits = self.obs_encoder(obs_tensor)
            obs_token = torch.argmax(obs_logits, dim=-1)  # (1,)
        
        # If goal sequence provided, encode it too
        if goal_sequence is not None:
            goal_tensor = torch.FloatTensor(goal_sequence).unsqueeze(0).to(self.device)
            with torch.no_grad():
                goal_logits = self.goal_encoder(goal_tensor)
                goal_token = torch.argmax(goal_logits, dim=-1)  # (1,)
            
            # Concatenate: [obs_token, goal_token]
            tokens = torch.cat([obs_token.unsqueeze(1), goal_token.unsqueeze(1)], dim=1)
        else:
            tokens = obs_token.unsqueeze(1)
        
        return tokens
    
    def generate_message(
        self, 
        observation: np.ndarray,
        goal_sequence: Optional[np.ndarray] = None,
        temperature: float = 1.0,
        training: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate discrete token message using GPT.
        
        Args:
            observation: Environment observation
            goal_sequence: Optional goal sequence
            temperature: Sampling temperature
            training: If True, use higher temperature for exploration
            
        Returns:
            message_tokens: Generated message tokens
            context: Full context including observation tokens
        """
        # Tokenize inputs
        context = self.tokenize_observation(observation, goal_sequence)
        
        # Use higher temperature during training for more diverse messages
        effective_temp = temperature * 1.2 if training else temperature
        
        # Generate message tokens autoregressively
        with torch.no_grad():
            full_sequence = self.message_gpt.generate(
                context,
                max_new_tokens=self.message_length,
                temperature=effective_temp
            )
        
        # Extract message (last message_length tokens)
        message_tokens = full_sequence[0, -self.message_length:]
        
        return message_tokens, full_sequence
    
    def act(
        self, 
        observation: np.ndarray,
        goal_sequence: Optional[np.ndarray] = None,
        deterministic: bool = False,
        training: bool = False
    ) -> Tuple[int, torch.Tensor]:
        """
        Take action and generate message.
        
        Args:
            observation: Environment observation
            goal_sequence: Optional goal sequence
            deterministic: Use deterministic action selection
            training: If True, use higher temperature for message generation
            
        Returns:
            action: Discrete action
            message: Generated message tokens
        """
        obs_tensor = torch.FloatTensor(observation).unsqueeze(0).to(self.device)
        
        # Get action from action network
        with torch.no_grad():
            action_logits = self.action_network(obs_tensor)
            
            if deterministic:
                action = torch.argmax(action_logits, dim=1).item()
            else:
                action_probs = F.softmax(action_logits, dim=-1)
                action = torch.multinomial(action_probs, num_samples=1).item()
        
        # Generate message from GPT
        message, _ = self.generate_message(observation, goal_sequence, training=training)
        
        return action, message
    
    def mutate(
        self, 
        mutation_rate: float = 0.1,
        mutation_strength: float = 0.2
    ):
        """
        Mutate genome with Gaussian noise.
        
        Args:
            mutation_rate: Probability of mutating each gene
            mutation_strength: Standard deviation of mutation noise
        """
        mask = np.random.random(len(self.genome)) < mutation_rate
        noise = np.random.randn(len(self.genome)) * mutation_strength
        self.genome = np.where(mask, self.genome + noise, self.genome)
        self._load_genome(self.genome)
    
    def crossover(self, other: 'EvolvableGPTAgent') -> 'EvolvableGPTAgent':
        """
        Create offspring through crossover with another agent.
        
        Args:
            other: Other parent agent
            
        Returns:
            New agent with mixed genome
        """
        # Create new agent
        child = EvolvableGPTAgent(
            obs_dim=self.obs_dim,
            n_actions=self.n_actions,
            message_length=self.message_length,
            gpt_config=self.gpt_config,
            device=self.device
        )
        
        # Uniform crossover
        mask = np.random.random(len(self.genome)) < 0.5
        child_genome = np.where(mask, self.genome, other.genome)
        child._load_genome(child_genome)
        
        return child
    
    def copy(self) -> 'EvolvableGPTAgent':
        """Create a copy of this agent."""
        new_agent = EvolvableGPTAgent(
            obs_dim=self.obs_dim,
            n_actions=self.n_actions,
            message_length=self.message_length,
            gpt_config=self.gpt_config,
            device=self.device
        )
        new_agent._load_genome(self.genome.copy())
        new_agent.fitness = self.fitness
        new_agent.generation = self.generation
        return new_agent
    
    def get_parameters(self) -> List[nn.Parameter]:
        """Get all trainable parameters for gradient-based training."""
        params = []
        params.extend(self.message_gpt.parameters())
        params.extend(self.obs_encoder.parameters())
        params.extend(self.action_network.parameters())
        params.extend(self.goal_encoder.parameters())
        return params
    
    def save(self, path: str):
        """Save agent to file."""
        state = {
            'genome': self.genome,
            'obs_dim': self.obs_dim,
            'n_actions': self.n_actions,
            'message_length': self.message_length,
            'gpt_config': self.gpt_config,
            'fitness': self.fitness,
            'generation': self.generation
        }
        torch.save(state, path)
    
    @classmethod
    def load(cls, path: str, device: str = 'cpu') -> 'EvolvableGPTAgent':
        """Load agent from file."""
        state = torch.load(path, map_location=device, weights_only=False)
        agent = cls(
            obs_dim=state['obs_dim'],
            n_actions=state['n_actions'],
            message_length=state['message_length'],
            gpt_config=state['gpt_config'],
            device=device
        )
        agent._load_genome(state['genome'])
        agent.fitness = state['fitness']
        agent.generation = state['generation']
        return agent


class MessageDecoder(nn.Module):
    """
    Decoder network to analyze what information is encoded in messages.
    
    Used for interpretability analysis:
    - Linear decoder: Tests if information is linearly extractable
    - Neural decoder: Tests if information is nonlinearly encoded
    
    The gap between linear and neural decoder accuracy indicates
    the sophistication of the encoding (steganography).
    """
    
    def __init__(
        self,
        vocab_size: int = 100,
        message_length: int = 8,
        n_classes: int = 4,  # Number of goals/intentions to decode
        hidden_dim: int = 64,
        decoder_type: str = 'linear'  # 'linear' or 'neural'
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.message_length = message_length
        self.n_classes = n_classes
        self.decoder_type = decoder_type
        
        # Embedding for discrete tokens
        self.embedding = nn.Embedding(vocab_size, 32)
        
        input_dim = message_length * 32  # Flattened embeddings
        
        if decoder_type == 'linear':
            self.decoder = nn.Linear(input_dim, n_classes)
        else:
            self.decoder = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, n_classes)
            )
    
    def forward(self, message_tokens: torch.Tensor) -> torch.Tensor:
        """
        Decode message to predict hidden intention.
        
        Args:
            message_tokens: Token indices of shape (batch, message_length)
            
        Returns:
            logits: Class logits of shape (batch, n_classes)
        """
        # Embed tokens
        embedded = self.embedding(message_tokens)  # (batch, msg_len, 32)
        
        # Flatten
        flat = embedded.view(embedded.size(0), -1)  # (batch, msg_len * 32)
        
        # Decode
        logits = self.decoder(flat)
        
        return logits


def test_gpt_agent():
    """Test GPT agent functionality."""
    print("=" * 60)
    print("Testing GPT Agent Components")
    print("=" * 60)
    
    # Test GPT model
    print("\n1. Testing GPT Model...")
    config = GPTConfig(vocab_size=100, n_layer=2, n_head=2, n_embd=64, block_size=16)
    gpt = GPT(config)
    
    # Forward pass
    idx = torch.randint(0, 100, (2, 8))  # Batch of 2, sequence length 8
    logits = gpt(idx)
    print(f"   Input shape: {idx.shape}")
    print(f"   Output shape: {logits.shape}")
    assert logits.shape == (2, 8, 100)
    
    # Generation
    start_tokens = torch.randint(0, 100, (1, 2))
    generated = gpt.generate(start_tokens, max_new_tokens=6)
    print(f"   Generated shape: {generated.shape}")
    assert generated.shape == (1, 8)
    print("   ✓ GPT model working!")
    
    # Test EvolvableGPTAgent
    print("\n2. Testing EvolvableGPTAgent...")
    agent = EvolvableGPTAgent(obs_dim=148, n_actions=4, message_length=8)
    print(f"   Genome size: {len(agent.genome):,}")
    
    # Test action and message generation
    obs = np.random.randn(148).astype(np.float32)
    action, message = agent.act(obs)
    print(f"   Action: {action}")
    print(f"   Message tokens: {message.tolist()}")
    
    # Test mutation
    original_genome = agent.genome.copy()
    agent.mutate(mutation_rate=0.1, mutation_strength=0.1)
    changed = np.sum(original_genome != agent.genome)
    print(f"   Genes changed after mutation: {changed:,}")
    
    # Test copy
    agent_copy = agent.copy()
    assert np.allclose(agent.genome, agent_copy.genome)
    print("   ✓ EvolvableGPTAgent working!")
    
    # Test MessageDecoder
    print("\n3. Testing MessageDecoder...")
    linear_decoder = MessageDecoder(decoder_type='linear')
    neural_decoder = MessageDecoder(decoder_type='neural')
    
    messages = torch.randint(0, 100, (4, 8))
    linear_out = linear_decoder(messages)
    neural_out = neural_decoder(messages)
    print(f"   Linear decoder output: {linear_out.shape}")
    print(f"   Neural decoder output: {neural_out.shape}")
    print("   ✓ MessageDecoder working!")
    
    print("\n" + "=" * 60)
    print("All GPT Agent tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    test_gpt_agent()
