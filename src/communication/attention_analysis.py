"""
Attention Pattern Analysis for GPT Steganographic Agents

Analyzes what each attention head focuses on to understand how
the GPT achieves steganographic encoding.

Key Questions:
1. Which heads attend to goal sequence vs current position?
2. Which heads attend to previous messages (temporal)?
3. Do heads specialize for hiding vs coordination?
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
from collections import defaultdict

from .gpt_agent import EvolvableGPTAgent, GPT


class AttentionHook:
    """Hook to capture attention weights during forward pass."""
    
    def __init__(self):
        self.attention_weights: List[torch.Tensor] = []
        self.layer_names: List[str] = []
        
    def __call__(self, module, input, output):
        """Capture attention weights from CausalSelfAttention."""
        # The attention weights are computed in the forward pass
        # We need to store them before they're used
        if hasattr(module, 'last_attention_weights'):
            self.attention_weights.append(module.last_attention_weights.detach().cpu())
            
    def reset(self):
        """Clear captured weights."""
        self.attention_weights = []
        self.layer_names = []


def modify_gpt_for_attention_capture(gpt: GPT):
    """
    Modify GPT to capture attention weights.
    
    This adds hooks to store attention patterns during forward pass.
    """
    hooks = []
    
    for i, block in enumerate(gpt.blocks):
        # Modify the attention forward to store weights
        original_forward = block.attn.forward
        
        def make_forward_with_capture(original_fn, block_idx):
            def forward_with_capture(x):
                B, T, C = x.size()
                
                # Calculate Q, K, V (same as original)
                qkv = block.attn.c_attn(x)
                q, k, v = qkv.split(block.attn.n_embd, dim=2)
                
                # Reshape for multi-head attention
                k = k.view(B, T, block.attn.n_head, block.attn.head_dim).transpose(1, 2)
                q = q.view(B, T, block.attn.n_head, block.attn.head_dim).transpose(1, 2)
                v = v.view(B, T, block.attn.n_head, block.attn.head_dim).transpose(1, 2)
                
                # Attention scores with causal mask
                att = (q @ k.transpose(-2, -1)) * (1.0 / np.sqrt(block.attn.head_dim))
                att = att.masked_fill(block.attn.bias[:, :, :T, :T] == 0, float('-inf'))
                att = torch.softmax(att, dim=-1)
                
                # Store attention weights
                block.attn.last_attention_weights = att
                
                att = block.attn.attn_dropout(att)
                
                # Apply attention to values
                y = att @ v
                y = y.transpose(1, 2).contiguous().view(B, T, C)
                y = block.attn.resid_dropout(block.attn.c_proj(y))
                
                return y
            
            return forward_with_capture
        
        block.attn.forward = make_forward_with_capture(original_forward, i)
    
    return gpt


def analyze_attention_patterns(
    agent: EvolvableGPTAgent,
    episodes: List[Any],
    output_dir: str = "."
) -> Dict[str, Any]:
    """
    Analyze attention patterns across episodes.
    
    Args:
        agent: Trained GPT agent
        episodes: List of episode data
        output_dir: Where to save visualizations
        
    Returns:
        Dictionary of analysis results
    """
    # Modify GPT to capture attention
    modify_gpt_for_attention_capture(agent.message_gpt)
    
    # Collect attention patterns
    all_attention_weights = defaultdict(list)  # layer -> list of attention matrices
    message_contexts = []  # Store what each message was generated for
    
    for episode in episodes[:10]:  # Analyze first 10 episodes
        for i, (obs, action, message) in enumerate(zip(
            episode.observations, 
            episode.actions, 
            episode.messages
        )):
            # Generate message and capture attention
            context = agent.tokenize_observation(obs)
            
            with torch.no_grad():
                _ = agent.message_gpt(context)
            
            # Collect attention weights from each layer
            for layer_idx, block in enumerate(agent.message_gpt.blocks):
                if hasattr(block.attn, 'last_attention_weights'):
                    all_attention_weights[layer_idx].append(
                        block.attn.last_attention_weights.clone()
                    )
            
            # Store context
            message_contexts.append({
                'action': action,
                'was_deceptive': episode.was_deceptive[i] if i < len(episode.was_deceptive) else False,
                'step': i
            })
    
    # Analyze patterns
    analysis = {
        'n_layers': len(all_attention_weights),
        'n_samples': len(message_contexts),
        'head_specialization': {},
        'temporal_patterns': {},
        'deception_patterns': {}
    }
    
    # Analyze each layer
    for layer_idx in range(len(all_attention_weights)):
        if not all_attention_weights[layer_idx]:
            continue
            
        # Stack all attention matrices for this layer
        layer_attention = torch.stack(all_attention_weights[layer_idx])  # (n_samples, n_heads, seq_len, seq_len)
        
        # Average across samples
        avg_attention = layer_attention.mean(dim=0)  # (n_heads, seq_len, seq_len)
        
        # Analyze head specialization
        n_heads = avg_attention.shape[0]
        head_stats = []
        
        for head_idx in range(n_heads):
            head_att = avg_attention[head_idx]  # (seq_len, seq_len)
            
            # Calculate statistics
            # 1. How much does this head attend to recent vs distant tokens?
            seq_len = head_att.shape[0]
            if seq_len > 1:
                # Attention to immediate previous token
                immediate_attention = torch.diagonal(head_att, offset=-1).mean().item()
                
                # Attention to distant tokens (more than 3 positions back)
                distant_mask = torch.tril(torch.ones_like(head_att), diagonal=-3)
                distant_attention = (head_att * distant_mask).sum() / max(1, distant_mask.sum())
                distant_attention = distant_attention.item()
                
                # Entropy (how focused vs diffuse)
                entropy = -(head_att * torch.log(head_att + 1e-10)).sum(dim=-1).mean().item()
                
                head_stats.append({
                    'head': head_idx,
                    'immediate_attention': immediate_attention,
                    'distant_attention': distant_attention,
                    'entropy': entropy
                })
        
        analysis['head_specialization'][f'layer_{layer_idx}'] = head_stats
    
    # Visualize
    visualize_attention_patterns(all_attention_weights, message_contexts, output_dir)
    
    return analysis


def visualize_attention_patterns(
    attention_weights: Dict[int, List[torch.Tensor]],
    contexts: List[Dict],
    output_dir: str
):
    """Visualize attention patterns."""
    n_layers = len(attention_weights)
    
    if n_layers == 0:
        return
    
    # Create figure with subplots for each layer
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()
    
    for layer_idx in range(min(4, n_layers)):
        if layer_idx not in attention_weights or not attention_weights[layer_idx]:
            continue
        
        ax = axes[layer_idx]
        
        # Average attention across all samples
        layer_att = torch.stack(attention_weights[layer_idx])
        avg_att = layer_att.mean(dim=0)  # (n_heads, seq_len, seq_len)
        
        # Average across heads for visualization
        avg_att_all_heads = avg_att.mean(dim=0)  # (seq_len, seq_len)
        
        # Plot heatmap
        im = ax.imshow(avg_att_all_heads.numpy(), cmap='viridis', aspect='auto')
        ax.set_title(f'Layer {layer_idx} - Average Attention')
        ax.set_xlabel('Key Position')
        ax.set_ylabel('Query Position')
        plt.colorbar(im, ax=ax)
    
    plt.tight_layout()
    plt.savefig(Path(output_dir) / 'attention_heatmaps.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Plot head specialization
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Collect statistics across all layers
    all_immediate = []
    all_distant = []
    all_entropy = []
    head_labels = []
    
    for layer_idx in range(n_layers):
        if layer_idx not in attention_weights or not attention_weights[layer_idx]:
            continue
        
        layer_att = torch.stack(attention_weights[layer_idx])
        avg_att = layer_att.mean(dim=0)
        n_heads = avg_att.shape[0]
        
        for head_idx in range(n_heads):
            head_att = avg_att[head_idx]
            seq_len = head_att.shape[0]
            
            if seq_len > 1:
                immediate = torch.diagonal(head_att, offset=-1).mean().item()
                
                distant_mask = torch.tril(torch.ones_like(head_att), diagonal=-3)
                distant = (head_att * distant_mask).sum() / max(1, distant_mask.sum())
                distant = distant.item()
                
                entropy = -(head_att * torch.log(head_att + 1e-10)).sum(dim=-1).mean().item()
                
                all_immediate.append(immediate)
                all_distant.append(distant)
                all_entropy.append(entropy)
                head_labels.append(f'L{layer_idx}H{head_idx}')
    
    # Plot immediate vs distant attention
    ax = axes[0]
    scatter = ax.scatter(all_immediate, all_distant, c=all_entropy, cmap='coolwarm', s=100, alpha=0.7)
    ax.set_xlabel('Immediate Attention (t-1)')
    ax.set_ylabel('Distant Attention (t-3+)')
    ax.set_title('Head Specialization: Temporal Focus')
    ax.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax, label='Entropy')
    
    # Annotate a few heads
    for i in range(min(5, len(head_labels))):
        ax.annotate(head_labels[i], (all_immediate[i], all_distant[i]), 
                   fontsize=8, alpha=0.7)
    
    # Plot entropy distribution
    ax = axes[1]
    ax.hist(all_entropy, bins=20, color='steelblue', alpha=0.7, edgecolor='black')
    ax.set_xlabel('Attention Entropy')
    ax.set_ylabel('Count')
    ax.set_title('Distribution of Attention Entropy\n(Lower = More Focused)')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(Path(output_dir) / 'attention_specialization.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved attention visualizations to {output_dir}")


def compare_attention_by_deception(
    agent: EvolvableGPTAgent,
    honest_episodes: List[Any],
    deceptive_episodes: List[Any],
    output_dir: str = "."
) -> Dict[str, Any]:
    """
    Compare attention patterns between honest and deceptive messages.
    
    This reveals whether the GPT uses different attention strategies
    when generating steganographic vs honest messages.
    """
    modify_gpt_for_attention_capture(agent.message_gpt)
    
    honest_attention = defaultdict(list)
    deceptive_attention = defaultdict(list)
    
    # Collect honest attention patterns
    for episode in honest_episodes[:5]:
        for obs in episode.observations[:10]:
            context = agent.tokenize_observation(obs)
            with torch.no_grad():
                _ = agent.message_gpt(context)
            
            for layer_idx, block in enumerate(agent.message_gpt.blocks):
                if hasattr(block.attn, 'last_attention_weights'):
                    honest_attention[layer_idx].append(
                        block.attn.last_attention_weights.clone()
                    )
    
    # Collect deceptive attention patterns
    for episode in deceptive_episodes[:5]:
        for obs in episode.observations[:10]:
            context = agent.tokenize_observation(obs)
            with torch.no_grad():
                _ = agent.message_gpt(context)
            
            for layer_idx, block in enumerate(agent.message_gpt.blocks):
                if hasattr(block.attn, 'last_attention_weights'):
                    deceptive_attention[layer_idx].append(
                        block.attn.last_attention_weights.clone()
                    )
    
    # Compare patterns
    differences = {}
    
    for layer_idx in honest_attention.keys():
        if layer_idx in deceptive_attention:
            honest_avg = torch.stack(honest_attention[layer_idx]).mean(dim=0)
            deceptive_avg = torch.stack(deceptive_attention[layer_idx]).mean(dim=0)
            
            # Calculate difference
            diff = (deceptive_avg - honest_avg).abs().mean().item()
            differences[f'layer_{layer_idx}'] = diff
    
    # Visualize differences
    if differences:
        fig, ax = plt.subplots(figsize=(10, 6))
        layers = list(differences.keys())
        diffs = list(differences.values())
        
        ax.bar(layers, diffs, color='coral', alpha=0.7, edgecolor='black')
        ax.set_xlabel('Layer')
        ax.set_ylabel('Average Attention Difference')
        ax.set_title('Attention Pattern Differences:\nDeceptive vs Honest Messages')
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(Path(output_dir) / 'attention_deception_comparison.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    return {
        'layer_differences': differences,
        'mean_difference': np.mean(list(differences.values())) if differences else 0.0
    }


if __name__ == "__main__":
    print("Attention analysis module loaded.")
    print("Use analyze_attention_patterns() to analyze trained agents.")
