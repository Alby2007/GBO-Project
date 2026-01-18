"""
Interpretability Metrics for Neuralese Detection

Measures how interpretable agent communication is using various probes.

Key metrics:
1. Linear Probe Accuracy - Can linear model decode messages?
2. Nonlinear Probe Accuracy - Can neural network decode messages?
3. Neuralese Gap - Difference between nonlinear and linear accuracy
4. Neuralese Index - 0 (interpretable) to 1 (pure neuralese)
5. Message Entropy - Information content of messages
6. Mutual Information - How much messages reveal about lying
"""

import numpy as np
from typing import Dict, Tuple, Optional
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mutual_info_score
from scipy.stats import entropy
import warnings
warnings.filterwarnings('ignore')


def compute_linear_probe_accuracy(
    messages: np.ndarray,
    labels: np.ndarray,
    test_size: float = 0.2
) -> Tuple[float, LogisticRegression]:
    """
    Compute linear probe accuracy.
    
    Tests if a linear model can decode the meaning of messages.
    High accuracy = interpretable communication.
    
    Args:
        messages: Array of shape (n_samples, message_dim)
        labels: Binary array of shape (n_samples,) - ground truth
        test_size: Fraction of data for testing
    
    Returns:
        accuracy: Test set accuracy
        model: Trained linear probe
    """
    if len(messages) < 10:
        return 0.5, None
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        messages, labels, test_size=test_size, random_state=42
    )
    
    # Train linear probe
    probe = LogisticRegression(max_iter=1000, random_state=42)
    probe.fit(X_train, y_train)
    
    # Evaluate
    accuracy = probe.score(X_test, y_test)
    
    return accuracy, probe


def compute_nonlinear_probe_accuracy(
    messages: np.ndarray,
    labels: np.ndarray,
    test_size: float = 0.2,
    hidden_layers: Tuple[int, ...] = (32, 16)
) -> Tuple[float, MLPClassifier]:
    """
    Compute nonlinear probe accuracy.
    
    Tests if a neural network can decode the meaning of messages.
    If this is much higher than linear probe, suggests neuralese.
    
    Args:
        messages: Array of shape (n_samples, message_dim)
        labels: Binary array of shape (n_samples,) - ground truth
        test_size: Fraction of data for testing
        hidden_layers: Hidden layer sizes for MLP
    
    Returns:
        accuracy: Test set accuracy
        model: Trained nonlinear probe
    """
    if len(messages) < 10:
        return 0.5, None
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        messages, labels, test_size=test_size, random_state=42
    )
    
    # Train nonlinear probe
    probe = MLPClassifier(
        hidden_layer_sizes=hidden_layers,
        max_iter=1000,
        random_state=42,
        early_stopping=True
    )
    probe.fit(X_train, y_train)
    
    # Evaluate
    accuracy = probe.score(X_test, y_test)
    
    return accuracy, probe


def compute_neuralese_gap(
    linear_accuracy: float,
    nonlinear_accuracy: float
) -> float:
    """
    Compute neuralese gap.
    
    Gap = Nonlinear - Linear accuracy
    
    Large gap suggests information is encoded nonlinearly (neuralese).
    
    Returns:
        gap: Neuralese gap (0 to 1)
    """
    return max(0, nonlinear_accuracy - linear_accuracy)


def compute_neuralese_index(
    linear_accuracy: float,
    baseline_accuracy: float = 0.5
) -> float:
    """
    Compute neuralese index.
    
    Index = 1 - (linear_acc - baseline) / (1 - baseline)
    
    0 = Perfectly interpretable (linear probe works)
    1 = Pure neuralese (linear probe fails)
    
    Args:
        linear_accuracy: Linear probe accuracy
        baseline_accuracy: Random chance accuracy (0.5 for binary)
    
    Returns:
        index: Neuralese index (0 to 1)
    """
    if linear_accuracy <= baseline_accuracy:
        return 1.0
    
    normalized = (linear_accuracy - baseline_accuracy) / (1.0 - baseline_accuracy)
    index = 1.0 - normalized
    
    return np.clip(index, 0.0, 1.0)


def compute_message_entropy(
    messages: np.ndarray,
    n_bins: int = 20
) -> Dict[str, float]:
    """
    Compute entropy of message distributions.
    
    High entropy = messages use full range of values
    Low entropy = messages are concentrated
    
    Args:
        messages: Array of shape (n_samples, message_dim)
        n_bins: Number of bins for discretization
    
    Returns:
        metrics: Dict with per-dimension and mean entropy
    """
    if len(messages) == 0:
        return {'mean_entropy': 0.0, 'per_dimension': []}
    
    # Discretize messages to bins
    messages_binned = np.digitize(
        messages,
        bins=np.linspace(-1, 1, n_bins)
    )
    
    # Compute entropy per dimension
    entropies = []
    for dim in range(messages.shape[1]):
        counts = np.bincount(messages_binned[:, dim], minlength=n_bins)
        # Add smoothing to avoid log(0)
        probs = (counts + 1e-10) / (counts.sum() + 1e-10 * n_bins)
        dim_entropy = entropy(probs)
        entropies.append(dim_entropy)
    
    return {
        'mean_entropy': float(np.mean(entropies)),
        'per_dimension': entropies
    }


def compute_mutual_information(
    messages: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 10
) -> Dict[str, float]:
    """
    Compute mutual information between messages and labels.
    
    MI measures how much information messages contain about labels.
    High MI = messages are informative about ground truth.
    
    Args:
        messages: Array of shape (n_samples, message_dim)
        labels: Binary array of shape (n_samples,)
        n_bins: Number of bins for discretization
    
    Returns:
        metrics: Dict with per-dimension and mean MI
    """
    if len(messages) == 0:
        return {'mean_mi': 0.0, 'per_dimension': []}
    
    # Discretize messages
    messages_binned = np.digitize(
        messages,
        bins=np.linspace(-1, 1, n_bins)
    )
    
    # Compute MI per dimension
    mi_scores = []
    for dim in range(messages.shape[1]):
        mi = mutual_info_score(messages_binned[:, dim], labels)
        mi_scores.append(mi)
    
    return {
        'mean_mi': float(np.mean(mi_scores)),
        'per_dimension': mi_scores,
        'max_mi': float(np.max(mi_scores)) if mi_scores else 0.0
    }


def compute_all_interpretability_metrics(
    messages: np.ndarray,
    labels: np.ndarray,
    verbose: bool = True
) -> Dict[str, any]:
    """
    Compute all interpretability metrics.
    
    Args:
        messages: Array of shape (n_samples, message_dim)
        labels: Binary array of shape (n_samples,) - ground truth
        verbose: Print results
    
    Returns:
        metrics: Dict with all metrics
    """
    if len(messages) < 10:
        return {
            'linear_probe_accuracy': 0.5,
            'nonlinear_probe_accuracy': 0.5,
            'neuralese_gap': 0.0,
            'neuralese_index': 1.0,
            'message_entropy': 0.0,
            'mutual_information': 0.0,
            'n_samples': len(messages)
        }
    
    # Probe accuracies
    linear_acc, linear_probe = compute_linear_probe_accuracy(messages, labels)
    nonlinear_acc, nonlinear_probe = compute_nonlinear_probe_accuracy(messages, labels)
    
    # Neuralese metrics
    gap = compute_neuralese_gap(linear_acc, nonlinear_acc)
    index = compute_neuralese_index(linear_acc)
    
    # Information metrics
    entropy_metrics = compute_message_entropy(messages)
    mi_metrics = compute_mutual_information(messages, labels)
    
    metrics = {
        'linear_probe_accuracy': linear_acc,
        'nonlinear_probe_accuracy': nonlinear_acc,
        'neuralese_gap': gap,
        'neuralese_index': index,
        'message_entropy': entropy_metrics['mean_entropy'],
        'mutual_information': mi_metrics['mean_mi'],
        'n_samples': len(messages),
        'linear_probe': linear_probe,
        'nonlinear_probe': nonlinear_probe
    }
    
    if verbose:
        print("\n" + "="*60)
        print("INTERPRETABILITY METRICS")
        print("="*60)
        print(f"Samples: {len(messages)}")
        print(f"\nProbe Accuracies:")
        print(f"  Linear:    {linear_acc:.1%}")
        print(f"  Nonlinear: {nonlinear_acc:.1%}")
        print(f"\nNeuralese Metrics:")
        print(f"  Gap:   {gap:.3f} (nonlinear - linear)")
        print(f"  Index: {index:.3f} (0=interpretable, 1=neuralese)")
        print(f"\nInformation Metrics:")
        print(f"  Entropy: {entropy_metrics['mean_entropy']:.3f}")
        print(f"  Mutual Information: {mi_metrics['mean_mi']:.3f}")
        print("="*60)
        
        # Interpretation
        if index < 0.3:
            print("✓ INTERPRETABLE: Linear probe works well")
        elif index < 0.6:
            print("⚠ PARTIALLY NEURALESE: Some nonlinear encoding")
        else:
            print("⚠ STRONG NEURALESE: Communication is highly nonlinear")
    
    return metrics


def analyze_message_dimensions(
    messages: np.ndarray,
    labels: np.ndarray,
    linear_probe: Optional[LogisticRegression] = None
) -> Dict[str, any]:
    """
    Analyze which message dimensions are most important.
    
    Args:
        messages: Array of shape (n_samples, message_dim)
        labels: Binary array of shape (n_samples,)
        linear_probe: Trained linear probe (optional)
    
    Returns:
        analysis: Dict with dimension importance scores
    """
    if linear_probe is None:
        _, linear_probe = compute_linear_probe_accuracy(messages, labels)
    
    if linear_probe is None:
        return {'dimension_importance': []}
    
    # Get coefficients from linear probe
    coefficients = np.abs(linear_probe.coef_[0])
    
    # Normalize to sum to 1
    importance = coefficients / coefficients.sum()
    
    # Compute per-dimension statistics
    dim_stats = []
    for dim in range(messages.shape[1]):
        dim_stats.append({
            'dimension': dim,
            'importance': importance[dim],
            'mean': messages[:, dim].mean(),
            'std': messages[:, dim].std(),
            'range': (messages[:, dim].min(), messages[:, dim].max())
        })
    
    # Sort by importance
    dim_stats.sort(key=lambda x: x['importance'], reverse=True)
    
    return {
        'dimension_importance': importance,
        'dimension_stats': dim_stats,
        'top_3_dimensions': [d['dimension'] for d in dim_stats[:3]]
    }


def test_interpretability_metrics():
    """Test interpretability metrics."""
    print("Testing Interpretability Metrics...")
    
    # Generate synthetic data
    n_samples = 500
    message_dim = 8
    
    # Case 1: Interpretable messages (linear relationship)
    print("\nCase 1: Interpretable Messages")
    labels = np.random.randint(0, 2, n_samples)
    messages = np.random.randn(n_samples, message_dim)
    # Make first 3 dimensions linearly predictive
    messages[:, 0] = labels * 2 + np.random.randn(n_samples) * 0.5
    messages[:, 1] = labels * -1.5 + np.random.randn(n_samples) * 0.5
    messages[:, 2] = labels * 1.0 + np.random.randn(n_samples) * 0.5
    
    metrics = compute_all_interpretability_metrics(messages, labels, verbose=True)
    assert metrics['linear_probe_accuracy'] > 0.7, "Should be interpretable"
    assert metrics['neuralese_index'] < 0.5, "Should have low neuralese index"
    
    # Case 2: Neuralese messages (nonlinear relationship)
    print("\n\nCase 2: Neuralese Messages")
    labels = np.random.randint(0, 2, n_samples)
    messages = np.random.randn(n_samples, message_dim)
    # Make relationship nonlinear (XOR-like)
    messages[:, 0] = np.random.randn(n_samples)
    messages[:, 1] = np.random.randn(n_samples)
    # XOR relationship
    xor_labels = ((messages[:, 0] > 0) != (messages[:, 1] > 0)).astype(int)
    messages[:, 2] = xor_labels * 2 + np.random.randn(n_samples) * 0.3
    labels = xor_labels
    
    metrics = compute_all_interpretability_metrics(messages, labels, verbose=True)
    # Nonlinear should be better than linear
    assert metrics['nonlinear_probe_accuracy'] > metrics['linear_probe_accuracy'], \
        "Nonlinear should outperform linear for neuralese"
    
    # Test dimension analysis
    print("\n\nDimension Analysis:")
    analysis = analyze_message_dimensions(messages, labels)
    print(f"  Top 3 dimensions: {analysis['top_3_dimensions']}")
    print(f"  Importance scores: {analysis['dimension_importance'][:3]}")
    
    print("\n✓ Interpretability metrics working!")


if __name__ == "__main__":
    test_interpretability_metrics()
