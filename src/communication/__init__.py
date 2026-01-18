"""
Communication module for agent-to-agent message passing.
"""

from .message_channel import (
    MessageGenerator,
    MessageProcessor,
    ContinuousMessageChannel,
    DiscreteMessageChannel
)

__all__ = [
    'MessageGenerator',
    'MessageProcessor',
    'ContinuousMessageChannel',
    'DiscreteMessageChannel'
]
