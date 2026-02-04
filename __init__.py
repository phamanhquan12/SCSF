"""
RL-based Reward Assignment for Deep Gamblers

This package contains modules for automatically learning optimal
per-class reward values using reinforcement learning approaches.

Approach 1: Multi-Armed Bandit (bandit_reward.py)
- Epsilon-Greedy Bandit
- UCB Bandit  
- Thompson Sampling Bandit

Future approaches can be added here:
- Approach 2: Contextual Bandit
- Approach 3: Policy Gradient
- Approach 4: Meta-RL
"""

from .bandit_reward import (
    BaseRewardBandit,
    EpsilonGreedyBandit,
    UCBBandit,
    ThompsonSamplingBandit,
    ClassSpecificRewardManager,
    create_reward_manager
)

__all__ = [
    'BaseRewardBandit',
    'EpsilonGreedyBandit', 
    'UCBBandit',
    'ThompsonSamplingBandit',
    'ClassSpecificRewardManager',
    'create_reward_manager'
]
