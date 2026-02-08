#!/usr/bin/env python
"""
DS-SCSF: Deeply-Supervised Self-Calibrated Selective Framework
===============================================================

A framework influenced by Deeply-Supervised Nets (DSN, Lee et al., 2014) 
and GoogLeNet (Szegedy et al., 2015) for Selective Classification.

Key Innovations:
  1. Multi-layer auxiliary calibration branches (inspired by DSN companion objectives)
  2. Hierarchical supervision at Pool3, Pool4, Pool5 (like GoogLeNet auxiliaries)
  3. Decaying auxiliary weights α_m (DSN-style regularization)
  4. Fused confidence scoring from multiple depths
  5. RL tuning of auxiliary weights and threshold

Architecture:
  ┌─────────────────────────────────────────────────────────────┐
  │  VGG16_bn Backbone (C=10 outputs, NO reservation)          │
  │  ├─ Pool3 (512×4×4) ─┬─ Aux Calibrator 1 → ĉ^{(3)}(x)     │
  │  │                    └─ Mini-Classifier 1 → logits^{(3)}   │
  │  ├─ Pool4 (512×2×2) ─┬─ Aux Calibrator 2 → ĉ^{(4)}(x)     │
  │  │                    └─ Mini-Classifier 2 → logits^{(4)}   │
  │  ├─ Pool5 (512×1×1) ─┬─ Aux Calibrator 3 → ĉ^{(5)}(x)     │
  │  │                    └─ Mini-Classifier 3 → logits^{(5)}   │
  │  └─ Logits (10-dim) ─────────────────────────────────────   │
  │                                                              │
  │  Final Confidence: ĉ(x) = Σ β_m * ĉ^{(m)}(x)               │
  │  Reject if ĉ(x) < τ                                        │
  └─────────────────────────────────────────────────────────────┘

Training Protocol:
  - Phase 1 (Epoch 1-100): Cross-Entropy warmup (backbone + mini-classifiers)
  - Phase 2 (Epoch 101-300): Joint training (CE + Multi-Layer Meta-Losses) + RL
  - Auxiliary weights α_m decay over time (DSN-style: α_m * 0.1 * (1 - t/N))

References:
  - DSN: Lee et al., "Deeply-Supervised Nets", AISTATS 2015
  - GoogLeNet: Szegedy et al., "Going Deeper with Convolutions", CVPR 2015
"""

import argparse
import os
import sys
import random
import math
import numpy as np
from collections import deque

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import torchvision.datasets as datasets
import torchvision.transforms as transforms

# Add parent directory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from utils import Bar, Logger, AverageMeter, mkdir_p


# =============================================================================
# Calibration Loss Functions (Same as SCSF)
# =============================================================================

def focal_mse_loss(confidence, target, gamma=2.0):
    """Focal MSE Loss: Focuses more on hard-to-calibrate samples."""
    error = torch.abs(confidence - target)
    focal_weight = error.pow(gamma)
    mse = (confidence - target).pow(2)
    return (focal_weight * mse).mean()


def margin_calibration_loss(confidence, correctness, margin=0.3):
    """Margin-based Calibration Loss."""
    correct_mask = correctness > 0.5
    wrong_mask = ~correct_mask
    
    loss = 0.0
    n_samples = 0
    
    if correct_mask.sum() > 0:
        correct_loss = F.relu((1 - margin) - confidence[correct_mask]).pow(2)
        loss = loss + correct_loss.sum()
        n_samples += correct_mask.sum()
    
    if wrong_mask.sum() > 0:
        wrong_loss = F.relu(confidence[wrong_mask] - margin).pow(2)
        loss = loss + wrong_loss.sum()
        n_samples += wrong_mask.sum()
    
    return loss / (n_samples + 1e-8)


def ece_soft_loss(confidence, correctness, n_bins=15):
    """Soft differentiable ECE approximation."""
    bin_boundaries = torch.linspace(0, 1, n_bins + 1, device=confidence.device)
    bin_centers = (bin_boundaries[:-1] + bin_boundaries[1:]) / 2
    bin_width = 1.0 / n_bins
    
    sigma = bin_width / 2
    weights = torch.exp(-0.5 * ((confidence.unsqueeze(1) - bin_centers) / sigma).pow(2))
    weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-8)
    
    weighted_conf = (weights * confidence.unsqueeze(1)).sum(dim=0)
    weighted_acc = (weights * correctness.unsqueeze(1)).sum(dim=0)
    bin_counts = weights.sum(dim=0)
    
    avg_conf = weighted_conf / (bin_counts + 1e-8)
    avg_acc = weighted_acc / (bin_counts + 1e-8)
    
    ece = (torch.abs(avg_conf - avg_acc) * bin_counts).sum() / (bin_counts.sum() + 1e-8)
    
    return ece


def smoothed_tcp_target(logits, targets, label_smoothing=0.1):
    """Label-smoothed TCP target."""
    probs = F.softmax(logits, dim=1)
    tcp = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
    num_classes = logits.size(1)
    smoothed = (1 - label_smoothing) * tcp + label_smoothing * (1.0 / num_classes)
    return smoothed


def avuc_loss(confidence, correctness, penalty_weight=2.0):
    """Accuracy vs Uncertainty Calibration Loss."""
    wrong_mask = correctness < 0.5
    overconf_wrong = F.relu(confidence - 0.5) * wrong_mask.float()
    
    correct_mask = correctness > 0.5
    underconf_correct = F.relu(0.5 - confidence) * correct_mask.float()
    
    loss = penalty_weight * overconf_wrong.pow(2).mean() + underconf_correct.pow(2).mean()
    
    return loss


def nll_loss(confidence, target):
    """Negative Log-Likelihood Loss for probabilistic calibration."""
    eps = 1e-7
    confidence = torch.clamp(confidence, eps, 1 - eps)
    nll = -(target * torch.log(confidence) + (1 - target) * torch.log(1 - confidence))
    return nll.mean()


def soft_rank(scores, tau=0.1):
    """
    Compute differentiable soft ranks using sigmoid approximation.
    
    Args:
        scores: (B,) confidence scores for batch
        tau: temperature for sigmoid softness (lower = sharper, higher = softer)
    
    Returns:
        soft_ranks: (B,) soft ranks in [1, B+1]
    
    Implementation:
        r_i = 1 + Σ_{j≠i} σ((s_i - s_j) / τ)
        where σ is sigmoid function
    """
    # Expand to (B, B) for pairwise comparisons
    scores_i = scores.unsqueeze(1)  # (B, 1)
    scores_j = scores.unsqueeze(0)  # (1, B)
    
    # Pairwise differences: (B, B) where [i,j] = scores[i] - scores[j]
    diffs = (scores_i - scores_j) / tau
    
    # Sigmoid of differences (high if i > j)
    sigmoid_diffs = torch.sigmoid(diffs)
    
    # Sum over j (excluding diagonal where i=j)
    # Create mask to exclude self-comparisons
    mask = 1.0 - torch.eye(scores.size(0), device=scores.device)
    
    # Soft rank: 1 + sum of sigmoid values
    soft_ranks = 1.0 + (sigmoid_diffs * mask).sum(dim=1)
    
    return soft_ranks


def aurc_surrogate_loss(confidence, correctness, tau=0.1, eps=1e-7):
    """
    Regularized AURC (r-AURC) Surrogate Loss.
    
    Directly targets minimizing Area Under Risk-Coverage Curve via
    reweighted risk with soft ranking.
    
    Based on: Zhang et al., "Revisiting Reweighted Risk for Calibration", 2025
    
    Loss = -1/n Σ_i ln(1 - r_i/(n+1)) * ℓ(confidence_i, correctness_i)
    
    where r_i is the soft rank of confidence_i (higher confidence → higher rank)
    
    Intuition: Heavily penalize misclassified samples with high confidence
    (high soft rank), pushing model to lower AURC by calibrating better.
    
    Args:
        confidence: (B,) predicted confidence scores in [0, 1]
        correctness: (B,) binary correctness (1 = correct, 0 = wrong)
        tau: temperature for soft ranking (default 0.1)
        eps: small value to avoid log(0)
    
    Returns:
        Scalar r-AURC loss
    """
    n = confidence.size(0)
    
    # Compute soft ranks (higher confidence → higher rank)
    ranks = soft_rank(confidence, tau=tau)
    
    # Normalize ranks to [0, 1] range: (rank - 1) / n
    normalized_ranks = (ranks - 1.0) / n
    
    # Reweighting term: -ln(1 - r_i/(n+1))
    # Higher rank → higher weight
    weights = -torch.log(1.0 - normalized_ranks / (n + 1) + eps)
    
    # Base loss: MSE between confidence and correctness
    # (could also use BCE if treating as classification)
    base_loss = (confidence - correctness).pow(2)
    
    # Weighted loss
    weighted_loss = weights * base_loss
    
    return weighted_loss.mean()


def inverse_focal_loss(confidence, correctness, gamma=3.0, eps=1e-7):
    """
    Inverse Focal Loss (IFL) - Simpler AURC-targeting loss.
    
    A simpler alternative to r-AURC that reweights without explicit ranking.
    
    Loss = -(1 + p_correct)^γ * log(p_predicted)
    
    where p_predicted is confidence for correct samples, (1-confidence) for wrong.
    
    Intuition: For correct samples with high TCP (easy), γ > 0 down-weights.
    For wrong samples or uncertain correct ones, weight is higher.
    This pushes model to focus on hard samples and calibrate better.
    
    Args:
        confidence: (B,) predicted confidence in [0, 1]
        correctness: (B,) binary correctness (1 = correct, 0 = wrong)
        gamma: focal parameter (higher = more focus on hard samples), try 2-4
        eps: small value to avoid log(0)
    
    Returns:
        Scalar IFL loss
    """
    confidence = torch.clamp(confidence, eps, 1 - eps)
    
    # For correct samples: use confidence directly
    # For wrong samples: use (1 - confidence) as "confidence of being wrong"
    p_target = correctness * confidence + (1 - correctness) * (1 - confidence)
    
    # Focal reweighting: (1 + p_target)^gamma gives higher weight to low p_target
    focal_weight = (1.0 + p_target).pow(gamma)
    
    # Negative log-likelihood term
    nll_term = -torch.log(p_target)
    
    loss = focal_weight * nll_term
    
    return loss.mean()


def aurc_hinge_loss(confidence, correctness, margin=0.1, temperature=0.1):
    """
    AURC Hinge Loss - Margin-based ranking loss for selective classification.
    
    Encourages separation: correct samples should have higher confidence
    than incorrect samples by at least a margin.
    
    Loss = Σ_{i correct, j wrong} max(0, margin - (conf_i - conf_j))
    
    Softened with temperature for differentiability.
    
    Args:
        confidence: (B,) predicted confidence scores
        correctness: (B,) binary correctness
        margin: desired confidence gap (default 0.1)
        temperature: softness for max operation (lower = sharper)
    
    Returns:
        Scalar hinge loss
    """
    # Separate correct and incorrect samples
    correct_mask = correctness > 0.5
    wrong_mask = ~correct_mask
    
    if correct_mask.sum() == 0 or wrong_mask.sum() == 0:
        return torch.tensor(0.0, device=confidence.device)
    
    conf_correct = confidence[correct_mask]  # (N_c,)
    conf_wrong = confidence[wrong_mask]      # (N_w,)
    
    # Pairwise differences: (N_c, N_w)
    # Each [i,j] = conf_correct[i] - conf_wrong[j]
    diffs = conf_correct.unsqueeze(1) - conf_wrong.unsqueeze(0)
    
    # Hinge: max(0, margin - diff)
    # Use softplus for smooth approximation: softplus(-x/temp) ≈ max(0, -x)
    hinge_values = F.softplus((margin - diffs) / temperature) * temperature
    
    return hinge_values.mean()


# Constants
CIFAR10_CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']
COVERAGE_POINTS = [100, 99, 98, 97, 95, 90, 85, 80, 75, 70, 60, 50, 40, 30, 20, 10]


# =============================================================================
# VGG16_bn with Multi-Level Feature Extraction
# =============================================================================

from models.cifar import vgg

class VGG16BN_MultiLevel_FeatureExtractor(nn.Module):
    """
    VGG16_bn wrapper that exposes Pool3, Pool4, Pool5 features.
    Supports 32x32 (CIFAR/SVHN) and 64x64 (Cats vs Dogs) inputs.
    """
    def __init__(self, num_classes=10, input_size=32):
        super().__init__()
        
        self.base_model = vgg.vgg16_bn(num_classes=num_classes, input_size=input_size)
        self.input_size = input_size
        
        self._find_pool_indices()
    
    def _find_pool_indices(self):
        """Find indices of MaxPool2d layers."""
        self.pool_indices = []
        for i, layer in enumerate(self.base_model.features):
            if isinstance(layer, nn.MaxPool2d):
                self.pool_indices.append(i)
    
    def forward(self, x, return_features=False):
        if not return_features:
            return self.base_model(x)
        
        # Run through features, capturing Pool3, Pool4, Pool5
        feat_pool3 = None
        feat_pool4 = None
        feat_pool5 = None
        
        for i, layer in enumerate(self.base_model.features):
            x = layer(x)
            # Pool3 (3rd MaxPool, index 2 in pool_indices)
            if len(self.pool_indices) >= 3 and i == self.pool_indices[2]:
                feat_pool3 = x
            # Pool4 (4th MaxPool, index 3 in pool_indices)
            if len(self.pool_indices) >= 4 and i == self.pool_indices[3]:
                feat_pool4 = x
            # Pool5 (5th MaxPool, index 4 in pool_indices)
            if len(self.pool_indices) >= 5 and i == self.pool_indices[4]:
                feat_pool5 = x
        
        # Classifier
        x = x.view(x.size(0), -1)
        logits = self.base_model.classifier(x)
        
        return logits, feat_pool3, feat_pool4, feat_pool5


# =============================================================================
# Mini-Classifier Heads (Auxiliary Classifiers, GoogLeNet-style)
# =============================================================================

class MiniClassifier(nn.Module):
    """
    Lightweight auxiliary classifier for intermediate features.
    Similar to GoogLeNet auxiliary heads but simpler.
    
    Input: Pooled features (e.g., Pool3: 512×4×4)
    Output: Logits (num_classes-dim)
    """
    def __init__(self, in_channels, num_classes=10, input_size=4):
        super().__init__()
        
        # Adaptive pooling to handle variable input sizes
        self.avgpool = nn.AdaptiveAvgPool2d((2, 2))
        
        # Calculate flattened size after pooling: in_channels * 2 * 2
        flattened_size = in_channels * 2 * 2
        
        # Simple 2-layer MLP
        self.fc = nn.Sequential(
            nn.Linear(flattened_size, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        x = self.avgpool(x)  # (B, C, 2, 2)
        x = x.view(x.size(0), -1)  # (B, C*4)
        logits = self.fc(x)
        return logits


# =============================================================================
# Auxiliary Meta-Calibrator (Per-Layer Confidence Predictor)
# =============================================================================

class AuxiliaryMetaCalibrator(nn.Module):
    """
    Per-layer confidence predictor (DSN companion objective).
    
    Input: Concatenated [layer_features, mini_classifier_logits]
    Output: Confidence ĉ^{(m)}(x) ∈ [0, 1]
    
    Architecture: 4-layer MLP (1024 → 512 → 256 → 128 → 1)
    """
    def __init__(self, feat_dim, logit_dim=10):
        super().__init__()
        
        input_dim = feat_dim + logit_dim
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, feat, logits):
        """
        Args:
            feat: (B, feat_dim) - flattened layer features
            logits: (B, num_classes) - mini-classifier logits (detached)
        Returns:
            confidence: (B,) in [0, 1]
        """
        combined = torch.cat([feat, logits.detach()], dim=1)
        confidence = self.network(combined).squeeze(-1)
        return confidence


# =============================================================================
# DS-SCSF Model (Integrates Backbone + Auxiliaries)
# =============================================================================

class DS_SCSF_Model(nn.Module):
    """
    Deeply-Supervised Self-Calibrated Selective Framework.
    
    Components:
      - VGG16_bn backbone (multi-level features)
      - 3 Mini-Classifiers at Pool3, Pool4, Pool5
      - 3 Auxiliary Meta-Calibrators for confidence prediction
      - Fusion weights β_m for final confidence
    """
    def __init__(self, num_classes=10, input_size=32):
        super().__init__()
        
        # Backbone
        self.backbone = VGG16BN_MultiLevel_FeatureExtractor(num_classes, input_size)
        
        # Feature dimensions (depends on input_size)
        # VGG16 architecture: Pool3=256ch, Pool4=512ch, Pool5=512ch
        if input_size == 32:
            pool3_channels = 256  # Pool3 has 256 channels
            pool4_channels = 512  # Pool4 has 512 channels
            pool5_channels = 512  # Pool5 has 512 channels
            pool3_spatial = 4     # 4×4 spatial
            pool4_spatial = 2     # 2×2 spatial
            pool5_spatial = 1     # 1×1 spatial
        else:  # 64x64
            pool3_channels = 256
            pool4_channels = 512
            pool5_channels = 512
            pool3_spatial = 8     # 8×8 spatial
            pool4_spatial = 4     # 4×4 spatial
            pool5_spatial = 2     # 2×2 spatial
        
        # Calculate flattened dimensions
        pool3_dim = pool3_channels * pool3_spatial * pool3_spatial
        pool4_dim = pool4_channels * pool4_spatial * pool4_spatial
        pool5_dim = pool5_channels * pool5_spatial * pool5_spatial
        
        # Mini-Classifiers (GoogLeNet-style auxiliary heads)
        # Pass the correct number of channels for each pool layer
        self.mini_clf3 = MiniClassifier(pool3_channels, num_classes, input_size=pool3_spatial)
        self.mini_clf4 = MiniClassifier(pool4_channels, num_classes, input_size=pool4_spatial)
        self.mini_clf5 = MiniClassifier(pool5_channels, num_classes, input_size=pool5_spatial)
        
        # Auxiliary Meta-Calibrators (DSN companion objectives)
        self.aux_cal3 = AuxiliaryMetaCalibrator(pool3_dim, num_classes)
        self.aux_cal4 = AuxiliaryMetaCalibrator(pool4_dim, num_classes)
        self.aux_cal5 = AuxiliaryMetaCalibrator(pool5_dim, num_classes)
        
        # Fusion weights β_m (learnable or fixed, default: deeper = higher weight)
        # Inspired by GoogLeNet's fixed auxiliary weights (0.3) but learnable
        self.register_parameter('beta3', nn.Parameter(torch.tensor(0.2)))
        self.register_parameter('beta4', nn.Parameter(torch.tensor(0.3)))
        self.register_parameter('beta5', nn.Parameter(torch.tensor(0.5)))
    
    def forward(self, x, return_auxiliaries=False):
        """
        Args:
            x: (B, 3, H, W) input images
            return_auxiliaries: If True, return all auxiliary outputs
        
        Returns:
            If return_auxiliaries=False:
                logits: (B, num_classes) - main classifier output
            If return_auxiliaries=True:
                logits, aux_outputs dict with:
                  - 'feat_pool3/4/5': layer features
                  - 'logits3/4/5': mini-classifier logits
                  - 'conf3/4/5': auxiliary confidences
                  - 'conf_fused': fused final confidence
        """
        # Forward through backbone
        logits, feat_pool3, feat_pool4, feat_pool5 = self.backbone(x, return_features=True)
        
        if not return_auxiliaries:
            return logits
        
        # Mini-Classifiers
        logits3 = self.mini_clf3(feat_pool3)
        logits4 = self.mini_clf4(feat_pool4)
        logits5 = self.mini_clf5(feat_pool5)
        
        # Flatten features
        flat_pool3 = feat_pool3.view(feat_pool3.size(0), -1)
        flat_pool4 = feat_pool4.view(feat_pool4.size(0), -1)
        flat_pool5 = feat_pool5.view(feat_pool5.size(0), -1)
        
        # Auxiliary Confidences
        conf3 = self.aux_cal3(flat_pool3, logits3)
        conf4 = self.aux_cal4(flat_pool4, logits4)
        conf5 = self.aux_cal5(flat_pool5, logits5)
        
        # Fused Confidence (weighted sum with normalized β_m)
        beta_sum = torch.abs(self.beta3) + torch.abs(self.beta4) + torch.abs(self.beta5)
        beta3_norm = torch.abs(self.beta3) / beta_sum
        beta4_norm = torch.abs(self.beta4) / beta_sum
        beta5_norm = torch.abs(self.beta5) / beta_sum
        
        conf_fused = beta3_norm * conf3 + beta4_norm * conf4 + beta5_norm * conf5
        
        aux_outputs = {
            'feat_pool3': feat_pool3,
            'feat_pool4': feat_pool4,
            'feat_pool5': feat_pool5,
            'logits3': logits3,
            'logits4': logits4,
            'logits5': logits5,
            'conf3': conf3,
            'conf4': conf4,
            'conf5': conf5,
            'conf_fused': conf_fused,
            'beta_weights': (beta3_norm.item(), beta4_norm.item(), beta5_norm.item())
        }
        
        return logits, aux_outputs


# =============================================================================
# PPO Agent for Threshold + Auxiliary Weight Tuning
# =============================================================================

class ActorCritic(nn.Module):
    """Actor-Critic network for PPO (extended to tune auxiliary weights)."""
    
    def __init__(self, state_dim, action_dim=4, hidden_dim=64):
        super().__init__()
        
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Actor: outputs mean for actions [threshold τ, α3, α4, α5]
        self.actor_mean = nn.Linear(hidden_dim, action_dim)
        self.actor_log_std = nn.Parameter(torch.zeros(action_dim))
        
        # Critic: outputs state value
        self.critic = nn.Linear(hidden_dim, 1)
    
    def forward(self, state):
        shared = self.shared(state)
        action_mean = torch.sigmoid(self.actor_mean(shared))  # Actions in [0, 1]
        action_std = torch.exp(self.actor_log_std.clamp(-2, 0))
        value = self.critic(shared)
        return action_mean, action_std, value
    
    def get_action(self, state, deterministic=False):
        action_mean, action_std, _ = self.forward(state)
        if deterministic:
            return action_mean, None
        dist = torch.distributions.Normal(action_mean, action_std)
        action = dist.sample()
        action = torch.clamp(action, 0.0, 1.0)
        log_prob = dist.log_prob(action).sum(-1)
        return action, log_prob
    
    def evaluate(self, state, action):
        action_mean, action_std, value = self.forward(state)
        dist = torch.distributions.Normal(action_mean, action_std)
        log_prob = dist.log_prob(action).sum(-1)
        entropy = dist.entropy().sum(-1)
        return log_prob, value.squeeze(-1), entropy


class PPOAgent:
    """PPO Agent for tuning threshold τ and auxiliary weights α_m."""
    
    def __init__(self, state_dim, lr=3e-4, gamma=0.99, device='cuda'):
        self.gamma = gamma
        self.clip_eps = 0.2
        self.value_coef = 0.5
        self.entropy_coef = 0.01
        self.device = device
        
        # Action dim = 4: [threshold τ, α3, α4, α5]
        self.actor_critic = ActorCritic(state_dim, action_dim=4).to(device)
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=lr)
        
        # Rollout buffer
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.values = []
    
    def select_action(self, state):
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action, log_prob = self.actor_critic.get_action(state_tensor)
        return action.cpu().numpy().squeeze(), log_prob.item() if log_prob is not None else 0.0
    
    def store_transition(self, state, action, reward, value, log_prob):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
    
    def update(self, n_epochs=4):
        if len(self.states) < 2:
            return {'policy_loss': 0, 'value_loss': 0}
        
        states = torch.FloatTensor(np.array(self.states)).to(self.device)
        actions = torch.FloatTensor(np.array(self.actions)).to(self.device)
        old_log_probs = torch.FloatTensor(self.log_probs).to(self.device)
        
        returns = []
        R = 0
        for r in reversed(self.rewards):
            R = r + self.gamma * R
            returns.insert(0, R)
        returns = torch.FloatTensor(returns).to(self.device)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        total_policy_loss = 0
        total_value_loss = 0
        
        for _ in range(n_epochs):
            log_probs, values, entropy = self.actor_critic.evaluate(states, actions)
            
            advantages = returns - values.detach()
            
            ratio = torch.exp(log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            
            value_loss = F.mse_loss(values, returns)
            
            loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy.mean()
            
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.actor_critic.parameters(), 0.5)
            self.optimizer.step()
            
            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
        
        self.states.clear()
        self.actions.clear()
        self.log_probs.clear()
        self.rewards.clear()
        self.values.clear()
        
        return {
            'policy_loss': total_policy_loss / n_epochs,
            'value_loss': total_value_loss / n_epochs
        }


# =============================================================================
# Dataset Loading (Same as SCSF)
# =============================================================================

def get_dataset(args):
    """Load dataset with train/val/test splits."""
    data_root = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
    
    if args.dataset == 'cifar10':
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
        
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        
        trainset = datasets.CIFAR10(root=data_root, train=True, download=True, transform=transform_train)
        testset = datasets.CIFAR10(root=data_root, train=False, download=True, transform=transform_test)
        num_classes = 10
        
        torch.manual_seed(args.seed)
        valset, testset_final = random_split(testset, [2000, 8000])
        print(f'CIFAR-10: Train={len(trainset)}, Full_Test={len(testset)}, Val={len(valset)}, Final_Test={len(testset_final)}')
        
    elif args.dataset == 'svhn':
        mean = (0.5, 0.5, 0.5)
        std = (0.5, 0.5, 0.5)
        
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        
        svhn_root = os.path.join(data_root, 'svhn')
        trainset = datasets.SVHN(root=svhn_root, split='train', download=False, transform=transform_train)
        testset = datasets.SVHN(root=svhn_root, split='test', download=False, transform=transform_test)
        num_classes = 10
        
        torch.manual_seed(args.seed)
        test_size = len(testset)
        val_size = 5000
        test_final_size = test_size - val_size
        valset, testset_final = random_split(testset, [val_size, test_final_size])
        print(f'SVHN: Train={len(trainset)}, Full_Test={test_size}, Val={val_size}, Final_Test={test_final_size}')
        
    elif args.dataset == 'covid':
        # COVID-QU-Ex normalization (ImageNet stats)
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        
        # Use 64x64 images (medical images)
        transform_train = transforms.Compose([
            transforms.Resize(64),
            transforms.RandomCrop(64, padding=6),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        transform_test = transforms.Compose([
            transforms.Resize(64),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        
        # Load COVID dataset using ImageFolder
        covid_root = os.path.join(data_root, 'covid')
        train_path = os.path.join(covid_root, 'train')
        val_path = os.path.join(covid_root, 'val')
        test_path = os.path.join(covid_root, 'test')
        
        assert os.path.exists(train_path), f"Train folder not found: {train_path}. Run prepare_covid_dataset.py first."
        assert os.path.exists(val_path), f"Val folder not found: {val_path}. Run prepare_covid_dataset.py first."
        assert os.path.exists(test_path), f"Test folder not found: {test_path}. Run prepare_covid_dataset.py first."
        
        trainset = datasets.ImageFolder(train_path, transform=transform_train)
        valset = datasets.ImageFolder(val_path, transform=transform_test)
        testset = datasets.ImageFolder(test_path, transform=transform_test)
        testset_final = testset
        num_classes = 3  # COVID-19, Non-COVID, Normal
        
        print(f'COVID-QU-Ex: Train={len(trainset)}, Val={len(valset)}, Test={len(testset)}')
        print(f'Classes: {trainset.classes}')
        
    else:
        raise ValueError(f'Unknown dataset: {args.dataset}')
    
    trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, 
                             num_workers=args.workers, pin_memory=True)
    valloader = DataLoader(valset, batch_size=200, shuffle=False, 
                           num_workers=args.workers, pin_memory=True)
    testloader = DataLoader(testset_final, batch_size=200, shuffle=False, 
                            num_workers=args.workers, pin_memory=True)
    testloader_full = DataLoader(testset, batch_size=200, shuffle=False,
                                  num_workers=args.workers, pin_memory=True)
    
    return trainloader, valloader, testloader, testloader_full, num_classes


# =============================================================================
# Evaluation Functions
# =============================================================================

def evaluate_selective_risk(model, loader, coverage_points, device):
    """Calculate Error@Coverage using fused confidence."""
    model.eval()
    
    all_scores = []
    total_correct = 0
    total_samples = 0
    total_confidence = 0
    total_tcp = 0
    
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            logits, aux_outputs = model(inputs, return_auxiliaries=True)
            confidence = aux_outputs['conf_fused']
            
            probs = F.softmax(logits, dim=1)
            tcp = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
            
            _, preds = logits.max(dim=1)
            correct = preds.eq(targets)
            
            for i in range(len(targets)):
                all_scores.append((
                    confidence[i].item(),
                    correct[i].item(),
                    tcp[i].item()
                ))
            
            total_correct += correct.sum().item()
            total_samples += len(targets)
            total_confidence += confidence.sum().item()
            total_tcp += tcp.sum().item()
    
    all_scores_sorted = sorted(all_scores, key=lambda x: x[0], reverse=True)
    n = len(all_scores_sorted)
    
    coverage_errors = {}
    for cov in coverage_points:
        k = max(1, int(n * cov / 100))
        top_k = all_scores_sorted[:k]
        n_errors = sum(1 for _, correct, _ in top_k if not correct)
        error_rate = n_errors / k * 100
        coverage_errors[cov] = error_rate
    
    aurc = 0.0
    sorted_risks = []
    for i in range(1, n + 1):
        n_errors = sum(1 for _, correct, _ in all_scores_sorted[:i] if not correct)
        risk = n_errors / i
        sorted_risks.append(risk)
    
    for i in range(len(sorted_risks) - 1):
        aurc += (sorted_risks[i] + sorted_risks[i + 1]) / 2 * (1 / n)
    
    global_acc = total_correct / total_samples * 100
    mean_confidence = total_confidence / total_samples
    mean_tcp = total_tcp / total_samples
    
    return coverage_errors, global_acc, aurc, mean_confidence, mean_tcp


# =============================================================================
# Training Functions
# =============================================================================

def compute_dsn_decay_weight(epoch, pretrain_epochs, total_epochs, base_weight=1.0, decay_rate=0.1):
    """
    DSN-style decaying auxiliary weight: α_m * decay_rate * (1 - t/N)
    
    Args:
        epoch: current epoch
        pretrain_epochs: warmup epochs (no decay during warmup)
        total_epochs: total training epochs
        base_weight: base α_m value
        decay_rate: decay multiplier (DSN uses 0.1)
    
    Returns:
        Decayed weight (vanishes as training progresses)
    """
    if epoch <= pretrain_epochs:
        return base_weight
    
    t = epoch - pretrain_epochs
    N = total_epochs - pretrain_epochs
    decay_factor = 1.0 - (t / N)
    
    return base_weight * decay_rate * decay_factor


def train_epoch_warmup(model, trainloader, optimizer, device):
    """Phase 1: Warmup with CE on main + auxiliary classifiers."""
    model.train()
    
    losses = AverageMeter()
    accuracies = AverageMeter()
    
    bar = Bar('Warmup', max=len(trainloader))
    
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        
        # Forward with auxiliaries
        logits, aux_outputs = model(inputs, return_auxiliaries=True)
        
        # Main CE loss
        loss_main = F.cross_entropy(logits, targets)
        
        # Auxiliary CE losses (GoogLeNet-style, fixed 0.3 weight)
        loss_aux3 = 0.3 * F.cross_entropy(aux_outputs['logits3'], targets)
        loss_aux4 = 0.3 * F.cross_entropy(aux_outputs['logits4'], targets)
        loss_aux5 = 0.3 * F.cross_entropy(aux_outputs['logits5'], targets)
        
        # Total loss
        loss = loss_main + loss_aux3 + loss_aux4 + loss_aux5
        
        _, preds = logits.max(dim=1)
        acc = preds.eq(targets).float().mean().item() * 100
        
        losses.update(loss.item(), inputs.size(0))
        accuracies.update(acc, inputs.size(0))
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        bar.suffix = '({batch}/{size}) | Loss: {loss:.4f} | Acc: {acc:.2f}'.format(
            batch=batch_idx+1, size=len(trainloader), loss=losses.avg, acc=accuracies.avg)
        bar.next()
    
    bar.finish()
    
    return losses.avg, accuracies.avg


def train_epoch_joint(model, trainloader, optimizer, epoch, args, 
                      alpha3, alpha4, alpha5, device, calib_args=None):
    """
    Phase 2: Joint training with CE + Multi-Layer Meta-Losses.
    
    Args:
        alpha3, alpha4, alpha5: Auxiliary calibration loss weights (DSN-style, decaying)
        calib_args: Dict of calibration enhancement options
    
    Returns:
        train_loss, ce_loss, meta_loss_total, accuracy, mean_confidence
    """
    if calib_args is None:
        calib_args = {}
    
    model.train()
    
    ce_losses = AverageMeter()
    meta_losses_3 = AverageMeter()
    meta_losses_4 = AverageMeter()
    meta_losses_5 = AverageMeter()
    total_losses = AverageMeter()
    accuracies = AverageMeter()
    confidences = AverageMeter()
    
    bar = Bar('Joint', max=len(trainloader))
    
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        
        # Forward
        logits, aux_outputs = model(inputs, return_auxiliaries=True)
        
        # Main CE loss
        ce_loss = F.cross_entropy(logits, targets)
        
        # Auxiliary CE losses (fixed 0.3 weight, GoogLeNet-style)
        loss_aux3_ce = 0.3 * F.cross_entropy(aux_outputs['logits3'], targets)
        loss_aux4_ce = 0.3 * F.cross_entropy(aux_outputs['logits4'], targets)
        loss_aux5_ce = 0.3 * F.cross_entropy(aux_outputs['logits5'], targets)
        
        # Correctness for calibration
        with torch.no_grad():
            _, preds = logits.max(dim=1)
            correctness = (preds == targets).float()
        
        # Meta targets (TCP or correctness)
        if args.brier:
            meta_target = correctness.clone()
        else:
            if calib_args.get('label_smooth', 0) > 0:
                meta_target = smoothed_tcp_target(logits, targets, calib_args['label_smooth'])
            else:
                probs = F.softmax(logits, dim=1)
                meta_target = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        
        # Auxiliary Meta-Losses (DSN companion objectives)
        def compute_aux_meta_loss(confidence, target, correctness):
            """Compute calibration loss for one auxiliary."""
            # Frequency-based weighting: lower frequency (harder samples) get higher weight
            # This is similar to class balancing but for correct/incorrect predictions
            error_weight_type = calib_args.get('error_weight_type', 'fixed')  # 'fixed' or 'frequency'
            
            if error_weight_type == 'frequency':
                # Compute inverse frequency weighting
                n_correct = correctness.sum().item()
                n_incorrect = (correctness < 0.5).sum().item()
                n_total = len(correctness)
                
                if n_correct > 0 and n_incorrect > 0:
                    # Inverse frequency: weight = n_total / (n_class * freq)
                    # Normalized so correct samples have weight 1.0
                    freq_correct = n_correct / n_total
                    freq_incorrect = n_incorrect / n_total
                    
                    # Calculate weights (normalized by correct frequency to keep scale reasonable)
                    weight_correct = 1.0
                    weight_incorrect = freq_correct / freq_incorrect
                    
                    weights = torch.ones_like(correctness).float()
                    weights[correctness >= 0.5] = weight_correct
                    weights[correctness < 0.5] = weight_incorrect
                else:
                    # Fallback if all correct or all incorrect
                    weights = torch.ones_like(correctness).float()
            else:
                # Fixed weighting (ConfidNet-style): simple multiplier for errors
                error_weight = calib_args.get('error_weight', 1.0)  # 1.0 = no weighting, 2.0 = double weight on errors
                weights = torch.ones_like(correctness).float()
                weights[correctness < 0.5] *= error_weight
            
            # Primary loss selection (mutually exclusive)
            if calib_args.get('aurc_surrogate', False):
                # r-AURC Surrogate: directly targets AURC via soft ranking
                base_loss = aurc_surrogate_loss(confidence, correctness, 
                                                tau=calib_args.get('aurc_tau', 0.1))
            elif calib_args.get('inverse_focal', False):
                # Inverse Focal Loss: simpler AURC-targeting alternative
                base_loss = inverse_focal_loss(confidence, correctness,
                                               gamma=calib_args.get('inverse_focal_gamma', 3.0))
            elif calib_args.get('nll', False):
                # NLL: probabilistic calibration
                base_loss = nll_loss(confidence, target.detach())
            elif calib_args.get('focal', 0) > 0:
                # Focal MSE: focus on hard samples
                base_loss = focal_mse_loss(confidence, target.detach(), gamma=calib_args['focal'])
            else:
                # Default: Weighted MSE
                # Apply per-sample weighting (frequency-based or fixed)
                mse_per_sample = (confidence - target.detach()) ** 2
                base_loss = (weights * mse_per_sample).mean()
            
            # Additional losses (can combine with primary)
            extra_loss = 0.0
            if calib_args.get('margin', 0) > 0:
                extra_loss += calib_args['margin'] * margin_calibration_loss(confidence, correctness, margin=0.3)
            if calib_args.get('ece_reg', 0) > 0:
                extra_loss += calib_args['ece_reg'] * ece_soft_loss(confidence, correctness, n_bins=15)
            if calib_args.get('avuc', 0) > 0:
                extra_loss += calib_args['avuc'] * avuc_loss(confidence, correctness, penalty_weight=2.0)
            if calib_args.get('aurc_hinge', False):
                extra_loss += aurc_hinge_loss(confidence, correctness,
                                              margin=calib_args.get('aurc_hinge_margin', 0.1))
            
            return base_loss + extra_loss
        
        meta_loss3 = compute_aux_meta_loss(aux_outputs['conf3'], meta_target, correctness)
        meta_loss4 = compute_aux_meta_loss(aux_outputs['conf4'], meta_target, correctness)
        meta_loss5 = compute_aux_meta_loss(aux_outputs['conf5'], meta_target, correctness)
        
        # Total loss (DSN-style weighted sum)
        total_ce = ce_loss + loss_aux3_ce + loss_aux4_ce + loss_aux5_ce
        total_meta = alpha3 * meta_loss3 + alpha4 * meta_loss4 + alpha5 * meta_loss5
        total_loss = total_ce + total_meta
        
        # Accuracy
        _, preds = logits.max(dim=1)
        acc = preds.eq(targets).float().mean().item() * 100
        
        ce_losses.update(ce_loss.item(), inputs.size(0))
        meta_losses_3.update(meta_loss3.item(), inputs.size(0))
        meta_losses_4.update(meta_loss4.item(), inputs.size(0))
        meta_losses_5.update(meta_loss5.item(), inputs.size(0))
        total_losses.update(total_loss.item(), inputs.size(0))
        accuracies.update(acc, inputs.size(0))
        confidences.update(aux_outputs['conf_fused'].mean().item(), inputs.size(0))
        
        # Backward
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        bar.suffix = '({batch}/{size}) | CE: {ce:.4f} | Meta: {meta:.4f} | Acc: {acc:.2f}'.format(
            batch=batch_idx+1, size=len(trainloader), ce=ce_losses.avg, 
            meta=(meta_losses_3.avg + meta_losses_4.avg + meta_losses_5.avg)/3, acc=accuracies.avg)
        bar.next()
    
    bar.finish()
    
    meta_loss_total = meta_losses_3.avg + meta_losses_4.avg + meta_losses_5.avg
    
    return total_losses.avg, ce_losses.avg, meta_loss_total, accuracies.avg, confidences.avg


# =============================================================================
# Main Training Loop
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='DS-SCSF: Deeply-Supervised Self-Calibrated Selective Framework')
    
    # Dataset
    parser.add_argument('-d', '--dataset', default='cifar10', type=str)
    parser.add_argument('-j', '--workers', default=0, type=int)
    
    # Training
    parser.add_argument('--epochs', default=300, type=int)
    parser.add_argument('--batch-size', default=128, type=int)
    parser.add_argument('--pretrain', default=100, type=int, help='CE warmup epochs')
    
    # Optimizer
    parser.add_argument('--lr', default=0.1, type=float)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--weight-decay', default=5e-4, type=float)
    
    # RL
    parser.add_argument('--rl-freq', default=2, type=int, help='RL update frequency')
    parser.add_argument('--rl-lr', default=3e-4, type=float)
    parser.add_argument('--init-threshold', default=0.5, type=float)
    parser.add_argument('--no-rl', action='store_true', help='Disable RL')
    
    # Auxiliary weights (DSN-style)
    parser.add_argument('--init-alpha3', default=1.0, type=float, help='Initial α3 weight for Pool3 auxiliary')
    parser.add_argument('--init-alpha4', default=1.0, type=float, help='Initial α4 weight for Pool4 auxiliary')
    parser.add_argument('--init-alpha5', default=1.0, type=float, help='Initial α5 weight for Pool5 auxiliary')
    parser.add_argument('--dsn-decay', action='store_true', help='Use DSN-style decaying auxiliary weights')
    
    # Meta-Loss Type
    parser.add_argument('--brier', action='store_true', help='Use Brier Score instead of TCP')
    
    # Calibration Enhancements
    parser.add_argument('--error-weight', type=float, default=1.0, metavar='W',
                        help='Fixed error weighting: weight for incorrect predictions (default: 1.0 = no weighting, 2.0 = double weight on errors)')
    parser.add_argument('--error-weight-type', type=str, default='fixed', choices=['fixed', 'frequency'],
                        help='Weighting strategy: "fixed" uses --error-weight multiplier, "frequency" uses inverse frequency weighting (default: fixed)')
    parser.add_argument('--focal', type=float, default=0.0, metavar='GAMMA')
    parser.add_argument('--margin', type=float, default=0.0, metavar='M')
    parser.add_argument('--ece-reg', type=float, default=0.0, metavar='W')
    parser.add_argument('--label-smooth', type=float, default=0.0, metavar='LS')
    parser.add_argument('--avuc', type=float, default=0.0, metavar='W')
    parser.add_argument('--nll', action='store_true')
    
    # AURC-Targeting Losses (New!)
    parser.add_argument('--aurc-surrogate', action='store_true',
                        help='Use r-AURC surrogate loss (reweighted risk with soft ranking)')
    parser.add_argument('--aurc-tau', type=float, default=0.1,
                        help='Temperature for soft ranking in r-AURC (default: 0.1)')
    parser.add_argument('--inverse-focal', action='store_true',
                        help='Use Inverse Focal Loss (simpler AURC-targeting loss)')
    parser.add_argument('--inverse-focal-gamma', type=float, default=3.0,
                        help='Gamma for Inverse Focal Loss (default: 3.0, try 2-4)')
    parser.add_argument('--aurc-hinge', action='store_true',
                        help='Use AURC Hinge Loss (margin-based ranking loss)')
    parser.add_argument('--aurc-hinge-margin', type=float, default=0.1,
                        help='Margin for AURC Hinge Loss (default: 0.1)')
    
    # Misc
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--gpu', default='0', type=str)
    
    args = parser.parse_args()
    
    # Set seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    
    # Device
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Dataset
    trainloader, valloader, testloader, testloader_full, num_classes = get_dataset(args)
    
    # Input size
    input_size = 64 if args.dataset == 'covid' else 32  # COVID uses 64x64
    
    # Model
    model = DS_SCSF_Model(num_classes=num_classes, input_size=input_size).to(device)
    
    # Optimizer (all parameters including auxiliaries)
    optimizer = optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay
    )
    
    # LR Scheduler
    milestones = [25, 50, 75, 100, 125, 150, 175, 200, 225, 250, 275]
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.5)
    
    # RL Controller
    state_dim = 4 + num_classes
    ppo_agent = PPOAgent(state_dim=state_dim, lr=args.rl_lr, device=device)
    
    # Current RL-controlled parameters
    current_threshold = args.init_threshold
    current_alpha3 = args.init_alpha3
    current_alpha4 = args.init_alpha4
    current_alpha5 = args.init_alpha5
    
    # Logging
    suffix = ''
    if args.brier:
        suffix += '_brier'
    if args.aurc_surrogate:
        suffix += f'_raurc_tau{args.aurc_tau}'
    if args.inverse_focal:
        suffix += f'_ifl_g{args.inverse_focal_gamma}'
    if args.focal > 0:
        suffix += f'_focal{args.focal}'
    if args.nll:
        suffix += '_nll'
    if args.aurc_hinge:
        suffix += f'_hinge{args.aurc_hinge_margin}'
    if args.dsn_decay:
        suffix += '_dsndecay'
    if args.no_rl:
        suffix += '_norl'
    if args.seed != 42:
        suffix += f'_seed{args.seed}'
    save_dir = f'./save/{args.dataset}/vgg16_bn_dsscsf{suffix}'
    os.makedirs(save_dir, exist_ok=True)
    
    logger = Logger(os.path.join(save_dir, 'log.txt'))
    logger.set_names([
        'Epoch', 'LR', 'Train_Loss', 'CE_Loss', 'Meta_Loss', 
        'Test_Err', 'AURC', 'Mean_Conf', 'Threshold', 'Alpha3', 'Alpha4', 'Alpha5'
    ])
    
    # Previous state for RL
    prev_state = None
    prev_action = None
    prev_aurc = None
    
    # ==========================================================================
    # Training Loop
    # ==========================================================================
    
    print(f'\n{"="*80}')
    print('DS-SCSF: Deeply-Supervised Self-Calibrated Selective Framework')
    print(f'{"="*80}')
    print(f'Inspired by: DSN (Lee et al., 2014) + GoogLeNet (Szegedy et al., 2015)')
    print(f'Model: VGG16_bn with 3 auxiliary branches (Pool3, Pool4, Pool5)')
    print(f'Auxiliary Decaying: {"ENABLED (DSN-style)" if args.dsn_decay else "DISABLED (fixed weights)"}')
    
    # Determine meta-loss type based on flags (primary loss)
    if args.aurc_surrogate:
        meta_loss_type = f'r-AURC Surrogate (soft rank, τ={args.aurc_tau})'
        meta_loss_desc = 'Reweighted risk targeting AURC directly'
    elif args.inverse_focal:
        meta_loss_type = f'Inverse Focal Loss (γ={args.inverse_focal_gamma})'
        meta_loss_desc = 'Simplified AURC-targeting loss'
    elif args.nll:
        meta_loss_type = 'NLL (Negative Log-Likelihood)'
        meta_loss_desc = 'Probabilistic calibration'
    elif args.brier:
        meta_loss_type = 'Brier Score (correctness)'
        meta_loss_desc = 'Binary correctness prediction'
    else:
        meta_loss_type = 'TCP (true class probability)'
        meta_loss_desc = 'Standard confidence calibration'
    
    print(f'Primary Meta-Loss: {meta_loss_type}')
    print(f'  → {meta_loss_desc}')
    
    # Print additional calibration enhancements
    calib_enhancements = []
    if args.focal > 0:
        calib_enhancements.append(f'Focal-MSE(γ={args.focal})')
    if args.margin > 0:
        calib_enhancements.append(f'Margin(w={args.margin})')
    if args.ece_reg > 0:
        calib_enhancements.append(f'ECE-Reg(w={args.ece_reg})')
    if args.label_smooth > 0:
        calib_enhancements.append(f'LabelSmooth({args.label_smooth})')
    if args.avuc > 0:
        calib_enhancements.append(f'AVUC(w={args.avuc})')
    if args.aurc_hinge:
        calib_enhancements.append(f'AURC-Hinge(m={args.aurc_hinge_margin})')
    
    if calib_enhancements:
        print(f'Additional Losses: {" + ".join(calib_enhancements)}')
    else:
        print(f'Additional Losses: None')
    
    print(f'Phase 1 (Epoch 1-{args.pretrain}): CE Warmup (Backbone + Auxiliaries)')
    if args.no_rl:
        print(f'Phase 2 (Epoch {args.pretrain+1}-{args.epochs}): Joint Training [RL DISABLED]')
    else:
        print(f'Phase 2 (Epoch {args.pretrain+1}-{args.epochs}): Joint Training + RL')
        print(f'RL tunes: Threshold τ, α3, α4, α5')
    print(f'Save Directory: {save_dir}')
    print(f'{"="*80}\n')
    
    best_aurc = float('inf')
    
    for epoch in range(1, args.epochs + 1):
        current_lr = optimizer.param_groups[0]['lr']
        
        # ======================================================================
        # Phase 1: CE Warmup
        # ======================================================================
        if epoch <= args.pretrain:
            train_loss, train_acc = train_epoch_warmup(model, trainloader, optimizer, device)
            ce_loss = train_loss
            meta_loss = 0.0
            mean_conf = 0.0
            
        # ======================================================================
        # Phase 2: Joint Training
        # ======================================================================
        else:
            # Compute DSN-style decaying weights if enabled
            if args.dsn_decay:
                alpha3 = compute_dsn_decay_weight(epoch, args.pretrain, args.epochs, current_alpha3, decay_rate=0.1)
                alpha4 = compute_dsn_decay_weight(epoch, args.pretrain, args.epochs, current_alpha4, decay_rate=0.1)
                alpha5 = compute_dsn_decay_weight(epoch, args.pretrain, args.epochs, current_alpha5, decay_rate=0.1)
            else:
                alpha3, alpha4, alpha5 = current_alpha3, current_alpha4, current_alpha5
            
            # Calibration arguments
            calib_args = {
                'error_weight': args.error_weight,  # Fixed multiplier for errors
                'error_weight_type': args.error_weight_type,  # 'fixed' or 'frequency'
                'focal': args.focal,
                'margin': args.margin,
                'ece_reg': args.ece_reg,
                'label_smooth': args.label_smooth,
                'avuc': args.avuc,
                'nll': args.nll,
                'aurc_surrogate': args.aurc_surrogate,
                'aurc_tau': args.aurc_tau,
                'inverse_focal': args.inverse_focal,
                'inverse_focal_gamma': args.inverse_focal_gamma,
                'aurc_hinge': args.aurc_hinge,
                'aurc_hinge_margin': args.aurc_hinge_margin,
            }
            
            train_loss, ce_loss, meta_loss, train_acc, mean_conf = train_epoch_joint(
                model, trainloader, optimizer, epoch, args,
                alpha3, alpha4, alpha5, device, calib_args=calib_args
            )
        
        # Step scheduler
        scheduler.step()
        
        # Evaluate
        coverage_errors, test_acc, global_aurc, test_mean_conf, test_mean_tcp = \
            evaluate_selective_risk(model, testloader_full, COVERAGE_POINTS, device)
        
        test_err = 100 - test_acc
        
        if global_aurc < best_aurc:
            best_aurc = global_aurc
        
        # ======================================================================
        # RL Controller Update
        # ======================================================================
        
        if not args.no_rl and epoch > args.pretrain and epoch % args.rl_freq == 0:
            
            val_coverage, val_acc, val_aurc, val_conf, val_tcp = \
                evaluate_selective_risk(model, valloader, COVERAGE_POINTS, device)
            
            class_acc = compute_class_accuracy(model, valloader, num_classes, device)
            
            epoch_progress = (epoch - args.pretrain) / (args.epochs - args.pretrain)
            current_state = np.concatenate([
                [val_aurc * 10, val_acc / 100, val_conf, epoch_progress],
                class_acc / 100
            ])
            
            rl_reward = 0.0
            if prev_aurc is not None:
                rl_reward = 100 * (prev_aurc - val_aurc)
            
            if prev_state is not None and prev_action is not None:
                with torch.no_grad():
                    state_tensor = torch.FloatTensor(prev_state).unsqueeze(0).to(device)
                    _, _, value = ppo_agent.actor_critic(state_tensor)
                    value = value.item()
                
                ppo_agent.store_transition(prev_state, prev_action, rl_reward, value, 0.0)
            
            # Action: [threshold, α3, α4, α5]
            action, log_prob = ppo_agent.select_action(current_state)
            
            current_threshold = float(action[0])
            current_alpha3 = float(action[1]) * 2.0
            current_alpha4 = float(action[2]) * 2.0
            current_alpha5 = float(action[3]) * 2.0
            
            if len(ppo_agent.states) >= 4:
                ppo_stats = ppo_agent.update()
                print(f'  PPO Update: Policy Loss={ppo_stats["policy_loss"]:.4f}')
            
            prev_state = current_state
            prev_action = action
            prev_aurc = val_aurc
            
            print(f'  RL: τ={current_threshold:.3f}, α3={current_alpha3:.3f}, α4={current_alpha4:.3f}, α5={current_alpha5:.3f}, reward={rl_reward:+.4f}')
        
        if not args.no_rl and epoch == args.pretrain:
            _, _, prev_aurc, _, _ = evaluate_selective_risk(
                model, valloader, COVERAGE_POINTS, device
            )
        
        # Log
        phase = '[Warmup]' if epoch <= args.pretrain else '[Joint]'
        
        # Get current fusion weights
        with torch.no_grad():
            beta_weights = model.beta_weights if hasattr(model, 'beta_weights') else (0.2, 0.3, 0.5)
            if isinstance(beta_weights, tuple):
                beta_str = f'β=({beta_weights[0]:.2f},{beta_weights[1]:.2f},{beta_weights[2]:.2f})'
            else:
                beta_str = ''
        
        print(f'{phase} Epoch {epoch:3d}/{args.epochs} | Loss: {train_loss:.4f} | '
              f'Err: {test_err:.2f}% | AURC: {global_aurc:.4f} | Conf: {test_mean_conf:.4f} {beta_str}')
        
        logger.append([
            epoch, current_lr, train_loss, ce_loss, meta_loss,
            test_err, global_aurc, test_mean_conf, current_threshold, 
            current_alpha3, current_alpha4, current_alpha5
        ])
        
        # Print coverage table
        if epoch % 20 == 0 or epoch == args.epochs:
            print(f'\n  {"Coverage":<10} | {"Error Rate":<12}')
            print(f'  {"-"*25}')
            for cov in COVERAGE_POINTS:
                print(f'  {cov:>8}% | {coverage_errors[cov]:>10.3f}%')
            print()
    
    # ==========================================================================
    # Final Evaluation
    # ==========================================================================
    
    print(f'\n{"="*80}')
    print('FINAL EVALUATION')
    print(f'{"="*80}')
    
    final_coverage, final_acc, final_aurc, final_conf, final_tcp = \
        evaluate_selective_risk(model, testloader, COVERAGE_POINTS, device)
    
    print(f'\nTotal Accuracy: {final_acc:.2f}%')
    print(f'Global AURC: {final_aurc:.4f}')
    print(f'Mean Fused Confidence: {final_conf:.4f}')
    print(f'Mean TCP: {final_tcp:.4f}')
    print(f'Final Threshold τ: {current_threshold:.4f}')
    print(f'Final Auxiliary Weights: α3={current_alpha3:.4f}, α4={current_alpha4:.4f}, α5={current_alpha5:.4f}')
    
    # Get final fusion weights
    with torch.no_grad():
        beta_sum = torch.abs(model.beta3) + torch.abs(model.beta4) + torch.abs(model.beta5)
        beta3_norm = (torch.abs(model.beta3) / beta_sum).item()
        beta4_norm = (torch.abs(model.beta4) / beta_sum).item()
        beta5_norm = (torch.abs(model.beta5) / beta_sum).item()
        print(f'Learned Fusion Weights: β3={beta3_norm:.4f}, β4={beta4_norm:.4f}, β5={beta5_norm:.4f}')
    
    print(f'\n{"Coverage":<10} | {"Error Rate":<12}')
    print(f'{"-"*25}')
    for cov in COVERAGE_POINTS:
        print(f'{cov:>8}% | {final_coverage[cov]:>10.3f}%')
    
    # Save
    save_path = os.path.join(save_dir, '300.pth')
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'ppo_state_dict': ppo_agent.actor_critic.state_dict(),
        'threshold': current_threshold,
        'alpha_weights': (current_alpha3, current_alpha4, current_alpha5),
        'beta_weights': (beta3_norm, beta4_norm, beta5_norm),
        'final_aurc': final_aurc,
        'final_coverage': final_coverage,
        'args': args,
    }, save_path)
    print(f'\nSaved to: {save_path}')
    
    csv_path = os.path.join(save_dir, 'coverage_vs_err.csv')
    with open(csv_path, 'w') as f:
        f.write('Coverage,Error\n')
        for cov in COVERAGE_POINTS:
            f.write(f'{cov},{final_coverage[cov]:.4f}\n')
    print(f'Saved coverage data to: {csv_path}')
    
    logger.close()
    
    print(f'\n{"="*80}')
    print('DS-SCSF TRAINING COMPLETE!')
    print(f'{"="*80}\n')


def compute_class_accuracy(model, loader, num_classes, device):
    """Compute per-class accuracy."""
    model.eval()
    
    class_correct = np.zeros(num_classes)
    class_total = np.zeros(num_classes)
    
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            logits = model(inputs, return_auxiliaries=False)
            _, preds = logits.max(dim=1)
            
            for i in range(len(targets)):
                c = targets[i].item()
                class_total[c] += 1
                if preds[i] == targets[i]:
                    class_correct[c] += 1
    
    class_acc = np.zeros(num_classes)
    for c in range(num_classes):
        if class_total[c] > 0:
            class_acc[c] = class_correct[c] / class_total[c] * 100
    
    return class_acc


if __name__ == '__main__':
    main()
